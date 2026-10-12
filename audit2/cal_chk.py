import json, sys, os, glob, collections
import numpy as np
R = sys.argv[1]
STIM = {os.path.basename(f): {x['trial']: x for x in map(json.loads, open(f, encoding='utf-8'))} for f in glob.glob(os.path.join(R, 'exp2/stim/*.jsonl'))}
# check ks vs kso are same trials (same queries/golds/target words)
for k in (16, 22, 28):
    a, b = STIM[f'ks_n6_k{k}.jsonl'], STIM[f'kso_n6_k{k}.jsonl']
    same = all(a[t]['target_words'] == b[t]['target_words'] and [q['gold'] for q in a[t]['queries']] == [q['gold'] for q in b[t]['queries']]
               and sorted(a[t]['context'].split('\n\n')[-1].split('\n')) == sorted(b[t]['context'].split('\n\n')[-1].split('\n')) and a[t]['t'] == b[t]['t'] for t in a)
    print('ks vs kso same trials k=%d:' % k, same, len(a), len(b))
rng = np.random.default_rng(5)
names = {'q06b': 'Qwen3-0.6B-Base', 'q17': 'Qwen3-1.7B', 'smol17': 'SmolLM2-1.7B', 'q17b': 'Qwen3-1.7B-Base', 'q4b': 'Qwen3-4B-Base', 'q8b': 'Qwen3-8B-Base', 'olmo1b': 'OLMo-2-1B', 'olmo7b': 'OLMo-2-7B'}
instr_all = {}
for m, nm in names.items():
    recs = [json.loads(l) for l in open(os.path.join(R, f'exp2/res/{m}_cal.jsonl'), encoding='utf-8')]
    prob = collections.Counter(); seen = set()
    by = collections.defaultdict(list)
    for r in recs:
        key = (r['file'], r['trial'])
        if key in seen: prob['dup'] += 1
        seen.add(key)
        s = STIM[r['file']][r['trial']]
        if [ (q['cell'], q['gold'], q['class']) for q in r['queries']] != [(q['cell'], q['gold'], q['class']) for q in s['queries']]: prob['qmismatch'] += 1
        for q in r['queries']:
            if q['correct'] != int(q['pred'] == q['gold']): prob['corr'] += 1
        var = r['file'].split('_')[0]
        per = collections.defaultdict(lambda: [0, 0])
        for q in r['queries']:
            per[q['class']][0] += q['correct']; per[q['class']][1] += 1
        by[(var, r['k'], r['t'])].append(dict(per))
    for f, st in STIM.items():
        n = sum(1 for k in seen if k[0] == f)
        if n != len(st): prob['missing_' + f] = len(st) - n
    quants = collections.Counter(r.get('quant') for r in recs); models = collections.Counter(r['model'] for r in recs)
    def pool(var, k, cls, ts=(0, 4)):
        tr = [x for t in ts for x in by.get((var, k, t), [])]
        c = sum(x[cls][0] for x in tr if cls in x); n = sum(x[cls][1] for x in tr if cls in x)
        return c / n if n else float('nan'), n, tr
    raw = {k: pool('ks', k, 'L')[0] for k in (10, 16, 22, 28)}
    srt = {k: pool('kso', k, 'L')[0] for k in (16, 22, 28)}
    # bootstrap k28 - k10 (raw, L)
    _, _, t28 = pool('ks', 28, 'L'); _, _, t10 = pool('ks', 10, 'L')
    def pa(tr, idx):
        c = sum(tr[i]['L'][0] for i in idx if 'L' in tr[i]); n = sum(tr[i]['L'][1] for i in idx if 'L' in tr[i]); return c / n
    bs = [pa(t28, rng.integers(0, len(t28), len(t28))) - pa(t10, rng.integers(0, len(t10), len(t10))) for _ in range(2000)]
    # sorted minus raw at k28 (paired trials)
    _, _, s28 = pool('kso', 28, 'L')
    idxs = [rng.integers(0, len(t28), len(t28)) for _ in range(2000)]
    bsr = [pa(s28, i) - pa(t28, i) for i in idxs]
    # monotonic trend: slope of L acc on k
    print(f"{nm:16s} integ={dict(prob)} n={len(recs)} quant={dict(quants)} model={list(models)}")
    print('   raw L by k: ' + ' '.join(f'{k}:{v:.3f}' for k, v in raw.items()) + f"   k28-k10 = {raw[28]-raw[10]:+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]")
    print('   sorted L by k: ' + ' '.join(f'{k}:{v:.3f}' for k, v in srt.items()) + f"   sorted-raw@28 = {srt[28]-raw[28]:+.3f} [{np.percentile(bsr,2.5):+.3f},{np.percentile(bsr,97.5):+.3f}]" + f"  sorted-raw@16 {srt[16]-raw[16]:+.3f} @22 {srt[22]-raw[22]:+.3f}")
    il = []
    for k in (10, 22):
        for cls in ('L', 'G', 'O'):
            a, na, _ = pool('ki', k, cls); b, nb, _ = pool('ks', k, cls)
            il.append((k, cls, a - b, na, nb))
    instr_all[nm] = il
    print('   instr-raw pooled t: ' + '; '.join(f'k{k} {c}: {d:+.3f} (n={na})' for k, c, d, na, nb in il))
print()
for thr in (50, 30, 1):
    vals = [(nm, k, c, d) for nm, il in instr_all.items() for k, c, d, na, nb in il if c in ('L', 'G') and na >= thr and nb >= thr]
    print(f'instr-raw L/G pooled, n>={thr}: count={len(vals)} min={min(v[3] for v in vals):+.3f} max={max(v[3] for v in vals):+.3f} mean={np.mean([v[3] for v in vals]):+.4f}')
    six = [v for v in vals if v[0] in ('Qwen3-0.6B-Base','Qwen3-1.7B-Base','Qwen3-4B-Base','Qwen3-8B-Base','OLMo-2-1B','OLMo-2-7B')]
    print(f'    6 exp2 models only: count={len(six)} min={min(v[3] for v in six):+.3f} max={max(v[3] for v in six):+.3f} mean={np.mean([v[3] for v in six]):+.4f}')
