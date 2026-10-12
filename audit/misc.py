import json, os, glob
from collections import defaultdict, Counter
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
stim = {}
for d in ['exp1/stim', 'exp2/stim']:
    for p in glob.glob(os.path.join(R, d, '*.jsonl')):
        for l in open(p, encoding='utf-8'):
            s = json.loads(l); stim[(os.path.basename(p), s['trial'])] = s
# 1. G cells: gold in query words?
cnt = Counter()
for (f, t), s in stim.items():
    tw = s['target_words']
    for q in s['queries']:
        a, b = tw[q['cell'][0]], tw[q['cell'][1]]
        assert q['prompt_suffix'].strip() == f'{a} * {b} =', (q['prompt_suffix'], a, b)
        cnt[(q['class'], q['gold'] in (a, b))] += 1
print('class x gold-in-query', sorted(cnt.items()))
# 2. copy bias on G cells for exp1 models
for m in ['qwen3-0.6b', 'qwen3-1.7b', 'smollm2-1.7b']:
    per = defaultdict(lambda: [0, 0])
    for l in open(os.path.join(R, 'exp1/res', m + '.jsonl'), encoding='utf-8'):
        r = json.loads(l); s = stim[(r['file'], r['trial'])]; tw = s['target_words']
        for q in r['queries']:
            if q['class'] != 'G': continue
            a, b = tw[q['cell'][0]], tw[q['cell'][1]]
            per[r['file'][:-6]][0] += q['pred'] in (a, b); per[r['file'][:-6]][1] += 1
    tot = [sum(v[0] for v in per.values()), sum(v[1] for v in per.values())]
    n6 = [sum(v[0] for k, v in per.items() if k.startswith('n6')), sum(v[1] for k, v in per.items() if k.startswith('n6'))]
    print(m, 'copy-rate G all=%.3f n6=%.3f' % (tot[0] / tot[1], n6[0] / n6[1]), 'range over cells %.3f-%.3f' % (min(v[0] / v[1] for v in per.values()), max(v[0] / v[1] for v in per.values())),
          {k: round(v[0] / v[1], 3) for k, v in sorted(per.items())})
# 3. ctx_tokens base_t4 vs scramble_t4 and fact counts from stimuli
for m in ['exp1/res/qwen3-0.6b.jsonl', 'exp1/res/smollm2-1.7b.jsonl']:
    tok = defaultdict(list)
    for l in open(os.path.join(R, m), encoding='utf-8'):
        r = json.loads(l); tok[r['file'][:-6]].append(r['ctx_tokens'])
    print(m, {k: round(float(np.mean(v)), 1) for k, v in sorted(tok.items()) if '_t4' in k})
for pre in ['n6', 'n8']:
    for c in ['base_t4', 'scramble_t4', 'heldout_t4']:
        lens = [len(stim[(f'{pre}_{c}.jsonl', t)]['context'].split('\n')) for t in range(100)]
        print(pre, c, 'context lines mean', np.mean(lens), 'chars', np.mean([len(stim[(f'{pre}_{c}.jsonl', t)]['context']) for t in range(100)]))
# 4. calibration design
for f in sorted({k[0] for k in stim if not k[0].startswith('n')}):
    c = Counter((stim[(f, t)]['t'], stim[(f, t)]['k']) for t in range(100))
    print(f, dict(c))
# 5. word reuse across tables (exp1)
viol = Counter()
for (f, t), s in stim.items():
    if not f.startswith('n'): continue
    blocks = s['context'].split('\n\n')
    ws = []
    for b in blocks:
        w = set()
        for line in b.split('\n')[1:]:
            if '=' in line:
                lhs, rhs = line.split('='); a, _, b2 = lhs.split(); w |= {a, b2, rhs.strip()}
        ws.append(w)
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            if ws[i] & ws[j]: viol[f] += 1
print('word reuse across tables', dict(viol))
