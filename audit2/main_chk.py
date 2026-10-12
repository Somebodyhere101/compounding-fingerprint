"""Independent recomputation of main-cell numbers (audit2). Does not import project code."""
import json, os, sys, glob, collections
import numpy as np
R = sys.argv[1]
STIM = {os.path.basename(f): [json.loads(l) for l in open(f, encoding='utf-8')] for f in glob.glob(os.path.join(R, 'exp1/stim/*.jsonl'))}
MODELS = [('Qwen3-0.6B','exp1/res/qwen3-0.6b.jsonl'),('Qwen3-0.6B-Base','exp2/res/q06b_main.jsonl'),('OLMo-2-1B','exp2/res/olmo1b_main.jsonl'),
 ('Qwen3-1.7B','exp1/res/qwen3-1.7b.jsonl'),('Qwen3-1.7B-Base','exp2/res/q17b_main.jsonl'),('Qwen3-1.7B-Base-nf4','exp2/res/q17bnf4_main.jsonl'),
 ('SmolLM2-1.7B','exp1/res/smollm2-1.7b.jsonl'),('Qwen3-4B-Base','exp2/res/q4b_main.jsonl'),('OLMo-2-7B','exp2/res/olmo7b_main.jsonl'),('Qwen3-8B-Base','exp2/res/q8b_main.jsonl')] + \
 [(k, f'exp3/res/{k}.jsonl') for k in ['A1','A2','A4','A6','G4','P4','I4']]
only = sys.argv[2:] 
def integrity(recs, lab):
    probs = collections.Counter()
    seen = set()
    for r in recs:
        key = (r['file'], r['trial'])
        if key in seen: probs['dup'] += 1
        seen.add(key)
        st = STIM[r['file']]
        s = [x for x in st if x['trial'] == r['trial']]
        if len(s) != 1: probs['nostim'] += 1; continue
        s = s[0]
        if len(s['queries']) != len(r['queries']): probs['nq'] += 1; continue
        for q, sq in zip(r['queries'], s['queries']):
            if q['cell'] != sq['cell'] or q['gold'] != sq['gold'] or q['class'] != sq['class']: probs['qmismatch'] += 1
            if q['correct'] != int(q['pred'] == q['gold']): probs['correct_inconsistent'] += 1
            if q['pred'] not in sq['candidates']: probs['pred_not_cand'] += 1
            if not (0 <= q['p_gold'] <= 1.00001): probs['pgold_range'] += 1
            if q['correct'] and q['p_gold'] < 1.0/len(sq['candidates']) - 1e-4: probs['correct_but_low_p'] += 1
    for f, st in STIM.items():
        n = sum(1 for k in seen if k[0] == f)
        if n != len(st): probs[f'missing_{f}'] = len(st) - n
    return dict(probs)

def load(recs):
    cells = collections.defaultdict(list)
    for r in recs:
        per = {}
        for q in r['queries']:
            a = per.setdefault(q['class'], [0, 0, 0.0]); a[0] += q['correct']; a[1] += 1; a[2] += q['p_gold']
        cells[r['file'][:-6]].append(per)
    return cells

def acc(trials, cls, idx=None, f=0):
    idx = range(len(trials)) if idx is None else idx
    num = 0.0; den = 0
    for i in idx:
        v = trials[i].get(cls)
        if v: num += v[f]; den += v[1]
    return num / den if den else float('nan')

def rule(cells, pre, n, f=0, sel=None):
    need = ['base_t0','base_t1','base_t4','scramble_t4','heldout_t4']
    sel = sel or {}
    ch = 1/n
    aL = acc(cells[f'{pre}_base_t0'], 'L', sel.get(f'{pre}_base_t0'), f)
    G = {c: acc(cells[f'{pre}_{c}'], 'G', sel.get(f'{pre}_{c}'), f) for c in need}
    g = {c: (G[c]-ch)/(aL-ch) if aL-ch > 1e-9 else float('nan') for c in need}
    base = max(g['base_t0'], g['scramble_t4']); comp = g['base_t4']-base; held = g['heldout_t4']-base
    if aL < ch + 0.25: cls = 'R'
    elif g['base_t0'] >= 0.85: cls = 'B-type'
    elif comp < 0.15: cls = 'N'
    elif held >= 0.6*comp: cls = 'A'
    else: cls = 'N_iso'
    return cls, aL, G, comp, held, g['base_t4']-g['base_t1']

def boot(cells, fn, cellnames, nb=2000, seed=99):
    rng = np.random.default_rng(seed); out = []
    for _ in range(nb):
        sel = {c: rng.integers(0, len(cells[c]), len(cells[c])) for c in cellnames}
        out.append(fn(sel))
    return np.nanpercentile(out, 2.5), np.nanpercentile(out, 97.5)

res = {}
for lab, rel in MODELS:
    if only and lab not in only: continue
    p = os.path.join(R, rel)
    recs = [json.loads(l) for l in open(p, encoding='utf-8')]
    integ = integrity(recs, lab)
    cells = load(recs)
    out = {'integrity': integ, 'ntrials': len(recs)}
    for pre, n in [('n6', 6), ('n8', 8)]:
        for f in (0, 2):
            cls, aL, G, comp, held, shape = rule(cells, pre, n, f)
            out[f'{pre}_f{f}'] = dict(cls=cls, accL=aL, G=G, comp=comp, held=held, shape=shape)
        c4, cs, c0, ch = f'{pre}_base_t4', f'{pre}_scramble_t4', f'{pre}_base_t0', f'{pre}_heldout_t4'
        d = lambda a, b: (lambda sel: acc(cells[a], 'G', sel[a]) - acc(cells[b], 'G', sel[b]))
        for nm, a, b in [('t4_scr', c4, cs), ('held_scr', ch, cs), ('t4_t0', c4, c0)]:
            pt = acc(cells[a], 'G') - acc(cells[b], 'G')
            lo, hi = boot(cells, d(a, b), [a, b])
            out[f'{pre}_{nm}'] = (pt, lo, hi)
    out['leak_G'] = acc(cells['n6_leak_t1'], 'G')
    out['G_by_cell'] = {c: acc(cells[c], 'G') for c in sorted(cells)}
    out['accL_t0_n6_CIupper'] = None
    res[lab] = out
    print(lab, len(recs), 'integrity', integ)
    for pre in ('n6', 'n8'):
        a, b = out[f'{pre}_f0'], out[f'{pre}_f2']
        print(f"  {pre}: class acc={a['cls']} pg={b['cls']} accL={a['accL']:.3f} pgL={b['accL']:.3f} G t0={a['G']['base_t0']:.3f} t1={a['G']['base_t1']:.3f} t4={a['G']['base_t4']:.3f} scr={a['G']['scramble_t4']:.3f} held={a['G']['heldout_t4']:.3f}")
        for nm in ('t4_scr', 'held_scr', 't4_t0'):
            v = out[f'{pre}_{nm}']; print(f"     {nm}: {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]")
    print(f"  leak G {out['leak_G']:.3f}")
json.dump(res, open(os.path.join(R, 'audit2', 'main_res.json' if not only else 'main_res_part.json'), 'w'), indent=1, default=float)
