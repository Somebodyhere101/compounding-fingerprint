import json, os, sys
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS = ['exp1/ideal/res/ideal_R.jsonl','exp1/ideal/res/ideal_N.jsonl','exp1/ideal/res/ideal_N_iso.jsonl','exp1/ideal/res/ideal_A.jsonl',
 'exp1/ideal/res/ideal_Asl.jsonl','exp1/ideal/res/ideal_B.jsonl','exp1/ideal/res/ideal_Bh.jsonl','exp1/res/qwen3-0.6b.jsonl','exp2/res/q06b_main.jsonl',
 'exp2/res/olmo1b_main.jsonl','exp1/res/qwen3-1.7b.jsonl','exp2/res/q17b_main.jsonl','exp2/res/q17bnf4_main.jsonl','exp1/res/smollm2-1.7b.jsonl',
 'exp2/res/q4b_main.jsonl','exp2/res/olmo7b_main.jsonl','exp2/res/q8b_main.jsonl'] + [f'exp3/res/{k}.jsonl' for k in ['A1','A2','A4','A6','G4','P4','I4']]
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 10000

def load(p):
    cells = {}
    for l in open(p, encoding='utf-8'):
        r = json.loads(l)
        c = r['file'][:-6]
        d = cells.setdefault(c, {})
        # per trial: arrays of (correct,total,pg) per class
        per = {}
        for q in r['queries']:
            a = per.setdefault(q['class'], [0, 0, 0.0]); a[0] += q['correct']; a[1] += 1; a[2] += q['p_gold']
        d[r['trial']] = per
    out = {}
    for c, d in cells.items():
        out[c] = {cl: np.array([[d[t].get(cl, [0, 0, 0])[0], d[t].get(cl, [0, 0, 0])[1], d[t].get(cl, [0, 0, 0.])[2]] for t in sorted(d)], float) for cl in ['L', 'O', 'G', 'U', 'Ab', 'Cy']}
    return out

def acc(cells, c, cl, f=0, idx=None):
    a = cells[c][cl]
    if idx is not None: a = a[idx]
    return a[:, f].sum() / a[:, 1].sum() if a[:, 1].sum() else float('nan')

def rule(cells, n, pre, f=0):
    ch = 1 / n
    aL = acc(cells, f'{pre}_base_t0', 'L', f)
    if aL < ch + 0.25: return 'R', aL, None
    g = lambda c: (acc(cells, c, 'G', f) - ch) / (aL - ch)
    if g(f'{pre}_base_t0') >= 0.85: return 'B-type', aL, None
    base = max(g(f'{pre}_base_t0'), g(f'{pre}_scramble_t4'))
    comp = g(f'{pre}_base_t4') - base
    if comp < 0.15: return 'N', aL, (comp,)
    held = g(f'{pre}_heldout_t4') - base
    return ('A' if held >= 0.6 * comp else 'N_iso'), aL, (comp, held, held / comp)

def bootdiff(cells, c1, c2, rng, f=0):
    a, b = cells[c1]['G'], cells[c2]['G']
    out = np.empty(NB)
    for i in range(NB):
        ia = rng.integers(0, len(a), len(a)); ib = rng.integers(0, len(b), len(b))
        out[i] = a[ia, f].sum() / a[ia, 1].sum() - b[ib, f].sum() / b[ib, 1].sum()
    return np.percentile(out, [2.5, 97.5])

for rel in ROWS:
    p = os.path.join(R, rel)
    if not os.path.exists(p): continue
    nl = sum(1 for _ in open(p, encoding='utf-8'))
    if nl < 1800: print(rel, 'incomplete', nl); continue
    C = load(p); rng = np.random.default_rng(7)
    print(f'== {rel}')
    for c in ['n6_base_t0', 'n6_base_t1', 'n6_base_t4', 'n6_scramble_t4', 'n6_heldout_t4', 'n6_leak_t1']:
        print(f'  {c:16s} ' + ' '.join(f'{cl}={acc(C, c, cl):.3f}(n={int(C[c][cl][:,1].sum())})' for cl in ['G', 'O', 'L']))
    for pre, n in [('n6', 6), ('n8', 8)]:
        for f in [0, 2]:
            cls, aL, ex = rule(C, n, pre, f)
            print(f'  rule {pre} field{f}: {cls} accL={aL:.3f} extra={ex}')
    d = acc(C, 'n6_base_t4', 'G') - acc(C, 'n6_scramble_t4', 'G'); ci = bootdiff(C, 'n6_base_t4', 'n6_scramble_t4', rng)
    h = acc(C, 'n6_heldout_t4', 'G') - acc(C, 'n6_scramble_t4', 'G'); hci = bootdiff(C, 'n6_heldout_t4', 'n6_scramble_t4', rng)
    print(f'  t4-scr {d:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  held-scr {h:+.3f} [{hci[0]:+.3f},{hci[1]:+.3f}]')
    for pre in ['n6', 'n8']:
        d0 = acc(C, f'{pre}_base_t4', 'G') - acc(C, f'{pre}_base_t0', 'G'); c0 = bootdiff(C, f'{pre}_base_t4', f'{pre}_base_t0', rng)
        d1 = acc(C, f'{pre}_base_t4', 'G') - acc(C, f'{pre}_scramble_t4', 'G'); c1 = bootdiff(C, f'{pre}_base_t4', f'{pre}_scramble_t4', rng)
        print(f'  {pre}: G t0={acc(C,pre+"_base_t0","G"):.3f} t1={acc(C,pre+"_base_t1","G"):.3f} t4={acc(C,pre+"_base_t4","G"):.3f} scr={acc(C,pre+"_scramble_t4","G"):.3f} held={acc(C,pre+"_heldout_t4","G"):.3f} '
              f't4-t0 {d0:+.3f} [{c0[0]:+.3f},{c0[1]:+.3f}] t4-scr {d1:+.3f} [{c1[0]:+.3f},{c1[1]:+.3f}] Lcells/cell=' + ','.join(str(int(C[c]['L'][:,1].sum())) for c in sorted(C) if c.startswith(pre)))
