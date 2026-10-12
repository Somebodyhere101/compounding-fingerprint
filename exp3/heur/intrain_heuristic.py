"""Post-audit diagnostic: accuracy of the global count heuristic and of row/column elimination on the *in-training*
evaluation conditions (teacher-forced: remaining target facts are revealed one by one), using the training generator
and the exact eval_conditions/EVAL_K of train.py. Compare with eval.csv of the trained runs."""
import sys, os, zlib, json
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import train as T
from data import Gen
a = T.build_args(['--name', 'x', '--types', 'noheld', '--env_mix', 'group=0.5,latin=0.25,loop=0.25', '--orders', '4', '5', '6', '7', '8', '10'])
dcfg, _ = T.make_cfgs(a)[:2]
gen = Gen(dcfg)
res = {}
for name, n, t, hist, tgt in T.eval_conditions(dcfg):
    if tgt == 'isocopy':
        continue
    rng = np.random.default_rng([999, n, t, zlib.crc32(name.encode())]); k = T.EVAL_K[n]
    d = gen.make(n, t, 64, rng, hist=hist, tgt=tgt, k=k)
    tok, P, S = d['tok'], d['tgt_pos'], d['tgt_sym']
    cnt_acc, rc_acc, q1c, q1r = [], [], [], []
    for b in range(tok.shape[0]):
        syms = list(S[b]); res_seen = []; row = {}; col = {}
        for j, p in enumerate(P):
            x, y, z = tok[b, p - 4], tok[b, p - 2], tok[b, p]
            if j >= k:
                c = {s: res_seen.count(s) for s in syms}; m = min(c.values()); low = [s for s in syms if c[s] == m]
                feas = [s for s in syms if s not in row.get(x, set()) and s not in col.get(y, set())]
                ca, ra = (z in low) / len(low), (z in feas) / max(1, len(feas))
                cnt_acc.append(ca); rc_acc.append(ra)
                if j == k:
                    q1c.append(ca); q1r.append(ra)
            res_seen.append(z); row.setdefault(x, set()).add(z); col.setdefault(y, set()).add(z)
    res[name] = dict(count_all=np.mean(cnt_acc), rowcol_all=np.mean(rc_acc), count_q1=np.mean(q1c), rowcol_q1=np.mean(q1r))
    print(f'{name:12s} n={n} t={t}: count heuristic acc {np.mean(cnt_acc):.3f} (q1 {np.mean(q1c):.3f}); row/col elimination {np.mean(rc_acc):.3f} (q1 {np.mean(q1r):.3f})')
json.dump(res, open(os.path.join(HERE, 'heur', 'intrain_heuristic.json'), 'w'), indent=1, default=float)
