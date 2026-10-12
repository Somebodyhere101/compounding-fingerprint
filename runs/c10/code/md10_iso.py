"""Relabelling-only minimum defining sets at order 10 (class = Iso(T) only), fixed-k feasibility.
usage: python md10_iso.py  -> ../out/md_n10_iso.json"""
import json, time
import numpy as np
from tables import cyclic, closure, from_perms
from md_extra import labelled_coset_reps
from md10_fast import solve_fixed_k
D5 = from_perms(closure([(1, 2, 3, 4, 0), (0, 4, 3, 2, 1)]))
res = {}
for h, T in {'Z10': cyclic(10), 'D5': D5}.items():
    C = labelled_coset_reps(T)[0]
    for k in range(3, 8):
        t0 = time.time()
        S, r, st = solve_fixed_k(T, C, k, init=20000)
        print(f'{h} (Iso only, {len(C)} tables): k={k} -> {st} ({r} rounds, {time.time()-t0:.0f}s) cue={S}', flush=True)
        if S is not None:
            m = np.ones(len(C), bool)
            for a, b, v in S:
                m &= C[:, a, b] == v
            res[h] = {'md_iso': k, 'cue': S, 'consistent_tables': int(m.sum())}
            break
json.dump(res, open('../out/md_n10_iso.json', 'w'), indent=1)
print(res)
