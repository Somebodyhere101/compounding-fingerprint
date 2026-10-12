"""Cross-check SAT feasibility masks against enumeration (n=4,5 Latin; n=4,5,6 loops).
usage: python test_engine.py"""
import numpy as np, time
from engine import universe, analyze, loop_iso_classes, sat_solver, counts_for
from tables import groups_of_order, all_loops
rng = np.random.default_rng(0)
print('non-assoc loop iso classes n=5:', [len(c) for c in loop_iso_classes(5)])
for n in (4, 5, 6):
    U = universe(n)
    bad = 0; tot = 0; t0 = time.time()
    for H, T in groups_of_order(n).items():
        for trial in range(40):
            k = int(rng.integers(0, n * n))
            S = np.sort(rng.choice(n * n, size=k, replace=False))
            kinds = ['loop'] + (['latin'] if n <= 5 else [])
            for kind in kinds:
                N, cnt = counts_for(U[kind], T, S)
                F = sat_solver(n, kind == 'loop').feasible(T, S)
                tot += 1
                if not np.array_equal(cnt > 0, F):
                    bad += 1
    print(f'n={n}: {tot} cue/class checks, mismatches={bad} ({time.time()-t0:.1f}s)')
