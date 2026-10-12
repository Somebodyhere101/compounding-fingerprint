"""Missing-mass (coverage) law for the exemplar learner N_iso vs the abstraction learner A.

Prediction: acc_G(Niso, t) = c_t + (1 - c_t) * a0,  c_t = 1 - (1 - 1/m)^t  (target type covered by the history),
a0 = acc_G of the fallback (t=0) learner; acc_G(A, t>=1) = 1.  Gap(A - Niso) = (1-1/m)^t (1 - a0) = expected
missing mass of the target's type x (1 - a0).  Compared with the simulated env-averaged G-cell accuracy.
usage: python coverage.py
"""
from report_tables import load, env_curve
for n, kd, cc in ((6, 10, 'G'), (8, 16, 'G'), (4, 4, 'O')):
    d = load(n)
    if d is None:
        continue
    m = {4: 2, 6: 2, 8: 5}[n]
    ks, c0 = env_curve(d, 'base', 'Niso', 0, cc)
    a0 = c0[ks.index(kd)]
    print(f'n={n} k={kd} class={cc} m={m} a0={a0:.3f}')
    for t in range(5):
        ks, c = env_curve(d, 'base', 'Niso', t, cc)
        ks, cA = env_curve(d, 'base', 'A', t, cc)
        ct = 1 - (1 - 1 / m) ** t
        pred = ct + (1 - ct) * a0
        print(f'   t={t}: Niso sim={c[ks.index(kd)]:.3f}  law={pred:.3f}   A sim={cA[ks.index(kd)]:.3f}   missing mass={(1-1/m)**t:.3f}')
