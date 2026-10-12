"""'Loop learner' (identity + Latin/cancellation, NO associativity): possibilistic accuracy per cell class
at the design cue size, env-averaged over group targets.  This is the predicted fingerprint of a model
that has the In-Context-Algebra mechanisms (identity recognition, cancellation) but not associativity.
usage: python loop_learner.py n k M
"""
import sys
import numpy as np
from sim import Setup, analyze_cue, classify_cells, poss_acc, feas
from tables import groups_of_order
n, k, M = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
st = Setup(n); rng = np.random.default_rng(21)
tot = {}; cnt = {}
for h, T in groups_of_order(n).items():
    for rep in range(M):
        S = np.sort(rng.choice(n * n, size=k, replace=False))
        unseen, vstar, R = analyze_cue(st, T, S, need_lp=False)
        cls = classify_cells(st, R, h, 'grp')
        a = poss_acc(feas(R, 'loop'), vstar)
        for cc in ('L', 'O', 'G', 'all'):
            msk = (cls == cc) if cc != 'all' else np.ones(len(a), bool)
            tot[cc] = tot.get(cc, 0) + a[msk].sum(); cnt[cc] = cnt.get(cc, 0) + msk.sum()
print(f'n={n} k={k} M={M} loop-learner accuracy: ' + ', '.join(f'{cc}={tot[cc]/max(cnt[cc],1):.3f} (cells {cnt[cc]/(M*len(groups_of_order(n))):.2f}/trial)' for cc in tot))
