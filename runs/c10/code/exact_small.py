"""Exhaustive (all cue positions) check of selected Monte-Carlo numbers.

For order n and cue size k, enumerate ALL C(n^2, k) cue position sets for each target iso class and
compute the exact env-averaged accuracy per learner and cell class for t = 0..2 (baseline condition)
and for the held-out condition.

usage: python exact_small.py n k     -> prints table, writes ../out/exact_n{n}_k{k}.json
"""
import sys, json, itertools
import numpy as np
from collections import defaultdict
from sim import Setup, analyze_cue, classify_cells, learner_preds, latin_propagation, histories, CELLCLS
from tables import groups_of_order


def main(n, k, tmax=2):
    st = Setup(n)
    st.seen = {4: ['Z4'], 6: ['Z6']}.get(n, st.H)
    acc = defaultdict(float); cnt = defaultdict(float)
    ncue = 0
    for name, T in groups_of_order(n).items():
        conds = ['base'] + (['heldout'] if name not in st.seen else [])
        for S in itertools.combinations(range(n * n), k):
            S = np.array(S)
            unseen, vstar, R = analyze_cue(st, T, S, need_lp=False)
            cls = classify_cells(st, R, name, 'grp')
            cache = {'prop': latin_propagation(T, S)[unseen]}
            for cond in conds:
                for t in range(tmax + 1):
                    for p, hist in histories(st, cond, t):
                        pr = learner_preds(st, R, vstar, hist, 'grp', cache)
                        for L, a in pr.items():
                            if L.startswith('_'):
                                continue
                            for cc in CELLCLS + ['all']:
                                msk = (cls == cc) if cc != 'all' else np.ones(len(a), bool)
                                if msk.any():
                                    acc[(cond, name, L, t, cc)] += p * a[msk].sum()
                        for cc in CELLCLS + ['all']:
                            msk = (cls == cc) if cc != 'all' else np.ones(len(vstar), bool)
                            cnt[(cond, name, t, cc)] += p * msk.sum()
            ncue += 1
    res = {'|'.join(map(str, key)): v / cnt[(key[0], key[1], key[3], key[4])] for key, v in acc.items() if cnt[(key[0], key[1], key[3], key[4])] > 0}
    json.dump(res, open(f'../out/exact_n{n}_k{k}.json', 'w'), indent=0)
    print(f'n={n} k={k}: {ncue} cue sets enumerated (all targets)')
    for cond in ('base', 'heldout'):
        names = sorted({key[1] for key in acc if key[0] == cond})
        for L in ('N', 'Niso', 'A', 'B', 'Bh'):
            for cc in ('O', 'G', 'all'):
                vals = []
                for t in range(tmax + 1):
                    num = sum(acc.get((cond, x, L, t, cc), 0.0) for x in names)
                    den = sum(cnt.get((cond, x, t, cc), 0.0) for x in names)
                    vals.append(num / den if den else float('nan'))
                print(f'  {cond:8s} {L:5s} {cc:4s} t=0..{tmax}: ' + ', '.join(f'{v:.4f}' for v in vals))


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
