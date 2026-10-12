"""Conditions (iii) A->B->C curriculum and (iv) prior in weights.

(iii) n=6: history = 2 tables from all groups (Z6/S3, iso-uniform) then 2 abelian tables (Z6);
      target C = S3 (non-abelian group) or a group-like non-associative loop (top-50 by associativity rate).
      Learners: N, Niso, A_cum (axioms over all 4 tables), A_rec (axioms over the last 2 tables only),
      Bh (hierarchical Bayes, fixed environment), Bh_cp (hierarchical Bayes that resets at the phase
      boundary, i.e. uses the last 2 tables only).
(iv)  n=5: prior in weights.  A_w = rigid group prior regardless of context (loop/Latin fallback when the
      cue contradicts it); Bh_w = hierarchical Bayes whose hyperprior favours group-type environments by
      2^10 : 1 and which IS updated by in-context tables; N_w = notebook (weights hold memorised tables, useless
      under fresh labels) = N.  History = t non-associative loops (or t groups); target = Z5 or a loop.

usage: python abc_inweights.py abc M      -> ../out/abc_n6.json
       python abc_inweights.py w M        -> ../out/inweights_n5.json
"""
import sys, json, math, itertools
import numpy as np
from collections import defaultdict
from sim import (Setup, analyze_cue, classify_cells, learner_preds, poss_acc, map_acc, union_counts, feas,
                 latin_propagation, CELLCLS)
from tables import groups_of_order, relabel
from evidence import bh_acc


def grouplike_loops6(top=50):
    from stimuli import grouplike_loops
    return [X for (_, _, X) in grouplike_loops(6, top)]


def run_abc(M, seed=7):
    n = 6
    st = Setup(n); st.seen = st.H
    rng = np.random.default_rng(seed)
    loops = grouplike_loops6()
    targets = [('grp', 'S3', groups_of_order(6)['S3'])] + [('lp6', f'loop{j}', loops[j]) for j in range(0, 50, 10)]
    acc = defaultdict(float); cnt = defaultdict(float)
    for (kind, name, T) in targets:
        for k in range(0, n * n, 2):
            for rep in range(M):
                S = np.sort(rng.choice(n * n, size=k, replace=False))
                unseen, vstar, R = analyze_cue(st, T, S, need_lp=False)
                cls = classify_cells(st, R, name if kind == 'grp' else None, 'grp' if kind == 'grp' else 'lp')
                cache = {'prop': latin_propagation(T, S)[unseen]}
                tg = 'grp' if kind == 'grp' else 'lp'
                for a1, a2 in itertools.product(st.H, repeat=2):
                    p = 0.25
                    hist = (('grp', a1), ('grp', a2), ('grp', 'Z6'), ('grp', 'Z6'))
                    if tg == 'lp':
                        # B (true prior) undefined for n=6 loops: use dummy target kind 'grp' for static learners
                        pass
                    full = learner_preds(st, R, vstar, hist, 'grp', cache)
                    rec = learner_preds(st, R, vstar, hist[2:], 'grp', {'prop': cache['prop']})
                    out = {'N': full['N'], 'Niso': full['Niso'], 'A_cum': full['A'], 'A_rec': rec['A'],
                           'Bh': full['Bh'], 'Bh_cp': rec['Bh']}
                    for L, a in out.items():
                        for cc in CELLCLS + ['all']:
                            msk = (cls == cc) if cc != 'all' else np.ones(len(a), bool)
                            if msk.any():
                                acc[(name, L, k, cc)] += p * a[msk].sum()
                    for cc in CELLCLS + ['all']:
                        msk = (cls == cc) if cc != 'all' else np.ones(len(vstar), bool)
                        cnt[(name, k, cc)] += p * msk.sum()
            print('abc', name, k, flush=True)
    res = {'|'.join(map(str, kk)): v / max(cnt[(kk[0], kk[2], kk[3])], 1e-12) for kk, v in acc.items()}
    res_cnt = {'|'.join(map(str, kk)): v / M for kk, v in cnt.items()}
    json.dump({'acc': res, 'cnt': res_cnt, 'M': M}, open('../out/abc_n6.json', 'w'))


def run_w(M, seed=8):
    n = 5
    st = Setup(n); st.seen = st.H
    rng = np.random.default_rng(seed)
    targets = [('grp', 'Z5', groups_of_order(5)['Z5'])] + [('lp', j, st.U[j][0].reshape(n, n)) for j in st.lp]
    acc = defaultdict(float); cnt = defaultdict(float)
    for (kind, name, T) in targets:
        for k in range(0, n * n):
            for rep in range(M):
                S = np.sort(rng.choice(n * n, size=k, replace=False))
                unseen, vstar, R = analyze_cue(st, T, S, need_lp=True)
                cls = classify_cells(st, R, name, kind)
                allG = ['iso:' + h for h in st.H]
                NG, cG = union_counts(R, allG)
                Fw = (cG > 0) if NG > 0 else feas(R, 'loop')
                aw = poss_acc(Fw, vstar)
                for hk in ('loops', 'groups'):
                    for t in range(0, 5):
                        if hk == 'loops':
                            hs = list(itertools.combinations_with_replacement(st.lp, t))
                            probs = []
                            for h in hs:
                                # multinomial probability of the multiset under uniform choice of 5 classes
                                from collections import Counter
                                c = Counter(h); m = math.factorial(t)
                                for v in c.values(): m //= math.factorial(v)
                                probs.append(m / len(st.lp) ** t)
                            hist_list = [(p, tuple(('lp', j) for j in h)) for p, h in zip(probs, hs)]
                        else:
                            hist_list = [(1.0, tuple(('grp', 'Z5') for _ in range(t)))]
                        for p, hist in hist_list:
                            bw = bh_acc(st, R, vstar, hist, -10)
                            out = {'A_w': aw, 'Bh_w': bw}
                            for L, a in out.items():
                                for cc in CELLCLS + ['all']:
                                    msk = (cls == cc) if cc != 'all' else np.ones(len(a), bool)
                                    if msk.any():
                                        acc[(name, hk, L, t, k, cc)] += p * a[msk].sum()
                            for cc in CELLCLS + ['all']:
                                msk = (cls == cc) if cc != 'all' else np.ones(len(vstar), bool)
                                cnt[(name, hk, t, k, cc)] += p * msk.sum()
            print('w', name, k, flush=True)
    res = {'|'.join(map(str, kk)): v / max(cnt[(kk[0], kk[1], kk[3], kk[4], kk[5])], 1e-12) for kk, v in acc.items()}
    json.dump({'acc': res, 'M': M}, open('../out/inweights_n5.json', 'w'))


if __name__ == '__main__':
    if sys.argv[1] == 'abc':
        run_abc(int(sys.argv[2]))
    else:
        run_w(int(sys.argv[2]))
