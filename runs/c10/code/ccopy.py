"""Commutative-copy probe (n=6, S3 target).

Probe cells: unseen (a,b) such that (b,a) is in the cue and T(a,b) != T(b,a).  A learner that has induced
commutativity answers T(b,a) there ('commutative-copy error').  We report, per learner and history type,
P(answer = T(a,b)) [accuracy] and P(answer = T(b,a)) [copy-error rate] on probe cells.

Histories (n=6): 'abel'  = t tables of Z6 only;  'mixed' = t tables iid from {Z6,S3};
                 'abc'   = 2 tables iid from {Z6,S3} then 2 Z6 tables (t=4).
Learners: N (Latin), Niso (exemplar), A (axioms over all history), A_rec (axioms over last 2 tables),
          Bh (hierarchical Bayes, uniform hyperprior over loop/group/abelian/cyclic), Bh_rec (last 2 tables).
usage: python ccopy.py M   -> ../out/ccopy_n6.json
"""
import sys, json, math, itertools
import numpy as np
from collections import defaultdict
from sim import Setup, analyze_cue, union_counts, feas
from tables import groups_of_order


def decision(P):
    """possibilistic/MAP decision distribution from a nonneg matrix P [u, n] (uniform over argmax / support)."""
    mx = P.max(1, keepdims=True)
    tie = np.isclose(P, mx, rtol=1e-9, atol=1e-300) & (mx > 0)
    return tie / np.maximum(tie.sum(1, keepdims=True), 1)


def learner_class_dist(st, R, hist):
    """decision distributions for N, Niso, A, Bh given history."""
    allG = ['iso:' + h for h in st.H]; abG = ['iso:' + h for h in st.Hab]; cyG = ['iso:' + h for h in st.Hcyc]
    Fl = feas(R, 'latin').astype(float)
    out = {'N': Fl / Fl.sum(1, keepdims=True)}
    types = sorted(set(hist))
    keys = ['iso:' + nm for (_, nm) in types]
    N, c = union_counts(R, keys) if keys else (0, None)
    F = (c > 0).astype(float) if N > 0 else Fl
    out['Niso'] = F / F.sum(1, keepdims=True)
    ax = {a: all(st.type_axioms(tp)[a] for tp in hist) for a in ('assoc', 'comm', 'cyc')} if hist else None
    chain = []
    if ax and ax['assoc']:
        if ax['cyc']: chain.append(cyG)
        if ax['comm']: chain.append(abG)
        chain.append(allG)
    F = None
    for kk in chain:
        N, c = union_counts(R, kk)
        if N > 0:
            F = (c > 0).astype(float); break
    if F is None:
        Fo = feas(R, 'loop').astype(float) if hist else None
        F = Fo if Fo is not None else Fl
    out['A'] = F / F.sum(1, keepdims=True)
    # Bh
    logp = {}
    for e in st.envs:
        lp = math.log(1.0 / len(st.envs)); ok = True
        for tp in hist:
            l = st.lik_type(tp, e)
            if l <= 0: ok = False; break
            lp += math.log(l)
        if ok: logp[e] = lp
    preds, logev = {}, {}
    for e in logp:
        if e == 'loop':
            N, c = R['loop']; P = c / N; ev = N / st.nLp
        elif e == 'latin':
            continue
        else:
            hs = st.H if e == 'group' else (st.Hab if e == 'abelian' else st.Hcyc)
            ws = [(1.0 / len(hs)) / st.meta[h]['nlab'] for h in hs]
            N, c = union_counts(R, ['iso:' + h for h in hs], ws)
            if N == 0: continue
            P = c / N; ev = N
        preds[e] = P; logev[e] = logp[e] + math.log(ev)
    mx = max(logev.values()); Z = sum(math.exp(v - mx) for v in logev.values())
    Pm = sum(math.exp(logev[e] - mx) / Z * preds[e] for e in preds)
    out['Bh'] = decision(Pm)
    return out


def main(M, seed=12):
    n = 6
    st = Setup(n)
    T = groups_of_order(6)['S3']
    rng = np.random.default_rng(seed)
    res = defaultdict(float); cnt = defaultdict(float)
    for k in (3, 4, 5, 6, 8, 10):
        for rep in range(M):
            S = np.sort(rng.choice(n * n, size=k, replace=False))
            Sset = set(S.tolist())
            unseen, vstar, R = analyze_cue(st, T, S, need_lp=False)
            probe = []
            alt = []
            for i, c in enumerate(unseen):
                a, b = divmod(int(c), n)
                if (b * n + a) in Sset and T[a, b] != T[b, a]:
                    probe.append(i); alt.append(int(T[b, a]))
            if not probe:
                continue
            probe = np.array(probe); alt = np.array(alt)
            hsets = []
            for t in (1, 2, 4):
                hsets.append(('abel', t, [(1.0, tuple([('grp', 'Z6')] * t))]))
                mixed = defaultdict(float)
                for seq in itertools.product(st.H, repeat=t):
                    mixed[tuple(sorted(('grp', x) for x in seq))] += 0.5 ** t
                hsets.append(('mixed', t, [(p, h) for h, p in mixed.items()]))
            abc = [(0.25, (('grp', a1), ('grp', a2), ('grp', 'Z6'), ('grp', 'Z6'))) for a1 in st.H for a2 in st.H]
            hsets.append(('abc', 4, abc))
            for (hk, t, hl) in hsets:
                for p, hist in hl:
                    D = learner_class_dist(st, R, hist)
                    Drec = learner_class_dist(st, R, hist[-2:]) if len(hist) >= 2 else D
                    D['A_rec'] = Drec['A']; D['Bh_rec'] = Drec['Bh']
                    for L, Dm in D.items():
                        res[(hk, t, k, L, 'acc')] += p * Dm[probe, vstar[probe]].sum()
                        res[(hk, t, k, L, 'copy')] += p * Dm[probe, alt].sum()
                    cnt[(hk, t, k)] += p * len(probe)
        print('k', k, 'done', flush=True)
    out = {'|'.join(map(str, key)): v / cnt[key[:3]] for key, v in res.items()}
    out_cnt = {'|'.join(map(str, key)): v / M for key, v in cnt.items()}
    json.dump({'res': out, 'probe_cells_per_cue': out_cnt, 'M': M}, open('../out/ccopy_n6.json', 'w'), indent=0)
    for k in (3, 4, 5, 6, 8, 10):
        print(f'k={k}: probe cells per cue (abel,t=1) = {cnt.get(("abel", 1, k), 0) / M:.2f}')
        for hk, t in (('abel', 1), ('abel', 4), ('mixed', 4), ('abc', 4)):
            row = []
            for L in ('N', 'Niso', 'A', 'A_rec', 'Bh', 'Bh_rec'):
                a = out.get(f'{hk}|{t}|{k}|{L}|acc'); c = out.get(f'{hk}|{t}|{k}|{L}|copy')
                row.append(f'{L}:{a:.2f}/{c:.2f}' if a is not None else f'{L}:-')
            print(f'   {hk:5s} t={t}  acc/copy  ' + '  '.join(row))


if __name__ == '__main__':
    main(int(sys.argv[1]))
