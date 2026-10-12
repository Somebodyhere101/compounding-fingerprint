"""Evidence carried by a history table vs. by the cue, and Bayesian compounding under prior odds.

(a) per-table evidence D = log2 P_E(T)/P_E'(T) for a complete group table T of type H,
    E = groups (uniform over iso classes, uniform labelling), E' = Latin (uniform labelled) or
    loops (uniform labelled).  Exact from class sizes.
(b) cue evidence curves  e(k) = E_K log2 P_E(K)/P_E'(K)  for uniformly random k-cell cues of a group target.
(c) hierarchical Bayes learner with prior odds O = P(broad envs)/P(group-type envs):
    k*_theta(t) for t = 0,1,2 (history of complete tables drawn from E).

usage: python evidence.py a            -> table (a)
       python evidence.py b n M        -> cue evidence curves
       python evidence.py c n M        -> k*(t) for several prior odds (broad envs = Latin+loops)
       python evidence.py cL n M       -> same with broad env = Latin only
"""
import sys, math, json
import numpy as np
from sim import Setup, analyze_cue, union_counts, map_acc
from tables import groups_of_order

LATIN = {1: 1, 2: 2, 3: 12, 4: 576, 5: 161280, 6: 812851200, 7: 61479419904000}  # OEIS A002860; n<=5 enumerated, n=6 = 6!5!R(6) with R(6)=9408 enumerated
REDUCED = {1: 1, 2: 1, 3: 1, 4: 4, 5: 56, 6: 9408, 7: 16942080}  # labelled loops = n*R(n); n<=6 enumerated here


def part_a():
    rows = []
    for n in range(2, 8):
        H = groups_of_order(n)
        m = len(H)
        from tables import labelled_groups
        G, names = labelled_groups(n)
        for h in H:
            nl = names.count(h)
            pE = (1.0 / m) / nl
            pL = 1.0 / LATIN[n]
            pLp = 1.0 / (n * REDUCED[n])
            rows.append((n, h, nl, math.log2(pE / pL), math.log2(pE / pLp)))
            print(f'n={n} {h:5s} labelled={nl:5d}  D(groups||Latin)={math.log2(pE/pL):6.2f} bits   D(groups||loops)={math.log2(pE/pLp):6.2f} bits   log2|Latin|={math.log2(LATIN[n]):6.2f}')
    return rows


def part_b(n, M, seed=3):
    st = Setup(n)
    rng = np.random.default_rng(seed)
    res = {}
    for h, T in groups_of_order(n).items():
        for k in range(0, n * n + 1):
            vals_L, vals_O = [], []
            for rep in range(M):
                S = np.sort(rng.choice(n * n, size=k, replace=False))
                unseen, vstar, R = analyze_cue(st, T, S, need_lp=False)
                ws = [(1.0 / st.m) / st.meta[x]['nlab'] for x in st.H]
                NG, _ = union_counts(R, ['iso:' + x for x in st.H], ws)  # = P_E(K) up to position factor
                if st.enum_latin:
                    vals_L.append(math.log2(NG / (R['latin'][0] / st.nL)))
                if st.enum_loop:
                    vals_O.append(math.log2(NG / (R['loop'][0] / st.nLp)))
            res[f'{h}|{k}'] = (float(np.mean(vals_L)) if vals_L else None, float(np.mean(vals_O)) if vals_O else None)
            print(f'n={n} {h} k={k:2d}  E[log2 P_grp/P_latin]={res[f"{h}|{k}"][0]}  E[log2 P_grp/P_loop]={res[f"{h}|{k}"][1]}', flush=True)
    json.dump(res, open(f'../out/evidence_b_n{n}.json', 'w'))


def bh_acc(st, R, vstar, hist, logodds2, envs=None):
    """MAP accuracy of hierarchical Bayes with prior odds 2^logodds2 for broad envs vs group-type envs."""
    envs = envs if envs is not None else st.envs
    broad = [e for e in envs if e in ('latin', 'loop')]
    narrow = [e for e in envs if e not in ('latin', 'loop')]
    logp = {}
    for e in broad:
        logp[e] = math.log(1.0 / len(broad)) + logodds2 * math.log(2) - math.log(1 + 2.0 ** logodds2) if logodds2 < 900 else math.log(1.0 / len(broad))
    for e in narrow:
        logp[e] = math.log(1.0 / len(narrow)) - math.log(1 + 2.0 ** logodds2) if logodds2 < 900 else -1e300
    preds, logev = {}, {}
    for e in envs:
        lp = logp[e]
        ok = True
        for tp in hist:
            l = st.lik_type(tp, e)
            if l <= 0:
                ok = False; break
            lp += math.log(l)
        if not ok:
            continue
        if e in ('latin', 'loop'):
            N, c = R[e]
            if N is None or N <= 0: continue
            P = c / N; ev = N / (st.nL if e == 'latin' else st.nLp)
        else:
            hs = st.H if e == 'group' else (st.Hab if e == 'abelian' else st.Hcyc)
            ws = [(1.0 / len(hs)) / st.meta[h]['nlab'] for h in hs]
            N, c = union_counts(R, ['iso:' + h for h in hs], ws)
            if N == 0: continue
            P = c / N; ev = N
        preds[e] = P
        logev[e] = lp + math.log(ev)
    mx = max(logev.values())
    Z = sum(math.exp(v - mx) for v in logev.values())
    Pm = sum(math.exp(logev[e] - mx) / Z * preds[e] for e in preds)
    return map_acc(Pm, vstar)


def part_c(n, M, seed=4, theta=0.9, odds=(0, 5, 10, 20, 40), envs=None, tag=''):
    st = Setup(n)
    rng = np.random.default_rng(seed)
    out = {}
    for h, T in groups_of_order(n).items():
        for lo in odds:
            for t in (0, 1, 2):
                out[(h, lo, t)] = []
        for k in range(0, n * n):
            accs = {key: [] for key in out if key[0] == h}
            for rep in range(M):
                S = np.sort(rng.choice(n * n, size=k, replace=False))
                unseen, vstar, R = analyze_cue(st, T, S, need_lp=False)
                for lo in odds:
                    for t in (0, 1, 2):
                        # history: t tables of the SAME iso class as the target's environment draw:
                        # average over histories drawn from E (iso-uniform)
                        import itertools
                        tot = 0.0
                        for seq in itertools.product(st.H, repeat=t):
                            hist = tuple(('grp', x) for x in seq)
                            tot += bh_acc(st, R, vstar, hist, lo, envs).mean() / (len(st.H) ** t)
                        accs[(h, lo, t)].append(tot)
            for key in accs:
                out[key].append(float(np.mean(accs[key])))
        for lo in odds:
            ks = [next((k for k, a in enumerate(out[(h, lo, t)]) if a >= theta), None) for t in (0, 1, 2)]
            print(f'{tag} n={n} target={h} log2 prior odds={lo:3d}:  k*_{theta}(t=0,1,2) = {ks}   acc@k(t=0)={[round(a,2) for a in out[(h, lo, 0)]]}', flush=True)
    json.dump({f'{a}|{b}|{c}': v for (a, b, c), v in out.items()}, open(f'../out/evidence_c_n{n}{tag}.json', 'w'))


if __name__ == '__main__':
    if sys.argv[1] == 'a':
        part_a()
    elif sys.argv[1] == 'b':
        part_b(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == 'c':
        part_c(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == 'cL':   # competitor = Latin only (no loop environment)
        part_c(int(sys.argv[2]), int(sys.argv[3]), envs=['latin', 'group', 'abelian', 'cyclic'], tag='_latinonly')
