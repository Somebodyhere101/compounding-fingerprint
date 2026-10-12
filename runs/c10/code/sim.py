"""Exact/Monte-Carlo predictions of idealised learners R, N, N_iso, A (strict, sloppy), B (true prior),
B_hier (hierarchical Bayes over environment classes) for the compounding fingerprint experiment.

Each trial: history of t complete tables (fresh labels, so only their isomorphism TYPE matters),
then a target table with a uniformly random cue of k cells; every unseen cell is queried.
By label-equivariance of all learners we fix the target to a canonical representative of its
iso class and randomise only the cue positions (exact, see report).

usage: python sim.py n M [kstep] [tag]
  writes ../out/sim_n{n}{tag}.json with mean accuracies per (condition, learner, t, k, cell class)
"""
import sys, json, time, itertools, math
from collections import defaultdict
import numpy as np
from engine import universe, group_meta, counts_for, sat_solver, loop_iso_classes
from tables import groups_of_order, all_loops, is_comm, is_assoc, all_latin

CELLCLS = ['L', 'O', 'G', 'Ab', 'Cy', 'U']  # Latin-, loop-, group-, abelian-, cyclic-forced, undetermined
TMAX = 4


# ------------------------------------------------------------------ helpers
def poss_acc(F, vstar):
    """possibilistic learner: uniform guess among feasible values. F: [u,n] bool."""
    s = F.sum(1)
    ok = F[np.arange(len(vstar)), vstar]
    with np.errstate(invalid='ignore', divide='ignore'):
        a = np.where(s > 0, ok / np.maximum(s, 1), 1.0 / F.shape[1])
    return a


def map_acc(P, vstar):
    """MAP with uniform tie-breaking. P: [u,n] nonneg."""
    mx = P.max(1, keepdims=True)
    tie = np.isclose(P, mx, rtol=1e-9, atol=1e-300) & (mx > 0)
    nt = tie.sum(1)
    ok = tie[np.arange(len(vstar)), vstar]
    return np.where(nt > 0, ok / np.maximum(nt, 1), 1.0 / P.shape[1])


class Setup:
    def __init__(self, n):
        self.n = n
        self.U = universe(n)
        self.meta = group_meta(n)
        self.H = list(self.meta.keys())
        self.m = len(self.H)
        self.Hab = [h for h in self.H if self.meta[h]['abelian']]
        self.Hcyc = [h for h in self.H if self.meta[h]['cyclic']]
        self.enum_latin = n <= 5
        self.enum_loop = n <= 6
        self.nL = len(self.U['latin']) if self.enum_latin else None
        self.nLp = len(self.U['loop']) if self.enum_loop else None
        if self.enum_loop:
            Lp = self.U['loop']
            self.U['cloop'] = Lp[np.all(Lp.reshape(-1, n, n) == Lp.reshape(-1, n, n).transpose(0, 2, 1), axis=(1, 2))]
        if n == 5:
            self.lp = [k for k in self.U if k.startswith('lp:')]
            self.lp_comm = {k: is_comm(self.U[k][0].reshape(n, n)) for k in self.lp}
        else:
            self.lp, self.lp_comm = [], {}
        # environments for B_hier (only those whose likelihoods we can evaluate exactly)
        self.envs = (['latin'] if self.enum_latin else []) + (['loop'] if self.enum_loop else []) + ['group', 'abelian', 'cyclic']

    # likelihood of one complete history table of a given TYPE under env e (per labelled table)
    def lik_type(self, typ, e):
        kind, name = typ
        if kind == 'grp':
            nl = self.meta[name]['nlab']
            if e == 'latin': return 1.0 / self.nL
            if e == 'loop': return 1.0 / self.nLp
            if e == 'group': return (1.0 / self.m) / nl
            if e == 'abelian': return (1.0 / len(self.Hab)) / nl if name in self.Hab else 0.0
            if e == 'cyclic': return 1.0 / nl if name in self.Hcyc else 0.0
        else:  # non-associative loop
            if e == 'latin': return 1.0 / self.nL
            if e == 'loop': return 1.0 / self.nLp
            return 0.0

    def type_axioms(self, typ):
        kind, name = typ
        if kind == 'grp':
            return dict(ident=True, assoc=True, comm=self.meta[name]['abelian'], cyc=self.meta[name]['cyclic'])
        return dict(ident=True, assoc=False, comm=self.lp_comm.get(name, False), cyc=False)


def analyze_cue(st, T, S, need_lp):
    n = st.n
    flat = T.reshape(-1)
    unseen = np.setdiff1d(np.arange(n * n), S)
    vstar = flat[unseen].astype(np.int64)
    R = {}
    for h in st.H:
        N, c = counts_for(st.U['iso:' + h], T, S)
        R['iso:' + h] = (N, c[unseen])
    for kind in ('latin', 'loop'):
        if kind in st.U:
            N, c = counts_for(st.U[kind], T, S)
            R[kind] = (N, c[unseen])
        else:
            F = sat_solver(n, kind == 'loop').feasible(T, S)
            R[kind] = (None if F is None else -1, None if F is None else F[unseen])
    if 'cloop' in st.U:
        N, c = counts_for(st.U['cloop'], T, S)
        R['cloop'] = (N, c[unseen])
    if need_lp:
        for j in st.lp:
            N, c = counts_for(st.U[j], T, S)
            R[j] = (N, c[unseen])
    return unseen, vstar, R


def feas(R, key):
    N, c = R[key]
    if N is None or N == 0:
        return None
    return c > 0 if c.dtype != bool else c


def union_counts(R, keys, weights=None):
    tot = None
    N = 0.0
    for i, k in enumerate(keys):
        n_k, c = R[k]
        w = 1.0 if weights is None else weights[i]
        if n_k is None or n_k == 0:
            continue
        tot = c * w if tot is None else tot + c * w
        N += n_k * w
    return N, tot


def classify_cells(st, R, Tname, Tkind):
    """cell class per unseen cell."""
    Fl = feas(R, 'latin'); Fo = feas(R, 'loop')
    u = Fl.shape[0]
    NG, cG = union_counts(R, ['iso:' + h for h in st.H])
    sz = lambda F: F.sum(1)
    cls = np.full(u, 'U', dtype=object)
    forced_G = (cG > 0).sum(1) == 1 if NG > 0 else np.zeros(u, bool)
    if Tkind == 'grp' and st.meta[Tname]['abelian']:
        NA, cA = union_counts(R, ['iso:' + h for h in st.Hab])
        forced_A = (cA > 0).sum(1) == 1
    else:
        forced_A = np.zeros(u, bool)
    if Tkind == 'grp' and st.meta[Tname]['cyclic']:
        forced_C = (R['iso:' + Tname][1] > 0).sum(1) == 1
    else:
        forced_C = np.zeros(u, bool)
    if Tkind != 'grp':
        forced_G = np.zeros(u, bool)
    cls[forced_C] = 'Cy'
    cls[forced_A] = 'Ab'
    cls[forced_G] = 'G'
    cls[sz(Fo) == 1] = 'O'
    cls[sz(Fl) == 1] = 'L'
    return cls



def latin_propagation(T, S):
    """Bounded Latin inference: naked + hidden singles to a fixpoint (no search).
    Returns candidate mask [n*n, n] (sound: the true value is always a candidate)."""
    n = len(T)
    flat = T.reshape(-1)
    cand = np.ones((n, n, n), dtype=bool)
    val = -np.ones((n, n), dtype=np.int64)
    for c in S:
        r, q = divmod(int(c), n)
        val[r, q] = int(flat[c])
    changed = True
    while changed:
        changed = False
        for r in range(n):
            for q in range(n):
                v = val[r, q]
                if v >= 0:
                    if cand[r, q].sum() != 1 or not cand[r, q, v]:
                        cand[r, q] = False; cand[r, q, v] = True
                    rowmask = cand[r, :, v].copy(); rowmask[q] = False
                    colmask = cand[:, q, v].copy(); colmask[r] = False
                    if rowmask.any() or colmask.any():
                        cand[r, :, v] &= False; cand[:, q, v] &= False; cand[r, q, v] = True
                        changed = True
        # naked singles
        for r in range(n):
            for q in range(n):
                if val[r, q] < 0 and cand[r, q].sum() == 1:
                    val[r, q] = int(np.argmax(cand[r, q])); changed = True
        # hidden singles
        for v in range(n):
            for r in range(n):
                pos = np.where(cand[r, :, v])[0]
                if len(pos) == 1 and val[r, pos[0]] < 0:
                    val[r, pos[0]] = v; cand[r, pos[0]] = False; cand[r, pos[0], v] = True; changed = True
            for q in range(n):
                pos = np.where(cand[:, q, v])[0]
                if len(pos) == 1 and val[pos[0], q] < 0:
                    val[pos[0], q] = v; cand[pos[0], q] = False; cand[pos[0], q, v] = True; changed = True
    return cand.reshape(n * n, n)

# ------------------------------------------------------------------ learners
def _counts(R, key):
    N, c = R[key]
    if N is None or N <= 0 or c is None or c.dtype == bool:
        return None
    return c


def learner_preds(st, R, vstar, hist, Tkind, cache):
    """Per-cell accuracy of every learner given a history (tuple of types).
    Possibilistic learners (uniform guess among values feasible under the learner's class):
      R, N, Niso, A, Asl, Bposs.   MAP-Bayes learners: Nb, Nisob, Ab, B, Bh.
    cache: per-cue dict for history-independent / repeated computations."""
    from collections import Counter
    n = st.n
    out = {}
    Fl = feas(R, 'latin')
    Fo = feas(R, 'loop')
    allG = ['iso:' + h for h in st.H]
    abG = ['iso:' + h for h in st.Hab]
    cyG = ['iso:' + h for h in st.Hcyc]

    def Fclass(keys):
        N, c = union_counts(R, keys)
        return None if (N == 0 or c is None) else (c > 0)

    def Cclass(keys, w=None):
        N, c = union_counts(R, keys, w)
        return None if (N == 0 or c is None) else c

    if 'static' not in cache:
        s = {}
        s['R'] = np.full(len(vstar), 1.0 / n)            # recall only: chance under fresh labels
        s['N'] = poss_acc(Fl, vstar)                      # notebook: surface retrieval empty; Latin inference
        s['Np'] = poss_acc(cache['prop'], vstar)          # notebook with bounded (propagation-only) inference
        cl = _counts(R, 'latin')
        s['Nb'] = map_acc(cl, vstar) if cl is not None else None
        if Tkind == 'grp':
            tk = allG; w = [1.0 / st.meta[h]['nlab'] for h in st.H]
        else:
            tk = st.lp; w = [1.0 / len(st.U[j]) for j in st.lp]
        NB, cB = union_counts(R, tk, w)
        if cB is None:   # no table of the assumed true-prior class fits the cue (only in the abc loop-target probe)
            s['B'] = poss_acc(Fo if Fo is not None else Fl, vstar); s['Bposs'] = s['B']
        else:
            s['B'] = map_acc(cB, vstar)                  # true prior of the target environment
            s['Bposs'] = poss_acc(cB > 0, vstar)
        cache['static'] = s
    for k, v in cache['static'].items():
        if v is not None:
            out[k] = v

    hc = tuple(sorted(Counter(hist).items()))
    # ---- N_iso (exemplar matcher) and its MAP version
    key = ('niso', hc)
    if key not in cache:
        types = [tp for tp, _ in hc]
        keys = [('iso:' + nm) if kd == 'grp' else nm for (kd, nm) in types]
        F = Fclass(keys) if keys else None
        a = poss_acc(F if F is not None else Fl, vstar)
        b = None
        if keys:
            w = [c / max(len(hist), 1) / len(st.U[kk]) for kk, (_, c) in zip(keys, hc)]
            C = Cclass(keys, w)
            if C is not None:
                b = map_acc(C, vstar)
        if b is None:
            b = cache['static']['Nb']
        if b is None:  # Latin not enumerable (n>=6): MAP fallback = possibilistic Latin
            b = cache['static']['N']
        cache[key] = (a, b)
    a, b = cache[key]
    out['Niso'] = a
    if b is not None:
        out['Nisob'] = b
    # ---- A (axiom induction, strict) and MAP version
    if hist:
        ax = {x: all(st.type_axioms(tp)[x] for tp in hist) for x in ('ident', 'assoc', 'comm', 'cyc')}
    else:
        ax = dict(ident=False, assoc=False, comm=False, cyc=False)
    key = ('A', tuple(sorted(ax.items())))
    if key not in cache:
        chain = []
        if ax['assoc']:
            if ax['cyc']: chain.append(cyG)
            if ax['comm']: chain.append(abG)
            chain.append(allG)
        tail = []
        if ax['ident'] and ax['comm'] and not ax['assoc'] and 'cloop' in R:
            tail.append('cloop')
        if ax['ident']:
            tail.append('loop')
        tail.append('latin')
        F = None; C = None; usedC = False
        for keys in chain:
            F = Fclass(keys)
            if F is not None:
                C = Cclass(keys); usedC = True
                break
        if F is None:
            for kk in tail:
                F = feas(R, kk)
                if F is not None:
                    C = _counts(R, kk); usedC = True
                    break
        a = poss_acc(F, vstar)
        b = map_acc(C, vstar) if (usedC and C is not None) else a  # SAT-only class: possibilistic
        cache[key] = (a, b)
    a, b = cache[key]
    out['A'] = a
    if b is not None:
        out['Ab'] = b
    # ---- A sloppy: identity + Latin seen in every history table => assumes GROUP (no assoc check)
    key = ('Asl', bool(hist) and ax['ident'])
    if key not in cache:
        if key[1]:
            F = Fclass(allG)
            if F is None:
                F = Fo
        else:
            F = Fl
        cache[key] = poss_acc(F, vstar)
    out['Asl'] = cache[key]
    # ---- B_hier: hierarchical Bayes over environment classes
    if 'loop' in st.U:
        key = ('Bh', hc)
        if key not in cache:
            post = {}
            for e in st.envs:
                lp = math.log(1.0 / len(st.envs))
                ok = True
                for tp in hist:
                    l = st.lik_type(tp, e)
                    if l <= 0:
                        ok = False; break
                    lp += math.log(l)
                if ok:
                    post[e] = lp
            preds, logev = {}, {}
            for e in post:
                if e in ('latin', 'loop'):
                    N, c = R[e]; W = 1.0 / (st.nL if e == 'latin' else st.nLp)
                    if N is None or N <= 0: continue
                    P = c / N; ev = N * W
                else:
                    hs = st.H if e == 'group' else (st.Hab if e == 'abelian' else st.Hcyc)
                    ws = [(1.0 / len(hs)) / st.meta[h]['nlab'] for h in hs]
                    N, c = union_counts(R, ['iso:' + h for h in hs], ws)
                    if N == 0: continue
                    P = c / N; ev = N
                preds[e] = P
                logev[e] = post[e] + math.log(ev)
            mx = max(logev.values())
            Z = sum(math.exp(v - mx) for v in logev.values())
            Pm = sum(math.exp(logev[e] - mx) / Z * preds[e] for e in preds)
            wg = sum(math.exp(logev[e] - mx) / Z for e in preds if e in ('group', 'abelian', 'cyclic'))
            cache[key] = (map_acc(Pm, vstar), wg)
        out['Bh'], out['_wgrp'] = cache[key]
    return out


# ------------------------------------------------------------------ conditions
def histories(st, cond, t):
    """list of (prob, history tuple) for condition cond at length t."""
    if cond in ('base', 'tgtnew'):
        types = [('grp', h) for h in st.H]
    elif cond == 'heldout':  # history restricted to the 'seen' classes
        types = [('grp', h) for h in st.seen]
    elif cond == 'loops':
        types = [('lp', j) for j in st.lp]
    else:
        raise ValueError(cond)
    out = defaultdict(float)
    for seq in itertools.product(types, repeat=t):
        out[tuple(sorted(seq))] += (1.0 / len(types)) ** t
    return [(p, h) for h, p in out.items()]


def run(n, M, kstep=1, tag='', seed=1):
    st = Setup(n)
    rng = np.random.default_rng(seed)
    gtabs = groups_of_order(n)
    # held-out design: seen = cyclic class only (n=4,6) / abelian classes (n=8); target = others
    if n == 4: st.seen = ['Z4']
    elif n == 6: st.seen = ['Z6']
    elif n == 8: st.seen = ['Z8', 'Z4xZ2', 'Z2^3']
    else: st.seen = st.H
    targets = [('grp', h, gtabs[h]) for h in st.H]
    if n == 5:
        for j in st.lp:
            targets.append(('lp', j, st.U[j][0].reshape(n, n)))
    conds_for = {}
    for (kind, name, T) in targets:
        c = []
        if kind == 'grp':
            c.append('base')
            if name not in st.seen:
                c.append('heldout')
            if st.lp:
                c.append('loops')
        else:
            c.append('base')   # loop target after group history (A_sloppy / B misapply)
            c.append('loops')  # loop target after loop history
        conds_for[name] = c
    acc = defaultdict(float)   # key (cond, target, learner, t, k, cellcls) -> sum acc
    cnt = defaultdict(float)   # key (cond, target, t, k, cellcls) -> sum weight (cells)
    pfull = defaultdict(float) # key (cond, target, learner, t, k) -> sum P(full completion correct)
    ncue = defaultdict(int)
    wgrp = defaultdict(float)
    ks = list(range(0, n * n + 1, kstep))
    if ks[-1] != n * n: ks.append(n * n)
    t0 = time.time()
    for (kind, name, T) in targets:
        for k in ks:
            if k == n * n:
                continue
            for rep in range(M):
                S = np.sort(rng.choice(n * n, size=k, replace=False))
                unseen, vstar, R = analyze_cue(st, T, S, need_lp=bool(st.lp))
                cls = classify_cells(st, R, name, kind)
                cache = {'prop': latin_propagation(T, S)[unseen]}
                for cond in conds_for[name]:
                    for t in range(0, TMAX + 1):
                        for (p, hist) in histories(st, cond, t):
                            preds = learner_preds(st, R, vstar, hist, kind, cache)
                            for L, a in preds.items():
                                if L == '_wgrp':
                                    wgrp[(cond, name, t, k)] += p * a / M
                                    continue
                                for cc in CELLCLS:
                                    msk = cls == cc
                                    if msk.any():
                                        acc[(cond, name, L, t, k, cc)] += p * a[msk].sum()
                                acc[(cond, name, L, t, k, 'all')] += p * a.sum()
                                pfull[(cond, name, L, t, k)] += p * float(np.prod(a)) / M
                            for cc in CELLCLS:
                                cnt[(cond, name, t, k, cc)] += p * float((cls == cc).sum())
                            cnt[(cond, name, t, k, 'all')] += p * len(vstar)
                ncue[(name, k)] += 1
            print(f'n={n} target={name} k={k} done ({time.time()-t0:.0f}s)', flush=True)
    res = {'n': n, 'M': M, 'ks': ks, 'acc': {}, 'cnt': {}, 'pfull': {}, 'wgrp': {}}
    for key, v in acc.items():
        cond, name, L, t, k, cc = key
        res['acc']['|'.join(map(str, key))] = v / max(cnt[(cond, name, t, k, cc)], 1e-12)
    for key, v in cnt.items():
        cond, name, t, k, cc = key
        res['cnt']['|'.join(map(str, key))] = v / M
    for key, v in pfull.items():
        res['pfull']['|'.join(map(str, key))] = v
    for key, v in wgrp.items():
        res['wgrp']['|'.join(map(str, key))] = v
    with open(f'../out/sim_n{n}{tag}.json', 'w') as f:
        json.dump(res, f)
    print('saved', f'../out/sim_n{n}{tag}.json', f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    n = int(sys.argv[1]); M = int(sys.argv[2])
    kstep = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    tag = sys.argv[4] if len(sys.argv) > 4 else ''
    run(n, M, kstep, tag)
