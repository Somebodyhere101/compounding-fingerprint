"""Exact per-cue analysis engine.

For a target table T (n x n) and a cue S (flat cell indices), compute for each hypothesis
class X the set of tables in X consistent with the cue and, for each unseen cell, the
count vector over values (enumerable classes) or the feasible-value mask (SAT classes).

Classes (keys):
  'latin'  all Latin squares            (enumerated n<=5, SAT n>=6)
  'loop'   Latin squares with identity  (enumerated n<=6, SAT n>=7)
  'iso:H'  relabelings of group H       (enumerated, n<=8)
  'lp:j'   relabelings of non-assoc loop iso class j (n=5 only; used as exemplars)
Group / abelian / cyclic classes are unions of 'iso:H'.
"""
import functools, itertools
import numpy as np
from tables import (labelled_groups, groups_of_order, all_latin, all_loops, is_comm,
                    is_assoc, relabel)


# ---------------------------------------------------------------- universes
@functools.lru_cache(maxsize=None)
def universe(n):
    U = {}
    G, names = labelled_groups(n)
    names = np.array(names)
    gnames = list(groups_of_order(n).keys())
    for H in gnames:
        U['iso:' + H] = G[names == H].reshape(-1, n * n)
    if n <= 5:
        U['latin'] = all_latin(n).reshape(-1, n * n)
    if n <= 6:
        U['loop'] = all_loops(n).reshape(-1, n * n)
    if n == 5:
        for j, cls in enumerate(loop_iso_classes(5)):
            U[f'lp:{j}'] = cls.reshape(-1, n * n)
    return U


@functools.lru_cache(maxsize=None)
def group_meta(n):
    """iso class name -> dict(abelian, cyclic, nlab)"""
    meta = {}
    for H, T in groups_of_order(n).items():
        meta[H] = {'abelian': is_comm(T), 'cyclic': H == f'Z{n}',
                   'nlab': len(universe(n)['iso:' + H])}
    return meta


@functools.lru_cache(maxsize=None)
def loop_iso_classes(n):
    """Non-associative loops of order n (n<=5 practical) split into isomorphism classes."""
    L = all_loops(n)
    L = [x for x in L if not is_assoc(x)]
    keyset = {x.tobytes(): i for i, x in enumerate(L)}
    seen = set()
    classes = []
    perms = [np.array(p) for p in itertools.permutations(range(n))]
    for i, x in enumerate(L):
        if i in seen:
            continue
        cls = {}
        for s in perms:
            y = relabel(x, s)
            cls[y.tobytes()] = y
        idx = [keyset[k] for k in cls]
        seen.update(idx)
        classes.append(np.array(list(cls.values()), dtype=np.uint8))
    return classes


# ---------------------------------------------------------------- SAT classes
class SatLatin:
    """Feasible value masks for Latin squares / loops consistent with a cue (pysat, CaDiCaL)."""

    def __init__(self, n, loop=False):
        from pysat.solvers import Cadical153
        self.n = n
        self.loop = loop
        self.s = Cadical153()
        v = lambda r, c, x: 1 + (r * n + c) * n + x
        self.v = v
        cl = []
        def exactly_one(lits):
            cl.append(list(lits))
            for a, b in itertools.combinations(lits, 2):
                cl.append([-a, -b])
        for r in range(n):
            for c in range(n):
                exactly_one([v(r, c, x) for x in range(n)])
        for r in range(n):
            for x in range(n):
                exactly_one([v(r, c, x) for c in range(n)])
        for c in range(n):
            for x in range(n):
                exactly_one([v(r, c, x) for r in range(n)])
        if loop:
            base = n * n * n
            e = lambda i: base + 1 + i
            exactly_one([e(i) for i in range(n)])
            for i in range(n):
                for j in range(n):
                    cl.append([-e(i), v(i, j, j)])
                    cl.append([-e(i), v(j, i, j)])
        for c in cl:
            self.s.add_clause(c)

    def feasible(self, T, S):
        """Boolean mask F[cell, value] of values attainable at each cell by a completion of the cue."""
        n = self.n
        flat = T.reshape(-1)
        assum = [int(self.v(int(c) // n, int(c) % n, int(flat[c]))) for c in S]
        F = np.zeros((n * n, n), dtype=bool)
        if not self.s.solve(assumptions=assum):
            return None
        def mark(model):
            pos = [l for l in model if 0 < l <= n * n * n]
            for l in pos:
                q = l - 1
                cell, x = divmod(q, n)
                F[cell, x] = True
        mark(self.s.get_model())
        for cell in range(n * n):
            for x in range(n):
                if F[cell, x]:
                    continue
                if self.s.solve(assumptions=assum + [self.v(cell // n, cell % n, x)]):
                    mark(self.s.get_model())
        return F


@functools.lru_cache(maxsize=None)
def sat_solver(n, loop):
    return SatLatin(n, loop)


# ---------------------------------------------------------------- per-cue analysis
def counts_for(tabs, T, S):
    """tabs: [m, n*n] uint8.  Returns (N, counts[n*n, n]) for tables consistent with cue."""
    n = len(T)
    flat = T.reshape(-1)
    if len(S):
        mask = np.all(tabs[:, S] == flat[S], axis=1)
        C = tabs[mask]
    else:
        C = tabs
    cnt = np.zeros((n * n, n), dtype=np.int64)
    if len(C):
        for x in range(n):
            cnt[:, x] = (C == x).sum(0)
    return len(C), cnt


def analyze(T, S, classes, sat_kinds=()):
    """Return dict class -> (N or None, counts or mask)."""
    n = len(T)
    U = universe(n)
    out = {}
    for X in classes:
        out[X] = counts_for(U[X], T, S)
    for kind in sat_kinds:  # 'latin' or 'loop' via SAT
        F = sat_solver(n, kind == 'loop').feasible(T, S)
        out[kind] = (None, F)
    return out
