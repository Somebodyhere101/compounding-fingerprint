"""Independent cross-check of minimum defining sets via minimum hitting set (CP-SAT, 1 thread).

S is defining for T in C  <=>  S meets D(T') = {cells where T' != T} for every T' in C, T' != T.
md(T, C) = min hitting set.  Lazy constraint generation: start from the tables closest to T,
solve, check the optimum against ALL of C (numpy), add violated tables, repeat.

usage: python defsets_ilp.py n1,n2,... [prior=group|abelian] [outfile]
"""
import sys, json, time
import numpy as np
from ortools.sat.python import cp_model
from tables import labelled_groups, groups_of_order, is_comm


def md_hitting(T, C, init=2000, verbose=False):
    n = len(T)
    flatC = C.reshape(len(C), -1)
    flatT = T.reshape(-1)
    D = flatC != flatT  # [m, n*n]
    self_idx = np.where(~D.any(1))[0]
    assert len(self_idx) == 1
    dist = D.sum(1)
    order = np.argsort(dist)
    active = [int(i) for i in order[1:init + 1]]
    rounds = 0
    while True:
        rounds += 1
        m = cp_model.CpModel()
        x = [m.NewBoolVar(f'x{c}') for c in range(n * n)]
        for i in active:
            m.AddBoolOr([x[c] for c in np.where(D[i])[0]])
        m.Minimize(sum(x))
        s = cp_model.CpSolver()
        s.parameters.num_workers = 1
        s.parameters.max_time_in_seconds = 600
        st = s.Solve(m)
        assert st == cp_model.OPTIMAL, st
        S = [c for c in range(n * n) if s.Value(x[c])]
        # check against all
        hit = D[:, S].any(1)
        hit[self_idx] = True
        viol = np.where(~hit)[0]
        if verbose:
            print('  round', rounds, 'k=', len(S), 'violations', len(viol), flush=True)
        if len(viol) == 0:
            return len(S), [(c // n, c % n, int(T[c // n, c % n])) for c in S], rounds
        active += [int(i) for i in viol[:5000]]


def main(ns, prior='group', out=None):
    res = {}
    for n in ns:
        t0 = time.time()
        G, names = labelled_groups(n)
        print(f'n={n}: {len(G)} labelled group tables ({time.time()-t0:.1f}s)', flush=True)
        if prior == 'abelian':
            G = G[np.array([is_comm(x) for x in G])]
        for gname, T in groups_of_order(n).items():
            if prior == 'abelian' and not is_comm(T):
                continue
            t0 = time.time()
            k, S, r = md_hitting(T, G)
            print(f'n={n} {gname:6s} prior={prior} md={k} cue={S} rounds={r} ({time.time()-t0:.1f}s)', flush=True)
            res[f'{n}:{gname}:{prior}'] = {'k': k, 'cue': S}
    if out:
        with open(out, 'w') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    ns = [int(v) for v in sys.argv[1].split(',')]
    prior = sys.argv[2] if len(sys.argv) > 2 else 'group'
    out = sys.argv[3] if len(sys.argv) > 3 else None
    main(ns, prior, out)
