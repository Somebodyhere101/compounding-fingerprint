"""Order-10 group-prior minimum defining sets, fixed-k feasibility with lazy constraints.

Lower bound: the optimum of any relaxation (subset of competing tables) is a valid lower bound; the
first relaxation (3000 nearest tables) already gave 5 for Z10 (log_md10.txt).  Here: for k = kmin, kmin+1, ...
search for a k-cell cue hitting D(T') for every competing labelled group table T' (lazy: start with the
20000 nearest, add every violator found), with CP-SAT feasibility (1 thread).  Infeasible at k with a subset
of constraints => md > k (valid).  Feasible with zero violations over ALL 1088640 tables => md <= k.
usage: python md10_fast.py kmin  -> ../out/md_n10.json
"""
import sys, json, time
import numpy as np
from ortools.sat.python import cp_model
from tables import cyclic, closure, from_perms
from md_extra import labelled_coset_reps


def solve_fixed_k(T, C, k, init=20000, tmax=900):
    n = len(T)
    flatC = C.reshape(len(C), -1); flatT = T.reshape(-1)
    D = flatC != flatT
    selfi = np.where(~D.any(1))[0]
    dist = D.sum(1)
    order = np.argsort(dist, kind='stable')
    active = [int(i) for i in order[1:init + 1]]
    rounds = 0
    while True:
        rounds += 1
        m = cp_model.CpModel()
        x = [m.NewBoolVar(f'x{c}') for c in range(n * n)]
        for i in active:
            m.AddBoolOr([x[c] for c in np.where(D[i])[0]])
        m.Add(sum(x) == k)
        s = cp_model.CpSolver(); s.parameters.num_workers = 1; s.parameters.max_time_in_seconds = tmax
        st = s.Solve(m)
        if st == cp_model.INFEASIBLE:
            return None, rounds, 'infeasible'
        if st not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            return None, rounds, f'status {st}'
        S = [c for c in range(n * n) if s.Value(x[c])]
        hit = D[:, S].any(1); hit[selfi] = True
        viol = np.where(~hit)[0]
        print(f'   k={k} round {rounds}: {len(viol)} violators', flush=True)
        if len(viol) == 0:
            return [(c // n, c % n, int(T[c // n, c % n])) for c in S], rounds, 'ok'
        active += [int(i) for i in viol]


def main(kmin):
    D5 = from_perms(closure([(1, 2, 3, 4, 0), (0, 4, 3, 2, 1)]))
    gs = {'Z10': cyclic(10), 'D5': D5}
    C = np.concatenate([labelled_coset_reps(T)[0] for T in gs.values()])
    print('labelled groups of order 10:', len(C), flush=True)
    res = {}
    for h, T in gs.items():
        for k in range(kmin, 9):
            t0 = time.time()
            S, r, st = solve_fixed_k(T, C, k)
            print(f'{h}: k={k} -> {st} ({r} rounds, {time.time()-t0:.0f}s) cue={S}', flush=True)
            if S is not None:
                res[h] = {'md': k, 'cue': S, 'lower_bound_note': f'no defining cue of size {k-1} (infeasible relaxation) or from kmin'}
                break
            res[h] = {'md_gt': k}
        json.dump(res, open('../out/md_n10.json', 'w'), indent=1)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
