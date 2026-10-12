"""Extra minimum-defining-set computations.

 iso  : md within the relabelings of T only (the 'labelling-only' teaching dimension), all groups n<=9
 n10  : md under the group prior for Z10 and D5 (labelled tables enumerated as coset representatives)

usage: python md_extra.py iso     -> ../out/md_iso.json
       python md_extra.py n10     -> ../out/md_n10.json
"""
import sys, json, time, itertools
import numpy as np
from tables import labelled_groups, groups_of_order, relabel, closure, from_perms, cyclic
from defsets_ilp import md_hitting


def run_iso():
    res = {}
    for n in range(2, 10):
        G, names = labelled_groups(n)
        names = np.array(names)
        for h, T in groups_of_order(n).items():
            C = G[names == h]
            t0 = time.time()
            k, S, r = md_hitting(T, C)
            print(f'n={n} {h:6s} |Iso|={len(C):6d}  md_iso={k}  cue={S}  ({time.time()-t0:.1f}s)', flush=True)
            res[f'{n}:{h}'] = {'k': k, 'cue': S}
    json.dump(res, open('../out/md_iso.json', 'w'), indent=1)


def automorphisms(T):
    """All automorphisms of the group table T (identity = 0) as permutation arrays; brute force over
    images of a small generating set."""
    n = len(T)
    # find a generating set greedily
    def gen_closure(gs):
        S = {0}
        frontier = [0]
        while frontier:
            x = frontier.pop()
            for g in gs:
                y = int(T[x, g])
                if y not in S:
                    S.add(y); frontier.append(y)
        return S
    gens = []
    while len(gen_closure(gens)) < n:
        cand = [x for x in range(n) if x not in gen_closure(gens)]
        gens.append(cand[0])
    auts = []
    for imgs in itertools.permutations(range(1, n), len(gens)):
        phi = -np.ones(n, dtype=np.int64); phi[0] = 0
        for g, im in zip(gens, imgs):
            phi[g] = im
        # extend by closure: phi(x*g) = phi(x)*phi(g)
        ok = True
        frontier = [0]
        visited = {0}
        while frontier and ok:
            x = frontier.pop()
            for g in gens:
                y = int(T[x, g]); v = int(T[phi[x], phi[g]])
                if phi[y] < 0:
                    phi[y] = v
                elif phi[y] != v:
                    ok = False; break
                if y not in visited:
                    visited.add(y); frontier.append(y)
        if not ok or len(set(phi.tolist())) != n:
            continue
        if np.array_equal(phi[T], T[np.ix_(phi, phi)]):
            auts.append(phi.copy())
    return auts


def labelled_coset_reps(T, batch=20000):
    """All distinct labelled copies of T, one per coset sigma*Aut(T): keep sigma iff it is the
    lexicographically smallest element of {sigma o alpha} (compared as tuples of labels)."""
    n = len(T)
    auts = np.array(automorphisms(T))
    out = []
    it = itertools.permutations(range(n))
    while True:
        chunk = list(itertools.islice(it, batch))
        if not chunk:
            break
        P = np.array(chunk, dtype=np.int64)            # P[b, g] = label of group element g
        comp = P[:, auts]                              # comp[b, a, g] = P[b, aut_a[g]]
        # sigma is canonical iff P[b] <= comp[b, a] lexicographically for all a
        keep = np.ones(len(P), bool)
        for a in range(len(auts)):
            diff = comp[:, a, :] - P
            nz = diff != 0
            first = np.where(nz.any(1), diff[np.arange(len(P)), nz.argmax(1)], 0)
            keep &= first >= 0
        Pk = P[keep]
        if len(Pk) == 0:
            continue
        # labelled table: L[s[a], s[b]] = s[T[a,b]]
        L = np.empty((len(Pk), n, n), dtype=np.uint8)
        inv = np.argsort(Pk, axis=1)
        TQ = T[inv[:, :, None], inv[:, None, :]]
        L[:] = np.take_along_axis(Pk, TQ.reshape(len(Pk), -1), axis=1).reshape(len(Pk), n, n)
        out.append(L)
    return np.concatenate(out), len(auts)


def run_n10():
    D5 = from_perms(closure([(1, 2, 3, 4, 0), (0, 4, 3, 2, 1)]))
    gs = {'Z10': cyclic(10), 'D5': D5}
    tabs = {}
    for h, T in gs.items():
        t0 = time.time()
        L, na = labelled_coset_reps(T)
        print(f'{h}: |Aut|={na}, labelled tables={len(L)} ({time.time()-t0:.0f}s)', flush=True)
        tabs[h] = L
    C = np.concatenate([tabs['Z10'], tabs['D5']])
    print('total labelled groups of order 10:', len(C), '(OEIS A034383: 1088640)', flush=True)
    res = {}
    for h, T in gs.items():
        t0 = time.time()
        k, S, r = md_hitting(T, C, init=3000, verbose=True)
        print(f'n=10 {h} md_group={k} cue={S} rounds={r} ({time.time()-t0:.0f}s)', flush=True)
        res[h] = {'k': k, 'cue': S}
    json.dump(res, open('../out/md_n10.json', 'w'), indent=1)


if __name__ == '__main__':
    if sys.argv[1] == 'iso':
        run_iso()
    else:
        run_n10()
