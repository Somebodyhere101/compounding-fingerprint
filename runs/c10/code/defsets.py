"""Minimum defining sets ("teaching sets") of group Cayley tables under several priors.

For a target table T in a hypothesis class C (a set of labelled n x n tables),
a cue S (set of cells) is DEFINING if T is the only member of C agreeing with T on S.
md(T, C) = min |S| over defining cues.   Exhaustive search with bitsets.

usage: python defsets.py [n1,n2,...] [outfile]  (default 2..7 -> ../out/defsets.json)
"""
import sys, json, itertools, time
import numpy as np
from tables import labelled_groups, groups_of_order, all_latin, all_loops, is_comm, relabel


def bitsets_for(T, C):
    """For each cell (flattened index c) the python-int bitset of members of C agreeing with T at c."""
    n = len(T)
    flatC = C.reshape(len(C), -1)
    flatT = T.reshape(-1)
    bs = []
    for c in range(n * n):
        agree = (flatC[:, c] == flatT[c])
        # pack to python int
        b = int.from_bytes(np.packbits(agree, bitorder='little').tobytes(), 'little')
        bs.append(b)
    full = int.from_bytes(np.packbits(np.ones(len(C), bool), bitorder='little').tobytes(), 'little')
    return bs, full


def min_defining(T, C, kmax=None, count_all=True):
    """Return (k_min, number of defining cues of size k_min, example cue)."""
    n = len(T)
    bs, full = bitsets_for(T, C)
    cells = list(range(n * n))
    kmax = kmax if kmax is not None else n * n
    for k in range(0, kmax + 1):
        found, example = 0, None
        # DFS with incremental AND
        stack = [(0, full, ())]
        def rec(start, cur, chosen):
            nonlocal found, example
            if len(chosen) == k:
                if cur & (cur - 1) == 0:
                    found += 1
                    if example is None:
                        example = chosen
                return
            # remaining picks
            need = k - len(chosen)
            for c in range(start, n * n - need + 1):
                rec(c + 1, cur & bs[c], chosen + (c,))
                if found and not count_all:
                    return
        rec(0, full, ())
        if found:
            ex = [(c // n, c % n, int(T[c // n, c % n])) for c in example]
            return k, found, ex
    return None, 0, None


def main(ns=(2, 3, 4, 5, 6, 7), out='../out/defsets.json'):
    res = {}
    for n in ns:
        G, names = labelled_groups(n)
        names = np.array(names)
        comm = np.array([is_comm(x) for x in G])
        # cyclic: iso class name starts with 'Z' and equals Z{n}
        cyc = names == f'Z{n}'
        classes = {'group': G, 'abelian': G[comm], 'cyclic': G[cyc]}
        if n <= 6:
            classes['loop'] = all_loops(n)
        if n <= 5:
            classes['latin'] = all_latin(n)
        for gname, T in groups_of_order(n).items():
            row = {}
            for cname, C in classes.items():
                # T must be in C
                inC = any(np.array_equal(T, x) for x in C) if len(C) < 300000 else True
                if not inC:
                    continue
                t0 = time.time()
                k, cnt, ex = min_defining(T, C)
                row[cname] = {'k': k, 'count': cnt, 'example': ex, 'secs': round(time.time() - t0, 2)}
                print(f'n={n} {gname:6s} prior={cname:8s} |C|={len(C):7d}  md={k}  #min-cues={cnt}  ex={ex}  ({time.time()-t0:.1f}s)', flush=True)
            res[f'{n}:{gname}'] = row
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    ns = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [2, 3, 4, 5, 6, 7]
    out = sys.argv[2] if len(sys.argv) > 2 else '../out/defsets.json'
    main(ns, out)
