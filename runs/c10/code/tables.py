"""Basic objects: group Cayley tables of order <= 8, labelled relabelings,
Latin squares (n<=5 by enumeration), loops (n<=6 by enumeration).
A 'table' is an n x n uint8 array T with T[a,b] = a*b on labels 0..n-1."""
import itertools, functools
import numpy as np


def cyclic(n):
    return np.array([[(i + j) % n for j in range(n)] for i in range(n)], dtype=np.uint8)


def direct(T1, T2):
    n1, n2 = len(T1), len(T2)
    n = n1 * n2
    T = np.zeros((n, n), dtype=np.uint8)
    for a in range(n):
        for b in range(n):
            a1, a2 = divmod(a, n2)
            b1, b2 = divmod(b, n2)
            T[a, b] = T1[a1, b1] * n2 + T2[a2, b2]
    return T


def from_perms(elems):
    """Cayley table of a permutation group given as list of tuples (closed set)."""
    idx = {e: i for i, e in enumerate(elems)}
    n = len(elems)
    T = np.zeros((n, n), dtype=np.uint8)
    for i, p in enumerate(elems):
        for j, q in enumerate(elems):
            r = tuple(p[q[x]] for x in range(len(p)))  # p o q
            T[i, j] = idx[r]
    return T


def closure(gens):
    d = len(gens[0])
    e = tuple(range(d))
    S = {e}
    frontier = [e]
    while frontier:
        nf = []
        for x in frontier:
            for g in gens:
                y = tuple(x[g[i]] for i in range(d))
                if y not in S:
                    S.add(y)
                    nf.append(y)
        frontier = nf
    el = sorted(S)
    el.remove(e)
    return [e] + el


def S3():
    return from_perms(closure([(1, 0, 2), (1, 2, 0)]))


def D4():
    return from_perms(closure([(1, 2, 3, 0), (3, 2, 1, 0)]))


def Q8():
    # quaternions as signed units: represent by 8 elements via multiplication table
    # elements: 1,-1,i,-i,j,-j,k,-k encoded (sign, unit) unit in {1,i,j,k}
    units = ['1', 'i', 'j', 'k']
    mult = {('1', x): (1, x) for x in units}
    mult.update({(x, '1'): (1, x) for x in units})
    mult.update({('i', 'i'): (-1, '1'), ('j', 'j'): (-1, '1'), ('k', 'k'): (-1, '1'),
                 ('i', 'j'): (1, 'k'), ('j', 'k'): (1, 'i'), ('k', 'i'): (1, 'j'),
                 ('j', 'i'): (-1, 'k'), ('k', 'j'): (-1, 'i'), ('i', 'k'): (-1, 'j')})
    els = [(s, u) for u in units for s in (1, -1)]
    idx = {e: i for i, e in enumerate(els)}
    T = np.zeros((8, 8), dtype=np.uint8)
    for a, (s1, u1) in enumerate(els):
        for b, (s2, u2) in enumerate(els):
            s, u = mult[(u1, u2)]
            T[a, b] = idx[(s * s1 * s2, u)]
    return T


def groups_of_order(n):
    """Dict name -> Cayley table (labels 0..n-1, 0 = identity) for every group of order n <= 8."""
    Z = cyclic
    if n == 1:
        return {'Z1': Z(1)}
    if n in (2, 3, 5, 7):
        return {f'Z{n}': Z(n)}
    if n == 4:
        return {'Z4': Z(4), 'V4': direct(Z(2), Z(2))}
    if n == 6:
        return {'Z6': Z(6), 'S3': S3()}
    if n == 8:
        return {'Z8': Z(8), 'Z4xZ2': direct(Z(4), Z(2)), 'Z2^3': direct(direct(Z(2), Z(2)), Z(2)),
                'D4': D4(), 'Q8': Q8()}
    if n == 9:
        return {'Z9': Z(9), 'Z3xZ3': direct(Z(3), Z(3))}
    if n == 10:
        return {'Z10': Z(10), 'D5': from_perms(closure([(1, 2, 3, 4, 0), (0, 4, 3, 2, 1)]))}
    raise ValueError(n)


def is_latin(T):
    n = len(T)
    full = set(range(n))
    return all(set(T[i]) == full for i in range(n)) and all(set(T[:, j]) == full for j in range(n))


def is_assoc(T):
    n = len(T)
    A = T[T, :]  # A[a,b,c] = T[T[a,b],c]
    B = T[:, T]  # B[a,b,c] = T[a,T[b,c]]
    return bool(np.all(A == B))


def identity_of(T):
    n = len(T)
    ar = np.arange(n)
    for e in range(n):
        if np.all(T[e] == ar) and np.all(T[:, e] == ar):
            return e
    return None


def is_comm(T):
    return bool(np.all(T == T.T))


def relabel(T, s):
    """s: array, s[old] = new label.  Returns T' with T'[s[a], s[b]] = s[T[a,b]]."""
    n = len(T)
    inv = np.empty(n, dtype=np.int64)
    inv[s] = np.arange(n)
    return s[T[np.ix_(inv, inv)]].astype(np.uint8)


def all_relabelings(T):
    n = len(T)
    seen = {}
    for p in itertools.permutations(range(n)):
        s = np.array(p, dtype=np.int64)
        R = relabel(T, s)
        key = R.tobytes()
        if key not in seen:
            seen[key] = R
    return np.array(list(seen.values()), dtype=np.uint8)


@functools.lru_cache(maxsize=None)
def labelled_groups(n):
    """Returns (tables [G,n,n], names list) of all labelled group tables of order n."""
    tabs, names = [], []
    for name, T in groups_of_order(n).items():
        R = all_relabelings(T)
        tabs.append(R)
        names += [name] * len(R)
    return np.concatenate(tabs), names


def _latin_first_row_identity(n):
    """All Latin squares with first row = 0..n-1 (backtracking)."""
    out = []
    T = -np.ones((n, n), dtype=np.int64)
    T[0] = np.arange(n)
    rowused = [set() for _ in range(n)]
    colused = [set() for _ in range(n)]
    for j in range(n):
        rowused[0].add(j)
        colused[j].add(j)
    cells = [(i, j) for i in range(1, n) for j in range(n)]

    def rec(k):
        if k == len(cells):
            out.append(T.copy())
            return
        i, j = cells[k]
        for v in range(n):
            if v not in rowused[i] and v not in colused[j]:
                T[i, j] = v
                rowused[i].add(v)
                colused[j].add(v)
                rec(k + 1)
                rowused[i].discard(v)
                colused[j].discard(v)
                T[i, j] = -1
    rec(0)
    return np.array(out, dtype=np.uint8)


@functools.lru_cache(maxsize=None)
def all_latin(n):
    """All Latin squares of order n (n<=5): array [L, n, n]."""
    base = _latin_first_row_identity(n)
    perms = np.array(list(itertools.permutations(range(n))), dtype=np.uint8)
    # symbol permutation sigma applied to base square: sigma[base]
    res =np.empty((len(perms) * len(base), n, n), dtype=np.uint8)
    k = 0
    for s in perms:
        res[k:k + len(base)] = s[base]
        k += len(base)
    return res


@functools.lru_cache(maxsize=None)
def all_loops(n):
    """All labelled loops (Latin squares with a two-sided identity) of order n (n<=6)."""
    out = []
    T = -np.ones((n, n), dtype=np.int64)
    T[0] = np.arange(n)
    T[:, 0] = np.arange(n)
    rowused = [set() for _ in range(n)]
    colused = [set() for _ in range(n)]
    for j in range(n):
        rowused[0].add(j); colused[j].add(j)
    for i in range(1, n):
        rowused[i].add(i); colused[0].add(i)
    cells = [(i, j) for i in range(1, n) for j in range(1, n)]

    def rec(k):
        if k == len(cells):
            out.append(T.copy())
            return
        i, j = cells[k]
        for v in range(n):
            if v not in rowused[i] and v not in colused[j]:
                T[i, j] = v
                rowused[i].add(v); colused[j].add(v)
                rec(k + 1)
                rowused[i].discard(v); colused[j].discard(v)
                T[i, j] = -1
    rec(0)
    norm = np.array(out, dtype=np.uint8)
    res = [norm]
    for e in range(1, n):
        s = np.arange(n)
        s[0], s[e] = e, 0
        res.append(np.array([relabel(L, s) for L in norm], dtype=np.uint8))
    return np.concatenate(res)


if __name__ == '__main__':
    for n in range(1, 9):
        G, names = labelled_groups(n)
        print(n, len(G), {nm: names.count(nm) for nm in dict.fromkeys(names)})
        for name, T in groups_of_order(n).items():
            assert is_latin(T) and is_assoc(T) and identity_of(T) == 0, name
    for n in range(1, 6):
        L = all_latin(n)
        assert len({x.tobytes() for x in L}) == len(L)
        print('latin', n, len(L))
    for n in range(1, 7):
        Lp = all_loops(n)
        print('loops', n, len(Lp), 'assoc', sum(is_assoc(x) for x in Lp))
