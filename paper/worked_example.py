"""Worked forcing-class example for Section 3.2 (order 5): checks by brute force over all Latin squares of order 5, and
with the stimulus engine (runs/c10/code/engine.py), that the cell b*b is associativity-forced (G) for the cue below.

Labels e, a, b, c, d; the hidden table is Z5 with e = 0, a = 1, b = 2, c = 3, d = 4 (a*a = b, a*b = c, a*c = d).
Cue: e's row and column, a*a = b and a*b = c.

usage: python worked_example.py   -> writes worked_example.md
"""
import itertools, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
L, n = 'eabcd', 5
T = np.array([[(i + j) % n for j in range(n)] for i in range(n)])
CUE = {(0, j): int(T[0, j]) for j in range(n)}
CUE.update({(i, 0): int(T[i, 0]) for i in range(n)})
CUE[(1, 1)] = int(T[1, 1])   # a*a = b
CUE[(1, 2)] = int(T[1, 2])   # a*b = c
QUERY = (2, 2)               # b*b
SHOWN = ((0, 1, 2, 3, 4), (1, 2, 3, 4, 0), (2, 4, 0, 1, 3), (3, 0, 4, 2, 1), (4, 3, 1, 0, 2))   # the loop printed in the paper


def latin_squares():
    perms = list(itertools.permutations(range(n)))
    def rec(rows):
        if len(rows) == n:
            yield tuple(rows); return
        for p in perms:
            if all(p[j] != r[j] for r in rows for j in range(n)):
                yield from rec(rows + [p])
    yield from rec([])


def fits(t):
    return all(t[i][j] == v for (i, j), v in CUE.items())


def identity(t):
    return any(all(t[x][y] == y and t[y][x] == y for y in range(n)) for x in range(n))


def assoc(t):
    return all(t[t[x][y]][z] == t[x][t[y][z]] for x in range(n) for y in range(n) for z in range(n))


def main():
    lat = list(latin_squares())
    fit = [t for t in lat if fits(t)]
    loops = [t for t in fit if identity(t)]
    groups = [t for t in loops if assoc(t)]
    vals = lambda S: sorted({L[t[QUERY[0]][QUERY[1]]] for t in S})
    s = SHOWN
    assert s in loops and not assoc(s)
    sys.path.insert(0, os.path.join(ROOT, 'runs', 'c10', 'code'))
    import engine
    res = engine.analyze(T, sorted(i * n + j for (i, j) in CUE), ['latin', 'loop'] + [k for k in engine.universe(n) if k.startswith('iso:')])
    q = QUERY[0] * n + QUERY[1]
    eng = {k: (N, [L[v] for v in range(n) if cnt[q, v] > 0]) for k, (N, cnt) in res.items()}
    md = ['# Worked forcing-class example (Section 3.2)\n',
          'Order 5, labels e a b c d; cue: the row and column of e, a*a = b, a*b = c; query b*b.\n',
          f'- Latin squares of order 5: {len(lat):,}',
          f'- consistent with the cue: {len(fit)} Latin squares, {len(loops)} loops, {len(groups)} group table(s)',
          f'- values of b*b: Latin {vals(fit)}, loop {vals(loops)}, group {vals(groups)} -> G-forced',
          f'- a*c over all consistent Latin squares: {sorted({L[t[1][3]] for t in fit})} -> L-forced',
          f'- printed loop: b*b = {L[s[2][2]]}, (a*a)*b = {L[s[s[1][1]][2]]}, a*(a*b) = {L[s[1][s[1][2]]]}; associative: {assoc(s)}',
          f'- commutative loops among those consistent with the cue: {sum(1 for t in loops if all(t[x][y] == t[y][x] for x in range(n) for y in range(n)))}',
          '- engine: ' + '; '.join(f'{k} {N} tables, b*b in {v}' for k, (N, v) in eng.items())]
    open(os.path.join(HERE, 'worked_example.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
