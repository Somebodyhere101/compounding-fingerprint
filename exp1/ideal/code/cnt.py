"""ctypes wrapper for latc.dll: exact Latin / loop completion counts with per-cell value marginals."""
import ctypes, os
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_here, 'latc.dll'))
_lib.lat_count.restype = ctypes.c_double
_lib.lat_count.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.c_double]
_lib.lat_nodes.restype = ctypes.c_double

CAP = 4e9  # node cap per call


def latin_counts(n, grid, cap=CAP):
    """grid: flat length n*n ints (-1 empty). Returns (N, marg[n*n, n]) as floats; N=None if capped."""
    g = (ctypes.c_int * (n * n))(*[int(x) for x in grid])
    m = (ctypes.c_double * (n * n * n))()
    tot = _lib.lat_count(n, g, m, float(cap))
    if tot == -1.0:
        return None, None
    if tot == -2.0:
        return 0.0, np.zeros((n * n, n))
    return tot, np.frombuffer(m, dtype=np.float64).reshape(n * n, n).copy()


def loop_counts(n, grid, cap=CAP):
    """Exact count of loops (Latin squares with a two-sided identity) completing the cue, with marginals.
    Each loop has a unique identity e, so the sum over e of Latin completions with row e and column e
    fixed to the identity is exact."""
    tot = 0.0
    marg = np.zeros((n * n, n))
    for e in range(n):
        g = list(int(x) for x in grid)
        ok = True
        for j in range(n):
            for c, v in ((e * n + j, j), (j * n + e, j)):
                if g[c] >= 0 and g[c] != v:
                    ok = False
                g[c] = v
        if not ok:
            continue
        N, M = latin_counts(n, g, cap)
        if N is None:
            return None, None
        tot += N
        marg += M
    return tot, marg
