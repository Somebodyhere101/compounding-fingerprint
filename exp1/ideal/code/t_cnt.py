import sys, time, numpy as np
sys.path.insert(0, r'../../../runs/c10/code')
from cnt import latin_counts, loop_counts
from engine import universe, counts_for
from tables import all_latin, all_loops, groups_of_order, relabel
for n in (3,4,5):
    N,_ = latin_counts(n, [-1]*(n*n)); print('empty latin', n, N, len(all_latin(n)))
    N,_ = loop_counts(n, [-1]*(n*n)); print('empty loop', n, N, len(all_loops(n)))
rng = np.random.default_rng(0)
bad = 0; tests = 0
for n, k, reps in ((5, 6, 200), (5, 3, 100), (6, 10, 100)):
    U = universe(n)
    L = U.get('latin'); Lp = U['loop']
    for _ in range(reps):
        H = list(groups_of_order(n))
        T = relabel(groups_of_order(n)[H[rng.integers(len(H))]], rng.permutation(n))
        S = np.sort(rng.choice(n*n, k, replace=False))
        g = -np.ones(n*n, int); g[S] = T.reshape(-1)[S]
        if L is not None:
            Ne, ce = counts_for(L, T, S); Nc, cc = latin_counts(n, g)
            tests += 1; bad += (Ne != Nc) or not np.array_equal(ce, cc)
        Ne, ce = counts_for(Lp, T, S); Nc, cc = loop_counts(n, g)
        tests += 1; bad += (Ne != Nc) or not np.array_equal(ce, cc)
print('enum vs C mismatches', bad, 'of', tests)
# timing on n=8 group cues
for n, k in ((6, 10), (8, 16)):
    H = list(groups_of_order(n)); ts = []; Ns = []; Ls = []
    for _ in range(30):
        T = relabel(groups_of_order(n)[H[rng.integers(len(H))]], rng.permutation(n))
        S = np.sort(rng.choice(n*n, k, replace=False))
        g = -np.ones(n*n, int); g[S] = T.reshape(-1)[S]
        t0 = time.time(); N, _ = latin_counts(n, g); t1 = time.time(); Nl, _ = loop_counts(n, g); t2 = time.time()
        ts.append((t1-t0, t2-t1)); Ns.append(N); Ls.append(Nl)
    ts = np.array(ts)
    print(n, k, 'latin N median/max', np.median(Ns), max(Ns), 'loop N median/max', np.median(Ls), max(Ls), 'time lat mean/max', ts[:,0].mean(), ts[:,0].max(), 'loop', ts[:,1].mean(), ts[:,1].max())
