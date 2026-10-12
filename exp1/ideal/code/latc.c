/* Exact counting of Latin-square completions of a partial square, with per-cell value marginals.
   lat_count(n, grid[n*n] (-1 = empty), marg[n*n*n] (out), cap_nodes) -> total count (double),
   or -1.0 if the node cap was hit (result invalid) or -2.0 if the partial square is not Latin.
   marg[c*n+v] = number of completions with cell c = v (cue cells get the total on their value).
   DFS, branching on the empty cell with fewest candidates; marginals are accumulated as subtree
   counts at each assignment node (each non-cue cell is assigned exactly once on every root-leaf path).
   Build: gcc -O2 -shared -o latc.dll latc.c */
#include <string.h>

static int N, NN;
static int grid[64];
static unsigned rowm[8], colm[8];
static double *MARG;
static double nodes, cap;
static int aborted;

static double dfs(void) {
    if (aborted) return 0.0;
    nodes += 1.0;
    if (nodes > cap) { aborted = 1; return 0.0; }
    unsigned full = (1u << N) - 1u;
    int best = -1, bc = 99; unsigned bm = 0;
    for (int c = 0; c < NN; c++) {
        if (grid[c] >= 0) continue;
        unsigned m = ~(rowm[c / N] | colm[c % N]) & full;
        int k = __builtin_popcount(m);
        if (k < bc) { bc = k; best = c; bm = m; if (k <= 1) break; }
    }
    if (best < 0) return 1.0;
    if (bc == 0) return 0.0;
    /* dead-end check: every symbol missing from a row/col must have a free slot */
    for (int r = 0; r < N; r++) {
        unsigned avail = 0;
        for (int q = 0; q < N; q++) if (grid[r * N + q] < 0) avail |= ~(rowm[r] | colm[q]) & full;
        if ((avail | rowm[r]) != full) return 0.0;
    }
    for (int q = 0; q < N; q++) {
        unsigned avail = 0;
        for (int r = 0; r < N; r++) if (grid[r * N + q] < 0) avail |= ~(rowm[r] | colm[q]) & full;
        if ((avail | colm[q]) != full) return 0.0;
    }
    int r = best / N, q = best % N;
    double tot = 0.0;
    for (int v = 0; v < N; v++) {
        if (!(bm >> v & 1u)) continue;
        grid[best] = v; rowm[r] |= 1u << v; colm[q] |= 1u << v;
        double s = dfs();
        grid[best] = -1; rowm[r] &= ~(1u << v); colm[q] &= ~(1u << v);
        if (aborted) return 0.0;
        MARG[best * N + v] += s;
        tot += s;
    }
    return tot;
}

__declspec(dllexport) double lat_count(int n, const int *g, double *marg, double cap_nodes) {
    N = n; NN = n * n; MARG = marg; nodes = 0.0; cap = cap_nodes; aborted = 0;
    memset(rowm, 0, sizeof rowm); memset(colm, 0, sizeof colm);
    memset(marg, 0, sizeof(double) * NN * n);
    for (int c = 0; c < NN; c++) {
        grid[c] = g[c];
        if (g[c] >= 0) {
            unsigned b = 1u << g[c];
            if ((rowm[c / n] & b) || (colm[c % n] & b)) return -2.0;
            rowm[c / n] |= b; colm[c % n] |= b;
        }
    }
    double tot = dfs();
    if (aborted) return -1.0;
    for (int c = 0; c < NN; c++) if (g[c] >= 0) marg[c * n + g[c]] = tot;
    return tot;
}

__declspec(dllexport) double lat_nodes(void) { return nodes; }
