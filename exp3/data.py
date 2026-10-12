"""Experiment 3 data: in-context group-table completion sequences in a compact token vocabulary.

Vocabulary: PAD=0, BOS=1, MUL='*'=2, EQ='='=3, SEMI=';'=4 (end of fact), SEP='|'=5 (table separator),
then symbols s0..s(S-1) at ids 6..6+S-1.

A sequence (one trial):
    BOS  [table 1] | [table 2] | ... | [table t] | [target]
Each table is all n^2 facts 'a * b = c ;' in random order. The target is also a full table in random order; its
first k facts are the 'cue' and the rest are the 'remaining' facts (with fresh random order there is no other
difference, so k only matters for loss_on='target' and for evaluation metrics).
Symbols: a fresh random injection {0..n-1} -> symbols per table, no symbol reused across the tables of one
sequence (except in pool mode, see below). Because a uniformly random injection composed with a fixed base table
equals a uniformly random relabelling followed by a random injection, every labelled table of the chosen
isomorphism type is equally likely.

Loss (Todd et al., In-Context Algebra, arXiv 2512.16902): next-token loss only on answer tokens (the token after
'='), for every fact except the very first fact of the sequence ('all'), or only on the target's facts after its k
cues ('target').

Training distribution (Cfg):
  orders     orders n (default 4..8; 9, 10 available: Z9, Z3xZ3, Z10, D5)
  types      'all' | 'cyclic' | 'abelian' | 'noheld' (all except HELDOUT = S3@6, D4@8, Q8@8)
  t_min..t_max  number of history tables (capped per n so the sequence fits max_len)
  k_frac     k sampled uniformly in [ceil(lo*n^2), floor(hi*n^2)]
  pool_M     >0: low task diversity. A fixed pool of pool_M labelled symbol tables PER ORDER (type + fixed symbol
             injection, drawn once from the seed); every table of every sequence is drawn from the pool.
             Tables in one sequence are distinct pool entries, chosen to minimise symbol overlap, but symbols CAN
             repeat across tables of a sequence in this mode (documented deviation; unavoidable with S=64).
  env_mix    {'group': w, 'latin': w, 'loop': w}: per-sequence environment; history and target come from the same
             environment. 'latin' = identity-free Latin squares (as Exp-1's scramble control), 'loop' =
             non-associative loops (n>=5; at n=4 every loop is a group, so 'loop' falls back to 'latin').
             'isocopy' = exemplar environment: history tables are identity-free Latin squares and the target is a
             relabelled (isomorphic) copy of one uniformly chosen history table (t=0: a fresh Latin square).
             A model that fits it must do in-context isomorphism matching (N_iso by construction) and never
             sees a group or an identity element.
"""
import math
from dataclasses import dataclass, field, asdict
import numpy as np
from tables import groups_of_order, is_comm, is_assoc, identity_of

PAD, BOS, MUL, EQ, SEMI, SEP = 0, 1, 2, 3, 4, 5
NSPEC = 6
HELDOUT = {(6, 'S3'), (8, 'D4'), (8, 'Q8')}
FACT = 6  # tokens per fact: a * b = c ;


def vocab_size(S):
    return NSPEC + S


def seq_len(n, t):
    """Length of a full training sequence (BOS + (t+1) tables of n^2 facts + t separators)."""
    return 1 + (t + 1) * n * n * FACT + t


def allowed_types(n, types):
    G = groups_of_order(n)
    out = []
    for name, T in G.items():
        if types == 'all':
            ok = True
        elif types == 'cyclic':
            ok = name == f'Z{n}'
        elif types == 'abelian':
            ok = is_comm(T)
        elif types == 'noheld':
            ok = (n, name) not in HELDOUT
        else:
            raise ValueError(types)
        if ok:
            out.append(name)
    return out


@dataclass
class Cfg:
    orders: tuple = (4, 5, 6, 7, 8)
    types: str = 'noheld'
    t_min: int = 0
    t_max: int = 4
    k_frac: tuple = (0.15, 0.45)
    S: int = 64
    pool_M: int = 0
    env_mix: dict = field(default_factory=lambda: {'group': 1.0})
    max_len: int = 2400
    loss_on: str = 'all'
    seed: int = 0

    def to_dict(self):
        d = asdict(self)
        d['orders'] = list(self.orders); d['k_frac'] = list(self.k_frac)
        return d


def parse_env_mix(s):
    d = {}
    for part in s.split(','):
        k, v = part.split('=')
        d[k.strip()] = float(v)
    assert set(d) <= {'group', 'latin', 'loop', 'isocopy'}, d
    return d


# ---------------------------------------------------------------- non-group table pools
def _rand_latin(n, rng, loop=False, tries=200):
    """Random Latin square by randomised backtracking. loop=True fixes row 0 / column 0 to the identity."""
    for _ in range(tries):
        X = -np.ones((n, n), dtype=np.int64)
        start = 0
        if loop:
            X[0] = np.arange(n); X[:, 0] = np.arange(n)
        cells = [(r, c) for r in range(n) for c in range(n) if X[r, c] < 0]
        budget = [20000]

        def rec(i):
            if i == len(cells):
                return True
            budget[0] -= 1
            if budget[0] < 0:
                return False
            r, c = cells[i]
            for v in rng.permutation(n):
                if v in X[r] or v in X[:, c]:
                    continue
                X[r, c] = v
                if rec(i + 1):
                    return True
                X[r, c] = -1
            return False
        if rec(0):
            return X.astype(np.uint8)
    raise RuntimeError('latin generation failed')


def _has_identity(T):
    return identity_of(T) is not None


class Pools:
    """Lazily built, seed-deterministic pools of base tables for the 'latin' and 'loop' environments."""

    def __init__(self, seed, size=256):
        self.seed = seed; self.size = size; self.cache = {}

    def get(self, env, n):
        key = (env, n)
        if key not in self.cache:
            rng = np.random.default_rng([self.seed, 7919, n, 1 if env == 'loop' else 0])
            out, seen = [], set()
            for _ in range(self.size * 20):
                if env == 'latin':
                    X = _rand_latin(n, rng)
                else:
                    X = _rand_latin(n, rng, loop=True)
                    if is_assoc(X):
                        continue
                b = X.tobytes()
                if b in seen:
                    continue
                seen.add(b); out.append(X)
                if len(out) >= self.size:
                    break
            self.cache[key] = np.array(out, dtype=np.uint8).reshape(len(out), n * n)
        return self.cache[key]


# ---------------------------------------------------------------- generator
class Gen:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.V = vocab_size(cfg.S)
        self.base = {}  # n -> (names, array [ntypes, n*n])
        for n in set(cfg.orders) | {5, 6, 8}:
            G = groups_of_order(n)
            names = list(G.keys())
            self.base[n] = (names, np.array([G[h].reshape(-1) for h in names], dtype=np.int64))
        self.pools = Pools(cfg.seed)
        for e in cfg.env_mix:          # build non-group pools up front (a few seconds) so training never stalls
            if e != 'group':
                for n in cfg.orders:
                    self.pools.get('latin' if (e == 'isocopy' or (e == 'loop' and n < 5)) else e, n)
        self.tmax = {n: max([t for t in range(cfg.t_min, cfg.t_max + 1) if seq_len(n, t) <= cfg.max_len] or [-1])
                     for n in cfg.orders}
        for n in cfg.orders:
            assert self.tmax[n] >= cfg.t_min, f'order {n} does not fit max_len={cfg.max_len}'
            assert (self.tmax[n] + 1) * n <= cfg.S or cfg.pool_M > 0, 'not enough symbols for fresh labels'
            assert allowed_types(n, cfg.types), f'no allowed group types at n={n}'
        self.pool = {}
        if cfg.pool_M > 0:
            assert set(cfg.env_mix) == {'group'}, 'pool mode supports the group environment only'
            for n in cfg.orders:
                rng = np.random.default_rng([cfg.seed, 104729, n])
                names = allowed_types(n, cfg.types)
                tidx = rng.integers(0, len(names), cfg.pool_M)
                syms = np.array([rng.permutation(cfg.S)[:n] for _ in range(cfg.pool_M)])
                self.pool[n] = ([names[i] for i in tidx], np.array([self.base[n][0].index(names[i]) for i in tidx]), syms)

    # -------- distribution over (n, t)
    def nt_choices(self):
        out = []
        for n in self.cfg.orders:
            for t in range(self.cfg.t_min, self.tmax[n] + 1):
                out.append((n, t))
        return out

    def pick_nt(self, rng):
        """(n,t) for one homogeneous batch, with prob proportional to pi(n,t)*len(n,t) (pi uniform over n,
        then over t), so that with B ~ tokens/len the per-sequence distribution is pi in expectation."""
        ch = self.nt_choices()
        w = np.array([1.0 / len([1 for m, _ in ch if m == n]) * seq_len(n, t) for n, t in ch])
        i = rng.choice(len(ch), p=w / w.sum())
        return ch[i]

    # -------- tables
    def _group_tables(self, n, names, B, rng):
        allnames, arr = self.base[n]
        idx = np.array([allnames.index(h) for h in names])
        pick = idx[rng.integers(0, len(idx), B)]
        return arr[pick], [allnames[i] for i in pick]

    def _env_tables(self, n, env, B, rng):
        if env == 'loop' and n < 5:
            env = 'latin'
        P = self.pools.get(env, n).astype(np.int64)
        X = P[rng.integers(0, len(P), B)].reshape(B, n, n)
        if env == 'latin':
            # random isotopy (row perm, col perm); symbols are relabelled by the injection later.
            # Reject results that acquired a two-sided identity.
            out = X.copy()
            todo = np.arange(B)
            ar = np.arange(n)
            while len(todo):
                m = len(todo)
                pr = np.argsort(rng.random((m, n)), axis=1); pc = np.argsort(rng.random((m, n)), axis=1)
                Y = np.take_along_axis(X[todo], pr[:, :, None], axis=1)
                Y = np.take_along_axis(Y, pc[:, None, :], axis=2)
                rid = np.all(Y == ar[None, None, :], axis=2)              # row e is the identity map
                cid = np.all(Y == ar[None, :, None], axis=1)              # column e is the identity map
                bad = np.any(rid & cid, axis=1)
                out[todo[~bad]] = Y[~bad]
                todo = todo[bad]
            X = out
        return X.reshape(B, n * n), [env] * B

    def tables(self, n, spec, B, rng):
        """spec = ('group', [type names]) | ('latin', None) | ('loop', None). Returns ([B, n*n] int64, names)."""
        env, names = spec
        if env == 'group':
            return self._group_tables(n, names, B, rng)
        return self._env_tables(n, env, B, rng)

    # -------- sequences
    def make(self, n, t, B, rng, hist=None, tgt=None, k=None, env=None):
        """Build B sequences of order n with t history tables.
        hist / tgt: table specs (see tables()); default: training distribution (env from env_mix).
        k: int or None (sampled from k_frac). Returns dict of numpy arrays."""
        cfg = self.cfg
        n2 = n * n
        T1 = t + 1
        names_tr = allowed_types(n, cfg.types)
        if hist is None and tgt is None and env is None and set(cfg.env_mix) != {'group'}:
            envs = list(cfg.env_mix); w = np.array([cfg.env_mix[e] for e in envs]); w = w / w.sum()
            env_of = rng.choice(len(envs), size=B, p=w)
        else:
            env_of = None
        tabs = np.zeros((B, T1, n2), dtype=np.int64)
        tnames = [[None] * T1 for _ in range(B)]
        pool_syms = None
        if cfg.pool_M > 0 and hist is None and tgt is None:
            names_p, tix, syms_p = self.pool[n]
            M = len(names_p)
            pool_syms = np.zeros((B, T1, n), dtype=np.int64)
            for b in range(B):
                best = None
                for _ in range(10):
                    sel = rng.choice(M, size=T1, replace=M < T1)
                    ov = T1 * n - len(set(syms_p[sel].reshape(-1).tolist()))
                    if best is None or ov < best[0]:
                        best = (ov, sel)
                    if ov == 0:
                        break
                sel = best[1]
                tabs[b] = self.base[n][1][tix[sel]]
                pool_syms[b] = syms_p[sel]
                for j, s in enumerate(sel):
                    tnames[b][j] = names_p[s]
        else:
            def fill(cols, spec, rows):
                for j in cols:
                    X, nm = self.tables(n, spec, len(rows), rng)
                    tabs[rows, j] = X
                    for r, h in zip(rows, nm):
                        tnames[r][j] = h
            allrows = np.arange(B)
            if env_of is not None:
                for ei, e in enumerate(envs):
                    rows = allrows[env_of == ei]
                    if len(rows) and e == 'isocopy':
                        fill(range(T1), ('latin', None), rows)
                        if t >= 1:   # target = isomorphic copy of a random history table (fresh symbols below)
                            src = rng.integers(0, t, len(rows))
                            tabs[rows, t] = tabs[rows, src]
                    elif len(rows):
                        spec = ('group', names_tr) if e == 'group' else (e, None)
                        fill(range(T1), spec, rows)
            elif env == 'isocopy':
                fill(range(T1), ('latin', None), allrows)
                if t >= 1:
                    src = rng.integers(0, t, B)
                    tabs[allrows, t] = tabs[allrows, src]
            else:
                e = env or 'group'
                dflt = ('group', names_tr) if e == 'group' else (e, None)
                fill(range(t), hist or dflt, allrows)
                fill([t], tgt or dflt, allrows)
        # symbol injections: disjoint across tables of a sequence (fresh labels)
        if pool_syms is None:
            perm = np.argsort(rng.random((B, cfg.S)), axis=1)[:, :T1 * n]
            syms = perm.reshape(B, T1, n)
        else:
            syms = pool_syms
        order = np.argsort(rng.random((B, T1, n2)), axis=2)  # fact order per table
        a, bb = order // n, order % n
        c = np.take_along_axis(tabs, order, axis=2)
        sa = np.take_along_axis(syms, a, axis=2) + NSPEC
        sb = np.take_along_axis(syms, bb, axis=2) + NSPEC
        sc = np.take_along_axis(syms, c, axis=2) + NSPEC
        facts = np.stack([sa, np.full_like(sa, MUL), sb, np.full_like(sa, EQ), sc, np.full_like(sa, SEMI)], axis=3)
        body = facts.reshape(B, T1, n2 * FACT)
        body = np.concatenate([body, np.full((B, T1, 1), SEP, dtype=np.int64)], axis=2).reshape(B, -1)[:, :-1]
        tok = np.concatenate([np.full((B, 1), BOS, dtype=np.int64), body], axis=1)
        L = tok.shape[1]
        assert L == seq_len(n, t)
        # answer positions: table j, fact i -> 1 + j*(n2*FACT+1) + i*FACT + 4
        j_idx = np.arange(T1)[:, None]; i_idx = np.arange(n2)[None, :]
        ans_pos = 1 + j_idx * (n2 * FACT + 1) + i_idx * FACT + 4  # [T1, n2]
        if k is None:
            lo = max(1, math.ceil(cfg.k_frac[0] * n2)); hi = max(lo, math.floor(cfg.k_frac[1] * n2))
            kk = rng.integers(lo, hi + 1, B)
        else:
            kk = np.full(B, int(k))
        # labels for next-token prediction: y[p-1] = tok[p] at answer positions
        mask = np.zeros((B, L), dtype=bool)
        if cfg.loss_on == 'all':
            ap = ans_pos.reshape(-1)[1:]          # every fact except the first fact of the sequence
            mask[:, ap] = True
        elif cfg.loss_on == 'target':
            tp = ans_pos[t]
            mask[:, tp] = True
            for b in range(B):
                mask[b, tp[:kk[b]]] = False
        else:
            raise ValueError(cfg.loss_on)
        x = tok[:, :-1]
        y = np.where(mask[:, 1:], tok[:, 1:], -100)
        return {'tok': tok, 'x': x, 'y': y, 'n': n, 't': t, 'k': kk, 'tabs': tabs, 'syms': syms,
                'names': tnames, 'tgt_pos': ans_pos[t], 'tgt_sym': syms[:, t] + NSPEC}

    def sample(self, B, n, t, rng):
        return self.make(n, t, B, rng)


def decode(tok):
    """Parse one token sequence back to a list of tables; each table = list of (a, b, c) symbol ids.
    Raises AssertionError on any format violation."""
    tok = list(map(int, tok))
    assert tok[0] == BOS
    tables, cur, i = [], [], 1
    while i < len(tok):
        if tok[i] == SEP:
            tables.append(cur); cur = []; i += 1; continue
        f = tok[i:i + FACT]
        assert len(f) == FACT and f[1] == MUL and f[3] == EQ and f[5] == SEMI, (i, f)
        assert all(x >= NSPEC for x in (f[0], f[2], f[4])), f
        cur.append((f[0], f[2], f[4]))
        i += FACT
    tables.append(cur)
    return tables


def table_from_facts(facts):
    """facts (a,b,c) symbol ids -> (labels list, n x n array) or raise if not a complete operation table."""
    labs = sorted({x for f in facts for x in f[:2]})
    n = len(labs)
    ix = {s: i for i, s in enumerate(labs)}
    T = -np.ones((n, n), dtype=np.int64)
    for a, b, c in facts:
        assert T[ix[a], ix[b]] < 0, 'duplicate cell'
        assert c in ix, 'answer outside the table alphabet'
        T[ix[a], ix[b]] = ix[c]
    assert (T >= 0).all(), 'incomplete table'
    return labs, T.astype(np.uint8)


def group_type(T):
    """Identify the isomorphism type of a group table of order <= 10 (n in 1..10) by invariants."""
    n = len(T)
    e = identity_of(T)
    assert e is not None and is_assoc(T)
    def order(x):
        k, y = 1, x
        while y != e:
            y = T[y, x]; k += 1
        return k
    ords = sorted(order(x) for x in range(n))
    ab = is_comm(T)
    if max(ords) == n:
        return f'Z{n}'
    if n == 4: return 'V4'
    if n == 6: return 'S3'
    if n == 8:
        if ab: return 'Z4xZ2' if max(ords) == 4 else 'Z2^3'
        return 'D4' if ords.count(2) == 5 else 'Q8'
    if n == 9: return 'Z3xZ3'
    if n == 10: return 'D5'
    raise ValueError(n)
