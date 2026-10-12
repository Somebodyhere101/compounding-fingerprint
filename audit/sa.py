"""Independent stimulus audit (own group constructions, own SAT encoding with Glucose4, own enumeration).
usage: python sa.py OUT.jsonl FILE [FILE ...] [--every K] [--off J]
Checks every trial (or every K-th starting at J) in each file; writes one JSON line per problem/trial summary."""
import sys, json, re, itertools, functools, collections
import numpy as np
from pysat.solvers import Glucose4

# ---------------------------------------------------------------- own groups
def cayley(elems, mul):
    idx = {e: i for i, e in enumerate(elems)}
    n = len(elems)
    return np.array([[idx[mul(a, b)] for b in elems] for a in elems], dtype=np.int64)

def perm_group(gens):
    d = len(gens[0]); e = tuple(range(d)); S = {e}; fr = [e]
    comp = lambda p, q: tuple(p[q[i]] for i in range(d))
    while fr:
        nf = []
        for x in fr:
            for g in gens:
                y = comp(x, g)
                if y not in S: S.add(y); nf.append(y)
        fr = nf
    return cayley(sorted(S), comp)

def quat():
    # elements (sign, unit); unit products
    tbl = {('1','1'):(1,'1'),('1','i'):(1,'i'),('1','j'):(1,'j'),('1','k'):(1,'k'),
           ('i','1'):(1,'i'),('j','1'):(1,'j'),('k','1'):(1,'k'),
           ('i','i'):(-1,'1'),('j','j'):(-1,'1'),('k','k'):(-1,'1'),
           ('i','j'):(1,'k'),('j','i'):(-1,'k'),('j','k'):(1,'i'),('k','j'):(-1,'i'),
           ('k','i'):(1,'j'),('i','k'):(-1,'j')}
    els = [(s, u) for s in (1, -1) for u in '1ijk']
    def mul(a, b):
        s, u = tbl[(a[1], b[1])]
        return (a[0] * b[0] * s, u)
    return cayley(els, mul)

def prod(*ms):
    els = list(itertools.product(*[range(m) for m in ms]))
    return cayley(els, lambda a, b: tuple((x + y) % m for x, y, m in zip(a, b, ms)))

@functools.lru_cache(None)
def my_groups(n):
    if n == 5: return {'Z5': prod(5)}
    if n == 6: return {'Z6': prod(6), 'S3': perm_group([(1, 0, 2), (1, 2, 0)])}
    if n == 8: return {'Z8': prod(8), 'Z4xZ2': prod(4, 2), 'Z2^3': prod(2, 2, 2),
                       'D4': perm_group([(1, 2, 3, 0), (3, 2, 1, 0)]), 'Q8': quat()}
    raise ValueError(n)

def is_latin(X):
    n = len(X); f = set(range(n))
    return all(set(X[i].tolist()) == f for i in range(n)) and all(set(X[:, j].tolist()) == f for j in range(n))

def identities(X):
    n = len(X); ar = np.arange(n)
    two = [e for e in range(n) if (X[e] == ar).all() and (X[:, e] == ar).all()]
    left = [e for e in range(n) if (X[e] == ar).all()]
    right = [e for e in range(n) if (X[:, e] == ar).all()]
    return two, left, right

def assoc_frac(X):
    return float((X[X, :] == X[:, X]).mean())

@functools.lru_cache(None)
def perms(n):
    return np.array(list(itertools.permutations(range(n))), dtype=np.int64)

def iso(X, G):
    """exists bijection p (X-label -> G-label) with p[X[a,b]] = G[p[a],p[b]]"""
    P = perms(len(X))
    lhs = P[:, X.reshape(-1)]                      # m x n^2 : p[X[a,b]]
    a = np.repeat(np.arange(len(X)), len(X)); b = np.tile(np.arange(len(X)), len(X))
    rhs = G[P[:, a], P[:, b]]
    return bool((lhs == rhs).all(1).any())

@functools.lru_cache(None)
def labelled(n, name):
    G = my_groups(n)[name]; P = perms(n)
    Pinv = np.argsort(P, axis=1)
    R = np.take_along_axis(P, G[Pinv[:, :, None], Pinv[:, None, :]].reshape(len(P), -1), axis=1)
    return np.unique(R, axis=0)

# ---------------------------------------------------------------- own SAT
class Sat:
    def __init__(self, n):
        self.n = n; s = Glucose4(); self.s = s
        v = self.v
        def eo(l):
            s.add_clause(l)
            for x, y in itertools.combinations(l, 2): s.add_clause([-x, -y])
        R = range(n)
        for r in R:
            for c in R: eo([v(r, c, x) for x in R])
        for r in R:
            for x in R: eo([v(r, c, x) for c in R])
        for c in R:
            for x in R: eo([v(r, c, x) for r in R])
    def v(self, r, c, x):
        return 1 + (r * self.n + c) * self.n + x
    def ident(self, e):
        return [self.v(e, j, j) for j in range(self.n)] + [self.v(j, e, j) for j in range(self.n)]

@functools.lru_cache(None)
def sat(n): return Sat(n)

# ---------------------------------------------------------------- parsing
FACT = re.compile(r'^(\S+) \* (\S+) = (\S+)$')

def parse_blocks(ctx):
    blocks = ctx.split('\n\n'); pre = []; tabs = []
    for b in blocks:
        lines = b.split('\n')
        if re.fullmatch(r'Table [A-Z]:', lines[0]):
            facts = []
            for l in lines[1:]:
                m = FACT.match(l)
                if not m: raise ValueError('bad fact line %r' % l)
                facts.append(m.groups())
            tabs.append((lines[0], facts))
        else:
            if tabs: raise ValueError('non-table block after tables: %r' % b[:80])
            pre.append(b)
    return pre, tabs

def to_table(facts, words=None):
    if words is None:
        words = []
        for f in facts:
            for w in f:
                if w not in words: words.append(w)
    ix = {w: i for i, w in enumerate(words)}
    n = len(words)
    X = -np.ones((n, n), dtype=np.int64); dup = 0
    for a, b, c in facts:
        if X[ix[a], ix[b]] != -1: dup += 1
        X[ix[a], ix[b]] = ix[c]
    return X, words, dup

# ---------------------------------------------------------------- audit
def audit(tr, fname):
    P = []  # problems
    n = tr['n']; tw = tr['target_words']; k = tr['k']
    tid = tr['trial']
    def bad(msg, **kw): P.append(dict(file=fname, trial=tid, msg=msg, **kw))
    pre, tabs = parse_blocks(tr['context'])
    hist, (thead, tfacts) = tabs[:-1], tabs[-1]
    letters = [h for h, _ in tabs]
    if letters != ['Table %s:' % 'ABCDEFGH'[i] for i in range(len(tabs))]: bad('headers', h=letters)
    if len(hist) != tr['t'] or len(tr['history_types']) != tr['t']: bad('t mismatch')
    # ---- histories (a)
    allwords = []
    hwords = []
    for i, ((h, facts), ht) in enumerate(zip(hist, tr['history_types'])):
        X, ws, dup = to_table(facts)
        hwords.append(ws)
        if len(ws) != n or dup or len(facts) != n * n or (X < 0).any():
            bad('history incomplete', i=i, nw=len(ws), dup=dup, nf=len(facts)); continue
        if not is_latin(X): bad('history not Latin', i=i); continue
        two, left, right = identities(X)
        kind, name = ht
        if kind == 'grp':
            if not iso(X, my_groups(n)[name]): bad('history not iso to stated group', i=i, name=name)
        elif kind == 'loop':
            fr = assoc_frac(X)
            if not two: bad('loop history without identity', i=i)
            if fr >= 1.0: bad('loop history associative', i=i)
            if abs(fr - float(name.split('=')[1])) > 6e-4: bad('loop assoc frac mismatch', i=i, fr=fr, stated=name)
        elif kind == 'latin-noid':
            if two: bad('scramble history has two-sided identity', i=i)
            if left or right: bad('INFO scramble one-sided identity', i=i, left=len(left), right=len(right))
        else:
            bad('unknown history type', ht=ht)
    # ---- target (b)
    if sorted(set(tw)) != sorted(tw) or len(tw) != n: bad('target words not n distinct')
    T = -np.ones((n, n), dtype=np.int64)
    ix = {w: i for i, w in enumerate(tw)}
    try:
        Xc, _, dup = to_table(tfacts, tw)
    except KeyError as e:
        bad('target fact uses non-target word', w=str(e)); return P, None
    if dup: bad('dup cue facts')
    cue = [(int(a), int(b)) for a, b in zip(*np.nonzero(Xc >= 0))]
    if len(cue) != k or len(tfacts) != k: bad('cue size', ncue=len(cue), k=k)
    T[Xc >= 0] = Xc[Xc >= 0]
    qcells = [tuple(q['cell']) for q in tr['queries']]
    unseen = sorted(set(itertools.product(range(n), range(n))) - set(cue))
    if sorted(qcells) != unseen: bad('queries != unseen cells', nq=len(qcells), nu=len(unseen))
    for q in tr['queries']:
        a, b = q['cell']
        if q['candidates'] != tw: bad('candidates != target_words')
        if q['prompt_suffix'] != '\n%s * %s =' % (tw[a], tw[b]): bad('prompt_suffix mismatch', cell=q['cell'])
        T[a, b] = ix[q['gold']]
    if (T < 0).any(): bad('target not fully determined'); return P, None
    tk, tn = tr['target_type']
    if not is_latin(T): bad('target (cue+gold) not Latin'); return P, None
    if tk == 'grp':
        if not iso(T, my_groups(n)[tn]): bad('target (cue+gold) not iso to stated group', name=tn)
    else:
        if not identities(T)[0] or assoc_frac(T) >= 1: bad('loop target not a non-assoc loop')
    # condition-specific design checks
    cond = tr['cond']
    if cond == 'heldout':
        seen = {6: {'Z6'}, 8: {'Z8', 'Z4xZ2', 'Z2^3'}}[n]
        if any(h[1] not in seen for h in tr['history_types']) or tn in seen: bad('heldout design violated')
    # ---- (d) words
    tset = set(tw)
    for i, ws in enumerate(hwords):
        if cond == 'leak' and i == len(hwords) - 1:
            continue
        if set(ws) & tset: bad('history shares word with target', i=i)
        for j in range(i):
            if cond == 'leak' and j == len(hwords) - 1: continue
            if set(ws) & set(hwords[j]): bad('two histories share words', i=i, j=j)
    if cond == 'leak':
        h, facts = hist[-1]
        X, ws, dup = to_table(facts, tw) if set(w for f in facts for w in f) <= tset else (None, None, None)
        if X is None or not (X == T).all() or len(facts) != n * n: bad('leak table != target labelled table')
        for i in range(len(hwords) - 1):
            if set(hwords[i]) & tset: bad('leak: earlier history shares target words', i=i)
    # ---- (c) forcing classes
    S = sat(n)
    cue_lits = [S.v(a, b, int(T[a, b])) for a, b in cue]
    flat = T.reshape(-1); cidx = np.array([a * n + b for a, b in cue], dtype=np.int64)
    def forced_set(tabs):
        C = tabs[(tabs[:, cidx] == flat[cidx]).all(1)]
        if len(C) == 0: return None
        return (C == C[0]).all(0)       # cell forced iff all consistent tables agree
    G = my_groups(n)
    fG = fA = fC = None
    if tk == 'grp':
        fG = forced_set(np.concatenate([labelled(n, h) for h in G]))
        if (G[tn] == G[tn].T).all():
            fA = forced_set(np.concatenate([labelled(n, h) for h in G if (G[h] == G[h].T).all()]))
        if tn == 'Z%d' % n:
            fC = forced_set(labelled(n, tn))
    n_id_ok = [e for e in range(n) if S.s.solve(assumptions=cue_lits + S.ident(e))]
    mism = []
    cnt = collections.Counter()
    for q in tr['queries']:
        a, b = q['cell']; g = int(T[a, b]); c = a * n + b
        neg = -S.v(a, b, g)
        L = not S.s.solve(assumptions=cue_lits + [neg])
        O = all(not S.s.solve(assumptions=cue_lits + S.ident(e) + [neg]) for e in n_id_ok)
        if L: cl = 'L'
        elif O: cl = 'O'
        elif fG is not None and fG[c]: cl = 'G'
        elif fA is not None and fA[c]: cl = 'Ab'
        elif fC is not None and fC[c]: cl = 'Cy'
        else: cl = 'U'
        cnt[cl] += 1
        if cl != q['class']: mism.append((q['cell'], q['class'], cl))
    if mism: bad('class mismatch', mism=mism)
    return P, dict(file=fname, trial=tid, cls=dict(cnt))

def main():
    out = sys.argv[1]; args = sys.argv[2:]
    every, off = 1, 0
    if '--every' in args:
        i = args.index('--every'); every = int(args[i + 1]); del args[i:i + 2]
    if '--off' in args:
        i = args.index('--off'); off = int(args[i + 1]); del args[i:i + 2]
    # sanity: own groups are groups and non-isomorphic, and match tables.py names
    for n in (5, 6, 8):
        Gs = my_groups(n)
        for nm, X in Gs.items():
            assert is_latin(X) and assoc_frac(X) == 1.0 and identities(X)[0], nm
        for a, b in itertools.combinations(Gs, 2):
            assert not iso(Gs[a], Gs[b]), (a, b)
    with open(out, 'w') as fo:
        for f in args:
            name = f.replace('\\', '/').split('/')[-1]
            for li, line in enumerate(open(f)):
                if li % every != off % every: continue
                tr = json.loads(line)
                try:
                    P, summ = audit(tr, name)
                except Exception as e:
                    P, summ = [dict(file=name, trial=tr.get('trial'), msg='EXC ' + repr(e))], None
                for p in P: fo.write(json.dumps(p) + '\n')
                if summ: fo.write(json.dumps(dict(summary=summ)) + '\n')
                fo.flush()

if __name__ == '__main__':
    main()
