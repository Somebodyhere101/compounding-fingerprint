"""Exact idealised-learner predictions on the identical Experiment-1 trials (exp1/stim/*.jsonl).

Learners (adapted from runs/c10/code/sim.py; c10 report section 5):
  R     recall: copy only if the same labelled cell (same two words) appears in a history table, else uniform
  N     Latin feasibility (enumeration n=5, SAT n>=6 as in c10): uniform over Latin-feasible values
  N_iso exemplar matcher: union of all relabelings of the actual history tables consistent with the cue,
        uniform over their values; falls back to Latin feasibility when no relabeling fits
  A     strict axiom induction from the history tables (identity, associativity, commutativity, cyclicity
        holding in EVERY history table); class chain cyclic->abelian->group, tail [comm-loop]->loop->Latin
  Asl   sloppy: identity in every history table => group class (no associativity check)
  B     Bayes posterior predictive under the TRUE generator of the cell's target (iso-uniform over the
        target types stimuli.py draws from, uniform relabeling); MAP answer
  Bh    hierarchical Bayes over {latin, loop, group, abelian, cyclic}, uniform hyperprior; history tables
        enter through their exact likelihood under each environment; MAP of the mixture predictive
Possibilistic learners (R, N, N_iso, A, Asl): P = uniform over feasible values; MAP learners (B, Bh):
P = posterior predictive, prediction = argmax. Ties (rtol 1e-9, as c10 map_acc) are broken uniformly at
random with a seed fixed per (learner, file, trial). p_gold = P[gold].
Every learner except R also copies a verbatim-recalled cell (only possible in the leak cell) - 'recall first'.

usage: python ideal.py [--procs 8] [--files n6_base_t0.jsonl ...] [--limit N]
writes ../res/ideal_<L>.jsonl (Experiment-1 results format) and ../res/_diag.jsonl
"""
import sys, os, json, glob, math, zlib, itertools, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP1 = os.path.abspath(os.path.join(HERE, '..', '..'))
C10 = os.path.abspath(os.path.join(EXP1, '..', 'runs', 'c10', 'code'))
sys.path.insert(0, C10)
sys.path.insert(0, HERE)
from engine import universe, group_meta, counts_for, sat_solver          # noqa: E402
from tables import groups_of_order, is_assoc, is_comm, identity_of, relabel, all_loops  # noqa: E402
from cnt import latin_counts, loop_counts                                 # noqa: E402

LEARNERS = ['R', 'N', 'N_iso', 'A', 'Asl', 'B', 'Bh']
LATIN_TOTAL = {5: 161280, 6: 812851200, 8: 108776032459082956800}       # OEIS A002860
LOOP_TOTAL = {5: 280, 6: 56448, 8: 8 * 535281401856}                     # n * reduced Latin squares (A000315)
SEEN = {6: ['Z6'], 8: ['Z8', 'Z4xZ2', 'Z2^3']}                            # stimuli.py held-out design
LAT_CAP = float(os.environ.get('LAT_CAP', '3e8'))  # node cap for exact n=8 Latin counts (~5.7M nodes/s here)
RTOL = 1e-9


# ------------------------------------------------------------------ per-process setup
_ST = {}


def setup(n):
    if n in _ST:
        return _ST[n]
    U = universe(n)
    meta = group_meta(n)
    H = list(groups_of_order(n).keys())
    st = dict(n=n, U=U, meta=meta, H=H, m=len(H),
              Hab=[h for h in H if meta[h]['abelian']], Hcyc=[h for h in H if meta[h]['cyclic']])
    st['gkey'] = {}
    for h in H:
        for row in U['iso:' + h]:
            st['gkey'][row.tobytes()] = h
    if n <= 6:
        Lp = U['loop']
        st['cloop'] = Lp[np.all(Lp.reshape(-1, n, n) == Lp.reshape(-1, n, n).transpose(0, 2, 1), axis=(1, 2))]
    st['perms'] = np.array(list(itertools.permutations(range(n))), dtype=np.int64)
    if n == 5:  # true generator of loop targets: stimuli.grouplike_loops(5) (top 50), uniform relabeling
        import stimuli
        pool = stimuli.grouplike_loops(5)
        w = {}
        for frac, com, X in pool:
            for p in itertools.permutations(range(n)):
                Y = relabel(X, np.array(p)).reshape(-1)
                w[Y.tobytes()] = w.get(Y.tobytes(), 0.0) + 1.0 / (len(pool) * 120)
        st['lt_tabs'] = np.array([np.frombuffer(k, dtype=np.uint8) for k in w], dtype=np.uint8)
        st['lt_w'] = np.array(list(w.values()))
    _ST[n] = st
    return st


# ------------------------------------------------------------------ parsing
def parse_fact(line):
    lhs, rhs = line.split(' = ')
    a, b = lhs.split(' * ')
    return a.strip(), b.strip(), rhs.strip()


def parse_trial(tr):
    n = tr['n']
    tw = tr['target_words']
    widx = {w: i for i, w in enumerate(tw)}
    blocks = tr['context'].split('\n\n')
    hist = []
    for blk in blocks[:-1]:
        lines = blk.split('\n')
        assert lines[0].startswith('Table ') and lines[0].endswith(':'), lines[0]
        facts = [parse_fact(l) for l in lines[1:]]
        words = []
        for f in facts:
            for w in f:
                if w not in words:
                    words.append(w)
        assert len(words) == n and len(facts) == n * n, (len(words), len(facts))
        wi = {w: i for i, w in enumerate(words)}
        X = -np.ones((n, n), dtype=np.int64)
        for a, b, c in facts:
            assert X[wi[a], wi[b]] < 0
            X[wi[a], wi[b]] = wi[c]
        assert (X >= 0).all()
        assert all(len(set(X[i])) == n and len(set(X[:, i])) == n for i in range(n)), 'history not Latin'
        hist.append(dict(X=X.astype(np.uint8), words=words, facts=facts))
    lines = blocks[-1].split('\n')
    assert lines[0].startswith('Table ')
    T = -np.ones((n, n), dtype=np.int64)
    S = []
    for l in lines[1:]:
        a, b, c = parse_fact(l)
        T[widx[a], widx[b]] = widx[c]
        S.append(widx[a] * n + widx[b])
    assert len(S) == tr['k']
    for q in tr['queries']:
        a, b = q['cell']
        assert T[a, b] < 0
        T[a, b] = widx[q['gold']]
        assert q['candidates'] == tw
    assert (T >= 0).all(), 'cue + queries do not cover the table'
    return hist, T.astype(np.uint8), np.array(sorted(S), dtype=np.int64)


def table_type(st, X):
    """('grp', H) | ('loop', comm) | ('latin-noid', '')"""
    h = st['gkey'].get(X.reshape(-1).astype(np.uint8).tobytes())
    if h is not None:
        return ('grp', h)
    if identity_of(X) is not None:
        assert not is_assoc(X)
        return ('loop', 'comm' if is_comm(X) else 'noncomm')
    return ('latin-noid', '')


def axioms(st, X, typ):
    if typ[0] == 'grp':
        return dict(ident=True, assoc=True, comm=st['meta'][typ[1]]['abelian'], cyc=st['meta'][typ[1]]['cyclic'])
    return dict(ident=identity_of(X) is not None, assoc=bool(is_assoc(X)), comm=bool(is_comm(X)), cyc=False)


# ------------------------------------------------------------------ helpers
def poss(F, n):
    """uniform over feasible values; F [u, n] bool or None -> uniform"""
    if F is None:
        return None
    s = F.sum(1, keepdims=True)
    P = np.where(s > 0, F / np.maximum(s, 1), 1.0 / n)
    return P


def union_mask(R, keys):
    F = None
    for k in keys:
        Nk, c = R[k]
        if Nk is None or Nk == 0:
            continue
        F = (c > 0) if F is None else (F | (c > 0))
    return F


def weighted(R, keys, ws):
    N = 0.0; C = None
    for k, w in zip(keys, ws):
        Nk, c = R[k]
        if Nk is None or Nk == 0:
            continue
        C = c * w if C is None else C + c * w
        N += Nk * w
    return N, C


def niso_mask(st, X, T, S, unseen):
    """values at unseen cells over all relabelings of history table X consistent with cue (T,S)."""
    n = st['n']
    P = st['perms']
    flat = T.reshape(-1)
    ok = np.ones(len(P), dtype=bool)
    for c in S:
        i, j = divmod(int(c), n); v = int(flat[c])
        ok &= X[P[:, i], P[:, j]] == P[:, v]
    sel = P[ok]
    if len(sel) == 0:
        return None
    inv = np.argsort(sel, axis=1)
    F = np.zeros((len(unseen), n), dtype=bool)
    rows = np.arange(len(sel))
    for r, c in enumerate(unseen):
        a, b = divmod(int(c), n)
        x = X[sel[:, a], sel[:, b]]
        F[r, np.unique(inv[rows, x])] = True
    return F


# ------------------------------------------------------------------ one trial
def run_trial(args):
    fname, tr = args
    t_start = time.time()
    n = tr['n']
    st = setup(n)
    U = st['U']
    hist, T, S = parse_trial(tr)
    flat = T.reshape(-1)
    unseen = np.array([q['cell'][0] * n + q['cell'][1] for q in tr['queries']], dtype=np.int64)
    gold = flat[unseen].astype(np.int64)
    u = len(unseen)
    diag = {'file': fname, 'trial': tr['trial']}
    # --- history types (validated against the stimulus metadata)
    htypes = [table_type(st, h['X']) for h in hist]
    diag['htypes'] = [list(h) for h in htypes]
    for i, (ht, hm) in enumerate(zip(htypes, tr['history_types'])):
        if tr['cond'] == 'leak' and i == len(hist) - 1:
            # stimuli.py replaces the last history table by the target's own table AFTER recording its type,
            # so history_types[-1] is stale in the leak cell; the parsed table must be the target's type
            assert ht == tuple(tr['target_type']), (ht, tr['target_type'])
            diag['leak_meta_stale'] = ht != tuple(hm)
            continue
        if hm[0] == 'grp':
            assert ht == ('grp', hm[1]), (ht, hm)
        elif hm[0] == 'loop':
            assert ht[0] == 'loop', (ht, hm)
        elif hm[0] == 'latin-noid':
            assert ht[0] == 'latin-noid', (ht, hm)
    tkind, tname = tr['target_type']
    if tkind == 'grp':
        assert st['gkey'].get(T.reshape(-1).tobytes()) == tname, 'target is not the stated group'
    else:
        assert identity_of(T) is not None and not is_assoc(T)
    # --- C1 check: words disjoint across tables except the leak table
    tw = tr['target_words']
    for i, h in enumerate(hist):
        inter = set(h['words']) & set(tw)
        if tr['cond'] == 'leak' and i == len(hist) - 1:
            assert set(h['words']) == set(tw)
        else:
            assert not inter, 'word overlap'
    # --- exact per-cue analysis
    R = {}
    for h in st['H']:
        Nk, c = counts_for(U['iso:' + h], T, S)
        R['iso:' + h] = (Nk, c[unseen])
    grid = -np.ones(n * n, dtype=np.int64); grid[S] = flat[S]
    # Latin feasibility (N): enumeration n=5, SAT n>=6 (c10)
    if n == 5:
        NL, cL = counts_for(U['latin'], T, S)
        R['latin'] = (NL, cL[unseen]); Fl = cL[unseen] > 0
    else:
        Fsat = sat_solver(n, False).feasible(T, S)
        Fl = Fsat[unseen]
        need_exact = (n == 6) or (tr['t'] == 0) or (tr['cond'] == 'scramble')
        if need_exact:
            t0 = time.time()
            NL, cL = latin_counts(n, grid, cap=LAT_CAP)
            diag['latin_count_s'] = round(time.time() - t0, 2)
            diag['latin_N'] = NL
            if NL is not None:
                R['latin'] = (NL, cL[unseen])
                diag['sat_vs_count_mismatch'] = int(((cL[unseen] > 0) != Fl).sum())
            else:
                R['latin'] = (None, None); diag['latin_capped'] = True
        else:
            R['latin'] = (None, None)
    # loops: enumeration n<=6, exact C count n=8
    if n <= 6:
        NLp, cLp = counts_for(U['loop'], T, S)
        R['loop'] = (NLp, cLp[unseen])
        Nc, cc = counts_for(st['cloop'], T, S)
        R['cloop'] = (Nc, cc[unseen])
    else:
        NLp, cLp = loop_counts(n, grid)
        assert NLp is not None
        R['loop'] = (NLp, cLp[unseen])
    Fo = R['loop'][1] > 0 if R['loop'][0] else None
    # --- cell-class check against the stimulus labels (c10 stimuli.cell_classes definition)
    if tkind == 'grp':
        fg = union_mask(R, ['iso:' + h for h in st['H']]).sum(1) == 1
        fa = union_mask(R, ['iso:' + h for h in st['Hab']]).sum(1) == 1 if st['meta'][tname]['abelian'] else np.zeros(u, bool)
        fc = (R['iso:' + tname][1] > 0).sum(1) == 1 if st['meta'][tname]['cyclic'] else np.zeros(u, bool)
    else:
        fg = fa = fc = np.zeros(u, bool)
    fl = Fl.sum(1) == 1; fo = Fo.sum(1) == 1
    mycls = np.where(fl, 'L', np.where(fo, 'O', np.where(fg, 'G', np.where(fa, 'Ab', np.where(fc, 'Cy', 'U')))))
    diag['class_mismatch'] = int(sum(m != q['class'] for m, q in zip(mycls, tr['queries'])))

    P = {}
    # R
    recall = {}
    for h in hist:
        for a, b, c in h['facts']:
            recall[(a, b)] = c
    widx = {w: i for i, w in enumerate(tw)}
    rec = np.full(u, -1)
    for r, q in enumerate(tr['queries']):
        a, b = q['cell']
        c = recall.get((tw[a], tw[b]))
        if c is not None and c in widx:
            rec[r] = widx[c]
    diag['n_recalled'] = int((rec >= 0).sum())
    P['R'] = np.full((u, n), 1.0 / n)
    # N
    P['N'] = poss(Fl, n)
    # N_iso
    F = None
    for h in hist:
        m = niso_mask(st, h['X'], T, S, unseen)
        if m is not None:
            F = m if F is None else (F | m)
    diag['niso_match'] = F is not None
    P['N_iso'] = poss(F if F is not None else Fl, n)
    # A (strict)
    if hist:
        axs = [axioms(st, h['X'], ty) for h, ty in zip(hist, htypes)]
        ax = {x: all(a[x] for a in axs) for x in ('ident', 'assoc', 'comm', 'cyc')}
    else:
        ax = dict(ident=False, assoc=False, comm=False, cyc=False)
    chain = []
    if ax['assoc']:
        if ax['cyc']: chain.append(('cyclic', ['iso:' + h for h in st['Hcyc']]))
        if ax['comm']: chain.append(('abelian', ['iso:' + h for h in st['Hab']]))
        chain.append(('group', ['iso:' + h for h in st['H']]))
    F = None; used = None
    for nm, keys in chain:
        F = union_mask(R, keys)
        if F is not None:
            used = nm; break
    if F is None:
        tail = []
        if ax['ident'] and ax['comm'] and not ax['assoc'] and 'cloop' in R:
            tail.append('cloop')
        if ax['ident']:
            tail.append('loop')
        for kk in tail:
            if R[kk][0]:
                F = R[kk][1] > 0; used = kk; break
        if F is None:
            F = Fl; used = 'latin'
    diag['A_class'] = used
    P['A'] = poss(F, n)
    # Asl
    if hist and ax['ident']:
        F = union_mask(R, ['iso:' + h for h in st['H']])
        if F is None:
            F = Fo
    else:
        F = Fl
    P['Asl'] = poss(F, n)
    # B: true generator of this cell's target
    if tkind == 'grp':
        types = [h for h in st['H'] if h not in SEEN[n]] if tr['cond'] == 'heldout' else st['H']
        Nb, Cb = weighted(R, ['iso:' + h for h in types], [(1.0 / len(types)) / st['meta'][h]['nlab'] for h in types])
    else:
        mask = np.all(st['lt_tabs'][:, S] == flat[S], axis=1)
        tabs, w = st['lt_tabs'][mask], st['lt_w'][mask]
        Cb = np.zeros((n * n, n))
        for x in range(n):
            Cb[:, x] = ((tabs == x) * w[:, None]).sum(0)
        Cb = Cb[unseen]; Nb = w.sum()
    P['B'] = Cb / Nb
    # Bh: hierarchical Bayes over environments
    envs = ['latin', 'loop', 'group', 'abelian', 'cyclic']
    logpost = {}
    for e in envs:
        lp = math.log(1.0 / len(envs))
        for h, ty in zip(hist, htypes):
            if e == 'latin':
                l = 1.0 / LATIN_TOTAL[n]
            elif e == 'loop':
                l = 1.0 / LOOP_TOTAL[n] if ty[0] in ('grp', 'loop') else 0.0
            else:
                hs = st['H'] if e == 'group' else (st['Hab'] if e == 'abelian' else st['Hcyc'])
                l = (1.0 / len(hs)) / st['meta'][ty[1]]['nlab'] if (ty[0] == 'grp' and ty[1] in hs) else 0.0
            if l <= 0:
                lp = -math.inf; break
            lp += math.log(l)
        logpost[e] = lp
    preds, logev = {}, {}
    dropped = None
    for e in envs:
        if logpost[e] == -math.inf:
            continue
        if e in ('latin', 'loop'):
            Nk, c = R[e]
            if Nk is None:   # Latin counts not computed (n=8, t>=1 group history) or capped: bound below
                dropped = logpost[e]  # + log P_latin(K) <= 0
                continue
            if Nk == 0:
                continue
            preds[e] = c / Nk
            logev[e] = logpost[e] + math.log(Nk) - math.log(LATIN_TOTAL[n] if e == 'latin' else LOOP_TOTAL[n])
        else:
            hs = st['H'] if e == 'group' else (st['Hab'] if e == 'abelian' else st['Hcyc'])
            Nk, c = weighted(R, ['iso:' + h for h in hs], [(1.0 / len(hs)) / st['meta'][h]['nlab'] for h in hs])
            if Nk == 0:
                continue
            preds[e] = c / Nk
            logev[e] = logpost[e] + math.log(Nk)
    if logev:
        mx = max(logev.values())
        wts = {e: math.exp(v - mx) for e, v in logev.items()}
        Z = sum(wts.values())
        Pm = sum(wts[e] / Z * preds[e] for e in preds)
        diag['Bh_w'] = {e: wts[e] / Z for e in wts}
        if dropped is not None:
            # upper bound on the dropped Latin environment's posterior weight (P_latin(K) <= 1)
            diag['Bh_latin_dropped_wbound'] = math.exp(dropped - (mx + math.log(Z)))
    else:  # only Latin could explain the history but its counts are unavailable (capped): possibilistic Latin
        Pm = poss(Fl, n); diag['Bh_fallback'] = 'possLatin'
    P['Bh'] = Pm
    # --- recall first (all learners)
    for L in LEARNERS:
        X = P[L].astype(np.float64).copy()
        for r in np.where(rec >= 0)[0]:
            X[r] = 0.0; X[r, rec[r]] = 1.0
        P[L] = X
    # --- predictions
    out = {}
    exp_acc = {}
    for L in LEARNERS:
        X = P[L]
        rng = np.random.default_rng(zlib.crc32(f'{L}|{fname}|{tr["trial"]}'.encode()))
        mx = X.max(1, keepdims=True)
        tie = np.isclose(X, mx, rtol=RTOL, atol=1e-300)
        qs = []
        ea = 0.0
        for r, q in enumerate(tr['queries']):
            cand = np.where(tie[r])[0]
            j = int(rng.choice(cand))
            g = int(gold[r])
            ea += float(tie[r, g]) / len(cand)
            qs.append({'cell': q['cell'], 'class': q['class'], 'gold': q['gold'], 'pred': tw[j],
                       'correct': int(j == g), 'p_gold': round(float(X[r, g]), 6), 'cand_mass': 1.0})
        exp_acc[L] = ea / u
        out[L] = {'model': f'ideal_{L}', 'file': fname, 'trial': tr['trial'], 'cond': tr['cond'], 'n': n,
                  't': tr['t'], 'k': tr['k'], 'target_type': tr['target_type'], 'ctx_tokens': 0, 'queries': qs}
    diag['exp_acc'] = exp_acc
    diag['secs'] = round(time.time() - t_start, 2)
    return fname, tr['trial'], out, diag


def write_outputs(results, outdir):
    os.makedirs(outdir, exist_ok=True)
    keys = sorted(results)  # (file, trial): same order as exp1/run.py output
    for L in LEARNERS:
        with open(os.path.join(outdir, f'ideal_{L}.jsonl'), 'w', encoding='utf-8') as fo:
            for k in keys:
                fo.write(json.dumps(results[k][2][L]) + '\n')
    with open(os.path.join(outdir, '_diag.jsonl'), 'w', encoding='utf-8') as fo:
        for k in keys:
            fo.write(json.dumps(results[k][3], default=float) + '\n')
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--procs', type=int, default=1,
                    help='multiprocessing.Pool size; >1 fails in this sandbox (DuplicateHandle denied): use --shard')
    ap.add_argument('--shard', type=int, default=None, help='process only tasks with index %% nshards == shard')
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--partdir', default=os.path.join(HERE, '..', 'parts'))
    ap.add_argument('--merge', action='store_true', help='merge part files from --partdir into --outdir')
    ap.add_argument('--files', nargs='*', default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--outdir', default=os.path.join(HERE, '..', 'res'))
    args = ap.parse_args()
    if args.merge:
        results = {}
        for p in sorted(glob.glob(os.path.join(args.partdir, 'part*.jsonl'))):
            for l in open(p, encoding='utf-8'):
                r = json.loads(l)
                results[(r['file'], r['trial'])] = (r['file'], r['trial'], r['out'], r['diag'])
        keys = write_outputs(results, args.outdir)
        print(f'MERGED {len(keys)} trials -> {args.outdir}', flush=True)
        return
    files = args.files or sorted(os.path.basename(f) for f in glob.glob(os.path.join(EXP1, 'stim', '*.jsonl')))
    tasks = []
    for f in files:
        rows = [json.loads(l) for l in open(os.path.join(EXP1, 'stim', f), encoding='utf-8')]
        if args.limit:
            rows = rows[:args.limit]
        tasks += [(f, r) for r in rows]
    # heavy n=8 exact-Latin trials first for load balance
    heavy = lambda t: (t[1]['n'] == 8 and (t[1]['t'] == 0 or t[1]['cond'] == 'scramble'))
    tasks.sort(key=lambda t: (not heavy(t), -t[1]['n'], t[0], t[1]['trial']))
    t0 = time.time()
    if args.shard is not None:
        # independent OS process per shard (launched by run_shards.sh); results streamed to a part file
        mine = [tk for i, tk in enumerate(tasks) if i % args.nshards == args.shard]
        os.makedirs(args.partdir, exist_ok=True)
        path = os.path.join(args.partdir, f'part{args.shard:02d}.jsonl')
        done = set()
        if os.path.exists(path):  # resume
            for l in open(path, encoding='utf-8'):
                r = json.loads(l); done.add((r['file'], r['trial']))
        with open(path, 'a', encoding='utf-8') as fo:
            for i, tk in enumerate(mine):
                if (tk[0], tk[1]['trial']) in done:
                    continue
                f, trial, out, diag = run_trial(tk)
                fo.write(json.dumps({'file': f, 'trial': trial, 'out': out, 'diag': diag}, default=float) + '\n')
                fo.flush()
                if (i + 1) % 10 == 0:
                    print(f'shard {args.shard}: {i + 1}/{len(mine)} trials, {time.time() - t0:.0f}s', flush=True)
        print(f'SHARD {args.shard} DONE {len(mine)} trials in {time.time() - t0:.0f}s', flush=True)
        return
    results = {}
    if args.procs > 1:
        import multiprocessing as mp
        with mp.Pool(args.procs) as pool:
            for i, res in enumerate(pool.imap_unordered(run_trial, tasks, chunksize=1)):
                results[(res[0], res[1])] = res
                if (i + 1) % 100 == 0:
                    print(f'{i + 1}/{len(tasks)} trials, {time.time() - t0:.0f}s', flush=True)
    else:
        for i, tk in enumerate(tasks):
            res = run_trial(tk)
            results[(res[0], res[1])] = res
            if (i + 1) % 20 == 0:
                print(f'{i + 1}/{len(tasks)} trials, {time.time() - t0:.0f}s', flush=True)
    keys = write_outputs(results, args.outdir)
    print(f'DONE {len(keys)} trials in {time.time() - t0:.0f}s -> {args.outdir}', flush=True)


if __name__ == '__main__':
    main()
