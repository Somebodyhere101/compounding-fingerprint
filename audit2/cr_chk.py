import json, os, sys, collections
import numpy as np
R = sys.argv[1]
M = [('q06b', 'exp2/res/q06b_main.jsonl'), ('q17b', 'exp2/res/q17b_main.jsonl'), ('q17', 'exp1/res/qwen3-1.7b.jsonl'), ('q4b', 'exp2/res/q4b_main.jsonl'), ('q8b', 'exp2/res/q8b_main.jsonl')]
STIM = {f: {r['trial']: r for r in map(json.loads, open(os.path.join(R, 'exp1/stim', f + '.jsonl'), encoding='utf-8'))} for f in ['n6_base_t1', 'n6_leak_t1', 'n6_base_t0', 'n6_base_t4']}
def auc(p, n):  # P(p > n) + 0.5 P(tie), brute force by sorting
    p = np.asarray(p); n = np.sort(np.asarray(n))
    lt = np.searchsorted(n, p, 'left'); le = np.searchsorted(n, p, 'right')
    return (lt.sum() + 0.5 * (le - lt).sum()) / (len(p) * len(n))
def rank(x):
    x = np.asarray(x, float); o = np.argsort(x, kind='stable'); r = np.empty(len(x)); xs = x[o]; i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]: j += 1
        r[o[i:j + 1]] = (i + j) / 2; i = j + 1
    return r
def spear(x, y): return float(np.corrcoef(rank(x), rank(y))[0, 1])
RK = {'L': 0, 'O': 1, 'G': 2}
rng = np.random.default_rng(2024)
nsup = 0; opp = 0
for name, mp in M:
    meta = json.load(open(os.path.join(R, f'exp5/res/{name}_meta.json')))
    rows = [json.loads(l) for l in open(os.path.join(R, f'exp5/res/{name}.jsonl'), encoding='utf-8')]
    # layer choice
    au = {int(k): v for k, v in meta['calib_auc'].items()}
    best = None
    for l in sorted(au):
        if best is None or au[l] > au[best]: best = l
    L = str(meta['best_layer'])
    by = collections.defaultdict(list)
    integ = collections.Counter()
    for r in rows:
        by[r['file']].append(r)
        s = STIM[r['file']][r['trial']]
        sc = {tuple(q['cell']): q for q in s['queries']}
        for q in r['queries']:
            sq = sc.get(tuple(q['cell']))
            if sq is None or sq['class'] != q['class'] or sq['gold'] != q['gold']: integ['qmismatch'] += 1
        if len({tuple(q['cell']) for q in r['queries']}) != len(r['queries']): integ['dupq'] += 1
        integ[f"nq={len(r['queries'])}"] += 1
    ntr = {f: len(v) for f, v in by.items()}
    def tm(f, fmt='det', l=L): return np.array([np.mean([q['proj_' + fmt][l] for q in r['queries']]) for r in by[f]])
    b, k = tm('n6_base_t1'), tm('n6_leak_t1')
    d = b.mean() - k.mean()
    bs = np.array([b[rng.integers(0, len(b), len(b))].mean() - k[rng.integers(0, len(k), len(k))].mean() for _ in range(2000)])
    lo, hi = np.percentile(bs, [2.5, 97.5]); sd = np.sqrt((b.var(ddof=1) + k.var(ddof=1)) / 2)
    qb = [q['proj_det'][L] for r in by['n6_base_t1'] for q in r['queries']]; qk = [q['proj_det'][L] for r in by['n6_leak_t1'] for q in r['queries']]
    sup = lo > 0; nsup += sup; opp += hi < 0
    tr = by['n6_base_t1']
    xs = [RK[q['class']] for r in tr for q in r['queries'] if q['class'] in RK]; ys = [q['proj_det'][L] for r in tr for q in r['queries'] if q['class'] in RK]
    rho = spear(xs, ys); bsr = []
    for _ in range(1000):
        pick = [tr[i] for i in rng.integers(0, len(tr), len(tr))]
        bsr.append(spear([RK[q['class']] for r in pick for q in r['queries'] if q['class'] in RK], [q['proj_det'][L] for r in pick for q in r['queries'] if q['class'] in RK]))
    cm = {c: np.mean([q['proj_det'][L] for r in tr for q in r['queries'] if q['class'] == c]) for c in RK}
    h = tm('n6_base_t4').mean() - tm('n6_base_t0').mean()
    # correctness join
    corr = {}
    for l in open(os.path.join(R, mp), encoding='utf-8'):
        r = json.loads(l)
        if r['file'][:-6] in ('n6_base_t1', 'n6_leak_t1'):
            for q in r['queries']: corr[(r['file'][:-6], r['trial'], tuple(q['cell']))] = q['correct']
    cw = {}
    for f in ('n6_base_t1', 'n6_leak_t1'):
        pc = [q['proj_det'][L] for r in by[f] for q in r['queries'] if corr.get((f, r['trial'], tuple(q['cell']))) == 1]
        pw = [q['proj_det'][L] for r in by[f] for q in r['queries'] if corr.get((f, r['trial'], tuple(q['cell']))) == 0]
        miss = sum(1 for r in by[f] for q in r['queries'] if (f, r['trial'], tuple(q['cell'])) not in corr)
        cw[f] = (auc(pw, pc), len(pc), len(pw), miss)
    raw_b, raw_k = tm('n6_base_t1', 'raw'), tm('n6_leak_t1', 'raw')
    print(f"{name}: layer {L} (recomputed first-max {best}) trials {ntr} integ {dict(integ)}")
    print(f"   H1 det: {d:+.4f} [{lo:+.4f},{hi:+.4f}] d={d/sd:+.2f} AUC={auc(qb, qk):.3f} supported={sup}; raw-format same layer {raw_b.mean()-raw_k.mean():+.4f}")
    print(f"   H2 rho={rho:+.3f} [{np.percentile(bsr,2.5):+.3f},{np.percentile(bsr,97.5):+.3f}] means L {cm['L']:+.3f} O {cm['O']:+.3f} G {cm['G']:+.3f}; t4-t0 {h:+.4f}")
    print('   corr AUC', {f: (round(v[0], 3), v[1], v[2], 'missing', v[3]) for f, v in cw.items()})
print('H1 supported in', nsup, 'of 5; CI excludes 0 in OPPOSITE direction in', opp)
