"""Audit2: independent re-analysis of exp5 results (H1, H2, correctness join)."""
import json, os, collections, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = [('q06b', 'exp2/res/q06b_main.jsonl'), ('q17b', 'exp2/res/q17b_main.jsonl'), ('q17', 'exp1/res/qwen3-1.7b.jsonl'),
          ('q4b', 'exp2/res/q4b_main.jsonl'), ('q8b', 'exp2/res/q8b_main.jsonl')]
R5 = json.load(open(os.path.join(ROOT, 'exp5', 'results5.json')))
def auc(p, n):
    p, n = np.asarray(p), np.asarray(n)
    return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))
def avrank(x):
    x = np.asarray(x, float); o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(len(x)); 
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1: r[m] = r[m].mean()
    return r
def spear(x, y): return float(np.corrcoef(avrank(x), avrank(y))[0, 1])
for name, cp in MODELS:
    rows = [json.loads(l) for l in open(os.path.join(ROOT, 'exp5', 'res', name + '.jsonl'), encoding='utf-8')]
    meta = json.load(open(os.path.join(ROOT, 'exp5', 'res', name + '_meta.json'))); L = str(meta['best_layer'])
    # best layer = first max?
    ca = {int(k): v for k, v in meta['calib_auc'].items()}; mx = max(ca.values()); first = min(k for k in sorted(ca) if ca[k] == mx)
    cnt = collections.Counter((r['file'], r['trial']) for r in rows)
    by = collections.defaultdict(list)
    for r in rows: by[r['file']].append(r)
    nq = {f: [len(r['queries']) for r in v] for f, v in by.items()}
    b = np.array([np.mean([q['proj_det'][L] for q in r['queries']]) for r in by['n6_base_t1']])
    k = np.array([np.mean([q['proj_det'][L] for q in r['queries']]) for r in by['n6_leak_t1']])
    rng = np.random.default_rng(123)
    bs = np.array([rng.choice(b, len(b)).mean() - rng.choice(k, len(k)).mean() for _ in range(4000)])
    qb = [q['proj_det'][L] for r in by['n6_base_t1'] for q in r['queries']]; qk = [q['proj_det'][L] for r in by['n6_leak_t1'] for q in r['queries']]
    h1 = R5[name]['H1']
    print(f'== {name}: best layer {L} (first max {first}); dup (file,trial)={sum(v>1 for v in cnt.values())}; trials/file={ {f: len(v) for f, v in by.items()} }; q/trial range={ {f: (min(v), max(v)) for f, v in nq.items()} }')
    print(f'  H1 diff {b.mean()-k.mean():+.4f} (rep {h1["diff"]:+.4f}); CI {np.percentile(bs,[2.5,97.5]).round(3)} (rep {np.round(h1["ci"],3)}); d {(b.mean()-k.mean())/np.sqrt((b.var(ddof=1)+k.var(ddof=1))/2):+.3f} (rep {h1["d"]:+.3f}); AUC {auc(qb,qk):.4f} (rep {h1["auc"]:.4f})')
    # H2
    tb = by['n6_base_t1']; RK = {'L': 0, 'O': 1, 'G': 2}
    cls = collections.Counter(q['class'] for r in tb for q in r['queries']); clk = collections.Counter(q['class'] for r in by['n6_leak_t1'] for q in r['queries'])
    xs = [RK[q['class']] for r in tb for q in r['queries'] if q['class'] in RK]; ys = [q['proj_det'][L] for r in tb for q in r['queries'] if q['class'] in RK]
    print(f'  H2 rho {spear(xs, ys):+.4f} (rep {R5[name]["H2"]["rho"]:+.4f}); class counts base {dict(cls)} leak {dict(clk)}')
    # class-matched H1 (G and O only, the classes present in both conditions)
    for c in ('G', 'O'):
        bb = [q['proj_det'][L] for r in tb for q in r['queries'] if q['class'] == c]; kk = [q['proj_det'][L] for r in by['n6_leak_t1'] for q in r['queries'] if q['class'] == c]
        if bb and kk: print(f'  class {c}: base-leak {np.mean(bb)-np.mean(kk):+.3f} AUC {auc(bb,kk):.3f} (n {len(bb)},{len(kk)})')
    # correctness join
    corr = {}; dupc = 0; quant = set()
    for l in open(os.path.join(ROOT, cp), encoding='utf-8'):
        r = json.loads(l); quant.add(r.get('quant'))
        for q in r['queries']:
            key = (r['file'].replace('.jsonl', ''), r['trial'], tuple(q['cell']))
            if key in corr: dupc += 1
            corr[key] = q['correct']
    for f in ('n6_base_t1', 'n6_leak_t1'):
        keys = [(f, r['trial'], tuple(q['cell'])) for r in by[f] for q in r['queries']]
        miss = sum(kk not in corr for kk in keys)
        acc = np.mean([corr[kk] for kk in keys if kk in corr])
        print(f'  join {f}: {len(keys)} queries, missing {miss}, dup keys in corr file {dupc}, quant {quant}, acc on probed queries {acc:.3f}')
