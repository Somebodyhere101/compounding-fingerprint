"""Audit2 / Exp5: recompute CR projections with full uncached forwards and compare with exp5/res.
usage: python e5_gpu.py q06b Qwen/Qwen3-0.6B-Base [ntrials_per_file]"""
import os, sys, json, time
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UP = r'<path to a local clone of github.com/Somebodyhere101/compute-retrieve>'
sys.path.insert(0, UP)
from cr_axis.detector import CRDetector, _format_prompt, _find_layers

name, repo = sys.argv[1], sys.argv[2]
NT = int(sys.argv[3]) if len(sys.argv) > 3 else 6
HUB = os.path.expanduser('~/.cache/huggingface/hub')
d = os.path.join(HUB, 'models--' + repo.replace('/', '--'), 'snapshots'); path = os.path.join(d, os.listdir(d)[0])
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to('cuda').eval()
meta = json.load(open(os.path.join(ROOT, 'exp5', 'res', f'{name}_meta.json')))
TL = meta['test_layers']; BL = meta['best_layer']
axz = np.load(os.path.join(ROOT, 'exp5', 'res', f'{name}_axes.npz')); axes = {li: axz[f'L{li}'] for li in TL}
layers = _find_layers(model)
cap = {}
hooks = [layers[li].register_forward_hook(lambda m, i, o, li=li: cap.__setitem__(li, (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu().numpy())) for li in TL]

@torch.no_grad()
def full(text):  # uncached full forward of the whole CausalLM, as CRDetector._get_hidden
    ids = tok(text, return_tensors='pt').to('cuda')
    model(ids['input_ids'])
    return {li: cap[li].copy() for li in TL}

out = {'name': name, 'best_layer': BL}
# ---- calibration re-derivation (independent of cr_probe code)
C = [l.strip() for l in open(os.path.join(UP, 'data', 'calibration', 'compute.txt'), encoding='utf-8') if l.strip()]
R = [l.strip() for l in open(os.path.join(UP, 'data', 'calibration', 'retrieve.txt'), encoding='utf-8') if l.strip()]
hc = {li: [] for li in TL}; hr = {li: [] for li in TL}
for lst, st in [(C, hc), (R, hr)]:
    for p in lst:
        h = full(_format_prompt(p, tok))
        for li in TL: st[li].append(h[li])
for h in hooks: h.remove()
det = CRDetector(model, tok)
gd = max(float(np.abs(det._get_hidden(p, BL) - np.array(hc[BL])[k]).max()) for k, p in enumerate(C[:5]))
for h in [layers[li].register_forward_hook(lambda m, i, o, li=li: cap.__setitem__(li, (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu().numpy())) for li in TL]:
    pass
def aucf(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))
cal = {}
best, bl = 0, None
for li in TL:
    a, b = np.array(hc[li]), np.array(hr[li]); ax = a.mean(0) - b.mean(0); ax /= np.linalg.norm(ax)
    A = aucf(a @ ax, b @ ax)
    cal[li] = {'auc': A, 'auc_meta': meta['calib_auc'][str(li)], 'cos_axis_vs_npz': float(ax @ axes[li]), 'maxabs_axis_diff': float(np.abs(ax - axes[li]).max())}
    if A > best: best, bl = A, li
out['calib'] = cal; out['best_layer_recomputed'] = bl; out['getHidden_vs_ownhook_maxabs'] = gd
print(json.dumps(out, indent=1), flush=True)

# ---- test queries: uncached full forward, all queries of NT random trials per file, both formats
stored = {}
for l in open(os.path.join(ROOT, 'exp5', 'res', f'{name}.jsonl'), encoding='utf-8'):
    r = json.loads(l); stored[(r['file'], r['trial'])] = r
rng = np.random.default_rng(20261009)
recs = []
t0 = time.time()
for fname in ['n6_base_t1', 'n6_leak_t1']:
    rows = [json.loads(l) for l in open(os.path.join(ROOT, 'exp1', 'stim', fname + '.jsonl'), encoding='utf-8')]
    for ti in sorted(rng.choice(len(rows), NT, replace=False)):
        tr = rows[ti]; st = stored[(fname, tr['trial'])]
        assert [q['cell'] for q in st['queries']] == [q['cell'] for q in tr['queries']], 'query order mismatch'
        for fmt in ('det', 'raw'):
            U = []
            for q in tr['queries']:
                text = tr['context'] + q['prompt_suffix']
                if fmt == 'det': text = _format_prompt(text, tok)
                h = full(text); U.append({li: float(h[li] @ axes[li]) for li in TL})
            for j, q in enumerate(st['queries']):
                for li in TL:
                    s = q[f'proj_{fmt}'][str(li)]; u = U[j][li]
                    # nearest uncached query (misalignment check)
                    near = int(np.argmin([abs(U[k][li] - s) for k in range(len(U))]))
                    recs.append({'file': fname, 'trial': tr['trial'], 'fmt': fmt, 'layer': li, 'j': j, 'stored': s, 'uncached': u, 'near': near})
    print(fname, 'done', round(time.time() - t0), 's', flush=True)
json.dump(recs, open(os.path.join(ROOT, 'audit2', f'e5_gpu_{name}.json'), 'w'))
# summary
for fmt in ('det', 'raw'):
    for li in TL:
        rr = [r for r in recs if r['fmt'] == fmt and r['layer'] == li]
        s = np.array([r['stored'] for r in rr]); u = np.array([r['uncached'] for r in rr])
        ad = np.abs(s - u); rel = ad / (np.abs(u) + 1e-9)
        mis = sum(1 for r in rr if r['near'] != r['j'] and abs(r['stored'] - r['uncached']) > 1e-6)
        dd = {}
        for f in ('n6_base_t1', 'n6_leak_t1'):
            dd[f] = (s[[r['file'] == f for r in rr]].mean(), u[[r['file'] == f for r in rr]].mean())
        print(f'{name} {fmt} L{li}{"*" if li == BL else ""}: n={len(rr)} maxabs={ad.max():.4g} maxrel={rel.max():.3g} medrel={np.median(rel):.3g} '
              f'sd(u)={u.std():.4g} corr={np.corrcoef(s, u)[0, 1]:.6f} nearest!=own:{mis} | base-leak stored={dd["n6_base_t1"][0] - dd["n6_leak_t1"][0]:+.3f} uncached={dd["n6_base_t1"][1] - dd["n6_leak_t1"][1]:+.3f}', flush=True)
