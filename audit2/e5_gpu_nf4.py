"""Audit2 / Exp5: nf4 models (4B, 8B with CPU embeddings/head): recompute calibration + uncached projections (subset queries matched by cell).
usage: python e5_gpu_nf4.py q4b Qwen/Qwen3-4B-Base NT [offload]"""
import os, sys, json, time, glob
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UP = r'<path to a local clone of github.com/Somebodyhere101/compute-retrieve>'
sys.path.insert(0, UP)
from cr_axis.detector import _format_prompt, _find_layers
name, repo, NT = sys.argv[1], sys.argv[2], int(sys.argv[3]); off = len(sys.argv) > 4
HUB = os.path.expanduser('~/.cache/huggingface/hub')
d = os.path.join(HUB, 'models--' + repo.replace('/', '--'), 'snapshots'); path = os.path.join(d, os.listdir(d)[0])
tok = AutoTokenizer.from_pretrained(path)
torch.cuda.set_per_process_memory_fraction(0.6)
qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
dmap = {'': 0}
if off:
    qc.llm_int8_enable_fp32_cpu_offload = True
    dmap = {'model.embed_tokens': 'cpu', 'model.layers': 0, 'model.norm': 0, 'model.rotary_emb': 0, 'lm_head': 'cpu'}
model = AutoModelForCausalLM.from_pretrained(path, quantization_config=qc, dtype=torch.bfloat16, device_map=dmap).eval()
W = None
if off:
    from safetensors import safe_open
    for fn in sorted(glob.glob(os.path.join(path, '*.safetensors'))):
        with safe_open(fn, framework='pt') as fh:
            if 'model.embed_tokens.weight' in fh.keys():
                W = fh.get_tensor('model.embed_tokens.weight').to(torch.bfloat16)
    E = model.model.embed_tokens.weight
    print('W_emb found', W is not None, tuple(W.shape), E.device, flush=True)
meta = json.load(open(os.path.join(ROOT, 'exp5', 'res', f'{name}_meta.json')))
TL = meta['test_layers']; BL = meta['best_layer']
axz = np.load(os.path.join(ROOT, 'exp5', 'res', f'{name}_axes.npz')); axes = {li: axz[f'L{li}'] for li in TL}
layers = _find_layers(model); cap = {}
for li in TL:
    layers[li].register_forward_hook(lambda m, i, o, li=li: cap.__setitem__(li, (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu().numpy()))
@torch.inference_mode()
def full(text):  # uncached, no past
    ids = tok(text)['input_ids']
    if W is not None:
        model.model(inputs_embeds=W[torch.tensor([ids])].to('cuda'), use_cache=False)
    else:
        model.model(input_ids=torch.tensor([ids], device='cuda'), use_cache=False)
    return {li: cap[li].copy() for li in TL}
C = [l.strip() for l in open(os.path.join(UP, 'data', 'calibration', 'compute.txt'), encoding='utf-8') if l.strip()]
R = [l.strip() for l in open(os.path.join(UP, 'data', 'calibration', 'retrieve.txt'), encoding='utf-8') if l.strip()]
hc = {li: [] for li in TL}; hr = {li: [] for li in TL}
for lst, st in [(C, hc), (R, hr)]:
    for p in lst:
        h = full(_format_prompt(p, tok))
        for li in TL: st[li].append(h[li])
def aucf(pos, neg):
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))
for li in TL:
    a, b = np.array(hc[li]), np.array(hr[li]); ax = a.mean(0) - b.mean(0); ax /= np.linalg.norm(ax)
    print(f'calib L{li}: auc={aucf(a @ ax, b @ ax):.6f} meta={meta["calib_auc"][str(li)]:.6f} cos(axis,npz)={float(ax @ axes[li]):.7f}', flush=True)
stored = {}
for l in open(os.path.join(ROOT, 'exp5', 'res', f'{name}.jsonl'), encoding='utf-8'):
    r = json.loads(l); stored[(r['file'], r['trial'])] = r
rng = np.random.default_rng(7); recs = []; t0 = time.time()
for fname in ['n6_base_t1', 'n6_leak_t1']:
    rows = [json.loads(l) for l in open(os.path.join(ROOT, 'exp1', 'stim', fname + '.jsonl'), encoding='utf-8')]
    for ti in sorted(rng.choice(len(rows), NT, replace=False)):
        tr = rows[ti]; st = stored[(fname, tr['trial'])]; bycell = {tuple(q['cell']): q for q in tr['queries']}
        for fmt in ('det', 'raw'):
            for j, sq in enumerate(st['queries']):
                q = bycell[tuple(sq['cell'])]; text = tr['context'] + q['prompt_suffix']
                if fmt == 'det': text = _format_prompt(text, tok)
                h = full(text)
                for li in TL:
                    recs.append({'file': fname, 'trial': tr['trial'], 'fmt': fmt, 'layer': li, 'j': j, 'stored': sq[f'proj_{fmt}'][str(li)], 'uncached': float(h[li] @ axes[li])})
    print(fname, 'done', round(time.time() - t0), flush=True)
json.dump(recs, open(os.path.join(ROOT, 'audit2', f'e5_gpu_{name}.json'), 'w'))
for fmt in ('det', 'raw'):
    for li in TL:
        rr = [r for r in recs if r['fmt'] == fmt and r['layer'] == li]
        s = np.array([r['stored'] for r in rr]); u = np.array([r['uncached'] for r in rr]); ad = np.abs(s - u)
        fb = np.array([r['file'] == 'n6_base_t1' for r in rr])
        print(f'{name} {fmt} L{li}{"*" if li == BL else ""}: n={len(rr)} maxabs={ad.max():.4g} maxrel={(ad / (np.abs(u) + 1e-9)).max():.3g} medrel={np.median(ad / (np.abs(u) + 1e-9)):.3g} corr={np.corrcoef(s, u)[0, 1]:.5f} base-leak stored={s[fb].mean() - s[~fb].mean():+.3f} uncached={u[fb].mean() - u[~fb].mean():+.3f}', flush=True)
