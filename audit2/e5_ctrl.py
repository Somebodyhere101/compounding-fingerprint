"""Audit2 exploratory control (q17b / q06b, det format, all 4 layers): is the base-vs-leak CR difference carried by vocabulary overlap?
conds: leak (orig), base (orig), leakRel = leak with Table A words relabelled by a random derangement (same 6 words, pairs present, answers inconsistent with Table B),
       baseRel = base with Table A words mapped onto the target's 6 words (different group table, target vocabulary)."""
import os, sys, json, re
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'<path to a local clone of github.com/Somebodyhere101/compute-retrieve>')
from cr_axis.detector import _format_prompt, _find_layers
name, repo, NT = sys.argv[1], sys.argv[2], int(sys.argv[3])
H = os.path.expanduser('~/.cache/huggingface/hub'); d = os.path.join(H, 'models--' + repo.replace('/', '--'), 'snapshots'); path = os.path.join(d, os.listdir(d)[0])
tok = AutoTokenizer.from_pretrained(path); model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to('cuda').eval()
meta = json.load(open(os.path.join(ROOT, 'exp5', 'res', f'{name}_meta.json'))); TL = meta['test_layers']; BL = meta['best_layer']
axz = np.load(os.path.join(ROOT, 'exp5', 'res', f'{name}_axes.npz')); axes = {li: axz[f'L{li}'] for li in TL}
layers = _find_layers(model); cap = {}
for li in TL: layers[li].register_forward_hook(lambda m, i, o, li=li: cap.__setitem__(li, (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu().numpy()))
@torch.inference_mode()
def proj(text):
    model.model(input_ids=torch.tensor([tok(_format_prompt(text, tok))['input_ids']], device='cuda'), use_cache=False)
    return {li: float(cap[li] @ axes[li]) for li in TL}
def split(ctx):
    a, b = ctx.split('\n\nTable B:\n'); return a[len('Table A:\n'):], b
def relabel(block, mp):
    return re.sub(r'[a-z]+', lambda m: mp.get(m.group(0), m.group(0)), block)
rng = np.random.default_rng(11)
L = {f: [json.loads(l) for l in open(os.path.join(ROOT, 'exp1', 'stim', f + '.jsonl'), encoding='utf-8')] for f in ['n6_base_t1', 'n6_leak_t1']}
idx = sorted(rng.choice(100, NT, replace=False))
res = {c: {li: [] for li in TL} for c in ['leak', 'leakRel', 'base', 'baseRel']}
stats = {'leakRel_lines_shared_with_B': [], 'baseRel_lines_shared_with_B': []}
for ti in idx:
    for f, c0, c1 in [('n6_leak_t1', 'leak', 'leakRel'), ('n6_base_t1', 'base', 'baseRel')]:
        tr = L[f][ti]; A_, B_ = split(tr['context']); tw = tr['target_words']
        if f == 'n6_leak_t1':
            while True:
                p = rng.permutation(6)
                if all(p[i] != i for i in range(6)): break
            mp = {tw[i]: tw[p[i]] for i in range(6)}
        else:
            hw = list(dict.fromkeys(re.findall(r'[a-z]+', A_))); p = rng.permutation(6); mp = {hw[i]: tw[p[i]] for i in range(6)}
        A2 = relabel(A_, mp); ctx2 = 'Table A:\n' + A2 + '\n\nTable B:\n' + B_
        stats[c1 + '_lines_shared_with_B'].append(np.mean([l in A2.split('\n') for l in B_.split('\n')]))
        for c, ctx in [(c0, tr['context']), (c1, ctx2)]:
            ps = [proj(ctx + q['prompt_suffix']) for q in tr['queries']]
            for li in TL: res[c][li].append(np.mean([p[li] for p in ps]))
print(name, 'NT', NT, {k: round(float(np.mean(v)), 3) for k, v in stats.items()})
for li in TL:
    print(f'  L{li}{"*" if li == BL else ""}: ' + '  '.join(f'{c} {np.mean(res[c][li]):+.2f}' for c in res) + f'   | base-leak {np.mean(res["base"][li]) - np.mean(res["leak"][li]):+.2f}; baseRel-leak {np.mean(res["baseRel"][li]) - np.mean(res["leak"][li]):+.2f}; leakRel-leak {np.mean(res["leakRel"][li]) - np.mean(res["leak"][li]):+.2f}', flush=True)
json.dump({c: {str(li): v for li, v in d.items()} for c, d in res.items()}, open(os.path.join(ROOT, 'audit2', f'e5_ctrl_{name}.json'), 'w'))
