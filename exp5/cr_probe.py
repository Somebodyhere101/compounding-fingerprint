"""Experiment 5: CR-axis projections on the fingerprint trials (see PREREGISTRATION.md).

Replicates cr_axis.detector.CRDetector (read-only import of _format_prompt / _find_layers): decoder-layer-output hidden state
at the last token, mean-difference axis on the paper's calibration prompts, best of the detector's 4 candidate layers by AUC.
Forward passes call the decoder stack only (no lm_head), with a shared-prefix KV cache per trial.

usage: python cr_probe.py --model Qwen/Qwen3-1.7B-Base --name q17b [--quant nf4] [--offload_head] [--max_q 12] [--verify]
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse, glob, json, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
UP = r'<path to a local clone of github.com/Somebodyhere101/compute-retrieve>'
sys.path.insert(0, UP)
from cr_axis.detector import _format_prompt, _find_layers  # noqa: E402  (read-only use of the published detector)

HUB = os.path.expanduser('~/.cache/huggingface/hub')
STIM = os.path.join(HERE, '..', 'exp1', 'stim')
FILES = ['n6_base_t1', 'n6_leak_t1', 'n6_base_t0', 'n6_base_t4']


def snapshot(repo):
    d = os.path.join(HUB, 'models--' + repo.replace('/', '--'), 'snapshots')
    return os.path.join(d, os.listdir(d)[0])


def auc(pos, neg):
    """Mann-Whitney AUC with tie averaging (equals sklearn roc_auc_score)."""
    x = np.concatenate([pos, neg]); r = np.empty(len(x))
    order = np.argsort(x, kind='mergesort'); xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--name', required=True)
    ap.add_argument('--quant', default='none', choices=['none', 'nf4'])
    ap.add_argument('--offload_head', action='store_true')
    ap.add_argument('--mem_frac', type=float, default=0.6)
    ap.add_argument('--max_q', type=int, default=0, help='queries per trial (0 = all), stratified by class')
    ap.add_argument('--limit', type=int, default=None, help='trials per file (testing)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--verify', action='store_true', help='check hidden states against CRDetector._get_hidden')
    a = ap.parse_args()
    dev = a.device
    if dev == 'cuda' and a.mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    path = snapshot(a.model)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if a.quant == 'none':
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16).to(dev).eval()
        W_emb = None
    else:
        from transformers import BitsAndBytesConfig
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        dmap = {'': 0}
        if a.offload_head:
            qc.llm_int8_enable_fp32_cpu_offload = True
            dmap = {'model.embed_tokens': 'cpu', 'model.layers': 0, 'model.norm': 0, 'model.rotary_emb': 0, 'lm_head': 'cpu'}
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, quantization_config=qc, dtype=torch.bfloat16, device_map=dmap).eval()
        W_emb = None
        if a.offload_head:
            from safetensors import safe_open
            for fn in sorted(glob.glob(os.path.join(path, '*.safetensors'))):
                with safe_open(fn, framework='pt') as fh:
                    if 'model.embed_tokens.weight' in fh.keys():
                        W_emb = fh.get_tensor('model.embed_tokens.weight').to(torch.bfloat16)
    layers = _find_layers(model); nl = len(layers)
    test_layers = sorted(set([max(1, nl // 4), nl // 2, int(nl * 0.75), nl - 2]))  # exactly as CRDetector.calibrate
    cap = {}
    hooks = [layers[li].register_forward_hook(lambda m, i, o, li=li: cap.__setitem__(li, (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu().numpy()))
             for li in test_layers]

    @torch.inference_mode()
    def run(ids, past=None):
        if W_emb is not None:
            o = model.model(inputs_embeds=W_emb[torch.tensor([ids])].to('cuda'), past_key_values=past, use_cache=True)
        else:
            o = model.model(input_ids=torch.tensor([ids], device=dev), past_key_values=past, use_cache=True)
        return {li: cap[li].copy() for li in test_layers}, o.past_key_values

    def enc(text):
        return tok(text)['input_ids']  # CRDetector: tok(text, return_tensors='pt') with default special tokens

    out = {'model': a.model, 'name': a.name, 'quant': a.quant, 'n_layers': nl, 'test_layers': test_layers}
    # ---------------- verification against the published detector
    if a.verify:
        from cr_axis.detector import CRDetector
        det = CRDetector(model, tok)
        prompts = ['What is 347 + 289?', 'The capital of France is', open(os.path.join(UP, 'data', 'calibration', 'compute.txt'), encoding='utf-8').readline().strip()]
        md = 0.0
        for p in prompts:
            h, _ = run(enc(_format_prompt(p, tok)))
            for li in test_layers:
                md = max(md, float(np.abs(h[li] - det._get_hidden(p, li)).max() / (np.abs(h[li]).max() + 1e-9)))
        out['verify_max_rel_diff'] = md
        print('verify: max relative diff vs CRDetector._get_hidden =', md, flush=True)
    # ---------------- calibration
    C = [l.strip() for l in open(os.path.join(UP, 'data', 'calibration', 'compute.txt'), encoding='utf-8') if l.strip()]
    R = [l.strip() for l in open(os.path.join(UP, 'data', 'calibration', 'retrieve.txt'), encoding='utf-8') if l.strip()]
    HC = {li: [] for li in test_layers}; HR = {li: [] for li in test_layers}
    for lst, store in [(C, HC), (R, HR)]:
        for p in lst:
            h, _ = run(enc(_format_prompt(p, tok)))
            for li in test_layers:
                store[li].append(h[li])
    axes, aucs = {}, {}
    best, best_l = 0, None
    for li in test_layers:
        hc, hr = np.array(HC[li]), np.array(HR[li])
        ax = hc.mean(0) - hr.mean(0); ax /= np.linalg.norm(ax)
        axes[li] = ax; aucs[li] = float(auc(hc @ ax, hr @ ax))
        if aucs[li] > best:
            best, best_l = aucs[li], li
    out.update({'calib_auc': aucs, 'best_layer': best_l, 'best_auc': best})
    print(f'calibration: AUC by layer {aucs} -> layer {best_l}', flush=True)
    np.savez(os.path.join(HERE, 'res', f'{a.name}_axes.npz'), **{f'L{li}': axes[li] for li in test_layers})
    # ---------------- test trials
    fout = open(os.path.join(HERE, 'res', f'{a.name}.jsonl'), 'w', encoding='utf-8')
    t0 = time.time(); nq = 0
    for fname in FILES:
        rows = [json.loads(l) for l in open(os.path.join(STIM, fname + '.jsonl'), encoding='utf-8')][:a.limit]
        for tr in rows:
            qs = tr['queries']
            if a.max_q and len(qs) > a.max_q:
                rng = np.random.default_rng(1000 + tr['trial'])
                by = {}
                for q in qs:
                    by.setdefault(q['class'], []).append(q)
                quota = {c: max(1, round(a.max_q * len(v) / len(qs))) for c, v in by.items()}
                sel = []
                for c in sorted(by):
                    v = by[c]
                    sel += [v[i] for i in rng.permutation(len(v))[:quota[c]]]
                sel = [sel[i] for i in rng.permutation(len(sel))]
                qs = sel[:a.max_q]
            rec = {'file': fname, 'trial': tr['trial'], 'cond': tr['cond'], 't': tr['t'], 'queries': []}
            for fmt in ('det', 'raw'):
                texts = [(_format_prompt(tr['context'] + q['prompt_suffix'], tok) if fmt == 'det' else tr['context'] + q['prompt_suffix']) for q in qs]
                toks = [enc(x) for x in texts]
                L = min(len(t) for t in toks)
                for i in range(L):
                    if any(t[i] != toks[0][i] for t in toks):
                        L = i; break
                L = min(L, min(len(t) for t in toks) - 1)  # every query keeps >= 1 own token
                _, cache = run(toks[0][:L])
                for j, (q, t) in enumerate(zip(qs, toks)):
                    h, _ = run(t[L:], past=cache); cache.crop(L)
                    if fmt == 'det':
                        rec['queries'].append({'cell': q['cell'], 'class': q['class'], 'gold': q['gold']})
                    rec['queries'][j][f'proj_{fmt}'] = {str(li): float(h[li] @ axes[li]) for li in test_layers}
                rec[f'ctx_tokens_{fmt}'] = L
                del cache
            fout.write(json.dumps(rec) + '\n'); fout.flush(); nq += len(qs)
            torch.cuda.empty_cache() if dev == 'cuda' else None
        print(f'{fname} done, queries so far {nq}, {time.time() - t0:.0f}s', flush=True)
    out['n_queries'] = nq; out['seconds'] = round(time.time() - t0)
    json.dump(out, open(os.path.join(HERE, 'res', f'{a.name}_meta.json'), 'w'), indent=1, default=float)
    for h in hooks:
        h.remove()
    print(f'DONE cr_probe {a.name}: {nq} queries, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
