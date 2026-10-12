"""Experiment 5b (exploratory, see CONTROL_vocab.md): vocabulary-overlap control for the base_t1 vs leak_t1 CR-axis difference.

usage: python ctrl_vocab.py --model Qwen/Qwen3-1.7B-Base --name q17b [--quant nf4]
Resumable: trials already in res_ctrl/{name}.jsonl are skipped.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse, json, re, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
UP = r'<path to a local clone of github.com/Somebodyhere101/compute-retrieve>'
sys.path.insert(0, UP)
from cr_axis.detector import _format_prompt, _find_layers  # noqa: E402  (read-only use of the published detector)

HUB = os.path.expanduser('~/.cache/huggingface/hub')
STIM = os.path.join(HERE, '..', 'exp1', 'stim')


def snapshot(repo):
    d = os.path.join(HUB, 'models--' + repo.replace('/', '--'), 'snapshots')
    return os.path.join(d, os.listdir(d)[0])


def split(ctx):
    a, b = ctx.split('\n\nTable B:\n')
    return a[len('Table A:\n'):], b


def relabel(block, mp):
    return re.sub(r'[a-z]+', lambda m: mp.get(m.group(0), m.group(0)), block)


def variants(leak_tr, base_tr):
    """-> {cond: context}; leak/base unchanged, leakRel/baseRel with Table A relabelled (per-trial seed)."""
    rng = np.random.default_rng([11, leak_tr['trial']])
    out = {'leak': leak_tr['context'], 'base': base_tr['context']}
    A_, B_ = split(leak_tr['context']); tw = leak_tr['target_words']
    while True:
        p = rng.permutation(6)
        if all(p[i] != i for i in range(6)):
            break
    out['leakRel'] = 'Table A:\n' + relabel(A_, {tw[i]: tw[p[i]] for i in range(6)}) + '\n\nTable B:\n' + B_
    A_, B_ = split(base_tr['context']); tw = base_tr['target_words']
    hw = list(dict.fromkeys(re.findall(r'[a-z]+', A_)))
    assert len(hw) == 6 and not set(hw) & set(tw)
    p = rng.permutation(6)
    out['baseRel'] = 'Table A:\n' + relabel(A_, {hw[i]: tw[p[i]] for i in range(6)}) + '\n\nTable B:\n' + B_
    for c, ctx in out.items():  # Table B (target) and length are untouched
        assert ctx.split('\n\nTable B:\n')[1] == (leak_tr if c.startswith('leak') else base_tr)['context'].split('\n\nTable B:\n')[1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--name', required=True)
    ap.add_argument('--quant', default='none', choices=['none', 'nf4'])
    ap.add_argument('--mem_frac', type=float, default=0.6)
    ap.add_argument('--batch', type=int, default=26, help='queries per right-padded forward (1 = unbatched reference path)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    path = snapshot(a.model)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if a.quant == 'none':
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval()
    else:
        from transformers import BitsAndBytesConfig
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, quantization_config=qc, dtype=torch.bfloat16, device_map={'': 0}).eval()
    meta = json.load(open(os.path.join(HERE, 'res', f'{a.name}_meta.json')))
    TL = meta['test_layers']
    axz = np.load(os.path.join(HERE, 'res', f'{a.name}_axes.npz')); axes = {li: axz[f'L{li}'] for li in TL}
    layers = _find_layers(model); cap = {}
    for li in TL:
        layers[li].register_forward_hook(lambda m, i, o, li=li: cap.__setitem__(li, (o[0] if isinstance(o, tuple) else o)[sel['rows'], sel['last']].float().cpu().numpy()))
    sel = {}
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    @torch.inference_mode()
    def projs(texts):
        """right-padded batches: real tokens never attend to padding (causal) and keep their positions, so each row equals its
        unpadded forward up to bf16 kernel noise; the hidden state is read at each row's last real token."""
        out = []
        for s0 in range(0, len(texts), a.batch):
            ids = [tok(t)['input_ids'] for t in texts[s0:s0 + a.batch]]
            T = max(map(len, ids)); lens = torch.tensor([len(x) for x in ids])
            x = torch.tensor([r + [pad] * (T - len(r)) for r in ids], device='cuda')
            m = torch.tensor([[1] * len(r) + [0] * (T - len(r)) for r in ids], device='cuda')
            sel['rows'], sel['last'] = torch.arange(len(ids), device='cuda'), (lens - 1).to('cuda')
            model.model(input_ids=x, attention_mask=m, use_cache=False)
            H = {li: cap[li] for li in TL}
            out += [{str(li): float(H[li][j] @ axes[li]) for li in TL} for j in range(len(ids))]
        return out

    leak = [json.loads(l) for l in open(os.path.join(STIM, 'n6_leak_t1.jsonl'), encoding='utf-8')]
    base = [json.loads(l) for l in open(os.path.join(STIM, 'n6_base_t1.jsonl'), encoding='utf-8')]
    os.makedirs(os.path.join(HERE, 'res_ctrl'), exist_ok=True)
    outp = a.out or os.path.join(HERE, 'res_ctrl', f'{a.name}.jsonl')
    done = {json.loads(l)['trial'] for l in open(outp, encoding='utf-8')} if os.path.exists(outp) else set()
    fout = open(outp, 'a', encoding='utf-8'); t0 = time.time()
    for ti in range(len(leak))[:a.limit]:
        if ti in done:
            continue
        assert leak[ti]['trial'] == base[ti]['trial'] == ti
        V = variants(leak[ti], base[ti])
        rec = {'trial': ti, 'model': a.model, 'quant': a.quant, 'best_layer': meta['best_layer'], 'batch': a.batch, 'conds': {}}
        for c, ctx in V.items():
            src = leak[ti] if c.startswith('leak') else base[ti]
            A2 = ctx.split('\n\nTable B:\n')[0].split('\n')[1:]
            Bl = ctx.split('\n\nTable B:\n')[1].split('\n')
            rec['conds'][c] = {'A_lines_shared_with_B': float(np.mean([l in A2 for l in Bl])),
                               'det': projs([_format_prompt(ctx + q['prompt_suffix'], tok) for q in src['queries']]),
                               'raw': projs([ctx + q['prompt_suffix'] for q in src['queries']]),
                               'classes': [q['class'] for q in src['queries']]}
        fout.write(json.dumps(rec) + '\n'); fout.flush()
        if ti % 10 == 0:
            print(f'trial {ti} {time.time() - t0:.0f}s', flush=True)
    print(f'DONE ctrl_vocab {a.name} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
