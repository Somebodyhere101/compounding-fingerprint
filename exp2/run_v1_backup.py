"""Score the Experiment 1 stimuli with a local causal LM (forced choice by next-token log-prob).

usage: python run.py --model Qwen/Qwen3-1.7B --out res/qwen3-1.7b.jsonl [--limit N] [--files stim/n6_*.jsonl ...]
Resumable: trials already present in --out are skipped.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse, copy, glob, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
HUB = os.path.expanduser('~/.cache/huggingface/hub')


def snapshot(repo):
    d = os.path.join(HUB, 'models--' + repo.replace('/', '--'), 'snapshots')
    return os.path.join(d, os.listdir(d)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--files', nargs='*', default=None)
    ap.add_argument('--limit', type=int, default=None, help='max trials per stimulus file')
    ap.add_argument('--batch', type=int, default=16, help='max queries scored together on a copied prefix cache')
    ap.add_argument('--kv_budget_gb', type=float, default=1.5)
    ap.add_argument('--quant', default='none', choices=['none', 'nf4', 'int8'])
    ap.add_argument('--stimdir', default=os.path.join(HERE, 'stim'))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    files = args.files or sorted(glob.glob(os.path.join(args.stimdir, '*.jsonl')))
    path = snapshot(args.model)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if args.quant == 'none':
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval()
    else:
        from transformers import BitsAndBytesConfig
        qc = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
              if args.quant == 'nf4' else BitsAndBytesConfig(load_in_8bit=True))
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, quantization_config=qc, dtype=torch.bfloat16, device_map={'': 0}).eval()
    print(f'loaded {args.model} quant={args.quant} vram={torch.cuda.memory_allocated()/2**30:.2f}GiB', flush=True)
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out, encoding='utf-8'):
            r = json.loads(l)
            done.add((r['file'], r['trial']))
    fout = open(args.out, 'a', encoding='utf-8')
    cand_id = {}
    multi_words = set()
    t0 = time.time(); ntr = 0; nq = 0; nfallback = 0
    for f in files:
        fname = os.path.basename(f)
        rows = [json.loads(l) for l in open(f, encoding='utf-8')]
        if args.limit:
            rows = rows[:args.limit]
        for tr in rows:
            if (fname, tr['trial']) in done:
                continue
            ctx = tr['context']
            ctx_ids = tok(ctx, add_special_tokens=True)['input_ids']
            L = len(ctx_ids)
            with torch.inference_mode():
                m0 = torch.cuda.memory_allocated()
                out = model(input_ids=torch.tensor([ctx_ids], device='cuda'), use_cache=True, logits_to_keep=1)
                cache = out.past_key_values
                kv_bytes = max(torch.cuda.memory_allocated() - m0, 1)
                B = max(1, min(args.batch, int(args.kv_budget_gb * 2**30 / kv_bytes)))
                items = []
                for q in tr['queries']:
                    for w in q['candidates']:
                        if w not in cand_id:
                            ids = tok.encode(' ' + w, add_special_tokens=False)
                            cand_id[w] = ids[0]
                            if len(ids) != 1:
                                multi_words.add(w)
                    firsts = [cand_id[w] for w in q['candidates']]
                    assert len(set(firsts)) == len(firsts), ('first-token collision', q['candidates'])
                    full = tok(ctx + q['prompt_suffix'], add_special_tokens=True)['input_ids']
                    if full[:L] != ctx_ids:
                        nfallback += 1
                        lg = model(input_ids=torch.tensor([full], device='cuda'), logits_to_keep=1).logits[0, -1].float()
                        items.append((q, None, lg))
                    else:
                        items.append((q, full[L:], None))
                # batch the cached queries by suffix length
                bylen = {}
                for i, (q, suf, lg) in enumerate(items):
                    if suf is not None:
                        bylen.setdefault(len(suf), []).append(i)
                logits_of = {i: lg for i, (q, suf, lg) in enumerate(items) if lg is not None}
                for ln, idxs in bylen.items():
                    for s0 in range(0, len(idxs), B):
                        chunk = idxs[s0:s0 + B]
                        cb = copy.deepcopy(cache)
                        if len(chunk) > 1:
                            cb.batch_repeat_interleave(len(chunk))
                        inp = torch.tensor([items[i][1] for i in chunk], device='cuda')
                        lg = model(input_ids=inp, past_key_values=cb, use_cache=True, logits_to_keep=1).logits[:, -1].float()
                        del cb
                        for r, i in enumerate(chunk):
                            logits_of[i] = lg[r]
                recs = []
                for i, (q, suf, _) in enumerate(items):
                    lp = torch.log_softmax(logits_of[i], -1)
                    ids = torch.tensor([cand_id[w] for w in q['candidates']], device='cuda')
                    cl = lp[ids]
                    pc = torch.softmax(cl, -1)
                    j = int(torch.argmax(cl))
                    gi = q['candidates'].index(q['gold'])
                    recs.append({'cell': q['cell'], 'class': q['class'], 'gold': q['gold'], 'pred': q['candidates'][j],
                                 'correct': int(j == gi), 'p_gold': round(float(pc[gi]), 5),
                                 'cand_mass': round(float(torch.exp(cl).sum()), 5)})
                del cache, out
            fout.write(json.dumps({'model': args.model, 'file': fname, 'trial': tr['trial'], 'cond': tr['cond'], 'n': tr['n'],
                                   't': tr['t'], 'k': tr['k'], 'target_type': tr['target_type'], 'ctx_tokens': L, 'quant': args.quant,
                                   'queries': recs}) + '\n')
            fout.flush()
            ntr += 1; nq += len(recs)
            if ntr % 25 == 0:
                el = time.time() - t0
                print(f'{args.model} {fname} trials={ntr} queries={nq} {el:.0f}s ({el / ntr:.2f}s/trial) fallback={nfallback}', flush=True)
    print(f'DONE {args.model}: trials={ntr} queries={nq} time={time.time() - t0:.0f}s fallback={nfallback} '
          f'peakVRAM={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB multi_token_words={sorted(multi_words)}', flush=True)


if __name__ == '__main__':
    main()
