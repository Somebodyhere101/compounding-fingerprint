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
    ap.add_argument('--mem_frac', type=float, default=0.0, help='cap this process at a fraction of GPU memory (0 = no cap)')
    ap.add_argument('--offload_head', action='store_true', help='keep input embeddings and lm_head on CPU (saves VRAM for 7-8B)')
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if args.mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    files = args.files or sorted(glob.glob(os.path.join(args.stimdir, '*.jsonl')))
    path = snapshot(args.model)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if args.quant == 'none':
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval()
    else:
        from transformers import BitsAndBytesConfig
        qc = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
              if args.quant == 'nf4' else BitsAndBytesConfig(load_in_8bit=True))
        dmap = {'': 0}
        if args.offload_head:
            qc.llm_int8_enable_fp32_cpu_offload = True
            dmap = {'model.embed_tokens': 'cpu', 'model.layers': 0, 'model.norm': 0, 'model.rotary_emb': 0, 'lm_head': 'cpu'}
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, quantization_config=qc, dtype=torch.bfloat16, device_map=dmap).eval()
    W_emb = W_headT = None
    if args.offload_head:
        from safetensors import safe_open
        def load_key(key):
            for fn in sorted(glob.glob(os.path.join(path, '*.safetensors'))):
                with safe_open(fn, framework='pt') as fh:
                    if key in fh.keys():
                        return fh.get_tensor(key)
            return None
        W_emb = load_key('model.embed_tokens.weight').to(torch.bfloat16)
        Wh = load_key('lm_head.weight')
        W_headT = (W_emb if Wh is None else Wh).float().t().contiguous()
        del Wh
    print(f'loaded {args.model} quant={args.quant} offload_head={args.offload_head} vram={torch.cuda.memory_allocated()/2**30:.2f}GiB', flush=True)

    def fwd(ids2d, past=None, need_logits=True):
        """-> (last-position logits [B,V] float or None, cache)"""
        if not args.offload_head:
            o = model(input_ids=torch.tensor(ids2d, device='cuda'), past_key_values=past, use_cache=True, logits_to_keep=1)
            return (o.logits[:, -1].float() if need_logits else None), o.past_key_values
        x = W_emb[torch.tensor(ids2d)].to('cuda')
        o = model.model(inputs_embeds=x, past_key_values=past, use_cache=True)
        lg = (o.last_hidden_state[:, -1].float().cpu() @ W_headT) if need_logits else None
        return lg, o.past_key_values
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out, encoding='utf-8'):
            r = json.loads(l)
            done.add((r['file'], r['trial']))
    fout = open(args.out, 'a', encoding='utf-8')
    cand_id = {}
    multi_words = set()
    t0 = time.time(); ntr = 0; nq = 0; nfallback = 0; n_oom = [0]
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
                _, cache = fwd([ctx_ids], need_logits=False)
                out = None
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
                        lg = fwd([full])[0][0]
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
                    s0 = 0
                    while s0 < len(idxs):
                        chunk = idxs[s0:s0 + B]
                        try:
                            if len(chunk) == 1:  # no copy: run on the prefix cache, then crop it back (saves one full KV copy)
                                lg, _ = fwd([items[chunk[0]][1]], past=cache)
                                cache.crop(L)
                            else:
                                cb = copy.deepcopy(cache)
                                cb.batch_repeat_interleave(len(chunk))
                                lg, _ = fwd([items[i][1] for i in chunk], past=cb)
                                del cb
                        except torch.OutOfMemoryError:
                            cb = None; cache.crop(L); torch.cuda.empty_cache()  # undo any partial in-place append
                            if B == 1:
                                raise
                            B = max(1, B // 2); n_oom[0] += 1
                            continue
                        for r, i in enumerate(chunk):
                            logits_of[i] = lg[r]
                        s0 += len(chunk)
                recs = []
                for i, (q, suf, _) in enumerate(items):
                    lp = torch.log_softmax(logits_of[i], -1)
                    ids = torch.tensor([cand_id[w] for w in q['candidates']], device=lp.device)
                    cl = lp[ids]
                    pc = torch.softmax(cl, -1)
                    j = int(torch.argmax(cl))
                    gi = q['candidates'].index(q['gold'])
                    recs.append({'cell': q['cell'], 'class': q['class'], 'gold': q['gold'], 'pred': q['candidates'][j],
                                 'correct': int(j == gi), 'p_gold': round(float(pc[gi]), 5),
                                 'cand_mass': round(float(torch.exp(cl).sum()), 5)})
                del cache, out
            torch.cuda.empty_cache()
            fout.write(json.dumps({'model': args.model, 'file': fname, 'trial': tr['trial'], 'cond': tr['cond'], 'n': tr['n'],
                                   't': tr['t'], 'k': tr['k'], 'target_type': tr['target_type'], 'ctx_tokens': L, 'quant': args.quant, 'offload_head': args.offload_head, 'batch_max': args.batch, 'kv_budget_gb': args.kv_budget_gb, 'B_used': B,
                                   'queries': recs}) + '\n')
            fout.flush()
            ntr += 1; nq += len(recs)
            if ntr % 25 == 0:
                el = time.time() - t0
                print(f'{args.model} {fname} trials={ntr} queries={nq} {el:.0f}s ({el / ntr:.2f}s/trial) fallback={nfallback}', flush=True)
    print(f'DONE {args.model}: trials={ntr} queries={nq} time={time.time() - t0:.0f}s fallback={nfallback} '
          f'peakVRAM={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB multi_token_words={sorted(multi_words)} oom_backoffs={n_oom[0]}', flush=True)


if __name__ == '__main__':
    main()
