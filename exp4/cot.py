"""Experiment 4 (exploratory): does test-time reasoning unlock the Latin inference that direct answering lacks?
Qwen3-1.7B (post-trained, thinking-capable) answers L-forced (and G-forced) cells of the Experiment-2 sweep trials,
with enable_thinking=True vs False, from the same chat prompt. One query per trial (the first query of the target class).

usage: python cot.py --out res/qwen3-1.7b_cot.jsonl [--n_per_cell 30] [--max_new 1536]
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse, json, re, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
HUB = os.path.expanduser('~/.cache/huggingface/hub')
STIM = os.path.join(HERE, '..', 'exp2', 'stim')
CELLS = [('kso_n6_k28.jsonl', 'L'), ('ks_n6_k28.jsonl', 'L')]


def snapshot(repo):
    d = os.path.join(HUB, 'models--' + repo.replace('/', '--'), 'snapshots')
    return os.path.join(d, os.listdir(d)[0])


def prompt_for(tr, q):
    target = tr['context'].split('\n\n')[-1]
    words = ', '.join(tr['target_words'])
    a, b = q['prompt_suffix'].strip().rstrip('=').strip().split(' * ')
    return (f"Below is a partial table of a binary operation * on the set {{{words}}}. Each line 'x * y = z' gives one entry. "
            f"Every row and every column of the full table contains each word exactly once (a Latin square).\n\n{target}\n\n"
            f"What is {a} * {b}? Reply with the single word only, in the form 'ANSWER: <word>'.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-1.7B')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_per_cell', type=int, default=30)
    ap.add_argument('--max_new', type=int, default=1536)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    path = snapshot(args.model)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16).to(args.device).eval()
    done = set()
    if os.path.exists(args.out):
        done = {(r['file'], r['trial'], r['thinking']) for r in map(json.loads, open(args.out, encoding='utf-8'))}
    fout = open(args.out, 'a', encoding='utf-8')
    t0 = time.time()
    for fname, cls in CELLS:
        rows = [json.loads(l) for l in open(os.path.join(STIM, fname), encoding='utf-8')]
        rows = [r for r in rows if r['t'] == 0][:args.n_per_cell]
        for tr in rows:
            qs = [q for q in tr['queries'] if q['class'] == cls]
            if not qs:
                continue
            q = qs[0]
            for thinking in (False, True):
                if (fname, tr['trial'], thinking) in done:
                    continue
                msgs = [{'role': 'user', 'content': prompt_for(tr, q)}]
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
                ids = tok(text, return_tensors='pt').to(args.device)
                with torch.inference_mode():
                    out = model.generate(**ids, max_new_tokens=args.max_new if thinking else 400, do_sample=False,
                                         pad_token_id=tok.eos_token_id)
                gen = tok.decode(out[0, ids['input_ids'].shape[1]:], skip_special_tokens=True)
                final = gen.split('</think>')[-1]
                trunc = thinking and '</think>' not in gen
                m = re.findall(r'ANSWER:\s*\**\s*([A-Za-z]+)', final)
                if not m and not trunc:  # fall back to the last candidate word mentioned in the final answer
                    m = re.findall(r'\b(' + '|'.join(tr['target_words']) + r')\b', final)
                pred = m[-1].lower() if (m and not trunc) else None
                rec = {'file': fname, 'trial': tr['trial'], 'k': tr['k'], 'class': cls, 'thinking': thinking, 'gold': q['gold'],
                       'pred': pred, 'correct': int(pred == q['gold']), 'n_gen_tokens': int(out.shape[1] - ids['input_ids'].shape[1]),
                       'truncated': int(trunc), 'explicit_answer': int(bool(re.search(r'ANSWER:', final))),
                       'final_text': final.strip()[:300]}
                fout.write(json.dumps(rec) + '\n'); fout.flush()
        print(f'{fname} done, {time.time() - t0:.0f}s', flush=True)
    print('DONE cot', flush=True)


if __name__ == '__main__':
    main()
