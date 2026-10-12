"""Audit2: surface differences between n6_base_t1 and n6_leak_t1 prompts (computed from stimulus files)."""
import os, json, re, collections, numpy as np
os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import AutoTokenizer
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = os.path.expanduser('~/.cache/huggingface/hub'); d = os.path.join(H, 'models--Qwen--Qwen3-1.7B-Base', 'snapshots')
tok = AutoTokenizer.from_pretrained(os.path.join(d, os.listdir(d)[0]))
def tables(ctx):
    parts = re.split(r'\n\nTable [A-Z]:\n', '\n\n' + ctx.split('Table A:\n', 1)[1] if False else ctx)
    blocks = re.findall(r'Table [A-Z]:\n((?:.+\n?)+?)(?:\n\n|$)', ctx + '\n\n')
    return [[l for l in b.strip().split('\n')] for b in blocks]
S = {}
for f in ['n6_base_t1', 'n6_leak_t1', 'n6_base_t0', 'n6_base_t4']:
    rows = [json.loads(l) for l in open(os.path.join(ROOT, 'exp1', 'stim', f + '.jsonl'), encoding='utf-8')]
    st = collections.defaultdict(list)
    for r in rows:
        ctx = r['context']; tb = tables(ctx); target = tb[-1]; hist = tb[:-1]
        words = re.findall(r'[a-z]+', ctx); tw = set(r['target_words'])
        st['n_tables'].append(len(tb)); st['target_lines'].append(len(target)); st['hist_lines'].append(sum(len(h) for h in hist))
        st['distinct_words_ctx'].append(len(set(words)))
        st['target_lines_repeated_in_history'].append(np.mean([any(l in h for h in hist) for l in target]) if hist else 0)
        st['hist_words_eq_target_words'].append(float(any(set(re.findall(r'[a-z]+', '\n'.join(h))) == tw for h in hist)))
        ids = tok(ctx)['input_ids']; st['ctx_tokens'].append(len(ids)); st['distinct_tok_frac'].append(len(set(ids)) / len(ids))
        # repeated 4-gram fraction of tokens
        g4 = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]; c4 = collections.Counter(g4); st['repeated_4gram_frac'].append(np.mean([c4[g] > 1 for g in g4]))
        for q in r['queries']:
            a, b = re.findall(r'[a-z]+', q['prompt_suffix'])
            pair = f'{a} * {b} ='
            st['q_pair_in_ctx'].append(float(pair in ctx)); st['q_pair_in_target'].append(float(any(l.startswith(pair) for l in target)))
            st['q_full_line_in_ctx_with_gold'].append(float(f'{pair} {q["gold"]}' in ctx))
            st['q_word_occurrences_ctx'].append(words.count(a) + words.count(b))
            st['q_suffix_tokens'].append(len(tok(q['prompt_suffix'])['input_ids']))
            st['full_tokens'].append(len(tok(ctx + q['prompt_suffix'])['input_ids']))
            # distance (in lines from end) to last line in ctx mentioning a word of the query
            lines = ctx.split('\n'); last = max([i for i, l in enumerate(lines) if a in re.findall(r'[a-z]+', l)] + [-1])
            st['lines_since_query_word_a'].append(len(lines) - last)
            st['class_' + q['class']].append(1)
    S[f] = st
    print(f'== {f} ({len(rows)} trials, {sum(len(r["queries"]) for r in rows)} queries)')
    for k, v in st.items():
        if k.startswith('class_'): continue
        print(f'   {k:36s} mean {np.mean(v):9.3f}  sd {np.std(v):8.3f}  min {np.min(v):7.2f} max {np.max(v):7.2f}')
    print('   classes', {k[6:]: len(v) for k, v in st.items() if k.startswith('class_')})
    print('   history types', collections.Counter(str(r['history_types']) for r in rows).most_common(3), 'target types', collections.Counter(str(r['target_type']) for r in rows).most_common(4))
