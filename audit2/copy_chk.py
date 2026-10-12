import json, os, sys, glob
R = sys.argv[1]
STIM = {os.path.basename(f): {r['trial']: r for r in map(json.loads, open(f, encoding='utf-8'))} for f in glob.glob(os.path.join(R, 'exp1/stim/*.jsonl'))}
files = [('Qwen3-0.6B','exp1/res/qwen3-0.6b.jsonl'),('Qwen3-1.7B','exp1/res/qwen3-1.7b.jsonl'),('SmolLM2','exp1/res/smollm2-1.7b.jsonl'),('Q0.6B-Base','exp2/res/q06b_main.jsonl'),('Q1.7B-Base','exp2/res/q17b_main.jsonl'),('Q4B','exp2/res/q4b_main.jsonl'),('Q8B','exp2/res/q8b_main.jsonl'),('OLMo1B','exp2/res/olmo1b_main.jsonl'),('OLMo7B','exp2/res/olmo7b_main.jsonl')]
gq = gt = None
for lab, p in files:
    rep = tot = gold_in = 0
    for l in open(os.path.join(R, p), encoding='utf-8'):
        r = json.loads(l); s = STIM[r['file']][r['trial']]
        sq = {tuple(q['cell']): q for q in s['queries']}
        for q in r['queries']:
            if q['class'] != 'G': continue
            a, b = sq[tuple(q['cell'])]['prompt_suffix'].strip().rstrip('=').strip().split(' * ')
            tot += 1; rep += q['pred'] in (a, b); gold_in += q['gold'] in (a, b)
    print(f'{lab:12s} G answers repeating a query word: {rep/tot:.3f} (n={tot}); gold is a query word: {gold_in/tot:.4f}')
