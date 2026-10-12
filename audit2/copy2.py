import json, os, sys, glob, collections
R = sys.argv[1]
STIM = {os.path.basename(f): {r['trial']: r for r in map(json.loads, open(f, encoding='utf-8'))} for f in glob.glob(os.path.join(R, 'exp1/stim/*.jsonl'))}
for lab, p in [('Qwen3-0.6B','exp1/res/qwen3-0.6b.jsonl'),('Qwen3-1.7B','exp1/res/qwen3-1.7b.jsonl'),('SmolLM2','exp1/res/smollm2-1.7b.jsonl')]:
    c = collections.Counter()
    for l in open(os.path.join(R, p), encoding='utf-8'):
        r = json.loads(l); s = STIM[r['file']][r['trial']]; tw = s['target_words']
        sq = {tuple(q['cell']): q for q in s['queries']}
        for q in r['queries']:
            if q['class'] != 'G': continue
            a, b = sq[tuple(q['cell'])]['prompt_suffix'].strip().rstrip('=').strip().split(' * ')
            i, j = q['cell']; pre = r['file'][:2]
            c[pre, 'n'] += 1; c[pre, 'rep'] += q['pred'] in (a, b); c[pre, 'rep_tw'] += q['pred'] in (tw[i], tw[j]); c[pre, 'same'] += (a, b) == (tw[i], tw[j])
            c[pre, 'gold'] += q['gold'] in (a, b); c[pre,'gold_tw'] += q['gold'] in (tw[i], tw[j])
    for pre in ('n5', 'n6', 'n8'):
        n = c[pre, 'n']; print(lab, pre, n, 'rep %.3f rep_tw %.3f suffix==tw %.3f gold %d gold_tw %d' % (c[pre,'rep']/n, c[pre,'rep_tw']/n, c[pre,'same']/n, c[pre,'gold'], c[pre,'gold_tw']))
