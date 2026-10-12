import json, os, glob
from collections import Counter, defaultdict
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
stim = {}
for p in glob.glob(os.path.join(R, 'exp1/stim/*.jsonl')):
    for l in open(p, encoding='utf-8'):
        s = json.loads(l); stim[(os.path.basename(p), s['trial'])] = s['target_words']
for m in ['qwen3-0.6b', 'qwen3-1.7b', 'smollm2-1.7b']:
    agg = defaultdict(Counter)
    for l in open(os.path.join(R, 'exp1/res', m + '.jsonl'), encoding='utf-8'):
        r = json.loads(l); tw = stim[(r['file'], r['trial'])]
        for q in r['queries']:
            if q['class'] != 'G': continue
            a, b = q['cell']; c = agg[r['file'][:2]]
            c['n'] += 1; c['ca'] += q['pred'] == tw[a]; c['cb'] += q['pred'] == tw[b]; c['u'] += q['pred'] in (tw[a], tw[b]); c['diag'] += a == b
            c['ca_offdiag'] += (q['pred'] == tw[a]) and a != b
    for k, c in sorted(agg.items()):
        N = c['n']; print(m, k, f"copy_a={c['ca']/N:.3f} copy_b={c['cb']/N:.3f} sum={(c['ca']+c['cb'])/N:.3f} union={c['u']/N:.3f} diagfrac={c['diag']/N:.3f}")
