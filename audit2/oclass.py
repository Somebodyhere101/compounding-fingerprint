import json, sys, os, collections
R = sys.argv[1]
for k in ['A1','A2','A4','A6','G4','I4','P4']:
    agg = collections.defaultdict(lambda: [0, 0])
    for l in open(os.path.join(R, f'exp3/res/{k}.jsonl'), encoding='utf-8'):
        r = json.loads(l)
        if not r['file'].startswith('n6_base'): continue
        for q in r['queries']:
            a = agg[(r['t'], q['class'])]; a[0] += q['correct']; a[1] += 1
    print(k, ' '.join(f"t{t}:{c}={v[0]/v[1]:.3f}" for (t, c), v in sorted(agg.items()) if c in ('L','O','G')))
