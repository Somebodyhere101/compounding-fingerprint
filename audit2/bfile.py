import json, glob, os, collections
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
for name in ['olmo7b_main','olmo1b_main','q17bnf4_main','q17b_main','olmo7b_cal','q17b_cal','q06b_cal','q17_cal','smol17_cal']:
    f = os.path.join(R, 'exp2/res', name + '.jsonl')
    by = collections.defaultdict(collections.Counter)
    for l in open(f, encoding='utf-8'):
        r = json.loads(l)
        by[r['file']][(r.get('B_used','v?'), r.get('batch_max','-'))] += 1
    print(name)
    for k in sorted(by):
        print('   ', k, dict(by[k]))
