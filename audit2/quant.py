import json, os, collections
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
def load(n):
    d = {}
    for l in open(os.path.join(R, 'exp2/res', n + '.jsonl'), encoding='utf-8'):
        r = json.loads(l)
        for q in r['queries']:
            d[(r['file'], r['trial'], tuple(q['cell']))] = (q['class'], q['correct'], q['pred'])
    return d
a, b = load('q17b_main'), load('q17bnf4_main')
assert a.keys() == b.keys()
agree = sum(a[k][2] == b[k][2] for k in a) / len(a)
print('n queries', len(a), 'argmax agreement', round(agree, 4))
acc = collections.defaultdict(lambda: [0, 0, 0])
for k in a:
    if a[k][0] == 'G':
        acc[k[0]][0] += 1; acc[k[0]][1] += a[k][1]; acc[k[0]][2] += b[k][1]
ds = []
for f in sorted(acc):
    n, ca, cb = acc[f]
    ds.append(cb / n - ca / n)
    print(f, n, 'bf16 %.3f nf4 %.3f d %+.3f' % (ca / n, cb / n, cb / n - ca / n))
print('dG range', min(ds), max(ds), 'all negative', all(d < 0 for d in ds))
# version split: agreement on v1-scored vs final-scored bf16 records
v = {}
for l in open(os.path.join(R, 'exp2/res/q17b_main.jsonl'), encoding='utf-8'):
    r = json.loads(l); v[(r['file'], r['trial'])] = 'final' if 'B_used' in r else 'v1'
ag = collections.defaultdict(lambda: [0, 0])
for k in a:
    g = v[(k[0], k[1])]; ag[g][0] += 1; ag[g][1] += a[k][2] == b[k][2]
print({g: round(x[1] / x[0], 4) for g, x in ag.items()})
# cal t split
c = collections.Counter()
for l in open(os.path.join(R, 'exp2/res/q8b_cal.jsonl'), encoding='utf-8'):
    r = json.loads(l); c[(r['file'], r['t'])] += 1
print(sorted(c.items()))
