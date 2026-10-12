import json, sys, os, collections
R = sys.argv[1]
def ld(p):
    return {(r['file'], r['trial']): r for r in map(json.loads, open(os.path.join(R, p), encoding='utf-8'))}
a, b = ld('exp2/res/q17b_main.jsonl'), ld('exp2/res/q17bnf4_main.jsonl')
assert set(a) == set(b)
cell = collections.defaultdict(lambda: [0, 0, 0, 0])  # bf16 correct, nf4 correct, n, agree
allcls = collections.defaultdict(lambda: [0, 0, 0])
agree = tot = 0; aggG = [0,0]; dp=[]
for k in a:
    for qa, qb in zip(a[k]['queries'], b[k]['queries']):
        assert qa['cell'] == qb['cell']
        tot += 1; agree += qa['pred'] == qb['pred']
        dp.append(abs(qa['p_gold']-qb['p_gold']))
        c = allcls[qa['class']]; c[0]+=qa['correct']; c[1]+=qb['correct']; c[2]+=1
        if qa['class'] == 'G':
            aggG[0]+=1; aggG[1]+= qa['pred']==qb['pred']
            c = cell[k[0][:-6]]; c[0] += qa['correct']; c[1] += qb['correct']; c[2] += 1
print('argmax agreement all queries %.4f (%d/%d); G-only %.4f' % (agree/tot, agree, tot, aggG[1]/aggG[0]))
import statistics
print('mean |dp_gold| %.4f' % statistics.mean(dp))
ds = []
for c, v in sorted(cell.items()):
    d = v[1]/v[2] - v[0]/v[2]; ds.append(d)
    print(f'{c:18s} bf16 {v[0]/v[2]:.3f} nf4 {v[1]/v[2]:.3f} d {d:+.3f} n={v[2]}')
print('min d %.3f max d %.3f mean d %.3f maxabs %.3f; cells with d<0: %d/%d' % (min(ds), max(ds), sum(ds)/len(ds), max(map(abs, ds)), sum(x<0 for x in ds), len(ds)))
n6 = [c for c in cell if c.startswith('n6')]
print('n6-only mean d %.3f' % (sum(cell[c][1]/cell[c][2]-cell[c][0]/cell[c][2] for c in n6)/len(n6)))
for cl, v in allcls.items(): print(cl, 'bf16 %.3f nf4 %.3f d %+.3f n=%d' % (v[0]/v[2], v[1]/v[2], (v[1]-v[0])/v[2], v[2]))
