import json, glob, os, sys, collections
R=sys.argv[1]
sys.path.insert(0, R+'/exp1'); import analyze as A1
def mine(path, n, pre, field='correct'):
    acc=collections.defaultdict(lambda:[0.0,0])
    for l in open(path,encoding='utf-8'):
        r=json.loads(l); c=r['file'][:-6]
        for q in r['queries']:
            a=acc[(c,q['class'])]; a[0]+=q[field]; a[1]+=1
    f=lambda c,cl: acc[(f'{pre}_{c}',cl)][0]/acc[(f'{pre}_{c}',cl)][1]
    ch=1/n; aL=f('base_t0','L')
    if aL< ch+0.25: return 'R', aL
    g=lambda c:(f(c,'G')-ch)/(aL-ch)
    if g('base_t0')>=0.85: return 'B-type', aL
    base=max(g('base_t0'),g('scramble_t4')); comp=g('base_t4')-base
    if comp<0.15: return 'N', aL
    held=g('heldout_t4')-base
    return ('A' if held>=0.6*comp else 'N_iso'), aL
files=sorted(glob.glob(f'{R}/exp1/res/*.jsonl')+glob.glob(f'{R}/exp2/res/*_main.jsonl')+glob.glob(f'{R}/exp1/ideal/res/ideal_*.jsonl')+[f'{R}/exp3/res/A4.jsonl'])
for p in files:
    cells=A1.load(p)
    out=[]
    for pre,n in [('n6',6),('n8',8)]:
        for fld,fi in [('correct',0),('p_gold',2)]:
            d=A1.decision(cells,n,pre,fi,boot=False)
            if d is None: out.append(f'{pre}/{fld}: None'); continue
            m=mine(p,n,pre,fld)
            out.append(f'{pre}/{fld}: code={d["class"]} mine={m[0]} accL={d["accL"]:.3f}/{m[1]:.3f}' + ('' if d['class']==m[0] else ' MISMATCH'))
    print(os.path.basename(p), '; '.join(out))
# L counts at n8 base t0 and G-cells where gold is a query word
for p in [f'{R}/exp1/stim/n8_base_t0.jsonl', f'{R}/exp1/stim/n6_base_t0.jsonl']:
    Ls=[];
    for l in open(p,encoding='utf-8'):
        r=json.loads(l); Ls.append(sum(q['class']=='L' for q in r['queries']))
    print(os.path.basename(p),'L cells total',sum(Ls))
cnt=collections.Counter()
for f in glob.glob(f'{R}/exp1/stim/n6_*.jsonl')+glob.glob(f'{R}/exp1/stim/n8_*.jsonl'):
    for l in open(f,encoding='utf-8'):
        r=json.loads(l); W=r['target_words']
        for q in r['queries']:
            if q['class']!='G': continue
            a,b=W[q['cell'][0]],W[q['cell'][1]]
            cnt['G']+=1; cnt['gold_in_query']+= q['gold'] in (a,b)
print(cnt)
