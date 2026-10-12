import json, glob, os, sys, collections
from scipy.stats import chisquare
R=sys.argv[1]
stim={}
for d in ['exp1/stim','exp2/stim']:
    for f in glob.glob(f'{R}/{d}/*.jsonl'):
        fn=os.path.basename(f)
        for l in open(f,encoding='utf-8'):
            r=json.loads(l); stim[(fn,r['trial'])]=r
files=glob.glob(f'{R}/exp1/res/*.jsonl')+glob.glob(f'{R}/exp2/res/*.jsonl')+glob.glob(f'{R}/exp1/ideal/res/ideal_*.jsonl')+[f'{R}/exp3/res/A4.jsonl']
for p in sorted(files):
    seen=collections.Counter(); percell=collections.Counter(); bad=0; nq=0; mism=0
    predidx=collections.Counter(); goldidx=collections.Counter(); fpos=collections.Counter(); fall=collections.Counter()
    quants=collections.Counter(); models=collections.Counter()
    for l in open(p,encoding='utf-8'):
        r=json.loads(l); key=(r['file'],r['trial']); seen[key]+=1; percell[r['file']]+=1
        quants[r.get('quant')]+=1; models[r.get('model')]+=1
        s=stim[key]
        sq={tuple(q['cell']):q for q in s['queries']}
        if len(sq)!=len(r['queries']): bad+=1
        for q in r['queries']:
            nq+=1
            t=sq[tuple(q['cell'])]
            if t['class']!=q['class'] or t['gold']!=q['gold']: mism+=1
            c=t['candidates']; pi=c.index(q['pred']); gi=c.index(q['gold'])
            predidx[pi]+=1; goldidx[gi]+=1
            fall[gi]+=1; fpos[gi]+=q['correct']
    dup=sum(1 for v in seen.values() if v>1)
    n6=[predidx[i] for i in range(6)]
    print(os.path.basename(p), 'lines',sum(seen.values()),'dup',dup,'qcount_mismatch',bad,'class/gold mismatch',mism,'quant',dict(quants),'model',dict(models))
    print('   cells(min,max)',min(percell.values()),max(percell.values()),len(percell))
    tot=sum(predidx.values())
    print('   pred idx frac', [round(predidx[i]/tot,3) for i in range(8)], ' gold idx frac',[round(goldidx[i]/tot,3) for i in range(8)])
    print('   acc by gold idx', [round(fpos[i]/fall[i],3) if fall[i] else None for i in range(8)])
