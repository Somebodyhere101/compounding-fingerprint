import json, glob, os, sys, collections, math
import numpy as np
R=sys.argv[1]
files=glob.glob(f'{R}/exp1/res/*.jsonl')+glob.glob(f'{R}/exp2/res/*.jsonl')
for p in sorted(files):
    nan=0; n=0; wrong_hi=0; pg=[]; cm=[]; corr_lo=0; eq=0
    for l in open(p,encoding='utf-8'):
        r=json.loads(l)
        for q in r['queries']:
            n+=1; x=q['p_gold']
            if x!=x: nan+=1; continue
            pg.append(x); cm.append(q['cand_mass'])
            if not q['correct'] and x>=0.5: wrong_hi+=1
            if q['correct'] and x<1/r['n']: corr_lo+=1
    pg=np.array(pg); cm=np.array(cm)
    print(os.path.basename(p), 'n',n,'nan',nan,'wrong&p>=.5',wrong_hi, 'correct&p<1/n',corr_lo, 'distinct p_gold',len(set(pg.round(5))), 'cand_mass median %.3f p10 %.3f'%(np.median(cm),np.percentile(cm,10)))
