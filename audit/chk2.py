import json, glob, os, sys, collections, re
R=sys.argv[1]
for d in ['exp1/stim','exp2/stim']:
  for f in sorted(glob.glob(f'{R}/{d}/*.jsonl')):
    fn=os.path.basename(f)
    idpos=collections.Counter(); gG=collections.Counter(); gAll=collections.Counter(); leakq=0; nid=0
    for l in open(f,encoding='utf-8'):
        r=json.loads(l); W=r['target_words']; n=r['n']
        last=r['context'].split('Table ')[-1]
        facts={}
        for a,b,c in re.findall(r'^(\w+) \* (\w+) = (\w+)$', last, re.M): facts[(a,b)]=c
        # queried cells not revealed
        for q in r['queries']:
            a,b=W[q['cell'][0]],W[q['cell'][1]]
            assert q['prompt_suffix']==f'\n{a} * {b} =', q
            if (a,b) in facts: leakq+=1
            facts[(a,b)]=q['gold']
            gi=W.index(q['gold']); gAll[gi]+=1
            if q['class']=='G': gG[gi]+=1
        if len(facts)!=n*n: print('incomplete', fn, r['trial'], len(facts)); continue
        ids=[e for e in W if all(facts[(e,x)]==x and facts[(x,e)]==x for x in W)]
        if ids: idpos[W.index(ids[0])]+=1; nid+=1
    tot=sum(gG.values())
    print(fn, 'revealed-and-queried:',leakq, 'identity pos',dict(sorted(idpos.items())), 'G gold idx', [round(gG[i]/tot,3) for i in range(n)] if tot else None)
