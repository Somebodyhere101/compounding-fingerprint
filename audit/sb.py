import json,collections,glob
def load(f): return [json.loads(l) for l in open(f)]
def blocks(ctx): return ctx.split('\n\n')
def chk_sorted(raw, so, k):
    bad=collections.Counter(); order=collections.Counter()
    assert len(raw)==len(so)
    for a,b in zip(raw,so):
        if a['trial']!=b['trial']: bad['trial id']+=1; continue
        for key in a:
            if key in ('context','cond'): continue
            if a[key]!=b[key]: bad['field '+key]+=1
        A,B=blocks(a['context']),blocks(b['context'])
        if A[:-1]!=B[:-1]: bad['history blocks differ']+=1
        la,lb=A[-1].split('\n'),B[-1].split('\n')
        if la[0]!=lb[0]: bad['target header']+=1
        if sorted(la[1:])!=sorted(lb[1:]): bad['target facts differ']+=1
        tw=a['target_words']; ix={w:i for i,w in enumerate(tw)}
        keys=[(ix[l.split(' ')[0]],ix[l.split(' ')[2]]) for l in lb[1:]]
        wk=[(l.split(' ')[0],l.split(' ')[2]) for l in lb[1:]]
        order['label-sorted' if keys==sorted(keys) else 'NOT label-sorted']+=1
        order['alpha-sorted' if wk==sorted(wk) else 'not alpha']+=1
        if la[1:]==lb[1:]: order['raw already identical order']+=1
    print('kso',k,dict(bad),dict(order))
def chk_instr(raw, ins, k):
    bad=collections.Counter(); hdrs=collections.Counter()
    for a,b in zip(raw,ins):
        if a['trial']!=b['trial']: bad['trial id']+=1; continue
        for key in a:
            if key in ('context','cond'): continue
            if a[key]!=b[key]: bad['field '+key]+=1
        if not b['context'].endswith(a['context']): bad['raw ctx not suffix']+=1; continue
        h=b['context'][:-len(a['context'])]; hdrs[h]+=1
    print('ki',k,dict(bad),'distinct headers',len(hdrs)); 
    for h in hdrs: print(repr(h))
for k in (16,22,28): chk_sorted(load(f'../exp2/stim/ks_n6_k{k}.jsonl'),load(f'../exp2/stim/kso_n6_k{k}.jsonl'),k)
for k in (10,22): chk_instr(load(f'../exp2/stim/ks_n6_k{k}.jsonl'),load(f'../exp2/stim/ki_n6_k{k}.jsonl'),k)
# overlap of exp2 with exp1 trials, and duplicates
ctxs=collections.defaultdict(list)
for f in glob.glob('../exp1/stim/*.jsonl')+glob.glob('../exp2/stim/ks_*.jsonl'):
    for x in load(f): ctxs[x['context']].append((f.replace(chr(92),'/').split('/')[-1],x['trial']))
print('dup contexts', [v for v in ctxs.values() if len(v)>1][:10])
tg=collections.defaultdict(list)
for f in glob.glob('../exp1/stim/*.jsonl')+glob.glob('../exp2/stim/ks_*.jsonl'):
    for x in load(f): tg[x['context'].split('\n\n')[-1]].append((f.replace(chr(92),'/').split('/')[-1],x['trial']))
print('dup target blocks', sum(len(v)>1 for v in tg.values()), [v for v in tg.values() if len(v)>1][:8])
for f in sorted(glob.glob('../exp*/stim/*.jsonl')):
    ids=[x['trial'] for x in load(f)]
    if len(set(ids))!=len(ids): print('dup ids',f)
