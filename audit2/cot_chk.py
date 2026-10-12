import json,collections,sys
for fn in ['exp4/res/_v1_trunc1536.jsonl','exp4/res/qwen3-1.7b_cot_v2.jsonl']:
    rs=[json.loads(l) for l in open(fn,encoding='utf-8')]
    print(fn,len(rs))
    keys=collections.Counter((r['file'],r['trial'],r['thinking']) for r in rs)
    print(' dup keys',[k for k,v in keys.items() if v>1])
    agg=collections.defaultdict(lambda:[0,0,0,0,0])
    for r in rs:
        a=agg[(r['file'],r['thinking'])]; a[0]+=r['correct']; a[1]+=1; a[2]+=r['truncated']; a[3]+=r['explicit_answer']; a[4]+=(r['pred'] is None)
    for k,v in sorted(agg.items()): print(' ',k,'correct %d/%d trunc %d explicit %d nopred %d'%tuple(v))
    # paired
    by=collections.defaultdict(dict)
    for r in rs: by[(r['file'],r['trial'])][r['thinking']]=r
    for f in sorted({r['file'] for r in rs}):
        pr=[v for (ff,t),v in by.items() if ff==f and True in v and False in v]
        on_only=sum(1 for v in pr if v[True]['correct'] and not v[False]['correct']); off_only=sum(1 for v in pr if v[False]['correct'] and not v[True]['correct'])
        nt=[v[True] for v in pr if not v[True]['truncated']]
        print('  paired',f,len(pr),'on-only',on_only,'off-only',off_only,'non-trunc thinking correct %d/%d'%(sum(x['correct'] for x in nt),len(nt)), 'gen tokens thinking', sorted(v[True]['n_gen_tokens'] for v in pr))
    # trials used
    print(' trials', sorted({(r['file'],r['trial']) for r in rs}))
