import json, glob, os, sys
from tokenizers import Tokenizer
R=sys.argv[1]; H=os.path.expanduser('~/.cache/huggingface/hub')
res={'Qwen--Qwen3-1.7B-Base':['exp2/res/q17b_main.jsonl','exp2/res/q17b_cal.jsonl'],'Qwen--Qwen3-8B-Base':['exp2/res/q8b_cal.jsonl'],
     'HuggingFaceTB--SmolLM2-1.7B-Instruct':['exp1/res/smollm2-1.7b.jsonl'],'allenai--OLMo-2-1124-7B':[],'allenai--OLMo-2-0425-1B':[]}
stims={os.path.basename(f):[json.loads(l) for l in open(f,encoding='utf-8')] for f in glob.glob(f'{R}/exp1/stim/*.jsonl')+glob.glob(f'{R}/exp2/stim/*.jsonl')}
for m,rf in res.items():
    tk=Tokenizer.from_file(glob.glob(f'{H}/models--{m}/snapshots/*/tokenizer.json')[0])
    nfb=0; nq=0; multi=set()
    for fn,rows in stims.items():
        for r in rows[:8]:
            c=tk.encode(r['context']).ids
            fulls=tk.encode_batch([r['context']+q['prompt_suffix'] for q in r['queries']])
            for e in fulls:
                nq+=1; nfb+= e.ids[:len(c)]!=c
            for w in r['target_words']:
                if len(tk.encode(' '+w, add_special_tokens=False).ids)!=1: multi.add(w)
    mis=0; tot=0
    for p in rf:
        lines=[json.loads(l) for l in open(f'{R}/{p}',encoding='utf-8')][::10]
        ctxs=[next(t for t in stims[r['file']] if t['trial']==r['trial'])['context'] for r in lines]
        for r,e in zip(lines, tk.encode_batch(ctxs)):
            tot+=1; mis+= r['ctx_tokens']!=len(e.ids)
    print(m,'sampled queries',nq,'boundary fallbacks',nfb,'multi-token words',sorted(multi),'ctx_tokens mismatches',mis,'/',tot, flush=True)
