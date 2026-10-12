import json, glob, os, sys, collections
from tokenizers import Tokenizer
R=sys.argv[1]; H=os.path.expanduser('~/.cache/huggingface/hub')
res={'Qwen--Qwen3-1.7B-Base':['exp2/res/q17b_main.jsonl','exp2/res/q17b_cal.jsonl'],'Qwen--Qwen3-8B-Base':['exp2/res/q8b_cal.jsonl'],
     'HuggingFaceTB--SmolLM2-1.7B-Instruct':['exp1/res/smollm2-1.7b.jsonl'],'allenai--OLMo-2-1124-7B':[],'allenai--OLMo-2-0425-1B':[]}
stims=glob.glob(f'{R}/exp1/stim/*.jsonl')+glob.glob(f'{R}/exp2/stim/*.jsonl')
for m,rf in res.items():
    tk=Tokenizer.from_file(glob.glob(f'{H}/models--{m}/snapshots/*/tokenizer.json')[0])
    enc=lambda s: tk.encode(s, add_special_tokens=True).ids
    ctxlen={}; nfb=0; nq=0; multi=set()
    for f in stims:
        fn=os.path.basename(f)
        for l in open(f,encoding='utf-8'):
            r=json.loads(l); c=enc(r['context']); ctxlen[(fn,r['trial'])]=len(c)
            for q in r['queries']:
                nq+=1
                full=enc(r['context']+q['prompt_suffix'])
                if full[:len(c)]!=c: nfb+=1
            for w in r['target_words']:
                if len(tk.encode(' '+w, add_special_tokens=False).ids)!=1: multi.add(w)
    mis=0; tot=0
    for p in rf:
        for l in open(f'{R}/{p}',encoding='utf-8'):
            r=json.loads(l); tot+=1; mis+= r['ctx_tokens']!=ctxlen[(r['file'],r['trial'])]
    print(m,'queries',nq,'boundary fallbacks',nfb,'multi-token words',sorted(multi),'ctx_tokens mismatches',mis,'/',tot)
