import os, json, sys
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from run import snapshot
repo = sys.argv[1]
p = snapshot(repo); tok = AutoTokenizer.from_pretrained(p, local_files_only=True)
m = AutoModelForCausalLM.from_pretrained(p, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval()
tr = json.loads(open('stim/n8_base_t4.jsonl').readline())
ctx = tr['context']; ids = tok(ctx)['input_ids']; L = len(ids)
maxdiff = 0; agree = 0
with torch.inference_mode():
    cache = m(input_ids=torch.tensor([ids], device='cuda'), use_cache=True).past_key_values
    for q in tr['queries'][:6]:
        full = tok(ctx + q['prompt_suffix'])['input_ids']; assert full[:L] == ids
        a = m(input_ids=torch.tensor([full[L:]], device='cuda'), past_key_values=cache, use_cache=True).logits[0,-1].float(); cache.crop(L)
        b = m(input_ids=torch.tensor([full], device='cuda')).logits[0,-1].float()
        cid = [tok.encode(' '+w, add_special_tokens=False)[0] for w in q['candidates']]
        maxdiff = max(maxdiff, float((torch.log_softmax(a,-1)[cid]-torch.log_softmax(b,-1)[cid]).abs().max()))
        agree += int(int(a[cid].argmax()) == int(b[cid].argmax()))
print(repo, 'ctx tokens', L, 'max |dlogp| cache vs full', round(maxdiff,4), 'argmax agree', agree, '/6')
