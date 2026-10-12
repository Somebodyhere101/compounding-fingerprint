import json, os, sys, zlib, hashlib
import numpy as np, torch
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(R, 'exp3'))
import eval_fp
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FILES = ['n6_leak_t1.jsonl', 'n6_base_t4.jsonl', 'n8_heldout_t4.jsonl']
def load_res(p):
    d = {}
    for l in open(p, encoding='utf-8'):
        r = json.loads(l); d[(r['file'], r['trial'])] = r
    return d
pairs = [(f'{k}/final.pt', f'{k}.jsonl') for k in ['A1','A2','A4','A6','G4','I4','P4']] + [('A6/final_step13450.pt', '_A6_step13450.jsonl'), ('G4/final_step9934.pt', '_G4_step9934.jsonl'), ('A2_stalled/final.pt', None)]
for ck, res in pairs:
    st = torch.load(os.path.join(R,'exp3','runs',ck), map_location='cpu', weights_only=False)
    h = hashlib.md5(b''.join(v.cpu().numpy().tobytes() for v in st['model'].values())).hexdigest()[:10]
    line = f'{ck}: step {st["step"]} args.steps {st["args"]["steps"]} md5 {h}'
    if res:
        model, st = eval_fp.load_model(os.path.join(R,'exp3','runs',ck), dev)
        S = st['dcfg']['S']
        ref = load_res(os.path.join(R,'exp3','res',res))
        maxd = 0.0; nq = 0; agree = 0
        for f in FILES:
            rows = [json.loads(l) for l in open(os.path.join(R,'exp1','stim',f), encoding='utf-8')][:4]
            for tr in rows:
                recs, L = eval_fp.score_trial(model, tr, S, [0, zlib.crc32(f.encode()), tr['trial']], 4, dev)
                rr = ref[(f, tr['trial'])]['queries']
                for a, b in zip(recs, rr):
                    maxd = max(maxd, abs(a['p_gold'] - b['p_gold'])); nq += 1; agree += a['pred'] == b['pred']
        line += f' | rescored {nq} queries vs {res}: max|dp_gold| {maxd:.5f}, pred agree {agree}/{nq}'
    print(line, flush=True)
