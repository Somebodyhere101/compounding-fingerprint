import json, os, sys, zlib, collections
import numpy as np, torch
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(R, 'exp3'))
import eval_fp
dev = torch.device('cuda')
# 1) gold position among candidates, word overlap across tables in non-leak files
import glob
for f in sorted(glob.glob(os.path.join(R,'exp1','stim','*.jsonl'))):
    pos = collections.Counter(); overlap = 0; maxw = 0
    for l in open(f, encoding='utf-8'):
        tr = json.loads(l); tabs = eval_fp.parse_context(tr['context'])
        ws = [set(w for fct in t for w in fct) for t in tabs]
        hist = ws[:-1] if len(ws) > 1 else []
        # overlap between any two different tables (target included as last)
        allw = [w for s in ws for w in s]
        overlap += len(allw) != len(set(allw))
        maxw = max(maxw, len(eval_fp.trial_words(tr, tabs)))
        for q in tr['queries']: pos[q['candidates'].index(q['gold'])] += 1
    tot = sum(pos.values())
    print(os.path.basename(f), 'trials w/ cross-table word overlap', overlap, 'max distinct words', maxw, 'gold-pos freq', [round(pos[i]/tot,3) for i in sorted(pos)])
# 2) ties in A4 scoring for leak, and a 'copy' sanity: does A4 put mass on Table-A answer? (same as gold in leak)
model, st = eval_fp.load_model(os.path.join(R,'exp3','runs','A4','final.pt'), dev)
for f in ['n6_leak_t1.jsonl', 'n6_base_t1.jsonl']:
    rows = [json.loads(l) for l in open(os.path.join(R,'exp1','stim',f), encoding='utf-8')][:40]
    ties1 = 0; nq = 0; corr = 0; corr_fp32 = 0
    for tr in rows:
        rng = np.random.default_rng([0, zlib.crc32(f.encode()), tr['trial'], 0])
        prefix, qs, w2s = eval_fp.encode_trial(tr, 64, rng)
        idx, pos, mask, last = eval_fp.packed_inputs(prefix, qs, dev)
        cids = torch.tensor([[w2s[w] for w in q['candidates']] for q in tr['queries']], device=dev)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg = model(idx, pos=pos, attn_mask=mask)[0, last].float()
            lg32 = model(idx, pos=pos, attn_mask=mask)[0, last].float()
        for lgx, name in [(lg, 'bf16'), (lg32, 'fp32')]:
            cl = lgx.gather(1, cids)
            mx = cl.max(1, keepdim=True).values
            if name == 'bf16': ties1 += int(((cl == mx).sum(1) > 1).sum())
            g = torch.tensor([q['candidates'].index(q['gold']) for q in tr['queries']], device=dev)
            c = int((cl.argmax(1) == g).sum())
            if name == 'bf16': corr += c
            else: corr_fp32 += c
        nq += len(tr['queries'])
    print(f, 'A4 single-draw: queries', nq, 'bf16 argmax ties', ties1, 'acc bf16', round(corr/nq,3), 'acc fp32', round(corr_fp32/nq,3))
# 3) unpacked equivalence for 3 leak trials
f = 'n6_leak_t1.jsonl'; tr = json.loads(open(os.path.join(R,'exp1','stim',f), encoding='utf-8').readline())
rng = np.random.default_rng([0, zlib.crc32(f.encode()), tr['trial'], 0])
prefix, qs, w2s = eval_fp.encode_trial(tr, 64, rng)
idx, pos, mask, last = eval_fp.packed_inputs(prefix, qs, dev)
with torch.no_grad():
    pk = model(idx, pos=pos, attn_mask=mask)[0, last].float()
    un = torch.stack([model(torch.tensor(prefix + q, device=dev)[None])[0, -1].float() for q in qs])
print('packed vs unpacked fp32 max|dlogit|', float((pk - un).abs().max()))
