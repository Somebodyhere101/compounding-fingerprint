"""Experiment 6, added after the verification pass (2026-10-11).
(1) Why 0.959 under the upstream protocol, not the authors' >= 0.99 (99.4% in their Table 1)? Upstream selects
    algebra_gpt_best.pt as the maximum over ~290 training-time evaluations on ONE cached 512-sequence set (tester.py
    get_test_batches; trainer.py). Here every released checkpoint we hold is scored on the SAME 1,024 fresh sequences.
(2) Independent forward pass: a second implementation (explicit causal softmax attention, real-valued RoPE rotation,
    erf-GELU) compared with ica.GPT on full training sequences.  -> validate_ckpts.json"""
import json, math, os, random
import numpy as np
import torch
import torch.nn.functional as F
import ica

prng = random.Random(7); seqs = []
for _ in range(1024):        # identical generator and seed to validate_unshuffled.py
    groups, total = [], 0
    while True:
        opts = ica.rosette_groups(min(10, ica.NSYM - total))
        if not opts:
            break
        g = prng.choice(opts); groups.append(g); total += ica.table(*g).shape[0]
        if prng.random() > 0.7:
            break
    elems = [(gi, e) for gi, g in enumerate(groups) for e in range(ica.table(*g).shape[0])]
    w = {el: i for i, el in enumerate(elems)}
    facts = [((gi, a), (gi, b)) for gi, g in enumerate(groups) for a in range(ica.table(*g).shape[0]) for b in range(ica.table(*g).shape[0])]
    held = facts.pop(prng.randrange(len(facts)))
    if (held[1], held[0]) in facts:
        facts.remove((held[1], held[0]))
    toks = []
    for (a, b) in [prng.choice(facts) for _ in range(199)] + [held]:
        c = (a[0], int(ica.table(*groups[a[0]])[a[1], b[1]])); toks += [ica.SEP, w[a], w[b], ica.EQ, w[c]]
    seqs.append(toks)
X = torch.tensor(seqs)
out = {'n_sequences': len(seqs), 'checkpoints': {}}
errs = {}
for name in ['algebra_gpt_best.pt', 'algebra_gpt_100000.pt', 'algebra_gpt_40000.pt', 'algebra_gpt_20000.pt', 'algebra_gpt_10000.pt']:
    m = ica.load(name); pred = []
    with torch.no_grad():
        for s in range(0, len(X), 64):
            x = X[s:s + 64].cuda(); pred.append(m(x[:, :-1])[:, -1].float().argmax(-1).cpu())
    corr = (torch.cat(pred) == X[:, -1]).numpy()
    errs[name] = set(np.where(~corr)[0].tolist())
    out['checkpoints'][name] = {'acc': float(corr.mean()), 'acc_first_512': float(corr[:512].mean()), 'acc_last_512': float(corr[512:].mean())}
    print(name, out['checkpoints'][name], flush=True)
    del m; torch.cuda.empty_cache()
b = errs['algebra_gpt_best.pt']
out['error_overlap_with_best'] = {k: f'{len(v & b)} of best {len(b)} / own {len(v)}' for k, v in errs.items() if k != 'algebra_gpt_best.pt'}

# ---- (2) independent forward pass
sd = torch.load(os.path.join(ica.CKPT_DIR, 'algebra_gpt_best.pt'), map_location='cuda', weights_only=True)


def ln(x, p):
    mu = x.mean(-1, keepdim=True); var = ((x - mu) ** 2).mean(-1, keepdim=True)
    return (x - mu) / torch.sqrt(var + 1e-5) * sd[p + '.weight'] + sd[p + '.bias']


def rope(x):                                  # x: [B, H, T, D]; rotate adjacent pairs (2i, 2i+1) by t * theta_i
    D = x.shape[-1]; T = x.shape[2]
    th = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device=x.device).float() / D))
    ang = torch.arange(T, device=x.device).float()[:, None] * th[None]
    c, s = torch.cos(ang), torch.sin(ang)
    x0, x1 = x[..., 0::2], x[..., 1::2]
    y = torch.empty_like(x); y[..., 0::2] = x0 * c - x1 * s; y[..., 1::2] = x0 * s + x1 * c
    return y


@torch.no_grad()
def fwd2(idx):
    x = sd['transformer.wte.weight'][idx]
    B, T, C = x.shape; H = 8; D = C // H
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
    for i in range(4):
        p = f'transformer.h.{i}'
        h = ln(x, p + '.ln_1') @ sd[p + '.attn.c_attn.weight'].T + sd[p + '.attn.c_attn.bias']
        q, k, v = [z.view(B, T, H, D).transpose(1, 2) for z in h.split(C, -1)]
        q, k = rope(q), rope(k)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        att = att.masked_fill(mask, float('-inf')).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        x = x + y @ sd[p + '.attn.c_proj.weight'].T + sd[p + '.attn.c_proj.bias']
        hm = ln(x, p + '.ln_2') @ sd[p + '.mlp.c_fc.weight'].T + sd[p + '.mlp.c_fc.bias']
        hm = 0.5 * hm * (1 + torch.erf(hm / math.sqrt(2)))
        x = x + hm @ sd[p + '.mlp.c_proj.weight'].T + sd[p + '.mlp.c_proj.bias']
    return ln(x, 'transformer.ln_f') @ sd['lm_head.weight'].T


m = ica.load('algebra_gpt_best.pt')
x = X[:16].cuda()[:, :-1]
with torch.no_grad():
    l1 = m(x).float(); l2 = fwd2(x).float()
eqpos = (x == ica.EQ)
a1 = l1[..., :ica.NSYM].argmax(-1)[eqpos]; a2 = l2[..., :ica.NSYM].argmax(-1)[eqpos]
out['independent_forward'] = {'max_abs_logit_diff': float((l1 - l2).abs().max()), 'answer_positions': int(eqpos.sum()),
                              'argmax_agreement': float((a1 == a2).float().mean())}
print(out['independent_forward'])
json.dump(out, open(os.path.join(ica.HERE, 'validate_ckpts.json'), 'w'), indent=1)
