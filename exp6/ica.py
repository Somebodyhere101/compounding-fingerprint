"""Experiment 6: minimal re-implementation of the released In-Context Algebra model (Todd et al., ICLR 2026) and of its
training-data generator, written from reading the upstream code (github.com/ericwtodd/algebra at commit 532a6a2; MIT License, Copyright (c) 2025 Eric Todd)
without executing it. The RoPE helpers and parameter names follow its src/models/model.py.

Model: nanoGPT with RoPE (src/models/model.py): 4 layers, 8 heads, d=1024, MLP 4096, bias=True, GELU, tied embeddings,
block 1024, vocab 32 (task uses ids 0..18). Weights are loaded with torch.load(weights_only=True).
Tokens (src/tasks/mix_group_addition.py): symbols '0'..'9','a'..'f' = ids 0..15, '=' 16, '^' 17 (start/pad, unused when
leftpad=False), ',' 18. A fact is ',ab=c' (5 tokens); sequences start with ','.
"""
import math, os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, 'dl', 'weights')
SYMS = list('0123456789abcdef')
EQ, START, SEP = 16, 17, 18
NSYM = 16


# ------------------------------------------------------------------ model (same parameter names as upstream)
def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    return torch.polar(torch.ones(end, dim // 2), torch.outer(t, freqs).float())


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    fc = freqs_cis[:xq_.shape[2], :xq_.shape[-1]].unsqueeze(0).unsqueeze(0)
    return torch.view_as_real(xq_ * fc).flatten(3).type_as(xq), torch.view_as_real(xk_ * fc).flatten(3).type_as(xk)


class LayerNorm(nn.Module):
    def __init__(self, d):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d)); self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class Attn(nn.Module):
    def __init__(self, d, h, block):
        super().__init__()
        self.c_attn = nn.Linear(d, 3 * d); self.c_proj = nn.Linear(d, d); self.n_head = h; self.n_embd = d
        self.register_buffer('bias', torch.tril(torch.ones(block, block)).view(1, 1, block, block))
        self.freqs_cis = precompute_freqs_cis(d // h, block)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = [z.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for z in (q, k, v)]
        q, k = apply_rotary_emb(q, k, self.freqs_cis.to(q.device))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__(); self.c_fc = nn.Linear(d, 4 * d); self.gelu = nn.GELU(); self.c_proj = nn.Linear(4 * d, d)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, d, h, block):
        super().__init__(); self.ln_1 = LayerNorm(d); self.attn = Attn(d, h, block); self.ln_2 = LayerNorm(d); self.mlp = MLP(d)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, vocab=32, d=1024, L=4, h=8, block=1024):
        super().__init__()
        self.block = block
        self.transformer = nn.ModuleDict(dict(wte=nn.Embedding(vocab, d), wpe=nn.Embedding(block, d),
                                              h=nn.ModuleList([Block(d, h, block) for _ in range(L)]), ln_f=LayerNorm(d)))
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx):
        x = self.transformer.wte(idx)          # RoPE: no learned position embedding added
        for b in self.transformer.h:
            x = b(x)
        return self.lm_head(self.transformer.ln_f(x))


def load(name='algebra_gpt_best.pt', device='cuda'):
    m = GPT()
    sd = torch.load(os.path.join(CKPT_DIR, name), map_location='cpu', weights_only=True)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.endswith('attn.bias') for k in missing), (missing, unexpected)
    return m.to(device).eval()


# ------------------------------------------------------------------ group tables (own construction)
def cyclic(m):
    return np.array([[(a + b) % m for b in range(m)] for a in range(m)])


def dihedral(m):
    """D_m of order 2m; element (i, j) = r^i s^j, index i + m*j; (i1,j1)(i2,j2) = (i1 + (-1)^j1 i2, j1 xor j2)."""
    el = [(i, j) for j in (0, 1) for i in range(m)]
    idx = {e: k for k, e in enumerate(el)}
    return np.array([[idx[((a[0] + (-1) ** a[1] * b[0]) % m, a[1] ^ b[1])] for b in el] for a in el])


def rosette_groups(max_order):
    """upstream MixRosetteGroupAddition._sample_group: uniform over C3..C_max and D2..D_{max//2}"""
    return [('C', m) for m in range(3, max_order + 1)] + [('D', m) for m in range(2, max_order // 2 + 1)]


def table(kind, m):
    return cyclic(m) if kind == 'C' else dihedral(m)


def sample_run(prng, k_shots=200, mix=0.7, max_order=10, holdout_final=True):
    """re-implementation of sample_run(k_shots, hold_out=1, commute_out=True): groups, a random vocabulary that never
    maps an identity to '0', facts with replacement, ending with a held-out fact (and its commutative partner held out)."""
    groups, total = [], 0
    while True:
        opts = rosette_groups(min(max_order, NSYM - total))
        if not opts:
            break
        g = prng.choice(opts); groups.append(g); total += table(*g).shape[0]
        if prng.random() > mix:
            break
    elems = [(gi, e) for gi, g in enumerate(groups) for e in range(table(*g).shape[0])]
    while True:
        vocab = prng.sample(range(NSYM), len(elems))
        w = {el: vocab[i] for i, el in enumerate(elems)}
        if not any(w[(gi, 0)] == 0 for gi in range(len(groups))):   # element 0 is the identity in both constructions
            break
    facts = [((gi, a), (gi, b)) for gi, g in enumerate(groups) for a in range(table(*g).shape[0]) for b in range(table(*g).shape[0])]
    held = None
    if holdout_final:
        held = facts.pop(prng.randrange(len(facts)))
        if (held[1], held[0]) in facts:
            facts.remove((held[1], held[0]))
    toks = []
    for _ in range(k_shots - (1 if held else 0)):
        a, b = prng.choice(facts); c = (a[0], int(table(*groups[a[0]])[a[1], b[1]]))
        toks += [SEP, w[a], w[b], EQ, w[c]]
    if held:
        a, b = held; c = (a[0], int(table(*groups[a[0]])[a[1], b[1]]))
        toks += [SEP, w[a], w[b], EQ, w[c]]
    return toks, groups, held
