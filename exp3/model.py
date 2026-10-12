"""GPT-style decoder for Experiment 3: pre-LN, RoPE or learned absolute positions, tied embeddings.

forward(idx, targets=None, pos=None, attn_mask=None)
  idx       [B, T] token ids
  targets   [B, T] next-token labels, -100 = ignored. If given, returns (loss_sum, n_labels) where the
            loss is summed over labelled positions only (logits are computed only there).
  pos       [B, T] or [T] position ids (default arange(T)). Lets eval pack several independent queries
            after one shared prefix, each query re-using positions L..L+3.
  attn_mask [T, T] bool, True = may attend (default: causal).
Without targets returns logits [B, T, V].
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Cfg:
    def __init__(self, vocab, d_model=256, n_layers=4, n_heads=8, max_len=2400, pos='rope', mlp_mult=4,
                 dropout=0.0):
        self.vocab = vocab; self.d_model = d_model; self.n_layers = n_layers; self.n_heads = n_heads
        self.max_len = max_len; self.pos = pos; self.mlp_mult = mlp_mult; self.dropout = dropout

    def to_dict(self):
        return dict(self.__dict__)


def rope_cache(max_len, hd, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
    t = torch.arange(max_len).float()
    f = torch.outer(t, inv)  # [L, hd/2]
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):
    # x [B, H, T, hd]; cos/sin [B or 1, 1, T, hd/2]
    cos = cos.to(x.dtype); sin = sin.to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack([o1, o2], dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.ln1 = nn.LayerNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.fc = nn.Linear(c.d_model, c.mlp_mult * c.d_model, bias=False)
        self.fc2 = nn.Linear(c.mlp_mult * c.d_model, c.d_model, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, cos, sin, attn_mask):
        B, T, D = x.shape
        H = self.c.n_heads; hd = D // H
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q = q.view(B, T, H, hd).transpose(1, 2)
        k = k.view(B, T, H, hd).transpose(1, 2)
        v = v.view(B, T, H, hd).transpose(1, 2)
        if cos is not None:
            q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
        if attn_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=self.c.dropout if self.training else 0.0)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                               dropout_p=self.c.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.drop(self.proj(y))
        x = x + self.drop(self.fc2(F.gelu(self.fc(self.ln2(x)))))
        return x


class GPT(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d_model)
        if c.pos == 'learned':
            self.pemb = nn.Embedding(c.max_len, c.d_model)
        else:
            assert c.pos == 'rope'
            cos, sin = rope_cache(c.max_len, c.d_model // c.n_heads)
            self.register_buffer('rcos', cos, persistent=False)
            self.register_buffer('rsin', sin, persistent=False)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.lnf = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab, bias=False)
        self.head.weight = self.emb.weight  # tied
        self.apply(self._init)
        for b in self.blocks:  # GPT-2 style residual scaling
            nn.init.normal_(b.proj.weight, std=0.02 / math.sqrt(2 * c.n_layers))
            nn.init.normal_(b.fc2.weight, std=0.02 / math.sqrt(2 * c.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def nparams(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, pos=None, attn_mask=None):
        B, T = idx.shape
        assert T <= self.c.max_len or pos is not None
        if pos is None:
            pos = torch.arange(T, device=idx.device)
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)
        assert int(pos.max()) < self.c.max_len, 'position beyond max_len'
        x = self.emb(idx)
        cos = sin = None
        if self.c.pos == 'learned':
            x = x + self.pemb(pos)
        else:
            cos = self.rcos[pos].unsqueeze(1)  # [B|1, 1, T, hd/2]; cast to q's dtype in apply_rope
            sin = self.rsin[pos].unsqueeze(1)
        if attn_mask is not None and attn_mask.dim() == 2:
            attn_mask = attn_mask[None, None]
        for b in self.blocks:
            x = b(x, cos, sin, attn_mask)
        x = self.lnf(x)
        if targets is None:
            return self.head(x)
        sel = targets != -100
        h = x[sel]
        logits = self.head(h).float()
        loss_sum = F.cross_entropy(logits, targets[sel], reduction='sum')
        return loss_sum, int(sel.sum())
