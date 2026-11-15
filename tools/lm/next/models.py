"""The two prototypes of the next student (#41), compared at the same size and cost (plan Task 9-11):
a small Transformer and a GRU with one attention layer over everything before (context and run).

Only operations the engines can do in integers: RMSNorm (an integer square root), ReLU, ALiBi (a bias by distance,
no position vectors), softmax over attention through the exp table, the output layer tied to the embedding (+ bias).
Both take ids [B, T] (<s> + context + run) and give logits [B, T, V], each position reading only what is before it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

CANDS = 24  # (candidates scored per character in macs_per_step: about what a syllable has)

# (arch -> tier -> sizes). light: about the shipped XL GRU's size (int8 weights <= 12M) and cost per character
# (macs_per_step <= 1.10x); plus about 2.5x, max about 8x (plan, Global Constraints)
TIERS = {
    "tf": {"light": dict(d=384, layers=4, heads=6, ffn=1280),
           "plus": dict(d=512, layers=5, heads=8, ffn=2048),
           "max": dict(d=768, layers=8, heads=12, ffn=2560)},
    "ga": {"light": dict(emb=384, hidden=640, layers=2, heads=4),
           "plus": dict(emb=512, hidden=1024, layers=2, heads=8),
           "max": dict(emb=768, hidden=1792, layers=2, heads=8)},
}


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.g


def alibi_slopes(heads):
    """ALiBi's slope per head (Press et al.): 2^(-8 i / heads), i = 1..heads."""
    return torch.tensor([2.0 ** (-8.0 * (i + 1) / heads) for i in range(heads)])


class Attention(nn.Module):
    """Causal self-attention with ALiBi: the score of position j seen from i is q.k / sqrt(dh) - slope * (i - j)."""

    def __init__(self, d, heads):
        super().__init__()
        assert d % heads == 0
        self.heads, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.register_buffer("slopes", alibi_slopes(heads), persistent=False)

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        pos = torch.arange(T, device=x.device)
        dist = (pos[:, None] - pos[None, :]).to(x.dtype)                      # i - j
        bias = -self.slopes.to(x.dtype)[:, None, None] * dist                  # [H, T, T]
        bias = bias.masked_fill(dist < 0, float("-inf"))                       # (only what is before)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.unsqueeze(0))
        return self.out(y.transpose(1, 2).reshape(B, T, d))


class Block(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.n1, self.att, self.n2 = RMSNorm(d), Attention(d, heads), RMSNorm(d)
        self.up, self.down = nn.Linear(d, ffn), nn.Linear(ffn, d)

    def forward(self, x):
        x = x + self.att(self.n1(x))
        return x + self.down(F.relu(self.up(self.n2(x))))


class TinyTransformer(nn.Module):
    def __init__(self, vocab, d, layers, heads, ffn):
        super().__init__()
        self.cfg = dict(arch="tf", vocab=vocab, d=d, layers=layers, heads=heads, ffn=ffn)
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(d, heads, ffn) for _ in range(layers))
        self.norm = RMSNorm(d)
        self.bias = nn.Parameter(torch.zeros(vocab))

    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x)
        return self.norm(x) @ self.emb.weight.T + self.bias


class GruAttn(nn.Module):
    def __init__(self, vocab, emb, hidden, layers, heads):
        super().__init__()
        self.cfg = dict(arch="ga", vocab=vocab, emb=emb, hidden=hidden, layers=layers, heads=heads)
        self.emb = nn.Embedding(vocab, emb)
        self.rnn = nn.GRU(emb, hidden, num_layers=layers, batch_first=True)
        self.n1, self.att, self.n2 = RMSNorm(hidden), Attention(hidden, heads), RMSNorm(hidden)
        self.proj = nn.Linear(hidden, emb)
        self.bias = nn.Parameter(torch.zeros(vocab))

    def forward(self, ids):
        y, _ = self.rnn(self.emb(ids))
        y = y + self.att(self.n1(y))
        return self.proj(self.n2(y)) @ self.emb.weight.T + self.bias


def build(arch, tier, vocab):
    sizes = TIERS[arch][tier]
    return TinyTransformer(vocab, **sizes) if arch == "tf" else GruAttn(vocab, **sizes)


def from_cfg(cfg):
    c = {k: v for k, v in cfg.items() if k != "arch"}
    return TinyTransformer(**c) if cfg["arch"] == "tf" else GruAttn(**c)


def count_int8_weights(model):
    """The weights the file would keep as int8: every matrix (embedding, GRU, attention, feed-forward, projection)."""
    return sum(p.numel() for p in model.parameters() if p.dim() == 2)


def macs_per_step(model, pos):
    """Multiply-adds to read one more character at position pos (pos positions before it, itself included) and score
    CANDS candidates after it: what each key costs per new character. Works for train.CharLM (the shipped GRU) too."""
    if isinstance(model, TinyTransformer):
        c = model.cfg
        d, f = c["d"], c["ffn"]
        return c["layers"] * (4 * d * d + 2 * d * f + 2 * pos * d) + CANDS * d
    if isinstance(model, GruAttn):
        c = model.cfg
        e, h = c["emb"], c["hidden"]
        gru = sum(3 * h * ((e if l == 0 else h) + h) for l in range(c["layers"]))
        return gru + 4 * h * h + 2 * pos * h + h * e + CANDS * e
    # train.CharLM: emb E, GRU H x layers, projection H -> E
    rnn = model.rnn
    e, h, layers = rnn.input_size, rnn.hidden_size, rnn.num_layers
    gru = sum(3 * h * ((e if l == 0 else h) + h) for l in range(layers))
    return gru + h * e + CANDS * e
