"""LoRA (hand-written, no peft) for the teachers of ft_teacher.py and teacher_rerank.py / teacher_train.py (#37).

attach(model) wraps the projections of every layer (attention, the linear attention of Qwen3.5-style layers, the MLP)
in a LoRA; a model loaded in 4 bits (bitsandbytes) keeps its weights and trains the LoRA only (QLoRA), and since 4-bit
weights cannot take the LoRA merged in, save(path) writes the LoRA alone and load(model, path) puts it back.
"""
import math

import torch
import torch.nn as nn

RANK, ALPHA = 16, 32
PARTS = {"self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
         "linear_attn": ("in_proj_qkv", "in_proj_z", "out_proj"),
         "mlp": ("gate_proj", "up_proj", "down_proj")}


class LoRA(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        dev = next(base.parameters()).device
        self.a = nn.Parameter(torch.randn(RANK, base.in_features, device=dev) / math.sqrt(base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, RANK, device=dev))

    def forward(self, x):
        return self.base(x) + (x.float() @ self.a.t() @ self.b.t() * (ALPHA / RANK)).to(x.dtype)

    def merged(self):
        w = self.base.weight.float() + (self.b @ self.a) * (ALPHA / RANK)
        return w.to(self.base.weight.dtype)


def layers(model):
    inner = getattr(model, "model", model)
    return getattr(inner, "layers", None) or inner.language_model.layers


def attach(model):
    """{name: LoRA} of every projection wrapped."""
    out = {}
    for i, layer in enumerate(layers(model)):
        for part, names in PARTS.items():
            parent = getattr(layer, part, None)
            if parent is None:
                continue
            for n in names:
                base = getattr(parent, n, None)
                if base is None:
                    continue
                lora = LoRA(base)
                setattr(parent, n, lora)
                out[f"{i}.{part}.{n}"] = lora
    return out


def detach_merged(model):
    """bf16 weights: the LoRA merged into each projection, which is put back in its place (save_pretrained after)."""
    with torch.no_grad():
        for layer in layers(model):
            for part, names in PARTS.items():
                parent = getattr(layer, part, None)
                for n in names if parent is not None else ():
                    lora = getattr(parent, n, None)
                    if isinstance(lora, LoRA):
                        lora.base.weight.copy_(lora.merged())
                        setattr(parent, n, lora.base)


def save(adapters, path):
    torch.save({k: (v.a.detach().cpu(), v.b.detach().cpu()) for k, v in adapters.items()}, path)


def load(model, path):
    """The LoRA of save(path) attached to model (loaded as it was trained: the same base, 4 bits or not)."""
    saved = torch.load(path, map_location="cpu")
    adapters = attach(model)
    with torch.no_grad():
        for k, (a, b) in saved.items():
            adapters[k].a.copy_(a)
            adapters[k].b.copy_(b)
    return adapters
