"""Teach Qwen3-1.7B Taiwan Traditional Chinese: LoRA (hand-written, no peft) on the GRU's training text.

    py ft_teacher.py [--steps 3000]

Text: random runs of Chinese from <WORK>\\train.u16 (tools/lm/prepare.py: the training
files only; dev.u16 and the evaluation files are not read here except dev for the loss check).
Each run is scored by the teacher as <|endoftext|> + run, so training uses the same shape: blocks of
runs, each preceded by <|endoftext|>.
Writes the merged model to <WORK>\\teacher_rerank\\qwen3-1.7b-zhtw (WORK: tools/lm/common.py).
"""
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tools/lm
from common import WORK  # noqa: E402  (DUALITYIME_LM_WORK moves it)
OUT = os.path.join(WORK, "teacher_rerank", "qwen3-1.7b-zhtw")
BASE = "Qwen/Qwen3-1.7B"
RANK, ALPHA = 16, 32
BLOCK, BATCH, LR = 256, 16, 2e-4
STEPS = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 3000


class LoRA(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.randn(RANK, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, RANK, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + (x.float() @ self.a.t() @ self.b.t() * (ALPHA / RANK)).to(x.dtype)

    def merged(self):
        w = self.base.weight.float() + (self.b @ self.a) * (ALPHA / RANK)
        return w.to(self.base.weight.dtype)


def runs(path, vocab, rng, want_chars):
    """Random runs (texts) from a .u16 file until want_chars characters."""
    data = np.memmap(path, dtype=np.uint16, mode="r")
    starts = np.flatnonzero(data == 1)
    out, total = [], 0
    for s in rng.permutation(len(starts)):
        a = starts[s] + 1
        b = starts[s + 1] if s + 1 < len(starts) else len(data)
        ids = data[a:b]
        if len(ids) < 2 or len(ids) > 200 or (ids == 0).any():
            continue
        out.append("".join(vocab[i] for i in ids))
        total += len(ids)
        if total >= want_chars:
            break
    return out


def blocks(texts, tok, bos):
    """Pack <bos> + run token lists into BLOCK-token rows."""
    rows, cur = [], []
    for t in texts:
        ids = [bos] + tok(t, add_special_tokens=False)["input_ids"]
        if len(cur) + len(ids) > BLOCK:
            rows.append(cur + [bos] * (BLOCK - len(cur)))
            cur = []
        cur += ids[:BLOCK]
    return torch.tensor(rows, dtype=torch.long)


def loss_of(model, x, bos):
    logits = model(input_ids=x).logits[:, :-1].float()
    target = x[:, 1:]
    mask = (target != bos).float()  # (no loss on predicting the separator / padding)
    lp = torch.log_softmax(logits, -1).gather(2, target[..., None])[..., 0]
    return -(lp * mask).sum() / mask.sum()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    vocab = open(os.path.join(WORK, "vocab.txt"), encoding="utf-8").read().rstrip("\n").split("\n")
    tok = AutoTokenizer.from_pretrained(BASE)
    bos = tok.convert_tokens_to_ids("<|endoftext|>")
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    adapters = []
    for layer in model.model.layers:
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                              (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for n in names:
                lora = LoRA(getattr(parent, n))
                setattr(parent, n, lora)
                adapters.append(lora)
    params = [p for a in adapters for p in (a.a, a.b)]
    print("trainable", sum(p.numel() for p in params) / 1e6, "M", flush=True)
    model.enable_input_require_grads()

    train = blocks(runs(os.path.join(WORK, "train.u16"), vocab, rng, STEPS * BATCH * BLOCK * 1.1), tok, bos)
    dev = blocks(runs(os.path.join(WORK, "dev.u16"), vocab, np.random.default_rng(1), 200_000), tok, bos)[:256]
    print("train rows", len(train), "dev rows", len(dev), flush=True)

    def dev_loss():
        model.eval()
        with torch.no_grad():
            ls = [loss_of(model, dev[i:i + BATCH].cuda(), bos).item() for i in range(0, len(dev), BATCH)]
        model.train()
        return sum(ls) / len(ls)

    print("dev loss before", round(dev_loss(), 4), flush=True)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100) * max(0.1, 1 - s / STEPS))
    model.train()
    t0 = time.time()
    for step in range(STEPS):
        x = train[(step * BATCH) % len(train):(step * BATCH) % len(train) + BATCH].cuda()
        loss = loss_of(model, x, bos)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 200 == 0 or step == STEPS - 1:
            print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if step and step % 1000 == 0:
            print("dev loss", round(dev_loss(), 4), flush=True)
    print("dev loss after", round(dev_loss(), 4), flush=True)
    # merge and save
    with torch.no_grad():
        for layer in model.model.layers:
            for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                                  (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
                for n in names:
                    lora = getattr(parent, n)
                    lora.base.weight.copy_(lora.merged())
                    setattr(parent, n, lora.base)
    model.config.use_cache = True
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
