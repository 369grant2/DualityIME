"""Pre-train a prototype of the next student (#41) as a character language model, before distillation.

    py -3 tools/lm/next/pretrain.py <tf|ga> <light|plus|max> [--minutes N] [--seed S] [--quick]

Reads common.WORK/next/train.u16, dev.u16, vocab.txt (prepare_next.py). Each sample is <s> + SEQ units from anywhere
in the text (what the engines read: <s>, then a context that starts mid-text, then the run); a <s> inside it is a
paragraph's start. Predicting <s> costs nothing (ignored). bf16, AdamW, warm-up then cosine over the time budget,
gpu.tidy() every step. Keeps the checkpoint with the best development loss:
common.WORK/next/models/<arch>_<tier>_s<S>.pt (+ .json: the model's sizes, loss, characters seen).
Run it through tools/lm/gpu_run.py (CLAUDE.md: long GPU jobs).
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.stdout.reconfigure(encoding="utf-8")

from common import WORK  # noqa: E402
from gpu import tidy  # noqa: E402
import models  # noqa: E402

SEQ = 128
BATCH = 128
BOS = 1
WARMUP = 1000


def batch(data, rng, device):
    idx = rng.integers(0, len(data) - SEQ - 1, BATCH)
    x = np.stack([np.concatenate(([BOS], data[i:i + SEQ])) for i in idx]).astype(np.int64)
    x = torch.from_numpy(x).to(device, non_blocking=True)
    return x[:, :-1], x[:, 1:]


def loss_of(logits, y):
    return nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=BOS)


def main(argv):
    arch, tier = argv[0], argv[1]
    minutes = float(argv[argv.index("--minutes") + 1]) if "--minutes" in argv else 360.0
    seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 1
    quick = "--quick" in argv
    work = os.path.join(WORK, "next")
    vocab = open(os.path.join(work, "vocab.txt"), encoding="utf-8").read().rstrip("\n").split("\n")
    train = np.memmap(os.path.join(work, "train.u16"), dtype=np.uint16, mode="r")
    dev = np.memmap(os.path.join(work, "dev.u16"), dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    dev_batches = [batch(dev, np.random.default_rng(100 + k), "cuda") for k in range(20)]
    torch.manual_seed(seed)
    model = models.build(arch, tier, len(vocab)).cuda()
    params = sum(p.numel() for p in model.parameters())
    print(f"{arch} {tier}: {model.cfg}, {params / 1e6:.2f}M parameters ({models.count_int8_weights(model) / 1e6:.2f}M "
          f"int8), {models.macs_per_step(model, 72) / 1e6:.2f}M MACs per character; train {len(train)} ids", flush=True)
    peak = 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=peak, weight_decay=0.01, betas=(0.9, 0.95))
    budget = minutes * 60
    t0 = time.time()
    step, best, seen, skipped = 0, None, 0, 0
    os.makedirs(os.path.join(work, "models"), exist_ok=True)
    out = os.path.join(work, "models", f"{arch}_{tier}_s{seed}" + (".quick" if quick else ""))
    while True:
        elapsed = time.time() - t0
        if elapsed > budget:
            break
        lr = peak * min(1.0, (step + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(1.0, elapsed / budget)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = batch(train, rng, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_of(model(x).float(), y)
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):  # (a step gone wrong is skipped, not learned from)
            skipped += 1
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tidy()  # (tools/lm/gpu.py: without it the GPU spills into system memory and the job crawls)
        step += 1
        seen += x.numel()
        if step % 500 == 0 or time.time() - t0 > budget:
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                dl = sum(loss_of(model(dx).float(), dy).item() for dx, dy in dev_batches) / len(dev_batches)
            model.train()
            print(f"step {step} {elapsed / 60:.1f} min: train {loss.item():.3f} dev {dl:.3f} "
                  f"(perplexity {math.exp(dl):.1f}), {seen / 1e9:.2f}G characters, {skipped} skipped", flush=True)
            if best is None or dl < best:
                best = dl
                torch.save(model.state_dict(), out + ".pt")
                with open(out + ".json", "w", encoding="utf-8") as f:
                    json.dump(dict(model.cfg, tier=tier, seed=seed, dev_loss=dl, step=step, characters=seen), f)
    print(f"best dev loss {best:.3f} (perplexity {math.exp(best):.1f}) -> {out}.pt", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
