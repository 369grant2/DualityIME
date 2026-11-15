"""Teach Qwen3-1.7B Taiwan Traditional Chinese: LoRA (hand-written, no peft) on the GRU's training text.

    py ft_teacher.py [--steps 3000] [--base Qwen/Qwen3-4B-Base --out qwen3-4b-zhtw] [--batch 16] [--4bit] [--log 200]
                     [--text FILE]

--base: another model to teach (#37: a stronger teacher); --out: its folder name under <WORK>\\teacher_rerank;
--batch: rows per step (a bigger model may need fewer to fit in the GPU's memory; then also more steps).
--4bit: the model loaded in 4 bits (bitsandbytes NF4) and only the LoRA trained (QLoRA: a 27B in 24 GB); the LoRA is
then saved alone (<out>/lora.pt, lora.py), not merged. --log: steps between progress lines.
--text: half the training runs from this file instead (one document a line, e.g. the Taiwan web text of #37's
docs/superpowers/items/37/fineweb_trad.py), cut at 。！？ into runs of up to 200 characters.

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
from gpu import tidy  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lora  # noqa: E402
def _arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


OUT = os.path.join(WORK, "teacher_rerank", _arg("--out", "qwen3-1.7b-zhtw"))
BASE = _arg("--base", "Qwen/Qwen3-1.7B")
BLOCK, LR = 256, 2e-4
FOUR_BIT = "--4bit" in sys.argv
TEXT = _arg("--text", None)
LOG = int(_arg("--log", 200))
BATCH = int(_arg("--batch", 16))
STEPS = int(_arg("--steps", 3000))


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


def text_runs(path, rng, want_chars):
    """Runs of up to 200 characters from a file of one document a line, cut after 。！？, in a random order of
    documents, until want_chars characters."""
    with open(path, encoding="utf-8") as f:
        docs = f.read().split("\n")
    out, total = [], 0
    for d in rng.permutation(len(docs)):
        cur = ""
        for ch in docs[d]:
            cur += ch
            if (ch in "。！？" and len(cur) >= 20) or len(cur) >= 200:
                out.append(cur.strip())
                total += len(cur)
                cur = ""
        if total >= want_chars:
            break
    return [r for r in out if len(r) >= 2]


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
    if FOUR_BIT:
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=q, device_map="cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    adapters = lora.attach(model)
    params = [p for a in adapters.values() for p in (a.a, a.b)]
    print("trainable", sum(p.numel() for p in params) / 1e6, "M", flush=True)
    model.enable_input_require_grads()

    want = STEPS * BATCH * BLOCK * 1.1
    if TEXT:  # (half of it from --text, shuffled together)
        mixed = runs(os.path.join(WORK, "train.u16"), vocab, rng, want / 2) + text_runs(TEXT, rng, want / 2)
        train = blocks([mixed[i] for i in rng.permutation(len(mixed))], tok, bos)
    else:
        train = blocks(runs(os.path.join(WORK, "train.u16"), vocab, rng, want), tok, bos)
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
        tidy()  # (tools/lm/gpu.py: without it the GPU spills into system memory and the job crawls)
        if step % LOG == 0 or step == STEPS - 1:
            print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if step and step % 1000 == 0:
            print("dev loss", round(dev_loss(), 4), flush=True)
    print("dev loss after", round(dev_loss(), 4), flush=True)
    if FOUR_BIT:  # (4-bit weights cannot take the LoRA merged in: saved alone, with the base it belongs to)
        os.makedirs(OUT, exist_ok=True)
        lora.save(adapters, os.path.join(OUT, "lora.pt"))
        with open(os.path.join(OUT, "base.txt"), "w", encoding="utf-8") as f:
            f.write(BASE + "\n")
    else:  # merge and save
        lora.detach_merged(model)
        model.config.use_cache = True
        model.save_pretrained(OUT)
        tok.save_pretrained(OUT)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
