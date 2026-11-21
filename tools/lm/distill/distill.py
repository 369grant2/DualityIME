"""Distil the teacher's reranking into a character GRU (a size of tools/lm/train.py's SIZES), same scoring as the engine.

    py distill.py <size> <name> --init <pt> [--data train_nbest_500k.jsonl] [--teacher zhtw17_500k]
                   [--epochs 2] [--lr 3e-4] [--tlam 1.0] [--kdw 0] [--groups 32] [--limit N] [--goldw 0] [--seed 0]

How data/lm.bin of 3.2.2 was made (decisions.md 2026-11-09; the scripts of this folder, run from it, GPU
steps on a GPU with 24 GB; WORK is tools/lm/common.WORK: train.u16, dev.u16, vocab.txt from tools/lm/prepare.py,
models/XL.pt from tools/lm/train.py; outputs go to WORK/teacher_rerank/ and WORK/teacher_rerank/distill/,
never into this folder):
  0. py tools/lm/quantize.py XL               models/XL.npz: teacher_rerank's workers load it (prep, step 2)
     py teacher_rerank.py prep                the held-out sets WORK/teacher_rerank/{dev,wiki_6-20,...}.jsonl
                                              that step 4 evaluates on each epoch
  1. py ft_teacher.py                         the teacher: Qwen3-1.7B with LoRA on train.u16, 3000 steps
                                              -> WORK/teacher_rerank/qwen3-1.7b-zhtw (a second round on
                                              top of it did no better and was not used)
  2. py gen_train.py --count 500000 --out train_nbest_500k.jsonl
                                              n-best of 500k training runs with the engine's scores (CPU)
  3. py teacher_train.py WORK/teacher_rerank/qwen3-1.7b-zhtw zhtw17_500k --data train_nbest_500k.jsonl
                                              the teacher's log10 P of every candidate (--reuse an earlier
                                              teacher_<name>.json to skip texts already scored)
  4. py distill.py XL xl_kd5 --init WORK/models/XL.pt --data train_nbest_500k.jsonl --teacher zhtw17_500k
                   --epochs 2 --kdw 5            the student; --kdw 5 keeps every character's probability near
                                              the starting model (KL), without it <unk> and rare characters
                                              drift (很好醟). The learnt w (0.5546) is the engines' LM_WEIGHT.
  5. copy WORK/teacher_rerank/distill/xl_kd5.pt and .json to WORK/models, then py tools/lm/quantize.py xl_kd5 and
     py tools/lm/export.py WORK/models/xl_kd5.npz data/lm.bin
                                              (2026-11-14: from the kept xl_kd5.pt this gives data/lm.bin
                                              byte for byte)
  6. measure in the real engine: tools/user_sweep.py (its docstring: the 3,822 held-out sentences and the
     common words, the commit before against the one after), and redo tools/lm/order_thresholds.py.
teacher_rerank.py: also the first study (n-best + teacher scores on the held-out sets, report); eval_int8.py:
a student's accuracy in the engine's integer arithmetic. The n-best the students learn from (step 2, through
teacher_rerank._job) and this script's candidate lists come from tools/lm/decode.py, which lacks parts of
convert() (docs/review-backlog.md): step 6 is the measure that counts.

Student score of a candidate = sum over its characters of log10 P(ch | before) among the characters the
dictionary writes for that syllable (the engine's Scorer.phrase_score); a row with "ctx" (gen_train.py --context,
#37 接續前文選字) is read after it: <s> + ctx + the candidate, scored on the candidate's characters only, and its
teacher score is the one of the text after that context (key ctx + tab + text). Combined as the engine does:
engine + w * student, w learnt (starts at the shipped 0.5).
Target: softmax over each sentence's n-best of ln10 * (engine + tlam * teacher). Loss: cross-entropy
of the student's softmax against it (listwise distillation). --goldw g: the target is (1 - g) of that and g on the
sentence as written, where it is among the candidates (#41, the user 2026-11-19: a student can learn past its teacher
where the teacher is wrong - 在／再, the teacher about 65% with the answer among the candidates 76%).
--seed s: the order of the training rows (and torch's random numbers): runs from one start that differ by it can be averaged.
Rows keep their syllables and the index of each character in its syllable's list (numpy int32); the
candidate matrices are built per batch (prebuilt, 500k rows took tens of GB).
Out: distill/<name>.pt / .json / .log (eval per epoch vs shipped, on the held-out sets).
"""
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from torch import nn

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "tools", "lm"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools", "lm", "next"))
sys.stdout.reconfigure(encoding="utf-8")

from train import SIZES, CharLM, BOS  # noqa: E402

from common import WORK  # noqa: E402  (DUALITYIME_LM_WORK moves it)
from gpu import note, tidy  # noqa: E402
OUT = os.path.join(WORK, "teacher_rerank")
DIST = os.path.join(OUT, "distill")
LN10 = math.log(10)
SETS = ["dev", "wiki_6-20", "web_6-20", "wiki_15-40", "web_15-40"]


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


NEXT = "--next" in sys.argv  # (#41: the next student's vocabulary, <en> at 2, contexts as text.to_units units)


def is_han(ch):
    return "\u3400" <= ch <= "\u9fff" or "\uf900" <= ch <= "\ufaff"


def forward(model, x):
    """The logits of either kind of model (train.CharLM gives (logits, state); the next ones logits alone)."""
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


class Data:
    def __init__(self):
        from engine.lexicon import Lexicon
        from evaluate import reading_of
        from evaluate_lm import best_readings
        from decode import candidates_by_syllable
        lex = Lexicon()
        self.best = best_readings(lex)
        self.reading_of = reading_of
        vocab_dir = os.path.join(WORK, "next") if NEXT else WORK
        self.vocab = open(os.path.join(vocab_dir, "vocab.txt"), encoding="utf-8").read().rstrip("\n").split("\n")
        self.ids = {ch: i for i, ch in enumerate(self.vocab)}
        self.cands = candidates_by_syllable(lex, self.vocab)
        self.index = {s: {int(i): k for k, i in enumerate(a)} for s, a in self.cands.items()}

    def row(self, r, teacher=None):
        rd = self.reading_of(r["text"], self.best)
        if not rd:
            return None
        n = len(rd)
        c = "" if os.environ.get("TR_NOCTX") else r.get("ctx", "")  # (TR_NOCTX: the control without contexts)
        if NEXT:  # (the context as written -> units: Chinese characters and <en>, #41)
            from text import EN, to_units
            ctx_ids = [2 if u == EN else self.ids.get(u, 0) for u in to_units(c)]
        else:  # (today's model reads the Chinese characters alone: a context as written loses the rest)
            ctx_ids = [self.ids.get(ch, 0) for ch in c if is_han(ch)]
        if os.environ.get("TR_CTXLEN"):  # (the engine's 前文長度: only the last N units, #41)
            n_ctx = int(os.environ["TR_CTXLEN"])
            ctx_ids = ctx_ids[max(0, len(ctx_ids) - n_ctx):]  # (0: none; [-0:] would keep them all)
        hyps, eng, tgt, tch, full = [], [], [], [], []
        for t, e, _ in r["hyps"]:
            eng.append(e)
            # (a row's teacher score is the one after its own context, kept for the control without contexts too)
            full_c = r.get("ctx", "")
            tch.append(teacher.get(full_c + "\t" + t if full_c else t) if teacher else None)
            full.append(t)
            if len(t) != n:  # (the engine alone, as shipped)
                hyps.append(None)
                tgt.append(None)
                continue
            ids = [self.ids.get(ch, 0) for ch in t]
            pos = [self.index[s].get(i, -1) if s in self.index else -1 for s, i in zip(rd, ids)]
            hyps.append(np.array(ids, dtype=np.int32))
            tgt.append(np.array(pos, dtype=np.int32))
        return {"text": r["text"], "base": r["base"], "syl": rd, "hyps": hyps, "full": full, "eng": eng,
                "tgt": tgt, "teacher": tch, "ctx": np.array(ctx_ids, dtype=np.int32)}


def score_hyps(model, rows, data, device):
    seqs = [(ri, hi) for ri, row in enumerate(rows) for hi, ids in enumerate(row["hyps"]) if ids is not None]
    out = [[None] * len(r["hyps"]) for r in rows]
    if not seqs:
        return out
    L = max(len(rows[ri]["hyps"][hi]) for ri, hi in seqs)
    C = max(len(rows[ri]["ctx"]) for ri, _ in seqs)  # (the longest context: <s> + ctx + candidate)
    K = max((len(data.cands[s]) for ri, _ in seqs for s in rows[ri]["syl"] if s in data.cands), default=1)
    B = len(seqs)
    x = torch.full((B, C + L), BOS, dtype=torch.long)
    at = torch.zeros((B, L), dtype=torch.long)  # (the position whose output scores each character: after its context)
    cand = torch.full((B, L, K), -1, dtype=torch.long)
    tgt = torch.full((B, L), -1, dtype=torch.long)
    mats = {}
    for b, (ri, hi) in enumerate(seqs):
        row = rows[ri]
        ids = row["hyps"][hi]
        n, c = len(ids), len(row["ctx"])
        if c:
            x[b, 1:1 + c] = torch.from_numpy(row["ctx"].astype(np.int64))
        x[b, 1 + c:c + n] = torch.from_numpy(ids[:-1].astype(np.int64))
        at[b] = torch.arange(L) + c
        if ri not in mats:
            m = torch.full((n, K), -1, dtype=torch.long)
            for k, s in enumerate(row["syl"]):
                c = data.cands.get(s)
                if c is not None:
                    m[k, :len(c)] = torch.from_numpy(c)
            mats[ri] = m
        cand[b, :n] = mats[ri]
        tgt[b, :n] = torch.from_numpy(row["tgt"][hi].astype(np.int64))
    x, at, cand, tgt = x.to(device), at.to(device), cand.to(device), tgt.to(device)
    logits = forward(model, x)                               # [B, C + L, V]
    logits = logits.gather(1, at.clamp(max=C + L - 1)[..., None].expand(-1, -1, logits.size(-1)))  # [B, L, V]
    valid = cand >= 0
    g = logits.float().gather(2, cand.clamp(min=0))          # [B, L, K]
    g = g.masked_fill(~valid, -1e9)
    logp = g - torch.logsumexp(g, -1, keepdim=True)
    picked = logp.gather(2, tgt.clamp(min=0)[..., None])[..., 0]
    picked = torch.where(tgt >= 0, picked, torch.zeros_like(picked))
    s = picked.sum(1) / LN10
    for b, (ri, hi) in enumerate(seqs):
        out[ri][hi] = s[b]
    return out


def evaluate(model, w, ev, data, device, label, save=None):
    """Accuracy on each set; save: a path for every row's result ({set: [right, ...]}, to compare two models)."""
    model.eval()
    lines = []
    results = {}
    with torch.no_grad():
        for name, rows in ev.items():  # (teacher_rerank.SETS, and TR_EXTRA's: the sets with contexts)
            right = ship = w2r = r2w = 0
            for i in range(0, len(rows), 32):
                tidy()  # (the sets' shapes change too: contexts of every length)
                chunk = rows[i:i + 32]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    sc = score_hyps(model, chunk, data, device)
                for row, s in zip(chunk, sc):
                    best, bs = None, None
                    for t, e, v in zip(row["full"], row["eng"], s):
                        val = e + (w * float(v) if v is not None else 0.0)
                        if bs is None or val > bs:
                            best, bs = t, val
                    got = best if best is not None else row["base"]
                    ok, ok_s = got == row["text"], row["shipped"] == row["text"]
                    results.setdefault(name, []).append(bool(ok))
                    right += ok
                    ship += ok_s
                    w2r += ok and not ok_s
                    r2w += ok_s and not ok
            lines.append(f"  {name}: {right / len(rows):.1%} (shipped {ship / len(rows):.1%})  "
                         f"wrong→right {w2r}  right→wrong {r2w}")
    model.train()
    if save:
        with open(save, "w", encoding="utf-8") as f:
            json.dump(results, f)
    msg = f"[{label}] w={w:.3f}\n" + "\n".join(lines)
    print(msg, flush=True)
    return msg


def main():
    size, name = sys.argv[1], sys.argv[2]
    init = arg("--init", None)
    dfile = arg("--data", "train_nbest_500k.jsonl")
    tname = arg("--teacher", "zhtw17_500k")
    epochs = arg("--epochs", 2, int)
    lr = arg("--lr", 3e-4, float)
    tlam = arg("--tlam", 1.0, float)
    groups = arg("--groups", 32, int)
    limit = arg("--limit", None, int)
    hard = "--hard" in sys.argv
    goldw = arg("--goldw", 0.0, float)
    seed = arg("--seed", 0, int)
    lmw = arg("--lmw", 0.0, float)
    kdw = arg("--kdw", 0.0, float)
    unkw = arg("--unkw", 0.0, float)
    arch = arg("--arch", "gru")       # (#41: gru = train.CharLM of SIZES[size]; tf / ga = tools/lm/next/models.py)
    tier = arg("--tier", "light")
    eval_only = "--eval-only" in sys.argv  # (the start's evaluation alone, at --w: a model's numbers, nothing trained)
    w0 = arg("--w", 0.5, float)
    try:  # (overrides for a run already queued: distill/overrides.json {"distill": {"x50_d500k": {"epochs": 1}}})
        with open(os.path.join(DIST, "overrides.json"), encoding="utf-8") as f:
            epochs = int(json.load(f).get("distill", {}).get(name, {}).get("epochs", epochs))
    except (OSError, ValueError):
        pass
    print(f"{name}: {epochs} epochs", flush=True)
    device = "cuda"
    torch.manual_seed(seed)
    random.seed(seed)
    data = Data()
    t0 = time.time()
    teacher = {} if eval_only else json.load(open(os.path.join(DIST, f"teacher_{tname}.json"), encoding="utf-8"))
    train = []
    with open(os.path.join(DIST, dfile), encoding="utf-8") as f:
        for k, line in enumerate([] if eval_only else f, 1):
            row = data.row(json.loads(line), teacher)
            if row and sum(h is not None for h in row["hyps"]) >= 2:
                train.append(row)
            if limit and len(train) >= limit:
                break
            if k % 200000 == 0:  # (gpu_run.py watches the log grow: millions of rows take minutes)
                print(f"  read {k} rows, {len(train)} kept", flush=True)
    del teacher
    import teacher_rerank as tr
    ev = {}
    for s in tr.SETS:
        rows = []
        for r in tr.load_set(s):
            row = data.row(r)
            if row:
                row["shipped"] = tr.shipped(r)
                rows.append(row)
        ev[tr.set_name(s)] = rows
    print("train groups", len(train), "loaded in", round(time.time() - t0), "s", flush=True)

    if arch == "gru":
        emb, hidden, layers = SIZES[size]

        def make():
            return CharLM(len(data.vocab), emb, hidden, layers).to(device)
    else:
        import models

        def make():
            return models.build(arch, tier, len(data.vocab)).to(device)
    model = make()
    if init:
        model.load_state_dict(torch.load(init, map_location=device))
    ref = None
    if kdw > 0 or unkw > 0:
        ref = make()
        ref.load_state_dict(torch.load(init, map_location=device))
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    wlog = nn.Parameter(torch.tensor(math.log(w0), device=device))
    opt = torch.optim.AdamW(list(model.parameters()) + [wlog], lr=lr, weight_decay=0.0)
    steps = max(1, epochs * math.ceil(len(train) / groups))  # (1 with --eval-only: no rows read)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / 200) * max(0.05, 1 - k / steps))
    log = open(os.path.join(DIST, f"{name}.log"), "w", encoding="utf-8")
    log.write(evaluate(model, math.exp(wlog.item()), ev, data, device, "start",
                       os.path.join(DIST, f"{name}_eval_start.json")) + "\n")
    log.flush()
    if eval_only:
        return
    step, t0 = 0, time.time()
    for ep in range(epochs):
        random.shuffle(train)
        for i in range(0, len(train), groups):
            chunk = train[i:i + groups]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sc = score_hyps(model, chunk, data, device)
            w = wlog.exp()
            loss, used = 0.0, 0
            for row, s in zip(chunk, sc):
                keep = [k for k, v in enumerate(s) if v is not None and row["teacher"][k] is not None]
                if len(keep) < 2:
                    continue
                eng = torch.tensor([row["eng"][k] for k in keep], device=device)
                stu = torch.stack([s[k] for k in keep]).float()
                if hard:  # (the control: the sentence as written is the answer, no teacher)
                    right = [j for j, k in enumerate(keep) if row["full"][k] == row["text"]]
                    if not right:
                        continue
                    target = torch.zeros(len(keep), device=device)
                    target[right[0]] = 1.0
                else:
                    tch = torch.tensor([row["teacher"][k] for k in keep], device=device)
                    target = torch.softmax(LN10 * (eng + tlam * tch), 0)
                    right = [j for j, k in enumerate(keep) if row["full"][k] == row["text"]] if goldw > 0 else []
                    if right:  # (--goldw: part of the target on the sentence as written)
                        target = (1 - goldw) * target
                        target[right[0]] += goldw
                logq = torch.log_softmax(LN10 * (eng + w * stu), 0)
                loss = loss - (target * logq).sum()
                used += 1
            if not used:
                continue
            loss = loss / used
            if lmw > 0:  # (the sentence as written, next character over the whole vocabulary: keeps every
                # character's score tied to text, <unk> too, which the ranking alone leaves free)
                texts = [np.array([data.ids.get(ch, 0) for ch in row["text"]], dtype=np.int64) for row in chunk]
                T = max(len(t) for t in texts)
                xt = torch.full((len(texts), T), BOS, dtype=torch.long)
                yt = torch.full((len(texts), T), -100, dtype=torch.long)
                for b, t in enumerate(texts):
                    xt[b, 1:len(t)] = torch.from_numpy(t[:-1])
                    yt[b, :len(t)] = torch.from_numpy(t)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lt = forward(model, xt.to(device))
                lm_loss = nn.functional.cross_entropy(lt.float().reshape(-1, lt.size(-1)), yt.to(device).reshape(-1),
                                                      ignore_index=-100)
                loss = loss + lmw * lm_loss
            if unkw > 0:  # (<unk> only: at each position of the sentence as written, log P(<unk>) stays the
                # starting model's; the ranking otherwise uses <unk> as a knob, 很好用 -> 很好醟, and a
                # calibration afterwards undoes what it learnt)
                texts = [np.array([data.ids.get(ch, 0) for ch in row["text"]], dtype=np.int64) for row in chunk]
                T = max(len(t) for t in texts)
                xt = torch.full((len(texts), T), BOS, dtype=torch.long)
                mask = torch.zeros((len(texts), T), dtype=torch.bool)
                for b, t in enumerate(texts):
                    xt[b, 1:len(t)] = torch.from_numpy(t[:-1])
                    mask[b, :len(t)] = True
                xt, mask = xt.to(device), mask.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lt = forward(model, xt)
                    with torch.no_grad():
                        lo = forward(ref, xt)
                u_new = torch.log_softmax(lt.float(), -1)[..., 0]
                u_ref = torch.log_softmax(lo.float(), -1)[..., 0]
                loss = loss + unkw * ((u_new - u_ref)[mask] ** 2).mean()
            if kdw > 0:  # (stay near the starting model: KL(start || student) over the whole vocabulary on the
                # sentence as written, so characters the ranking never sees, <unk> too, keep their scores)
                texts = [np.array([data.ids.get(ch, 0) for ch in row["text"]], dtype=np.int64) for row in chunk]
                T = max(len(t) for t in texts)
                xt = torch.full((len(texts), T), BOS, dtype=torch.long)
                mask = torch.zeros((len(texts), T), dtype=torch.bool)
                for b, t in enumerate(texts):
                    xt[b, 1:len(t)] = torch.from_numpy(t[:-1])
                    mask[b, :len(t)] = True
                xt, mask = xt.to(device), mask.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lt = forward(model, xt)
                    with torch.no_grad():
                        lo = forward(ref, xt)
                p_ref = torch.log_softmax(lo.float(), -1)
                p_new = torch.log_softmax(lt.float(), -1)
                kl = (p_ref.exp() * (p_ref - p_new)).sum(-1)
                loss = loss + kdw * kl[mask].mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            tidy()  # (tools/lm/gpu.py: without it the GPU spills into system memory and the job crawls)
            if step % 500 == 0:
                msg = (f"ep {ep} step {step}/{steps} loss {loss.item():.4f} w {w.item():.3f} {time.time() - t0:.0f}s"
                       + note())
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
        log.write(evaluate(model, math.exp(wlog.item()), ev, data, device, f"epoch {ep + 1}",
                           os.path.join(DIST, f"{name}_eval_ep{ep + 1}.json")) + "\n")
        log.flush()
        torch.save(model.state_dict(), os.path.join(DIST, f"{name}.pt"))
        torch.save(model.state_dict(), os.path.join(DIST, f"{name}_ep{ep + 1}.pt"))  # (each epoch kept: the best is picked)
        shape = ({"size": size, "emb": emb, "hidden": hidden, "layers": layers} if arch == "gru" else
                 dict(model.cfg, tier=tier))
        json.dump(dict(shape, vocab=len(data.vocab), next=NEXT, w=math.exp(wlog.item()), teacher=tname, tlam=tlam,
                       init=init, data=dfile, epochs=ep + 1),
                  open(os.path.join(DIST, f"{name}.json"), "w", encoding="utf-8"))
    print("saved", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
