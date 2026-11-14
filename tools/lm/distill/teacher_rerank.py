"""Offline: how much would a strong LM gain by reranking DualityIME's n-best? (no repo changes)

    py teacher_rerank.py prep                 # n-best + engine/GRU scores for every sample (CPU)
    py teacher_rerank.py teacher <hf model>   # log10 P(text) of every candidate (GPU)
    py teacher_rerank.py report <hf model>... # accuracy, right->wrong / wrong->right vs shipped

Samples as tools/lm/evaluate_lm.py: dev (600, for λ) and the held-out wiki / web files,
runs of 6-20 and 15-40 characters (1000 each). "shipped" = nbest mode, XL int8, λ 0.5.
Output: <WORK>\\teacher_rerank\\ (WORK: tools/lm/common.py)
"""
import json
import math
import multiprocessing
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "tools", "lm"))
sys.stdout.reconfigure(encoding="utf-8")

from common import WORK  # noqa: E402  (DUALITYIME_LM_WORK moves it)
OUT = os.path.join(WORK, "teacher_rerank")
SIZE = "XL"
SHIPPED_LAM = 0.5
SETS = [("dev", None, None, 600), ("wiki", 6, 20, 1000), ("web", 6, 20, 1000),
        ("wiki", 15, 40, 1000), ("web", 15, 40, 1000)]


def set_name(s):
    name, lo, hi, _ = s
    return name if lo is None else f"{name}_{lo}-{hi}"


# ---- prep ---------------------------------------------------------------------------------------
_ctx = None


def _init():
    global _ctx
    from engine import load_resources
    from decode import candidates_by_syllable
    from intlm import IntLM
    _, lexicon, _, _, bigram, _ = load_resources(None)
    vocab = open(os.path.join(WORK, "vocab.txt"), encoding="utf-8").read().rstrip("\n").split("\n")
    lm = IntLM(os.path.join(WORK, "models", SIZE + ".npz"))
    _ctx = (lexicon, bigram, lm, vocab, candidates_by_syllable(lexicon, vocab))


def _job(args):
    text, readings = args
    lexicon, bigram, lm, vocab, cands = _ctx
    from engine.composer import _search, convert
    from decode import Scorer, nbest
    n = len(readings)
    base = "".join(p for _, _, p in convert(readings, lexicon, None, None, bigram))
    ends, options, _ = _search(readings, lexicon, None, {}, bigram, None)
    hyps = nbest(readings, options, bigram) if ends else []
    if base not in (t for _, t in hyps) and ends:
        top = max(ends.values(), key=lambda v: v[0])
        hyps.append((top[0], base))
    scorer = Scorer(lm, vocab, cands)
    out = []
    for score, t in hyps:
        gru = scorer.phrase_score("", t, readings) if len(t) == n else None
        out.append([t, score, gru])
    return {"text": text, "base": base, "hyps": out}


def prep():
    from engine.lexicon import Lexicon
    from evaluate_lm import best_readings, dev_samples, eval_samples
    from common import EVAL
    os.makedirs(OUT, exist_ok=True)
    best = best_readings(Lexicon())
    with multiprocessing.Pool(processes=max(1, os.cpu_count() // 2), initializer=_init) as pool:
        for s in SETS:
            name, lo, hi, count = s
            samples = dev_samples(WORK, count, best) if lo is None else eval_samples(EVAL[name], lo, hi, count, best)
            res = pool.map(_job, samples, chunksize=4)
            with open(os.path.join(OUT, set_name(s) + ".jsonl"), "w", encoding="utf-8") as f:
                for r in res:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(set_name(s), len(res), flush=True)


def load_set(s):
    with open(os.path.join(OUT, set_name(s) + ".jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---- teacher ------------------------------------------------------------------------------------
def teacher(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda().eval()
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.convert_tokens_to_ids("<|endoftext|>")
    texts = sorted({h[0] for s in SETS for r in load_set(s) for h in r["hyps"]})
    print(model_id, len(texts), "texts", flush=True)
    scores = {}
    batch = 32
    for i in range(0, len(texts), batch):
        if i % (batch * 100) == 0:
            print(f"  {i}/{len(texts)}", flush=True)
        chunk = texts[i:i + batch]
        ids = [[bos] + tok(t, add_special_tokens=False)["input_ids"] for t in chunk]
        width = max(map(len, ids))
        x = torch.full((len(ids), width), bos, dtype=torch.long)
        mask = torch.zeros((len(ids), width), dtype=torch.long)
        for j, seq in enumerate(ids):
            x[j, :len(seq)] = torch.tensor(seq)
            mask[j, :len(seq)] = 1
        x, mask = x.cuda(), mask.cuda()
        with torch.no_grad():
            logits = model(input_ids=x, attention_mask=mask).logits[:, :-1]
            # log p(next) = logit[next] - logsumexp(logits), without a float copy of all logits
            picked = logits.gather(2, x[:, 1:, None])[..., 0].float()
            norm = torch.logsumexp(logits.float(), -1) if logits.shape[0] <= 8 else torch.stack(
                [torch.logsumexp(logits[j].float(), -1) for j in range(logits.shape[0])])
            logp = picked - norm
            del logits
        logp = (logp * mask[:, 1:]).sum(1) / math.log(10)
        for t, v in zip(chunk, logp.tolist()):
            scores[t] = v
    name = sys.argv[3] if len(sys.argv) > 3 else model_id.replace("/", "_")
    path = os.path.join(OUT, "teacher_" + name + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False)
    print("saved", path, flush=True)


# ---- report -------------------------------------------------------------------------------------
def pick(r, f):
    best, bs = None, None
    for t, eng, gru in r["hyps"]:
        s = f(t, eng, gru)
        if s is not None and (bs is None or s > bs):
            best, bs = t, s
    return best if best is not None else r["base"]


def shipped(r):
    return pick(r, lambda t, eng, gru: eng + SHIPPED_LAM * gru if gru is not None else eng)


def report(model_ids):
    data = {set_name(s): load_set(s) for s in SETS}
    teachers = {}
    for m in model_ids:
        with open(os.path.join(OUT, "teacher_" + m.replace("/", "_") + ".json"), encoding="utf-8") as f:
            teachers[m] = json.load(f)
    lams = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    combos = {}
    for m, T in teachers.items():
        short = m.split("/")[-1]
        combos[f"{short} only"] = (lambda T: lambda lam: lambda t, eng, gru: T[t])(T), [1.0]
        combos[f"engine + λ·{short}"] = (lambda T: lambda lam: lambda t, eng, gru: eng + lam * T[t])(T), lams
        combos[f"engine + 0.5·GRU + λ·{short}"] = (lambda T: lambda lam: lambda t, eng, gru:
                                                   eng + (SHIPPED_LAM * gru if gru is not None else 0) + lam * T[t])(T), lams
    dev = data["dev"]
    lines = []
    def out(s=""):
        print(s, flush=True)
        lines.append(s)
    for name, rows in data.items():
        if name == "dev":
            continue
        oracle = sum(any(h[0] == r["text"] for h in r["hyps"]) for r in rows) / len(rows)
        base = sum(r["base"] == r["text"] for r in rows) / len(rows)
        ship = sum(shipped(r) == r["text"] for r in rows) / len(rows)
        out(f"{name} ({len(rows)}): convert {base:.1%}  shipped(XL λ0.5) {ship:.1%}  oracle(top-10) {oracle:.1%}")
    out()
    examples = []
    for cname, (mk, grid) in combos.items():
        # λ on dev
        best_lam = max(grid, key=lambda lam: sum(pick(r, mk(lam)) == r["text"] for r in dev))
        f = mk(best_lam)
        out(f"{cname}  (λ={best_lam} on dev)")
        for name, rows in data.items():
            if name == "dev":
                continue
            right = w2r = r2w = 0
            for r in rows:
                got = pick(r, f)
                s = shipped(r)
                ok, ok_s = got == r["text"], s == r["text"]
                right += ok
                w2r += ok and not ok_s
                r2w += ok_s and not ok
                if ok_s and not ok and len(examples) < 400:
                    examples.append((cname, name, r["text"], got))
            out(f"  {name}: {right / len(rows):.1%}   wrong→right {w2r}   right→wrong {r2w}")
    with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n\nright→wrong examples (vs shipped):\n")
        for c, n, t, g in examples:
            fh.write(f"{c} | {n} | {t} → {g}\n")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep()
    elif cmd == "teacher":
        teacher(sys.argv[2])
    elif cmd == "report":
        report(sys.argv[2:])
