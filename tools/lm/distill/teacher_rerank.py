"""Offline: how much would a strong LM gain by reranking DualityIME's n-best? (no repo changes)

    py teacher_rerank.py prep                 # n-best + engine/GRU scores for every sample (CPU)
    py teacher_rerank.py teacher <hf model> [name] [--4bit] [--adapter DIR]
                                              # log10 P(text) of every candidate (GPU); --4bit loads the
                                              # model in 4 bits (bitsandbytes NF4: a 27B in 24 GB, #37);
                                              # --adapter: the LoRA of ft_teacher.py --4bit (DIR/lora.pt)
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
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "tools", "lm"))
sys.stdout.reconfigure(encoding="utf-8")

from common import WORK  # noqa: E402  (DUALITYIME_LM_WORK moves it)
from gpu import tidy  # noqa: E402
OUT = os.path.join(WORK, "teacher_rerank")
SIZE = "XL"
SHIPPED_LAM = 0.5
SETS = [("dev", None, None, 600), ("wiki", 6, 20, 1000), ("web", 6, 20, 1000),
        ("wiki", 15, 40, 1000), ("web", 15, 40, 1000)]
# TR_EXTRA=zai,cases: more held-out sets already in OUT as <name>.jsonl (#37: docs/superpowers/items/37/make_sets.py)
BASE_SETS = list(SETS)  # (the ones prep makes: TR_EXTRA's are made elsewhere and must never be written over here)
SETS += [(n, None, None, 0) for n in os.environ.get("TR_EXTRA", "").split(",") if n]


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
        for s in BASE_SETS:
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
    if FOUR_BIT:
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=q, device_map="cuda").eval()
        if ADAPTER:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import lora
            lora.load(model, os.path.join(ADAPTER, "lora.pt"))
            model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda().eval()
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.convert_tokens_to_ids("<|endoftext|>")
    # (a row with "ctx", #37: the text after its context, P(text | ctx); TR_NOCTX=1 leaves the contexts out)
    texts = sorted({(row_ctx(r), h[0]) for s in SETS for r in load_set(s) for h in r["hyps"]})
    print(model_id, len(texts), "texts", flush=True)
    scores = {}
    # TR_SIMPLIFY=1: each text scored in Simplified characters (a model of mostly Simplified text prefers 于 并 后 to 於
    # 並 後: candidates that differ only so then score the same, #37); the score is kept under the text itself
    if os.environ.get("TR_SIMPLIFY"):
        from engine.script import Script
        simplify = Script.load(os.path.join(REPO, "data")).to_simplified
    else:
        simplify = str
    batch = int(os.environ.get("TR_BATCH", 32))  # (fewer for a model that fills the GPU: TR_BATCH=8)
    t0 = time.time()
    for i in range(0, len(texts), batch):
        if i % (batch * 25) == 0:
            print(f"  {i}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
        tidy()  # (tools/lm/gpu.py: without it the GPU spills into system memory and the job crawls)
        chunk = texts[i:i + batch]
        heads = [[bos] + (tok(simplify(c), add_special_tokens=False)["input_ids"] if c else []) for c, _ in chunk]
        ids = [h + tok(simplify(t), add_special_tokens=False)["input_ids"] for h, (_, t) in zip(heads, chunk)]
        width = max(map(len, ids))
        x = torch.full((len(ids), width), bos, dtype=torch.long)
        attn = torch.zeros((len(ids), width), dtype=torch.long)  # (every token of the row: the model reads them all)
        mask = torch.zeros((len(ids), width), dtype=torch.long)  # (the tokens scored: the text's only, P(text | <bos> + ctx))
        for j, seq in enumerate(ids):
            x[j, :len(seq)] = torch.tensor(seq)
            attn[j, :len(seq)] = 1
            mask[j, len(heads[j]):len(seq)] = 1
        x, attn, mask = x.cuda(), attn.cuda(), mask.cuda()
        with torch.no_grad():
            logits = model(input_ids=x, attention_mask=attn).logits[:, :-1]
            # log p(next) = logit[next] - logsumexp(logits), without a float copy of all logits
            picked = logits.gather(2, x[:, 1:, None])[..., 0].float()
            norm = torch.logsumexp(logits.float(), -1) if logits.shape[0] <= 8 else torch.stack(
                [torch.logsumexp(logits[j].float(), -1) for j in range(logits.shape[0])])
            logp = picked - norm
            del logits
        logp = (logp * mask[:, 1:]).sum(1) / math.log(10)
        for (c, t), v in zip(chunk, logp.tolist()):
            scores[c + "\t" + t if c else t] = v
    name = sys.argv[3] if len(sys.argv) > 3 else model_id.replace("/", "_")
    path = os.path.join(OUT, "teacher_" + name + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False)
    print("saved", path, flush=True)


def row_ctx(r):
    """The context of a held-out row ("" without one, or with TR_NOCTX set)."""
    return "" if os.environ.get("TR_NOCTX") else r.get("ctx", "")


def tkey(r, t):
    """The key of candidate t of row r in a teacher's scores: ctx + tab + t with a context, else t."""
    c = row_ctx(r)
    return c + "\t" + t if c else t


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
        # (each takes the row, for the key of its candidates: tkey; a candidate the teacher never scored is left out)
        combos[f"{short} only"] = (lambda T: lambda lam: lambda r: lambda t, eng, gru: T.get(tkey(r, t)))(T), [1.0]
        combos[f"engine + λ·{short}"] = (lambda T: lambda lam: lambda r: lambda t, eng, gru:
                                         None if tkey(r, t) not in T else eng + lam * T[tkey(r, t)])(T), lams
        combos[f"engine + 0.5·GRU + λ·{short}"] = (lambda T: lambda lam: lambda r: lambda t, eng, gru:
                                                   None if tkey(r, t) not in T else
                                                   eng + (SHIPPED_LAM * gru if gru is not None else 0) + lam * T[tkey(r, t)])(T), lams
    dev = data[os.environ.get("TR_DEV", "dev")]  # (TR_DEV=devctx: λ on the set with contexts)
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
        best_lam = max(grid, key=lambda lam: sum(pick(r, mk(lam)(r)) == r["text"] for r in dev))
        f = mk(best_lam)
        out(f"{cname}  (λ={best_lam} on dev)")
        for name, rows in data.items():
            if name in ("dev", "devctx"):
                continue
            right = w2r = r2w = 0
            for r in rows:
                got = pick(r, f(r))
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


FOUR_BIT = "--4bit" in sys.argv
if FOUR_BIT:
    sys.argv.remove("--4bit")
ADAPTER = None
if "--adapter" in sys.argv:
    i = sys.argv.index("--adapter")
    ADAPTER = sys.argv[i + 1]
    if not FOUR_BIT:  # (the adapter is loaded on the 4-bit model only: without it the teacher would score untuned)
        sys.exit("--adapter needs --4bit (the LoRA of ft_teacher.py --4bit)")
    del sys.argv[i:i + 2]

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep()
    elif cmd == "teacher":
        teacher(sys.argv[2])
    elif cmd == "report":
        report(sys.argv[2:])
