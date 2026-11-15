"""Training rows for the next student (#41): the n-best of runs of Chinese from the training corpus, each with its
context as written (English and all), so the teacher reads the real text and the student its units.

    py -3 tools/lm/next/gen_rows.py [--count 500000] [--zai 0.2] [--mixed 0.2] [--out train_next_500k.jsonl] [--jobs N]

Reads the corpus texts of common.TRAIN (never common.EVAL), NFKC. A run is a stretch of Chinese characters between
anything else (the engine converts each such run on its own); runs of 6-40 characters, a window of 15-40 from a longer
one, at most 2 per text. "ctx": the text before the run in the same text, cut at a unit boundary so it holds L units
(text.to_units): L 0 for 3 rows in 10, else 1-64 at random. Shares: rows whose run holds 在 or 再 (--zai) and rows whose
run comes right after English or letters and digits (--mixed: a space between allowed; their L is at least 1, so the
English is in the context). The rest are any other runs.
Each row: {"text", "base", "hyps": [[candidate, engine score, null], ...], "ctx"} (as tools/lm/distill/gen_train.py
--context --no-gru) -> WORK/teacher_rerank/distill/<out>.
"""
import json
import multiprocessing
import os
import random
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "distill"))
import teacher_rerank as tr  # noqa: E402  (sets sys.path for the repo's modules and tools/lm)
from text import UNITS  # noqa: E402

RUN = re.compile(r"[㐀-鿿豈-﫿]{6,}")
ALNUM_END = re.compile(r"[A-Za-z0-9] ?$")
CTX_MAX, CTX_NONE, PER_TEXT = 64, 0.3, 2


def arg(name, default, cast):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def context(before, want):
    """The end of before holding want units (text.to_units), cut where the first of them starts; "" for 0."""
    if want <= 0:
        return ""
    starts = [m.start() for m in UNITS.finditer(before)]
    return before[starts[-want]:] if len(starts) >= want else before[starts[0]:] if starts else ""


def samples(count, zai_share, mixed_share, seed=11):
    import pyarrow.parquet as pq
    from common import TRAIN, text_column
    from engine.lexicon import Lexicon
    from evaluate import reading_of
    from evaluate_lm import best_readings
    best = best_readings(Lexicon())
    rng = random.Random(seed)
    want = {"zai": int(count * zai_share), "mixed": int(count * mixed_share)}
    want["plain"] = count - want["zai"] - want["mixed"]
    got = {k: [] for k in want}
    groups = [(p, g) for p in TRAIN for g in range(pq.ParquetFile(p).num_row_groups)]
    rng.shuffle(groups)
    seen = 0
    for path, g in groups:
        pf = pq.ParquetFile(path)
        for text in pf.read_row_group(g, columns=[text_column(pf)]).column(0).to_pylist():
            if not text:
                continue
            text = unicodedata.normalize("NFKC", text)
            runs = list(RUN.finditer(text))
            rng.shuffle(runs)
            taken = 0
            for m in runs:
                a, b = m.span()
                if b - a > 40:  # (a window of 15-40 from a long run: what is before it in the run is context)
                    n = rng.randint(15, 40)
                    a = rng.randint(a, b - n)
                    b = a + n
                run = text[a:b]
                before = text[:a]
                kind = "mixed" if ALNUM_END.search(before) else "zai" if ("在" in run or "再" in run) else "plain"
                if kind == "mixed" and len(got["mixed"]) >= want["mixed"] and ("在" in run or "再" in run):
                    kind = "zai"
                if len(got[kind]) >= want[kind]:
                    continue
                rd = reading_of(run, best)
                if not rd:
                    continue
                n_ctx = rng.randint(1, CTX_MAX) if kind == "mixed" or rng.random() >= CTX_NONE else 0
                got[kind].append((run, rd, context(before, n_ctx)))
                taken += 1
                if taken >= PER_TEXT:
                    break
            seen += 1
            if all(len(got[k]) >= want[k] for k in want):
                out = got["zai"] + got["mixed"] + got["plain"]
                rng.shuffle(out)
                print(f"texts read {seen}; rows {len(out)} (zai {len(got['zai'])}, mixed {len(got['mixed'])}, "
                      f"plain {len(got['plain'])})", flush=True)
                return out
        print(f"  texts read {seen}: zai {len(got['zai'])}/{want['zai']}, mixed {len(got['mixed'])}/{want['mixed']}, "
              f"plain {len(got['plain'])}/{want['plain']}", flush=True)
    raise SystemExit("the corpus ran out before the shares were filled")


def job(args):
    """The n-best of the run (the engine's search reads no context) and its context (gen_train.py _job_no_gru)."""
    text, readings, ctx = args
    lexicon, bigram, _, _, _ = tr._ctx
    from engine.composer import _search, convert
    from decode import nbest
    base = "".join(p for _, _, p in convert(readings, lexicon, None, None, bigram))
    ends, options, _ = _search(readings, lexicon, None, {}, bigram, None)
    hyps = nbest(readings, options, bigram) if ends else []
    if base not in (t for _, t in hyps) and ends:
        top = max(ends.values(), key=lambda v: v[0])
        hyps.append((top[0], base))
    return {"text": text, "base": base, "hyps": [[t, score, None] for score, t in hyps], "ctx": ctx}


def main():
    count = arg("--count", 500000, int)
    out = os.path.join(tr.OUT, "distill", arg("--out", "train_next_500k.jsonl", str))
    jobs = arg("--jobs", max(1, (os.cpu_count() or 2) // 2), int)
    todo = samples(count, arg("--zai", 0.2, float), arg("--mixed", 0.2, float))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with multiprocessing.Pool(processes=jobs, initializer=tr._init) as pool, \
            open(out, "w", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(pool.imap(job, todo, chunksize=16)):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 20000 == 0:
                print(i, flush=True)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
