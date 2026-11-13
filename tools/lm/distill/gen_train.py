"""Training data for the students: n-best of runs from train.u16 (never dev / evaluation files).

    py gen_train.py [--count 100000]

Each run of 6-40 characters -> its readings (as the evaluation samples: the dictionary's best reading
per character) -> teacher_rerank._job (convert, n-best with engine scores, the shipped GRU's score).
Writes <WORK>\\teacher_rerank\\distill\\train_nbest.jsonl (or --out NAME there).
"""
import json
import multiprocessing
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teacher_rerank as tr  # noqa: E402  (sets sys.path for the repo's modules)

COUNT = int(sys.argv[sys.argv.index("--count") + 1]) if "--count" in sys.argv else 100000
DIR = os.path.join(tr.OUT, "distill")


def samples(count):
    from engine.lexicon import Lexicon
    from evaluate import reading_of
    from evaluate_lm import best_readings
    best = best_readings(Lexicon())
    vocab = open(os.path.join(tr.WORK, "vocab.txt"), encoding="utf-8").read().rstrip("\n").split("\n")
    data = np.memmap(os.path.join(tr.WORK, "train.u16"), dtype=np.uint16, mode="r")
    starts = np.flatnonzero(data == 1)
    rng = np.random.default_rng(3)
    out = []
    for s in rng.permutation(len(starts) - 1):
        ids = data[starts[s] + 1:starts[s + 1]]
        if (ids == 0).any() or len(ids) < 6:
            continue
        if len(ids) > 40:  # a window of 15-40 from a long run
            n = int(rng.integers(15, 41))
            a = int(rng.integers(0, len(ids) - n + 1))
            ids = ids[a:a + n]
        text = "".join(vocab[i] for i in ids)
        rd = reading_of(text, best)
        if rd:
            out.append((text, rd))
        if len(out) >= count:
            break
    return out


def _job_no_gru(args):
    """tr._job without the shipped GRU's score (distill.py never reads it): the n-best search alone."""
    text, readings = args
    lexicon, bigram, _, _, _ = tr._ctx
    from engine.composer import _search, convert
    from decode import nbest
    base = "".join(p for _, _, p in convert(readings, lexicon, None, None, bigram))
    ends, options, _ = _search(readings, lexicon, None, {}, bigram, None)
    hyps = nbest(readings, options, bigram) if ends else []
    if base not in (t for _, t in hyps) and ends:
        top = max(ends.values(), key=lambda v: v[0])
        hyps.append((top[0], base))
    return {"text": text, "base": base, "hyps": [[t, score, None] for score, t in hyps]}


def main():
    os.makedirs(DIR, exist_ok=True)
    todo = samples(COUNT)
    print("samples", len(todo), flush=True)
    path = os.path.join(DIR, sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "train_nbest.jsonl")
    with multiprocessing.Pool(processes=max(1, os.cpu_count() // 2), initializer=tr._init) as pool, open(path, "w", encoding="utf-8") as f:
        for i, r in enumerate(pool.imap(_job_no_gru if "--no-gru" in sys.argv else tr._job, todo, chunksize=16)):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 10000 == 0:
                print(i, flush=True)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
