"""Accuracy of an int8 student (.npz from tools/lm/quantize.py) in the engine's own arithmetic (IntLM,
decode.Scorer.phrase_score), on teacher_rerank's held-out sets, vs shipped (XL int8, λ 0.5).

    py eval_int8.py <npz> <lambda>
"""
import json
import multiprocessing
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import teacher_rerank as tr  # noqa: E402  (sets the repo paths)

_ctx = None


def _init(npz):
    global _ctx
    from engine.lexicon import Lexicon
    from evaluate import reading_of
    from evaluate_lm import best_readings
    from decode import candidates_by_syllable
    from intlm import IntLM
    lex = Lexicon()
    vocab = open(os.path.join(tr.WORK, "vocab.txt"), encoding="utf-8").read().rstrip("\n").split("\n")
    _ctx = (IntLM(npz), vocab, candidates_by_syllable(lex, vocab), best_readings(lex), reading_of)


def _job(args):
    r, lam = args
    from decode import Scorer
    lm, vocab, cands, best, reading_of = _ctx
    rd = reading_of(r["text"], best)
    scorer = Scorer(lm, vocab, cands)
    top, bs = None, None
    for t, eng, _ in r["hyps"]:
        s = eng + (lam * scorer.phrase_score("", t, rd) if rd and len(t) == len(rd) else 0.0)
        if bs is None or s > bs:
            top, bs = t, s
    return top if top is not None else r["base"]


def main():
    npz, lam = sys.argv[1], float(sys.argv[2])
    with multiprocessing.Pool(processes=max(1, os.cpu_count() // 2), initializer=_init, initargs=(npz,)) as pool:
        for s in tr.SETS:
            rows = tr.load_set(s)
            got = pool.map(_job, [(r, lam) for r in rows], chunksize=8)
            right = sum(g == r["text"] for g, r in zip(got, rows))
            ship = [tr.shipped(r) == r["text"] for r in rows]
            w2r = sum(g == r["text"] and not sh for g, r, sh in zip(got, rows, ship))
            r2w = sum(g != r["text"] and sh for g, r, sh in zip(got, rows, ship))
            print(f"  {tr.set_name(s)}: {right / len(rows):.1%} (shipped {sum(ship) / len(rows):.1%})  "
                  f"wrong→right {w2r}  right→wrong {r2w}", flush=True)


if __name__ == "__main__":
    main()
