"""長句選字加強's choice among the n-best, recomputed offline with other weights.

    py tools/lm/rerank_sweep.py [--rev REV] [--grid 0.5:0,0.5:0.4,0.75:0.5] [--base 0.5:0]
                                [--count 1911] [--files last|eval] [--save F | --load F] [--examples N] [--jobs N]

Each held-out sentence is converted once (convert() of the Python engine, no memory, no context); for each,
the texts _with_lm weighs are kept with their three numbers: the engine's score, the model's log10
probability and the characters inside cohesive words (WORD_BONUS). Then the choice is made again for
every LM_WEIGHT:WORD_BONUS of --grid (the engine's own text first, a later one only when strictly better,
as _with_lm does), and each is counted against --base: sentences fully right, fixed, broken.
Before any of that: the choice with the revision's own LM_WEIGHT and WORD_BONUS must be convert()'s
text sentence by sentence (else the numbers are not the engine's and it stops).

--rev: a git revision whose engine/, data/ and tools/ are taken out with `git archive` into
%TEMP%\\dualityime-sweep\\<commit> (kept for the next time, never deleted here) and used for everything
(its lexicon, model, reading_of); default: this folder as it is now. A revision before WORD_BONUS
(e.g. 1eb3691^) counts no cohesive words.
--files: last (default): the last parquet file of each corpus folder, as docs/superpowers/items/14/names_eval.py
takes them (1911 runs each: the 3,822 sentences of decisions.md); eval: tools/lm/common.EVAL.
Runs of 6-20 Chinese characters from the end of each file (tools/evaluate.py's way, never trained on).
Needs the corpus (DUALITYIME_CORPUS, tools/lm/common.py), numpy and pyarrow.
--save / --load: the candidates as JSON with the weights of the engine that made them (the check above
uses those), to try other weights without converting again.

Reproduces decisions.md 2026-11-03 (長句選字加強：凝聚的詞每個字加 0.4):
    py tools/lm/rerank_sweep.py --rev 1eb3691 --grid 0.5:0.4,0.75:0.5 --base 0.5:0
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TREE = os.environ.get("DUALITYIME_SWEEP_TREE", ROOT)  # (set when re-run inside a revision's folder)
for p in (os.path.join(TREE, "tools", "lm"), os.path.join(TREE, "tools"), TREE):
    sys.path.insert(0, p)
sys.stdout.reconfigure(encoding="utf-8")

CACHE = os.path.join(tempfile.gettempdir(), "dualityime-sweep")
RUN = re.compile("[一-鿿]{6,20}")


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, check=True).stdout


def tree_of(rev):
    """The folder of this revision (engine/, data/, tools/), taken out once."""
    commit = git("rev-parse", "--verify", rev + "^{commit}").decode().strip()
    tree = os.path.join(CACHE, commit[:12])
    done = os.path.join(tree, ".complete")  # (written last: a run stopped half way is taken out again)
    if not os.path.exists(done):
        print(f"{rev} ({commit[:7]}): taking it out into {tree} ...", flush=True)
        os.makedirs(tree, exist_ok=True)
        archive = subprocess.Popen(["git", "-C", ROOT, "archive", "--format=tar", commit, "engine", "data", "tools"],
                                   stdout=subprocess.PIPE)
        with tarfile.open(fileobj=archive.stdout, mode="r|") as tar:
            tar.extractall(tree, filter="data")
        archive.stdout.read()  # (the padding after the end-of-archive blocks: or git waits on a full pipe for ever)
        if archive.wait() != 0:
            sys.exit(f"git archive {rev} failed")
        with open(done, "w") as f:
            f.write(commit)
    return f"{rev} ({commit[:7]})", tree


def corpus_files(which):
    from common import CORPUS, EVAL
    if which == "eval":
        return [EVAL["wiki"], EVAL["web"]]
    folders = [os.path.join(CORPUS, "wikipedia-zhtw", "20231001"), os.path.join(CORPUS, "CC-100-zh-Hant", "data")]
    files = [sorted(glob.glob(os.path.join(f, "*.parquet"))) for f in folders]
    if not all(files):
        sys.exit(f"corpus not found in {CORPUS} (set DUALITYIME_CORPUS)")
    return [f[-1] for f in files]


def samples(lexicon, files, count):
    """[(text, readings)]: count runs of 6-20 characters from the end of each file."""
    import pyarrow.parquet as pq
    from evaluate import reading_of
    best = {}
    for reading, lst in lexicon.entries.items():
        for p, s in lst:
            if s > best.get(p, ("", -99))[1]:
                best[p] = (reading, s)
    best = {p: r for p, (r, _) in best.items()}
    out = []
    for path in files:
        pf = pq.ParquetFile(path)
        column = "text" if "text" in pf.schema_arrow.names else pf.schema_arrow.names[0]
        texts = pf.read_row_group(pf.num_row_groups - 1, columns=[column]).column(0).to_pylist()
        got = []
        for text in reversed(texts):
            for run in RUN.findall(text):
                rd = reading_of(run, best)
                if rd:
                    got.append((run, rd))
            if len(got) >= count:
                break
        out += got[:count]
    return out


_W = {}  # this worker's engine: lexicon, bigram, scorer


def _init():
    from engine import load_resources
    from engine.lm import Scorer, load
    _, lexicon, _, _, bigram, _ = load_resources(None)
    model = load()
    if model is None:
        raise SystemExit("no language model (data/lm.bin, numpy)")
    _W.update(lexicon=lexicon, bigram=bigram, scorer=Scorer(model, lexicon))


def _job(sample):
    """(text, [[text, engine score, model log10, cohesive chars]], convert()'s text): as _with_lm weighs them."""
    from engine import composer as C
    text, rd = sample
    lexicon, bigram, scorer = _W["lexicon"], _W["bigram"], _W["scorer"]
    word_chars = getattr(C, "_word_chars", lambda path, bigram: 0)
    actual = "".join(p for _, _, p in C.convert(rd, lexicon, None, None, bigram, None, scorer))
    ends, options, full = C._search(rd, lexicon, None, {}, bigram, None)
    if not ends:
        return text, [], actual
    score, path = max(ends.values(), key=lambda v: v[0])
    path = C.repick(path, rd, lexicon, options, full, None, bigram, None, {})
    first = "".join(p for _, _, p in path)
    n = len(rd)

    def lm(t):
        return scorer.text_score(t, rd) / (1024 * C.LN10)

    if len(first) != n:  # (the model cannot read it: the engine's text stays)
        return text, [[first, score, 0.0, 0]], actual
    cands = [[first, score, lm(first), word_chars(path, bigram)]]
    seen = {first}
    for s, hyp in C.nbest(rd, options, bigram, None, C.NBEST_K, C.NBEST_N):
        hyp = C.repick(hyp, rd, lexicon, options, full, None, bigram, None, {}, True)
        t = "".join(p for _, _, p in hyp)
        if t in seen:
            continue
        seen.add(t)
        if len(t) != n:  # (nothing is kept without memory: no other condition of _with_lm applies)
            continue
        cands.append([t, s, lm(t), word_chars(hyp, bigram)])
    return text, cands, actual


def choose(cands, weight, bonus):
    """_with_lm's choice: the first (the engine's) unless a later one is strictly better."""
    if not cands:
        return None
    best, best_score = cands[0][0], None
    for t, s, l, w in cands:
        v = s + weight * l + bonus * w
        if best_score is None or v > best_score:
            best, best_score = t, v
    return best


def parse_point(text):
    w, b = text.split(":")
    return float(w), float(b)


def main():
    ap = argparse.ArgumentParser(description="長句選字加強's choice recomputed with other weights.")
    ap.add_argument("--rev", help="a git revision instead of this folder as it is now")
    ap.add_argument("--grid", default="0.5:0,0.5:0.4,0.75:0.5", help="LM_WEIGHT:WORD_BONUS, comma separated")
    ap.add_argument("--base", help="the point the others are counted against (default: the revision's own)")
    ap.add_argument("--count", type=int, default=1911, help="sentences from each file")
    ap.add_argument("--files", choices=("last", "eval"), default="last")
    ap.add_argument("--save", help="write the candidates here (JSON)")
    ap.add_argument("--load", help="read the candidates from here instead of converting")
    ap.add_argument("--examples", type=int, default=0, help="fixed and broken sentences shown per point")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="default: half the CPUs")
    args = ap.parse_args()

    if args.rev and "DUALITYIME_SWEEP_TREE" not in os.environ:  # run again with the revision's engine
        name, tree = tree_of(args.rev)
        print("revision:", name, flush=True)
        env = dict(os.environ, DUALITYIME_SWEEP_TREE=tree)
        return subprocess.run([sys.executable, os.path.abspath(__file__)] + sys.argv[1:], env=env).returncode

    from engine import composer as C
    own = (C.LM_WEIGHT, getattr(C, "WORD_BONUS", 0.0))
    if args.load:  # (the weights of the engine that made them, not this folder's)
        with open(args.load, encoding="utf-8") as f:
            saved = json.load(f)
        if not isinstance(saved, dict):
            sys.exit(f"{args.load}: not a --save file of this version (no engine weights in it)")
        own, rows = tuple(saved["engine"]), saved["rows"]
    else:
        import multiprocessing
        from engine import load_resources
        files = corpus_files(args.files)
        todo = samples(load_resources(None)[1], files, args.count)
        print(f"sentences: {len(todo)} ({args.count} from each of {', '.join(os.path.basename(f) for f in files)})",
              flush=True)
        with multiprocessing.Pool(args.jobs, initializer=_init) as pool:
            rows = pool.map(_job, todo, chunksize=8)
        if args.save:
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump({"engine": own, "rows": rows}, f, ensure_ascii=False)
            print("candidates ->", args.save)

    differ = [(t, choose(c, *own), a) for t, c, a in rows if choose(c, *own) not in (None, a)]
    print(f"engine: LM_WEIGHT {own[0]}, WORD_BONUS {own[1]}; recomputed = convert(): "
          f"{len(rows) - len(differ)}/{len(rows)}")
    if differ:
        for t, mine, a in differ[:5]:
            print(f"  {t}: recomputed {mine}, convert() {a}")
        sys.exit("the recomputed choice is not the engine's: the numbers would not be its behaviour")

    base = parse_point(args.base) if args.base else own
    points = [parse_point(p) for p in args.grid.split(",")]
    right_base = [choose(c, *base) == t for t, c, _ in rows]
    print(f"base {base[0]}:{base[1]}: {sum(right_base)}/{len(rows)} fully right")
    for w, b in points:
        right = [choose(c, w, b) == t for t, c, _ in rows]
        fixed = [i for i, (r, r0) in enumerate(zip(right, right_base)) if r and not r0]
        broken = [i for i, (r, r0) in enumerate(zip(right, right_base)) if r0 and not r]
        print(f"  {w}:{b}: {sum(right)} right, fixed {len(fixed)}, broken {len(broken)}, "
              f"net {len(fixed) - len(broken):+d}")
        for what, idx in (("fixed", fixed), ("broken", broken)):
            for i in idx[:args.examples]:
                t, c, _ = rows[i]
                print(f"    {what}: {t}  (base {choose(c, *base)} -> {choose(c, w, b)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
