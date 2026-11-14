"""Typing with a copy of someone's memory (or none), before and after a change: what got right, what got broken.

    py tools/user_sweep.py run <memory folder | none> <out.json> [--rev REV] [--kinds words,after,docs]
                           [--words 6000] [--corpus 1911] [--set long_sentence=1 ...] [--jobs N]
    py tools/user_sweep.py compare <before.json> <after.json> [--examples N]

run: types, through the Python engine's Composer with that memory (user.json and the files beside it:
dictionary.json, keys.json, ...; none: nothing learned), every item of --kinds and records what was sent:
  - words: the --words most common dictionary words of two characters or more, each typed alone
  - after: each of them typed after 我 in the same run
  - docs: the Chinese of this repository's own text (README, docs but docs/superpowers, the settings
    page; runs of 6-24 characters, as tools/autocorrect_eval.natural_sentences takes them)
  - corpus: --corpus runs of 6-20 characters from the end of the last file of each corpus folder, never
    trained on (tools/lm/rerank_sweep.py's sentences: 1911 each, the 3,822 of decisions.md; needs the corpus,
    DUALITYIME_CORPUS in tools/lm/common.py)
Each item starts from the memory as it is in the folder (a copy in memory: nothing is written anywhere,
the folder is only read). Settings: as they come, then --set (default: long_sentence=1 autocorrect=1, as
decisions.md 2026-11-03 measured; --set none for as they come).

Reproduces decisions.md 2026-11-09 (the distilled model: the commit before it, then its own):
    py tools/user_sweep.py run none before.json --rev 4b39496^ --kinds corpus,words,after --words 3000 --set none
    py tools/user_sweep.py run none after.json --rev 4b39496 --kinds corpus,words,after --words 3000 --set none
    py tools/user_sweep.py compare before.json after.json
--rev: a git revision whose engine/, data/ and tools/ are taken out with `git archive` into
%TEMP%\\dualityime-sweep\\<commit> (shared with tools/lm/rerank_sweep.py, kept) and used for the typing;
the docs' text always comes from this folder, so two revisions type the same sentences.
compare: right per kind before and after, fixed and broken (examples of each).

The memory: always a copy (never %APPDATA%\\DualityIME itself, CLAUDE.md), and the copy stays out of the
repository (it is what someone typed). A real user.json fades by today's date: run both sides the same day.
Needs numpy (else the Python engine runs without the language model: CLAUDE.md Gotchas).
"""
import argparse
import copy
import glob
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TREE = os.environ.get("DUALITYIME_SWEEP_TREE", ROOT)  # (set when re-run inside a revision's folder)
for p in (os.path.join(TREE, "tools"), TREE):
    sys.path.insert(0, p)
sys.stdout.reconfigure(encoding="utf-8")

CACHE = os.path.join(tempfile.gettempdir(), "dualityime-sweep")
KINDS = ("words", "after", "docs", "corpus")
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


def best_readings(lexicon):
    """phrase -> its most common reading key ("ㄋㄧˇ-ㄏㄠˇ"), and phrase -> that score."""
    best = {}
    for reading, lst in lexicon.entries.items():
        for p, s in lst:
            if s > best.get(p, ("", -99.0))[1]:
                best[p] = (reading, s)
    return {p: r for p, (r, _) in best.items()}, {p: s for p, (_, s) in best.items()}


def doc_runs():
    """Chinese runs of 6-24 characters in this folder's own text (natural_sentences' files), once each."""
    files = [os.path.join(ROOT, "README.md")]
    files += [f for f in glob.glob(os.path.join(ROOT, "docs", "**", "*.md"), recursive=True)
              if os.sep + "superpowers" + os.sep not in f]
    files += glob.glob(os.path.join(ROOT, "settings", "ui", "*.js"))
    out, seen = [], set()
    for path in sorted(files):
        with open(path, encoding="utf-8") as f:
            for run in re.findall(r"[一-鿿]{6,24}", f.read()):
                if run not in seen:
                    seen.add(run)
                    out.append(run)
    return out


def corpus_runs(count):
    """count runs of 6-20 characters from the end of the last parquet file of each corpus folder."""
    import pyarrow.parquet as pq
    sys.path.insert(0, os.path.join(ROOT, "tools", "lm"))
    from common import CORPUS
    out = []
    for folder in (os.path.join(CORPUS, "wikipedia-zhtw", "20231001"), os.path.join(CORPUS, "CC-100-zh-Hant", "data")):
        files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
        if not files:
            sys.exit(f"corpus not found in {folder} (set DUALITYIME_CORPUS)")
        pf = pq.ParquetFile(files[-1])
        column = "text" if "text" in pf.schema_arrow.names else pf.schema_arrow.names[0]
        texts = pf.read_row_group(pf.num_row_groups - 1, columns=[column]).column(0).to_pylist()
        got = []
        for text in reversed(texts):
            got += RUN.findall(text)
            if len(got) >= 2 * count:  # (some are not typeable: dropped below, as rerank_sweep does)
                break
        out.append(got)
    return out


def items(lexicon, kinds, words, runs, corpus, count):
    """[(kind, text, readings)]"""
    from evaluate import reading_of
    best, score = best_readings(lexicon)
    common = sorted((p for p, r in best.items() if len(p) >= 2 and len(p) == r.count("-") + 1),
                    key=lambda p: (-score[p], p))[:words]
    out = []
    if "words" in kinds:
        out += [("words", p, best[p].split("-")) for p in common]
    if "after" in kinds:
        me = best["我"].split("-")
        out += [("after", "我" + p, me + best[p].split("-")) for p in common]
    if "docs" in kinds:
        for run in runs:
            rd = reading_of(run, best)
            if rd and len(rd) == len(run):
                out.append(("docs", run, rd))
    if "corpus" in kinds:
        for file_runs in corpus:  # (as rerank_sweep.samples: the first `count` typeable runs of each file)
            got = [(run, reading_of(run, best)) for run in file_runs]
            out += [("corpus", run, rd) for run, rd in got if rd][:count]
    return out


_W = {}


def _detach(user):
    """The memory with every file path taken away: saving does nothing (each memory's save() returns
    without a path)."""
    user.path = None
    for value in vars(user).values():
        if hasattr(value, "path") and hasattr(value, "save"):
            value.path = None
    return user


def _init(folder, settings):
    from engine import load_resources
    from engine.lexicon import UserModel
    table, lexicon, english, _, bigram, config = load_resources(None)
    if settings:
        config.set(**settings)
    _W.update(table=table, lexicon=lexicon, english=english, bigram=bigram, config=config,
              user=_detach(UserModel(os.path.join(folder, "user.json") if folder else None)))


def _job(item):
    from engine.composer import Composer
    from evaluate import keys_of
    kind, text, rd = item
    c = Composer(_W["table"], _W["lexicon"], _W["english"], copy.deepcopy(_W["user"]), _W["bigram"], _W["config"])
    out = ""
    for ch in keys_of(rd):
        c.key(ch)
        out += c.commit
    c.key(special="ENTER")
    return kind, text, out + c.commit, rd


def parse_settings(pairs):
    if pairs == ["none"]:
        return {}
    out = {}
    for pair in pairs:
        k, v = pair.split("=", 1)
        out[k] = True if v in ("1", "true") else False if v in ("0", "false") else int(v) if v.isdigit() else v
    return out


def run(args):
    folder = None if args.folder == "none" else os.path.abspath(args.folder)
    if folder and not os.path.isfile(os.path.join(folder, "user.json")):
        sys.exit(f"no user.json in {folder}")
    if folder and os.path.normcase(folder) == os.path.normcase(os.path.join(os.environ.get("APPDATA", ""), "DualityIME")):
        sys.exit("that is the real memory: copy it to a folder of its own first")
    kinds = args.kinds.split(",")
    if any(k not in KINDS for k in kinds):
        sys.exit(f"--kinds: {', '.join(KINDS)}")
    if args.rev and "DUALITYIME_SWEEP_TREE" not in os.environ:  # run again with the revision's engine
        name, tree = tree_of(args.rev)
        print("revision:", name, flush=True)
        env = dict(os.environ, DUALITYIME_SWEEP_TREE=tree, DUALITYIME_SWEEP_DOCS=json.dumps(doc_runs()))
        return subprocess.run([sys.executable, os.path.abspath(__file__)] + sys.argv[1:], env=env).returncode
    import multiprocessing
    from engine import load_resources
    from engine import lm
    if lm.load() is None:
        sys.exit("the language model does not load (numpy?): the results would not be the IME's")
    runs = json.loads(os.environ["DUALITYIME_SWEEP_DOCS"]) if "DUALITYIME_SWEEP_DOCS" in os.environ else doc_runs()
    corpus = corpus_runs(args.corpus) if "corpus" in kinds else []
    todo = items(load_resources(None)[1], kinds, args.words, runs, corpus, args.corpus)
    settings = parse_settings(args.set)
    print(f"items: {len(todo)} ({', '.join(f'{k} {sum(1 for t in todo if t[0] == k)}' for k in kinds)}); "
          f"memory {folder or 'none'}; settings {settings or 'as they come'}", flush=True)
    if "corpus" in kinds and sum(1 for t in todo if t[0] == "corpus") != 2 * args.corpus:
        sys.exit("fewer corpus sentences than asked: the numbers would not compare")
    with multiprocessing.Pool(args.jobs, initializer=_init, initargs=(folder, settings)) as pool:
        rows = pool.map(_job, todo, chunksize=16)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"tree": TREE, "settings": settings, "rows": rows}, f, ensure_ascii=False)
    for k in kinds:
        mine = [r for r in rows if r[0] == k]
        print(f"  {k}: {sum(1 for _, t, o, _ in mine if t == o)}/{len(mine)} right")
    print("->", args.out)
    return 0


def compare(args):
    with open(args.before, encoding="utf-8") as f:
        a = json.load(f)["rows"]
    with open(args.after, encoding="utf-8") as f:
        b = json.load(f)["rows"]
    if [(r[0], r[1], r[3:]) for r in a] != [(r[0], r[1], r[3:]) for r in b]:  # (r[3]: the readings typed)
        sys.exit("the two runs typed different items or readings (other --words, or the dictionary changed)")
    for k in KINDS:
        pairs = [(ra[1], ra[2], rb[2]) for ra, rb in zip(a, b) if ra[0] == k]
        if not pairs:
            continue
        fixed = [p for p in pairs if p[2] == p[0] != p[1]]
        broken = [p for p in pairs if p[1] == p[0] != p[2]]
        print(f"{k}: {sum(1 for t, x, _ in pairs if t == x)} -> {sum(1 for t, _, y in pairs if t == y)} "
              f"of {len(pairs)} right, fixed {len(fixed)}, broken {len(broken)}")
        for what, lst in (("fixed", fixed), ("broken", broken)):
            for t, x, y in lst[:args.examples]:
                print(f"    {what}: {t}  ({x} -> {y})")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Typing with a copy of a memory, before and after a change.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("folder", help="a copy of the memory folder (user.json, dictionary.json, keys.json ...), or none")
    r.add_argument("out")
    r.add_argument("--rev", help="a git revision instead of this folder as it is now")
    r.add_argument("--kinds", default="words,after,docs", help=f"comma separated, of {','.join(KINDS)}")
    r.add_argument("--words", type=int, default=6000)
    r.add_argument("--corpus", type=int, default=1911, help="corpus sentences from each file (kind corpus)")
    r.add_argument("--set", nargs="+", default=["long_sentence=1", "autocorrect=1"],
                   help="settings k=v (default long_sentence=1 autocorrect=1; none: as they come)")
    r.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="default: half the CPUs")
    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    c.add_argument("--examples", type=int, default=5)
    args = ap.parse_args()
    return run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
