"""The training text of the next student model (#41): Chinese characters with English kept as one mark.

    py -3 tools/lm/next/prepare_next.py [--limit-million N] [--jobs N]

Reads common.TRAIN (never common.EVAL) and common.WORK/vocab.txt (tools/lm/prepare.py: the characters the model
knows). Writes common.WORK/next/:
  vocab.txt   <unk> (0), <s> (1), <en> (2), then the characters of WORK/vocab.txt in their order
  train.u16, dev.u16   each paragraph (a line of a text) with 2+ units as <s> + its units (text.to_units: English and
              letters-and-digits runs one <en>, punctuation and spaces dropped without cutting the text), the last
              DEV_SHARE (tools/lm/prepare.py) of each file's row groups apart as the development set (as tools/lm/prepare.py)
  prepare.txt counts
"""
import multiprocessing
import os
import re
import sys
import unicodedata

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

UNK, BOS, EN_ID = 0, 1, 2
MARK = ""  # (an English run, replaced before the text becomes codes)
ALNUM = re.compile(r"[A-Za-z0-9]+(?:[-._][A-Za-z0-9]+)*")
HAN = [(0x3400, 0x9FFF), (0xF900, 0xFAFF)]

_lut = None


def _init(lut):
    global _lut
    _lut = lut


def encode_text(text, lut):
    """The ids of one text: each paragraph with 2+ units as BOS + its units (the same as text.to_units)."""
    t = ALNUM.sub(MARK, unicodedata.normalize("NFKC", text))
    codes = np.frombuffer(t.encode("utf-32-le"), dtype=np.uint32)
    keep = codes == ord(MARK)
    for lo, hi in HAN:
        keep |= (codes >= lo) & (codes <= hi)
    para = np.cumsum(codes == 10)[keep]   # (the paragraph of each unit kept)
    ids = lut[codes[keep]]
    if len(ids) == 0:
        return np.zeros(0, dtype=np.uint16)
    counts = np.bincount(para)
    good = counts[para] >= 2
    ids, para = ids[good], para[good]
    if len(ids) == 0:
        return np.zeros(0, dtype=np.uint16)
    starts = np.flatnonzero(np.concatenate(([True], para[1:] != para[:-1])))
    return np.insert(ids, starts, BOS).astype(np.uint16)


def _tools():
    """tools/lm on the path (only when run: tests import this file for encode_text alone)."""
    if os.path.dirname(HERE) not in sys.path:
        sys.path.insert(0, os.path.dirname(HERE))


def encode_job(job):
    _tools()
    from prepare import read_group
    path, i = job
    parts = [encode_text(text, _lut) for text in read_group(path, i) if text]
    parts = [p for p in parts if len(p)]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint16)


def main(argv):
    _tools()
    sys.stdout.reconfigure(encoding="utf-8")
    from common import TRAIN, EVAL, WORK
    from prepare import groups
    limit = float(argv[argv.index("--limit-million") + 1]) if "--limit-million" in argv else 0
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else max(1, (os.cpu_count() or 2) // 2)
    assert not set(TRAIN) & set(EVAL.values())
    old = open(os.path.join(WORK, "vocab.txt"), encoding="utf-8").read().rstrip("\n").split("\n")
    assert old[:2] == ["<unk>", "<s>"], "WORK/vocab.txt: run tools/lm/prepare.py first"
    vocab = ["<unk>", "<s>", "<en>"] + old[2:]
    lut = np.zeros(0x110000, dtype=np.uint16)  # (UNK for all the rest)
    for i, ch in enumerate(vocab[3:], 3):
        lut[ord(ch)] = i
    lut[ord(MARK)] = EN_ID
    out_dir = os.path.join(WORK, "next")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "vocab.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(vocab) + "\n")
    work = []
    for path in TRAIN:
        gs = groups(path)
        if limit:  # (a quick run: the first groups of each file, and its development groups)
            train = [g for g in gs if not g[1]][:max(1, int(limit))]
            gs = train + [g for g in gs if g[1]]
        work += [(path, i, dev) for i, dev in gs]
    report = []
    for name, want_dev in (("train", False), ("dev", True)):
        n = en = bos = 0
        with open(os.path.join(out_dir, name + ".u16"), "wb") as f, \
                multiprocessing.Pool(jobs, initializer=_init, initargs=(lut,)) as pool:
            for ids in pool.imap(encode_job, [(p, i) for p, i, dev in work if dev == want_dev], chunksize=2):
                ids.tofile(f)
                n += len(ids)
                en += int((ids == EN_ID).sum())
                bos += int((ids == BOS).sum())
        units = n - bos
        report.append(f"{name}.u16: {n} ids, {bos} paragraphs, {units} units, <en> {en} ({en / max(1, units):.2%})")
        print(report[-1], flush=True)
    report.append(f"vocabulary {len(vocab)} (WORK/vocab.txt + <en>)")
    with open(os.path.join(out_dir, "prepare.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
