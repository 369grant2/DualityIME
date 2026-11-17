"""data/after_english.txt (#42): the character usually typed right after an English word, for the syllable typed there
(CP + ㄓˊ is 值, not the syllable's most common 直; T + ㄒㄩˋ is 恤). Both engines give it a bonus there (LEAD_BONUS) when
nothing is between the English and the zhuyin.

    py -3 tools/build_after_english.py [--min 20] [--share 0.8] [--out data/after_english.txt]

Counts, in the training files of tools/lm/common.TRAIN (never the evaluation ones), each run of ASCII letters (not after a
letter or digit) followed at once by a Chinese character, lower-cased; a character that makes a dictionary word with the
next one is left out (the table is for the character typed alone after the English: YouTube已經 would teach YouTube已 for
YouTube以). For each word and syllable (the character's best reading, as tools/evaluate.py types it), the character most
often there goes in when the word was seen with that syllable --min times and that character is --share of them, and it
is not the syllable's first candidate already, it is among the PER_SPAN candidates the engine tries for the syllable (else
no input could ever give it), and it is no Simplified-only character (the corpus mixes scripts: 网 for AV + ㄨㄤˇ wrote
AV网 in Traditional - /code-review): the Simplified form of another character in OpenCC's TSCharacters.txt and in fewer
than SIMPLIFIED_WORDS words of the dictionary (克 制 周 恤 板 stay: Traditional characters too, in many words). Measured (docs/superpowers/items/42-context-across-english.md): English with
one character after it, 4000 held-out sentences, written right 49.5% -> 57.2%, 308 wrong->right, 3 right->wrong.

Lines: word TAB reading TAB character, sorted; lines starting with # are comments.
Environment: DUALITYIME_CORPUS (tools/lm/common.py). About 3 minutes.
"""
import argparse
import collections
import os
import re
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "lm"))

PAIR = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]+)([一-鿿])(?=(.?))")
SIMPLIFIED_WORDS = 3  # a Simplified form of another character in fewer dictionary words than this is Simplified-only (网 0,
                      # 体 2; 范 5, 恤 16, 后 36 are Traditional characters too)


def simplified_only(lexicon):
    """The characters that are the Simplified form of another one (data/opencc/TSCharacters.txt) and in fewer than
    SIMPLIFIED_WORDS words of 2+ characters of the dictionary."""
    simp = set()
    with open(os.path.join(ROOT, "data", "opencc", "TSCharacters.txt"), encoding="utf-8") as f:
        for line in f:
            trad, _, forms = line.rstrip("\n").partition("\t")
            simp.update(s for s in forms.split() if s != trad)
    words = collections.Counter()
    for lst in lexicon.entries.values():
        for p, _ in lst:
            if len(p) >= 2:
                words.update(set(p) & simp)
    return {c for c in simp if words[c] < SIMPLIFIED_WORDS}


def count_pairs(paths, words):
    """Counter of (word lower-cased, character) in these corpus files, words: the dictionary's (a character starting
    one with the next is left out)."""
    import pyarrow.parquet as pq
    from common import text_column
    counts = collections.Counter()
    for path in paths:
        pf = pq.ParquetFile(path)
        col = text_column(pf)
        for g in range(pf.num_row_groups):
            for text in pf.read_row_group(g, columns=[col]).column(0).to_pylist():
                if not text:
                    continue
                for m in PAIR.finditer(unicodedata.normalize("NFKC", text)):
                    if not (m.group(3) and m.group(2) + m.group(3) in words):
                        counts[(m.group(1).lower(), m.group(2))] += 1
    return counts


def build(counts, best, tried, least, share, leave_out=frozenset()):
    """{(word, reading): character} from the counts; best: character -> its best reading; tried: reading -> the
    syllable's candidates the engine tries, first first; leave_out: characters never put in."""
    by = collections.defaultdict(collections.Counter)
    for (w, c), n in counts.items():
        r = best.get(c)
        if r and "-" not in r:
            by[(w, r)][c] += n
    table = {}
    for (w, r), cs in by.items():
        total = sum(cs.values())
        c, n = cs.most_common(1)[0]
        cands = tried(r)
        if total >= least and n >= share * total and cands and c != cands[0] and c in cands and c not in leave_out:
            table[(w, r)] = c
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=20)
    ap.add_argument("--share", type=float, default=0.8)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "after_english.txt"))
    args = ap.parse_args()
    from common import TRAIN
    from engine.composer import PER_SPAN, phrase_candidates
    from engine.lexicon import Lexicon
    from evaluate_lm import best_readings
    lex = Lexicon()
    best = best_readings(lex)
    cache = {}

    def tried(reading):
        if reading not in cache:
            cache[reading] = [p for p, _ in phrase_candidates([reading], lex)[:PER_SPAN]]
        return cache[reading]

    table = build(count_pairs(TRAIN, best), best, tried, args.min, args.share, simplified_only(lex))
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# the character usually right after an English word, by syllable (tools/build_after_english.py "
                f"--min {args.min} --share {args.share})\n")
        for (w, r), c in sorted(table.items()):
            f.write(f"{w}\t{r}\t{c}\n")
    print(f"{len(table)} entries -> {args.out}")


if __name__ == "__main__":
    main()
