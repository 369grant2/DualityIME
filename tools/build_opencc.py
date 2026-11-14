"""OpenCC's tables for 簡體輸出 (#32) into data/opencc/, from a folder of OpenCC's data/dictionary files.

    py -3 tools/build_opencc.py <folder>

<folder> holds TSPhrases.txt, TSCharacters.txt, TWVariants.txt, TWVariantsRevPhrases.txt, TWPhrasesRev.txt,
TWPhrases.txt and OpenCC's LICENSE, downloaded from https://github.com/BYVoid/OpenCC (data/dictionary; a
COMMIT file with the commit they came from is copied into the header if there). This tool reads only that
folder: no network. It writes, in OpenCC's format (key<TAB>value value..., # comments):
  - the five tables as they are (TSPhrases, TSCharacters, TWVariantsRevPhrases, TWPhrasesRev, TWPhrases)
  - TWVariantsRev.txt: TWVariants.txt turned around (OpenCC makes it when it builds; not in its sources)
  - mainland_words.txt: the keys of TWPhrases.txt (Mainland words written in Traditional characters, 視頻)
    that the dictionary does not have, each with a reading made of every character's most common reading
    in the dictionary ("視頻<TAB>ㄕˋ-ㄆㄧㄣˊ"); words with a character it has no reading of, or not all Han,
    are left out and printed
  - data/LICENSE-OpenCC.txt (Apache-2.0)
Design: docs/superpowers/specs/2026-11-14-simplified-chinese-design.md.
"""
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

OUT = os.path.join(ROOT, "data", "opencc")
AS_THEY_ARE = ("TSPhrases.txt", "TSCharacters.txt", "TWVariantsRevPhrases.txt", "TWPhrasesRev.txt", "TWPhrases.txt")
HAN = re.compile(r"^[㐀-鿿豈-﫿]+$")


def rows(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n").split("\t") for line in f if line.strip() and not line.startswith("#")]


def header(name, note, commit):
    lines = ["# Open Chinese Convert (OpenCC) Dictionary", f"# File: {name}", note,
             "# License: Apache-2.0 (see LICENSE-OpenCC.txt)", "# Source: https://github.com/BYVoid/OpenCC"]
    if commit:
        lines.append(f"# Commit: {commit}")
    return "\n".join(lines) + "\n"


def best_char_readings(lexicon):
    """character -> its most common reading in the dictionary (single-character entries)."""
    best = {}
    for reading, lst in lexicon.entries.items():
        if "-" in reading:
            continue
        for p, s in lst:
            if len(p) == 1 and s > best.get(p, ("", -99.0))[1]:
                best[p] = (reading, s)
    return {p: r for p, (r, _) in best.items()}


def main(folder):
    from engine.lexicon import Lexicon
    commit = ""
    if os.path.exists(os.path.join(folder, "COMMIT")):
        with open(os.path.join(folder, "COMMIT"), encoding="utf-8") as f:
            commit = f.read().strip()
    os.makedirs(OUT, exist_ok=True)
    for name in AS_THEY_ARE:
        shutil.copyfile(os.path.join(folder, name), os.path.join(OUT, name))
        print(f"{name}: {len(rows(os.path.join(OUT, name)))} rows")
    turned = {}
    for key, values in rows(os.path.join(folder, "TWVariants.txt")):
        for v in values.split():
            turned.setdefault(v, key)
    with open(os.path.join(OUT, "TWVariantsRev.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(header("TWVariantsRev.txt", "# Made by tools/build_opencc.py: TWVariants.txt turned around", commit))
        for v, key in turned.items():
            f.write(f"{v}\t{key}\n")
    print(f"TWVariantsRev.txt: {len(turned)} rows")
    lexicon = Lexicon()
    known = {p for lst in lexicon.entries.values() for p, _ in lst}
    chars = best_char_readings(lexicon)
    words, skipped = [], []
    for key, _ in rows(os.path.join(folder, "TWPhrases.txt")):
        if key in known:
            continue
        if not HAN.match(key) or any(ch not in chars for ch in key):
            skipped.append(key)
            continue
        words.append((key, "-".join(chars[ch] for ch in key)))
    with open(os.path.join(OUT, "mainland_words.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(header("mainland_words.txt", "# Made by tools/build_opencc.py: TWPhrases.txt's keys the dictionary "
                       "lacks, each character's most common reading", commit))
        for word, reading in words:
            f.write(f"{word}\t{reading}\n")
    print(f"mainland_words.txt: {len(words)} words; left out ({len(skipped)}): {' '.join(skipped)}")
    shutil.copyfile(os.path.join(folder, "LICENSE"), os.path.join(ROOT, "data", "LICENSE-OpenCC.txt"))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
