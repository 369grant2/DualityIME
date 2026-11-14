"""簡體輸出 (#32): Traditional (Taiwan) text to Simplified, and the Mainland words for Taiwan ones, from OpenCC's
tables in data/opencc (tools/build_opencc.py). The engines choose in Traditional; this only turns what is shown
and sent. Design: docs/superpowers/specs/2026-11-14-simplified-chinese-design.md.
"""
import os

# tw2sp's order without its first table (TWPhrasesRev: a Mainland word is offered, never put in by itself):
# Taiwan variants to standard Traditional, then Traditional to Simplified; phrases before characters
STEPS = (("TWVariantsRevPhrases.txt", "TWVariantsRev.txt"), ("TSPhrases.txt", "TSCharacters.txt"))


def _read(path):
    """{key: [values]} of an OpenCC table, {} without the file."""
    table = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                key, _, values = line.rstrip("\n").partition("\t")
                if key and values:
                    table[key] = values.split()
    except OSError:
        pass
    return table


class Script:
    def __init__(self, steps, rev, mainland_words, mainland_keys):
        # each step: (longest key, [{key: value}] in order); at each place the longest key of the step's tables
        # wins, the earlier table on a tie
        self._steps = [(max((len(k) for t in tables for k in t), default=0), tables) for tables in steps]
        self._rev = rev
        self.mainland_words = mainland_words  # Mainland word (Traditional characters) the dictionary lacks -> reading
        self._mainland = mainland_keys
        self._by_reading = {}  # reading key -> those words, the file's order
        for word, reading in mainland_words.items():
            self._by_reading.setdefault(reading, []).append(word)

    @classmethod
    def load(cls, data_dir):
        folder = os.path.join(data_dir, "opencc")
        steps = [[{k: v[0] for k, v in _read(os.path.join(folder, name)).items()} for name in names]
                 for names in STEPS]
        words = {k: v[0] for k, v in _read(os.path.join(folder, "mainland_words.txt")).items()}
        keys = set(_read(os.path.join(folder, "TWPhrases.txt")))
        return cls(steps, _read(os.path.join(folder, "TWPhrasesRev.txt")), words, keys)

    def to_simplified(self, text):
        """text written in Simplified characters, words kept as they are (軟體 -> 软体)."""
        for longest, tables in self._steps:
            if not longest:
                continue
            out, i = [], 0
            while i < len(text):
                for n in range(min(longest, len(text) - i), 0, -1):
                    piece = text[i:i + n]
                    hit = next((t[piece] for t in tables if piece in t), None)
                    if hit is not None:
                        out.append(hit)
                        i += n
                        break
                else:
                    out.append(text[i])
                    i += 1
            text = "".join(out)
        return text

    def mainland_terms(self, phrase):
        """The Mainland words for a Taiwan one, Traditional characters, the table's order (影片 -> [視頻])."""
        return list(self._rev.get(phrase, ()))

    def mainland_for_reading(self, reading_key):
        """The Mainland words the dictionary lacks that read so (ㄍㄨㄤ-ㄑㄩ -> [光驅])."""
        return self._by_reading.get(reading_key, ())

    def is_mainland(self, word):
        """word is one of the Mainland words of the tables (視頻, 主板)."""
        return word in self._mainland
