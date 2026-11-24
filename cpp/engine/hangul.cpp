#include "hangul.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "json.h"  // read_file

namespace sz {

namespace {

// 두벌식: the key of each letter. Consonants by their place as a syllable's first letter (초성),
// vowels as its middle one (중성); a capital only where Shift makes another letter (ㄲ ㄸ ㅃ ㅆ ㅉ ㅒ ㅖ).
constexpr int NONE = -1;

struct Jamo {
    bool vowel = false;
    int index = NONE;  // 초성 0-18, or 중성 0-20
};

Jamo jamo_of(char key) {
    switch (key) {
        // 초성: ㄱ ㄲ ㄴ ㄷ ㄸ ㄹ ㅁ ㅂ ㅃ ㅅ ㅆ ㅇ ㅈ ㅉ ㅊ ㅋ ㅌ ㅍ ㅎ
        case 'r': return {false, 0};
        case 'R': return {false, 1};
        case 's': return {false, 2};
        case 'e': return {false, 3};
        case 'E': return {false, 4};
        case 'f': return {false, 5};
        case 'a': return {false, 6};
        case 'q': return {false, 7};
        case 'Q': return {false, 8};
        case 't': return {false, 9};
        case 'T': return {false, 10};
        case 'd': return {false, 11};
        case 'w': return {false, 12};
        case 'W': return {false, 13};
        case 'c': return {false, 14};
        case 'z': return {false, 15};
        case 'x': return {false, 16};
        case 'v': return {false, 17};
        case 'g': return {false, 18};
        // 중성: ㅏ ㅐ ㅑ ㅒ ㅓ ㅔ ㅕ ㅖ ㅗ (ㅘ ㅙ ㅚ) ㅛ ㅜ (ㅝ ㅞ ㅟ) ㅠ ㅡ (ㅢ) ㅣ
        case 'k': return {true, 0};
        case 'o': return {true, 1};
        case 'i': return {true, 2};
        case 'O': return {true, 3};
        case 'j': return {true, 4};
        case 'p': return {true, 5};
        case 'u': return {true, 6};
        case 'P': return {true, 7};
        case 'h': return {true, 8};
        case 'y': return {true, 12};
        case 'n': return {true, 13};
        case 'b': return {true, 17};
        case 'm': return {true, 18};
        case 'l': return {true, 20};
    }
    return {};
}

// two vowel keys as one vowel (ㅗ + ㅏ = ㅘ), or NONE
int joined_vowel(int a, int b) {
    if (a == 8) return b == 0 ? 9 : b == 1 ? 10 : b == 20 ? 11 : NONE;    // ㅘ ㅙ ㅚ
    if (a == 13) return b == 4 ? 14 : b == 5 ? 15 : b == 20 ? 16 : NONE;  // ㅝ ㅞ ㅟ
    if (a == 18) return b == 20 ? 19 : NONE;                              // ㅢ
    return NONE;
}

// a consonant as a syllable's last letter (종성 1-27), or NONE (ㄸ ㅃ ㅉ end none)
int final_of(int cho) {
    static const int table[19] = {1, 2, 4, 7, NONE, 8, 16, 17, NONE, 19, 20, 21, 22, NONE, 23, 24, 25, 26, 27};
    return table[cho];
}

// two consonants as one last letter (ㄱ + ㅅ = ㄳ), or NONE
int joined_final(int first, int cho) {
    switch (first) {
        case 1: return cho == 9 ? 3 : NONE;                                // ㄳ
        case 4: return cho == 12 ? 5 : cho == 18 ? 6 : NONE;               // ㄵ ㄶ
        case 8:                                                            // ㄺ ㄻ ㄼ ㄽ ㄾ ㄿ ㅀ
            return cho == 0 ? 9 : cho == 6 ? 10 : cho == 7 ? 11 : cho == 9 ? 12 : cho == 16 ? 13 : cho == 17 ? 14
                   : cho == 18 ? 15 : NONE;
        case 17: return cho == 9 ? 18 : NONE;                              // ㅄ
    }
    return NONE;
}

// the letters on their own (호환용 자모): a consonant, a vowel
char32_t lone(const Jamo& j) {
    static const char32_t consonants[19] = {0x3131, 0x3132, 0x3134, 0x3137, 0x3138, 0x3139, 0x3141, 0x3142, 0x3143, 0x3145,
                                            0x3146, 0x3147, 0x3148, 0x3149, 0x314A, 0x314B, 0x314C, 0x314D, 0x314E};
    return j.vowel ? static_cast<char32_t>(0x314F + j.index) : consonants[j.index];
}

const char* const kChoKeys[19] = {"r", "R", "s", "e", "E", "f", "a", "q", "Q", "t", "T", "d", "w", "W", "c", "z", "x", "v", "g"};
const char* const kJungKeys[21] = {"k", "o", "i", "O", "j", "p", "u", "P", "h", "hk", "ho", "hl", "y", "n", "nj", "np", "nl",
                                   "b", "m", "ml", "l"};
const char* const kJongKeys[28] = {"", "r", "R", "rt", "s", "sw", "sg", "e", "f", "fr", "fa", "fq", "ft", "fx", "fv", "fg",
                                   "a", "q", "qt", "t", "T", "d", "w", "c", "z", "x", "v", "g"};

}  // namespace

std::optional<std::vector<HangulUnit>> hangul_units(std::string_view keys) {
    std::vector<Jamo> js;
    js.reserve(keys.size());
    for (char k : keys) {
        Jamo j = jamo_of(k);
        if (j.index == NONE) return std::nullopt;
        js.push_back(j);
    }
    std::vector<HangulUnit> out;
    const size_t n = js.size();
    auto consonant = [&](size_t i) { return i < n && !js[i].vowel; };
    auto vowel = [&](size_t i) { return i < n && js[i].vowel; };
    size_t i = 0;
    while (i < n) {
        const size_t start = i;
        if (!(consonant(i) && vowel(i + 1))) {  // nothing to join it to
            out.push_back({static_cast<int>(start), 1, lone(js[i]), false});
            ++i;
            continue;
        }
        int cho = js[i].index, jung = js[i + 1].index, jong = 0;
        i += 2;
        if (vowel(i))
            if (int both = joined_vowel(jung, js[i].index); both != NONE) {
                jung = both;
                ++i;
            }
        // a consonant ends the syllable unless a vowel follows it: then it starts the next one
        if (consonant(i) && !vowel(i + 1) && final_of(js[i].index) != NONE) {
            jong = final_of(js[i].index);
            ++i;
            if (consonant(i) && !vowel(i + 1))
                if (int both = joined_final(jong, js[i].index); both != NONE) {
                    jong = both;
                    ++i;
                }
        }
        out.push_back({static_cast<int>(start), static_cast<int>(i - start),
                       static_cast<char32_t>(0xAC00 + (cho * 21 + jung) * 28 + jong), true});
    }
    return out;
}

std::string hangul_keys(const ustr& text) {
    std::string keys;
    for (char32_t c : text) {
        if (c < 0xAC00 || c > 0xD7A3) return "";
        int s = static_cast<int>(c - 0xAC00);
        keys += kChoKeys[s / (21 * 28)];
        keys += kJungKeys[s / 28 % 21];
        keys += kJongKeys[s % 28];
    }
    return keys;
}

Korean::Korean(const std::wstring& path) {
    std::string text;
    if (!read_file(path, text)) return;
    // each line: the word, a space, how many times it was counted
    std::vector<std::pair<std::string, double>> counted;
    double total = 0;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t nl = text.find('\n', pos);
        if (nl == std::string::npos) nl = text.size();
        std::string line = strip(std::string_view(text).substr(pos, nl - pos));
        pos = nl + 1;
        size_t sp = line.find(' ');
        if (line.empty() || line[0] == '#' || sp == std::string::npos) continue;
        std::string keys = hangul_keys(from_utf8(std::string_view(line).substr(0, sp)));
        double count = std::atof(line.c_str() + sp + 1);
        if (keys.empty() || count <= 0) continue;
        counted.emplace_back(std::move(keys), count);
        total += count;
    }
    for (auto& [keys, count] : counted) {
        if (words_.emplace(keys, std::log10(count / total)).second) sorted_.push_back(keys);
    }
    std::sort(sorted_.begin(), sorted_.end());
}

std::optional<double> Korean::score(const std::string& keys) const {
    auto it = words_.find(keys);
    if (it == words_.end()) return std::nullopt;
    return it->second;
}

bool Korean::starts_word(const std::string& keys) const {
    auto it = std::lower_bound(sorted_.begin(), sorted_.end(), keys);
    return it != sorted_.end() && it->compare(0, keys.size(), keys) == 0;
}

size_t Korean::known_start(const std::string& keys) const {
    for (size_t len = keys.size(); len > 0; --len)
        if (words_.count(keys.substr(0, len))) return len;
    return 0;
}

KoreanContext& korean_context() {
    thread_local KoreanContext context;
    return context;
}

}  // namespace sz
