// Korean (한글) typed on the same keys, 두벌식 (the standard Korean layout): the third language of
// the mixed mode. The keys of a run of letters are put together into syllables the way a Korean
// keyboard does, and the tokenizer (composer.cpp) weighs the run as Korean against English and
// zhuyin: a word of data/korean.txt is cheap, like a word of the English list.
#pragma once
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "util.h"

namespace sz {

// one thing the keys make: a syllable (안), or a letter on its own (ㄴ, ㅏ: nothing to join yet)
struct HangulUnit {
    int start = 0, length = 0;  // its keys in the run
    char32_t ch = 0;
    bool syllable = false;
};

// The keys as Korean, or nothing if one of them is no key of the layout (a letter of the other
// case, a digit, a mark). Greedy as a Korean keyboard is: a consonant after a vowel ends the
// syllable unless a vowel follows it (dkssud: 안녕, the second s starts 녕).
std::optional<std::vector<HangulUnit>> hangul_units(std::string_view keys);
// the keys that type this text, or "" if it is not all Hangul syllables
std::string hangul_keys(const ustr& text);

// data/korean.txt: a word and how often it is used, by the keys that type it
class Korean {
public:
    explicit Korean(const std::wstring& path);
    bool loaded() const { return !sorted_.empty(); }
    // log10 of the word's share of the text counted, if these keys type a word of the list
    std::optional<double> score(const std::string& keys) const;
    // some word of the list starts with these keys (the word being typed)
    bool starts_word(const std::string& keys) const;
    // the longest start of these keys that types a word of the list (0: none)
    size_t known_start(const std::string& keys) const;

private:
    std::unordered_map<std::string, double> words_;
    std::vector<std::string> sorted_;
};

// What the tokenizer goes by, set by the Composer before it reads the keys (one typist a thread)
struct KoreanContext {
    bool on = true;      // the setting "korean"
    bool after = false;  // what was just sent ended in Korean: the next short word is likelier Korean too
};
KoreanContext& korean_context();

}  // namespace sz
