// 簡體輸出 (#32): Traditional (Taiwan) text to Simplified, and the Mainland words for Taiwan ones, from OpenCC's
// tables in data/opencc (tools/build_opencc.py). Same as engine/script.py.
#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "util.h"

namespace sz {

class Script {
public:
    // data_dir: the folder of lexicon.txt (the tables are in its opencc folder); no files: converts nothing
    void load(const std::wstring& data_dir);

    ustr to_simplified(const ustr& text) const;                  // 軟體 -> 软体 (characters only)
    std::vector<ustr> mainland_terms(const ustr& phrase) const;   // 影片 -> {視頻} (Traditional characters)
    bool is_mainland(const ustr& word) const { return mainland_.count(word) > 0; }
    // Mainland words (Traditional characters) the dictionary lacks -> reading key (主板 -> ㄓㄨˇ-ㄅㄢˇ)
    const std::unordered_map<ustr, ustr>& mainland_words() const { return words_; }
    const std::vector<ustr>& mainland_for_reading(const ustr& reading_key) const;  // ㄍㄨㄤ-ㄑㄩ -> {光驅}

private:
    struct Step {
        size_t longest = 0;
        std::vector<std::unordered_map<ustr, ustr>> tables;  // in order: on a tie the earlier wins
    };
    std::vector<Step> steps_;
    std::unordered_map<ustr, std::vector<ustr>> rev_;
    std::unordered_map<ustr, ustr> words_;
    std::unordered_map<ustr, std::vector<ustr>> by_reading_;
    std::unordered_set<ustr> mainland_;
};

}  // namespace sz
