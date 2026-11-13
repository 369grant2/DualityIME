#include "script.h"

#include <algorithm>

#include "json.h"

namespace sz {

namespace {

// {key: values} of an OpenCC table ("key<TAB>value value", # comments); {} without the file
std::unordered_map<ustr, std::vector<ustr>> read_table(const std::wstring& path) {
    std::unordered_map<ustr, std::vector<ustr>> table;
    std::string text;
    if (!read_file(path, text)) return table;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t nl = text.find('\n', pos);
        if (nl == std::string::npos) nl = text.size();
        std::string_view line(text.data() + pos, nl - pos);
        pos = nl + 1;
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        if (line.empty() || line[0] == '#') continue;
        size_t tab = line.find('\t');
        if (tab == std::string_view::npos || tab == 0) continue;
        std::vector<ustr> values;
        std::string_view rest = line.substr(tab + 1);
        while (!rest.empty()) {
            size_t sp = rest.find(' ');
            std::string_view v = rest.substr(0, sp);
            if (!v.empty()) values.push_back(from_utf8(v));
            if (sp == std::string_view::npos) break;
            rest.remove_prefix(sp + 1);
        }
        if (!values.empty()) table.emplace(from_utf8(line.substr(0, tab)), std::move(values));
    }
    return table;
}

}  // namespace

void Script::load(const std::wstring& data_dir) {
    // tw2sp's order without TWPhrasesRev (a Mainland word is offered, never put in by itself): Taiwan
    // variants to standard Traditional, then Traditional to Simplified; phrases before characters
    static const wchar_t* const kSteps[][2] = {{L"TWVariantsRevPhrases.txt", L"TWVariantsRev.txt"},
                                               {L"TSPhrases.txt", L"TSCharacters.txt"}};
    std::wstring folder = data_dir;
    if (!folder.empty() && folder.back() != L'\\' && folder.back() != L'/') folder += L"\\";
    folder += L"opencc\\";
    steps_.clear();
    for (auto& names : kSteps) {
        Step step;
        for (const wchar_t* name : names) {
            std::unordered_map<ustr, ustr> t;
            for (auto& [k, v] : read_table(folder + name)) {
                step.longest = std::max(step.longest, k.size());
                t.emplace(k, v[0]);
            }
            step.tables.push_back(std::move(t));
        }
        steps_.push_back(std::move(step));
    }
    rev_ = read_table(folder + L"TWPhrasesRev.txt");
    words_.clear();
    for (auto& [k, v] : read_table(folder + L"mainland_words.txt")) words_.emplace(k, v[0]);
    mainland_.clear();
    for (auto& [k, v] : read_table(folder + L"TWPhrases.txt")) mainland_.insert(k);
}

ustr Script::to_simplified(const ustr& in) const {
    ustr text = in;
    for (const auto& step : steps_) {
        if (!step.longest) continue;
        ustr out;
        size_t i = 0;
        while (i < text.size()) {
            bool hit = false;
            for (size_t n = std::min(step.longest, text.size() - i); n > 0 && !hit; --n) {
                ustr piece = text.substr(i, n);
                for (const auto& t : step.tables) {
                    auto it = t.find(piece);
                    if (it != t.end()) {
                        out += it->second;
                        i += n;
                        hit = true;
                        break;
                    }
                }
            }
            if (!hit) out += text[i++];
        }
        text = std::move(out);
    }
    return text;
}

std::vector<ustr> Script::mainland_terms(const ustr& phrase) const {
    auto it = rev_.find(phrase);
    return it == rev_.end() ? std::vector<ustr>() : it->second;
}

}  // namespace sz
