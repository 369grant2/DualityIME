// The character language models (長句選字加強), read from data/lm.bin (and lm_plus.bin, lm_max.bin): integer-only
// inference, the same operations in the same order as engine/lm.py (version 1, CharLM) and engine/lm_next.py (version
// 2, NextLM), so both engines give exactly the same scores (tools/compare.py, tools/lm/check_cpp.py). The file layouts
// are in those two Python files.
#pragma once
#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "util.h"

namespace sz {

class Lexicon;
struct NextNode;

// What a model has read so far: a version 1 model's GRU state (Q10 [layers * hidden], within -1..1, so int16), or a
// version 2 model's position (its node: K and V of every layer, linked to the one before, never copied).
struct LmState {
    std::vector<int16_t> h;
    std::shared_ptr<const NextNode> node;
};

class LmModel {
public:
    virtual ~LmModel() = default;
    // the model in the file (version 1 or 2), or nullptr (no file, another format, wrong sizes, cut short)
    static std::unique_ptr<LmModel> load(const std::wstring& path);

    virtual LmState start() const = 0;                          // after <s> (id 1)
    virtual LmState step(const LmState& state, int c) const = 0;  // after character id c
    // step() for each state and character id at once (the same results)
    virtual std::vector<LmState> steps(const std::vector<const LmState*>& states, const std::vector<int>& cs) const;
    // Q10 natural log probability of each candidate id, among the candidates alone (<= 0)
    virtual std::vector<int64_t> scores(const LmState& state, const std::vector<int>& cands) const = 0;
    // Q16 probability that the next character is one of ids, over the whole vocabulary (integers only, the exp table)
    virtual int64_t mass_q16(const LmState& state, const std::vector<int>& ids) const = 0;
    virtual int id(char32_t ch) const = 0;  // the character's vocabulary id, 0 (<unk>) if the model does not know it
    virtual int vocab() const = 0;
    virtual bool avx2() const = 0;  // the vector path (off: no AVX2, or DUALITYIME_LM_SCALAR=1)
    // #41: a version 2 model reads EN_MARK (an English or letters-and-digits run) and brings its own weight (< 0: the
    // engine's LM_WEIGHT) and order thresholds (nullptr: the engine's table)
    virtual bool knows_english() const { return false; }
    virtual double lm_weight() const { return -1.0; }
    virtual const std::unordered_map<ustr, int64_t>* order_thresholds() const { return nullptr; }
    virtual int context_max() const { return 0; }  // the context it was made for (0: no limit; #41)
    virtual size_t state_bytes() const = 0;        // about what one state keeps in memory (LmScorer's budget)
};

// Version 1: a GRU over characters (engine/lm.py CharLM)
class CharLM : public LmModel {
public:
    static std::unique_ptr<CharLM> load(const std::wstring& path);

    LmState start() const override;
    LmState step(const LmState& state, int c) const override;
    std::vector<LmState> steps(const std::vector<const LmState*>& states, const std::vector<int>& cs) const override;
    std::vector<int64_t> scores(const LmState& state, const std::vector<int>& cands) const override;
    int64_t mass_q16(const LmState& state, const std::vector<int>& ids) const override;
    int id(char32_t ch) const override;
    int vocab() const override { return V_; }
    bool avx2() const override { return avx2_; }
    size_t state_bytes() const override { return static_cast<size_t>(layers_) * H_ * 2 + 64; }

private:
    CharLM() = default;
    struct Layer {
        int n_in = 0;
        std::vector<int8_t> w_ih_q, w_hh_q;  // [3H, n_in], [3H, H]
        std::vector<int32_t> w_ih_m, w_hh_m, b_ih, b_hh;
    };
    std::vector<int64_t> logits(const LmState& state, const std::vector<int>& cands) const;
    // int8 row x int16 vector (an int32 sum, exact within the export's bounds)
    int32_t dot(const int8_t* row, const int16_t* x, int n) const;
    void matvec(const std::vector<int8_t>& w, const std::vector<int32_t>& m, const int16_t* x, int rows, int n,
                std::vector<int64_t>& out) const;
    void matvecs(const std::vector<int8_t>& w, const std::vector<int32_t>& m, const std::vector<const int16_t*>& xs,
                 int rows, int n, std::vector<std::vector<int64_t>>& outs) const;

    int V_ = 0, E_ = 0, H_ = 0, layers_ = 0;
    bool avx2_ = false;
    std::unordered_map<char32_t, int> ids_;
    std::vector<int8_t> emb_q_, proj_q_;  // [V, E], [E, H]
    std::vector<int32_t> emb_m_, bias_, proj_m_, proj_b_;
    std::vector<Layer> layer_;
    std::vector<int32_t> sigmoid_, tanh_, exp_, log_;
    mutable std::vector<int> all_;  // 0 .. V-1 (mass_q16), made on first use
};

// Version 2: the next student (#41), a small Transformer (arch 1) or a GRU with an attention layer (arch 2)
// (engine/lm_next.py NextLM: every operation and the file layout there)
class NextLM : public LmModel {
public:
    static std::unique_ptr<NextLM> load(const std::wstring& path);

    LmState start() const override;
    LmState step(const LmState& state, int c) const override;
    std::vector<int64_t> scores(const LmState& state, const std::vector<int>& cands) const override;
    int64_t mass_q16(const LmState& state, const std::vector<int>& ids) const override;
    int id(char32_t ch) const override;
    int vocab() const override { return V_; }
    bool avx2() const override { return avx2_; }
    bool knows_english() const override { return true; }
    double lm_weight() const override { return weight_; }
    const std::unordered_map<ustr, int64_t>* order_thresholds() const override { return &thresholds_; }
    int context_max() const override { return context_max_; }
    size_t state_bytes() const override;

private:
    NextLM() = default;
    struct Linear {  // int8 rows [rows, n] with a multiplier each, and a bias (Q10)
        int rows = 0, n = 0;
        std::vector<int8_t> q;
        std::vector<int32_t> m, b;
    };
    struct GruLayer {
        int n_in = 0;
        std::vector<int8_t> w_ih_q, w_hh_q;
        std::vector<int32_t> w_ih_m, w_hh_m, b_ih, b_hh;
    };
    struct Block {  // arch 1: one layer; arch 2 uses n1, att (qkv, out), n2 alone
        std::vector<int32_t> n1, n2;
        Linear qkv, out, up, down;
    };
    std::vector<int64_t> matvec(const Linear& l, const std::vector<int64_t>& x) const;
    std::vector<int64_t> rmsnorm(const std::vector<int64_t>& x, const std::vector<int32_t>& g) const;
    // the attention of the position after prev, reading every position up to it: its k and v, and the output
    void attention(const std::vector<int64_t>& x, const Block& blk, const NextNode* prev, int layer,
                   std::vector<int32_t>& k, std::vector<int32_t>& v, std::vector<int64_t>& out) const;
    std::vector<int64_t> embed(int c) const;
    std::vector<int64_t> logits(const LmState& state, const std::vector<int>& cands) const;
    int32_t dot(const int8_t* row, const int16_t* x, int n) const;

    int arch_ = 0, V_ = 0, E_ = 0, D_ = 0, layers_ = 0, heads_ = 0, dh_ = 0;
    int64_t inv_sqrt_q16_ = 0;
    int context_max_ = 0;
    double weight_ = 0.6;
    bool avx2_ = false;
    std::unordered_map<char32_t, int> ids_;
    std::unordered_map<ustr, int64_t> thresholds_;
    std::vector<int8_t> emb_q_;
    std::vector<int32_t> emb_m_, bias_, norm_g_, slopes_q16_;
    std::vector<Block> blocks_;      // arch 1: one a layer; arch 2: one (its attention)
    std::vector<GruLayer> gru_;      // arch 2
    Linear proj_;                    // arch 2
    std::vector<int32_t> sigmoid_, tanh_, exp_, log_;
    mutable std::vector<int> all_;
};

// states (and scored syllables) kept across keys, then all dropped but the start (the same as
// engine/lm.py: a sentence of 47 syllables typed with 注音自動校正 on makes about 1100; a version 1 state is 3 KB)
constexpr size_t LM_STATES_KEPT = 3000;
// ...and the states within this many bytes (#41: a version 2 state is 15 KB for light, 55 KB for a max Transformer;
// what is kept changes the time, never a score - Task 18 measures the time against it)
constexpr size_t LM_STATES_BYTES = size_t(64) << 20;
// views (LmScorer::view) kept: the runs of one buffer read contexts of their own (#42), a full buffer about 33; their
// states share LM_STATES_KEPT (engine/lm.py)
constexpr size_t LM_VIEWS_KEPT = 64;
// #41: one English or letters-and-digits run in a context, for a model that knows English (engine/lm.py EN_MARK)
constexpr char32_t EN_MARK = 0xE000;

// The model's score of a text written for some syllables, as convert() uses it (engine/lm.py
// Scorer): each character among the characters its syllable may be (every one a word of the
// dictionary writes for it), the text from the start of the run of Chinese (<s>, nothing
// before it). States are kept by text: the hypotheses of one key and the keys after it share
// their beginnings (LM_STATES_KEPT, then started again). Not thread-safe (the engine handles
// one request at a time).
class LmScorer {
public:
    LmScorer(const LmModel& lm, const Lexicon& lexicon);
    // sorted ids of every character a word of the dictionary writes for the syllable (0 for one
    // the model does not know), or nullptr
    const std::vector<int>* candidates(const ustr& syllable) const;
    // Q10 log probability of ch after before among the syllable's characters; one not among them
    // gets the lowest of theirs, a syllable with no list 0
    int64_t char_score(const ustr& before, char32_t ch, const ustr& syllable) const;
    // the sum of the characters' scores (Q10), one character per syllable
    int64_t text_score(const ustr& text, const std::vector<ustr>& readings) const;
    // Q16 probability that the character after before is one of the syllable's (candidates, id 0 left
    // out), over the whole vocabulary; 0 for a syllable with none (engine/lm.py Scorer.syllable_mass_q16)
    int64_t syllable_mass_q16(const ustr& before, const ustr& syllable) const;
    // the states of every beginning of the texts, stepped together (text_score then finds them)
    void prepare(const std::vector<ustr>& texts) const;
    const LmModel& model() const { return lm_; }
    // 接續前文選字 (#37): a scorer of the same model whose texts come after context (<s> + context + text), with
    // caches of its own; the last LM_VIEWS_KEPT contexts asked for are kept and given again (engine/lm.py Scorer.view)
    const LmScorer* view(const ustr& context) const;

private:
    LmScorer(const LmScorer& base, const ustr& context);  // (view's)
    void clear() const;
    void room() const;
    LmState state(const ustr& text) const;

    const LmModel& lm_;
    const Lexicon& lexicon_;
    std::shared_ptr<const std::unordered_map<ustr, std::vector<int>>> cands_;  // (shared with the views)
    ustr context_;                    // a view's context ("" for the scorer itself)
    const LmScorer* base_ = nullptr;  // a view's scorer (its views share one budget: room)
    LmState start_;                   // <s>, then the context
    size_t kept_ = LM_STATES_KEPT;    // states kept: LM_STATES_KEPT, fewer for a large state (LM_STATES_BYTES)
    mutable std::list<std::unique_ptr<LmScorer>> views_;  // the views kept, the one used most lately first
    mutable std::unordered_map<ustr, LmState> states_;
    mutable std::unordered_map<ustr, std::vector<int64_t>> score_cache_;  // before + U'\0' + syllable
    mutable std::unordered_map<ustr, int64_t> mass_cache_;               // before + U'\0' + syllable
};

// 長句選字加強's models and scorers, one a tier (選字模型, #41: light data/lm.bin, plus lm_plus.bin, max lm_max.bin), each
// read from its file the first time it is asked for (with the setting off the engine never reads one: no memory, no
// time). Not thread-safe, as LmScorer. (engine/lm.py shared)
class LanguageModel {
public:
    LanguageModel(std::wstring data_dir, const Lexicon& lexicon) : dir_(std::move(data_dir)), lexicon_(lexicon) {}
    // the tier's scorer ("light", "plus", "max"); a tier without its file, one that cannot be read, or no tier at all
    // gives the light one's; nullptr when light has none either
    const LmScorer* get(const std::string& tier = "light") const;

private:
    struct Tier {
        bool tried = false;
        std::unique_ptr<LmModel> model;
        std::unique_ptr<LmScorer> scorer;  // (after model: it uses it)
    };
    std::wstring dir_;
    const Lexicon& lexicon_;
    mutable Tier tiers_[3];
};

}  // namespace sz
