#include "lm.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <set>
#include <unordered_set>

#include "lexicon.h"
#include "lm_detail.h"

namespace sz {

using namespace lmq;

namespace {

constexpr uint32_t VERSION = 1;
constexpr int64_t EMBED_MAX = 4096;  // tools/lm/export.py's bounds: the int32 sums cannot overflow

#ifdef SZ_LM_X86
bool cpu_has_avx2() {
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7) return false;
    __cpuid(info, 1);
    bool osxsave = (info[2] & (1 << 27)) != 0, avx = (info[2] & (1 << 28)) != 0;
    if (!osxsave || !avx || (_xgetbv(0) & 6) != 6) return false;  // (the OS saves the ymm registers)
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
}
#endif

}  // namespace

bool lmq::use_avx2() {
    const char* forced = std::getenv("DUALITYIME_LM_SCALAR");
    if (forced && std::strcmp(forced, "1") == 0) return false;
#ifdef SZ_LM_X86
    return cpu_has_avx2();
#else
    return false;
#endif
}

std::unique_ptr<LmModel> LmModel::load(const std::wstring& path) {
    char magic[4] = {};
    {
        File file;
        file.f = _wfopen(path.c_str(), L"rb");
        if (!file.f || !file.read(magic, 4)) return nullptr;
    }
    if (std::memcmp(magic, "DLM1", 4) == 0) return CharLM::load(path);
    if (std::memcmp(magic, "DLM2", 4) == 0) return NextLM::load(path);
    return nullptr;
}

std::vector<LmState> LmModel::steps(const std::vector<const LmState*>& states, const std::vector<int>& cs) const {
    std::vector<LmState> out;
    out.reserve(states.size());
    for (size_t b = 0; b < states.size(); ++b) out.push_back(step(*states[b], cs[b]));
    return out;
}

std::vector<int64_t> LmModel::unit_scores(const std::vector<const LmState*>& states, const std::vector<int>& ids) const {
    std::vector<int> every(static_cast<size_t>(vocab()));
    for (size_t k = 0; k < every.size(); ++k) every[k] = static_cast<int>(k);
    std::vector<int64_t> out;
    out.reserve(states.size());
    for (size_t b = 0; b < states.size(); ++b) out.push_back(scores(*states[b], every)[static_cast<size_t>(ids[b])]);
    return out;
}

std::vector<LmState> LmModel::tree(const std::vector<const LmState*>& prev, const std::vector<int>& link,
                                   const std::vector<int>& cs) const {
    // (depth by depth, each depth through steps(): a model's own batching of one step - the GRU's - as before)
    const size_t n = cs.size();
    std::vector<LmState> out(n);  // (sized once: the parents below point into it)
    std::vector<size_t> depth(n, 0);
    size_t deepest = 0;
    for (size_t b = 0; b < n; ++b) {
        if (link[b] >= 0) depth[b] = depth[static_cast<size_t>(link[b])] + 1;
        deepest = std::max(deepest, depth[b]);
    }
    for (size_t d = 0; n && d <= deepest; ++d) {
        std::vector<size_t> at;
        std::vector<const LmState*> parents;
        std::vector<int> ids;
        for (size_t b = 0; b < n; ++b)
            if (depth[b] == d) {
                at.push_back(b);
                parents.push_back(link[b] >= 0 ? &out[static_cast<size_t>(link[b])] : prev[b]);
                ids.push_back(cs[b]);
            }
        std::vector<LmState> next = steps(parents, ids);
        for (size_t k = 0; k < at.size(); ++k) out[at[k]] = std::move(next[k]);
    }
    return out;
}

std::vector<LmState> LmModel::run(const LmState& state, const std::vector<int>& cs) const {
    std::vector<LmState> out;
    out.reserve(cs.size());
    for (int c : cs) out.push_back(step(out.empty() ? state : out.back(), c));
    return out;
}

std::unique_ptr<CharLM> CharLM::load(const std::wstring& path) {
    File file;
    file.f = _wfopen(path.c_str(), L"rb");
    if (!file.f) return nullptr;
    if (_fseeki64(file.f, 0, SEEK_END) != 0) return nullptr;
    const int64_t size = _ftelli64(file.f);
    if (size < 0 || _fseeki64(file.f, 0, SEEK_SET) != 0) return nullptr;
    // the header first, and the whole file's size from it, before anything is allocated
    char magic[4];
    uint32_t h[10];  // version q mul_shift vocab emb hidden layers act_range exp_range log_bits
    uint32_t text_len = 0;
    if (!file.read(magic, 4) || std::memcmp(magic, "DLM1", 4) != 0 || !file.read(h, sizeof h) || !file.u32(text_len))
        return nullptr;
    const uint32_t mine[] = {VERSION, Q, MUL_SHIFT, 0, 0, 0, 0, ACT_RANGE, EXP_RANGE, LOG_BITS};
    for (int i : {0, 1, 2, 7, 8, 9})
        if (h[i] != mine[i]) return nullptr;
    const uint64_t V = h[3], E = h[4], H = h[5], layers = h[6];
    if (V < 2 || V > (1u << 22) || E < 1 || E > 4096 || H < 1 || H > 4096 || layers < 1 || layers > 16) return nullptr;
    // (the bounds the export checks: 32-bit sums of int8 weights x inputs never overflow)
    if (127 * EMBED_MAX * static_cast<int64_t>(E) >= INT32_LIMIT || 127 * ONE * static_cast<int64_t>(H) >= INT32_LIMIT)
        return nullptr;
    uint64_t expect = 4 + sizeof h + 4 + uint64_t(text_len);
    auto add = [&](uint64_t count, uint64_t item) { expect += 4 + count * item; };
    for (uint64_t count : {V * E}) add(count, 1);
    for (uint64_t count : {V, V}) add(count, 4);
    for (uint64_t l = 0; l < layers; ++l) {
        uint64_t n_in = l == 0 ? E : H;
        for (uint64_t count : {3 * H * n_in, 3 * H * H}) add(count, 1);
        for (int k = 0; k < 4; ++k) add(3 * H, 4);
    }
    add(E * H, 1);
    for (uint64_t count : {E, E, uint64_t(2 * ACT_RANGE + 1), uint64_t(2 * ACT_RANGE + 1), uint64_t(EXP_RANGE + 1),
                           uint64_t(LOG_SIZE)})
        add(count, 4);
    if (expect != static_cast<uint64_t>(size)) return nullptr;

    std::unique_ptr<CharLM> lm(new CharLM());
    lm->V_ = static_cast<int>(V), lm->E_ = static_cast<int>(E), lm->H_ = static_cast<int>(H);
    lm->layers_ = static_cast<int>(layers);
    std::string text(text_len, '\0');
    if (!file.read(text.data(), text.size())) return nullptr;
    // (valid UTF-8 only, as Python's decode: a file loads in both engines or in neither)
    ustr chars = from_utf8(text);
    if (to_utf8(chars) != text) return nullptr;  // (bytes that are not UTF-8, or written too long)
    for (char32_t cp : chars)
        if ((cp >= 0xD800 && cp <= 0xDFFF) || cp > 0x10FFFF) return nullptr;
    std::vector<ustr> vocab = split(chars, U'\n');
    if (vocab.size() != V) return nullptr;
    for (size_t i = 0; i < vocab.size(); ++i)
        if (vocab[i].size() == 1) lm->ids_[vocab[i][0]] = static_cast<int>(i);  // (later ones win, like a dict)
    bool ok = file.array(lm->emb_q_, V * E) && file.array(lm->emb_m_, V) && file.array(lm->bias_, V);
    lm->layer_.resize(static_cast<size_t>(layers));  // (all sizes checked small above: they fit size_t)
    for (size_t l = 0; ok && l < lm->layer_.size(); ++l) {
        Layer& y = lm->layer_[l];
        y.n_in = static_cast<int>(l == 0 ? E : H);
        ok = file.array(y.w_ih_q, 3 * H * y.n_in) && file.array(y.w_ih_m, 3 * H) && file.array(y.w_hh_q, 3 * H * H) &&
             file.array(y.w_hh_m, 3 * H) && file.array(y.b_ih, 3 * H) && file.array(y.b_hh, 3 * H);
    }
    ok = ok && file.array(lm->proj_q_, E * H) && file.array(lm->proj_m_, E) && file.array(lm->proj_b_, E) &&
         file.array(lm->sigmoid_, 2 * ACT_RANGE + 1) && file.array(lm->tanh_, 2 * ACT_RANGE + 1) &&
         file.array(lm->exp_, EXP_RANGE + 1) && file.array(lm->log_, LOG_SIZE);
    if (!ok) return nullptr;
    // int8 weights within -127..127 (the bounds behind the int32 sums; engine/lm.py checks the same)
    if (has_minus_128(lm->emb_q_) || has_minus_128(lm->proj_q_)) return nullptr;
    for (auto& y : lm->layer_)
        if (has_minus_128(y.w_ih_q) || has_minus_128(y.w_hh_q)) return nullptr;
    // what keeps the sums in 32 bits and the states within -1..1: embedded characters within
    // EMBED_MAX, gates within 0..1, tanh within -1..1; and the best candidate's exp() above 0
    for (size_t c = 0; c < static_cast<size_t>(V); ++c)
        for (size_t j = 0; j < static_cast<size_t>(E); ++j) {
            int64_t x = requant(lm->emb_q_[c * static_cast<size_t>(E) + j] * ONE, lm->emb_m_[c]);
            if (x > EMBED_MAX || x < -EMBED_MAX) return nullptr;
        }
    if (!tables_ok(lm->sigmoid_, lm->tanh_, lm->exp_)) return nullptr;
    lm->avx2_ = use_avx2();
    return lm;
}

int CharLM::id(char32_t ch) const {
    auto it = ids_.find(ch);
    return it == ids_.end() ? 0 : it->second;
}

int32_t CharLM::dot(const int8_t* row, const int16_t* x, int n) const { return lmq::dot(avx2_, row, x, n); }

void CharLM::matvec(const std::vector<int8_t>& w, const std::vector<int32_t>& m, const int16_t* x, int rows, int n,
                    std::vector<int64_t>& out) const {
    out.resize(rows);
    for (int r = 0; r < rows; ++r) out[r] = requant(dot(w.data() + static_cast<size_t>(r) * n, x, n), m[r]);
}

void CharLM::matvecs(const std::vector<int8_t>& w, const std::vector<int32_t>& m, const std::vector<const int16_t*>& xs,
                     int rows, int n, std::vector<std::vector<int64_t>>& outs) const {
    // (a row for every vector before the next row: the weights are read from memory once for all)
    outs.resize(xs.size());
    for (auto& out : outs) out.resize(rows);
    int r = 0;
#ifdef SZ_LM_X86
    if (avx2_)
        for (; r + 4 <= rows; r += 4) {
            const int8_t* rows4 = w.data() + static_cast<size_t>(r) * n;
            for (size_t b = 0; b < xs.size(); ++b) {
                int32_t acc[4];
                dot4_avx2(rows4, static_cast<size_t>(n), xs[b], n, acc);
                for (int i = 0; i < 4; ++i) outs[b][r + i] = requant(acc[i], m[r + i]);
            }
        }
#endif
    for (; r < rows; ++r) {
        const int8_t* row = w.data() + static_cast<size_t>(r) * n;
        for (size_t b = 0; b < xs.size(); ++b) outs[b][r] = requant(dot(row, xs[b], n), m[r]);
    }
}

LmState CharLM::start() const {
    LmState zero;
    zero.h.assign(static_cast<size_t>(layers_) * H_, 0);
    return step(zero, 1);
}

LmState CharLM::step(const LmState& state, int c) const { return std::move(steps({&state}, {c})[0]); }

std::vector<LmState> CharLM::steps(const std::vector<const LmState*>& states, const std::vector<int>& cs) const {
    const int H = H_;
    const size_t B = states.size();
    std::vector<std::vector<int16_t>> x(B, std::vector<int16_t>(E_));
    for (size_t b = 0; b < B; ++b)
        for (int j = 0; j < E_; ++j)
            x[b][j] = static_cast<int16_t>(requant(emb_q_[static_cast<size_t>(cs[b]) * E_ + j] * ONE, emb_m_[cs[b]]));
    std::vector<LmState> out(B);
    for (auto& o : out) o.h.resize(static_cast<size_t>(layers_) * H);
    std::vector<std::vector<int64_t>> gi, gh;
    std::vector<const int16_t*> xs(B), hs(B);
    for (int l = 0; l < layers_; ++l) {
        const Layer& y = layer_[l];
        for (size_t b = 0; b < B; ++b) {
            xs[b] = x[b].data();
            hs[b] = states[b]->h.data() + static_cast<size_t>(l) * H;
        }
        matvecs(y.w_ih_q, y.w_ih_m, xs, 3 * H, y.n_in, gi);
        matvecs(y.w_hh_q, y.w_hh_m, hs, 3 * H, H, gh);
        for (size_t b = 0; b < B; ++b) {
            std::vector<int64_t>& g = gi[b];
            std::vector<int64_t>& k = gh[b];
            for (int j = 0; j < 3 * H; ++j) {
                g[j] += y.b_ih[j];
                k[j] += y.b_hh[j];
            }
            const int16_t* h = hs[b];
            int16_t* o = out[b].h.data() + static_cast<size_t>(l) * H;
            for (int j = 0; j < H; ++j) {
                int64_t r = act(sigmoid_, g[j] + k[j]);
                int64_t z = act(sigmoid_, g[H + j] + k[H + j]);
                int64_t n = act(tanh_, g[2 * H + j] + ((r * k[2 * H + j]) >> Q));
                // (within -1..1: a mix of n and h, both within it, the tables checked at load)
                o[j] = static_cast<int16_t>(((ONE - z) * n + z * h[j]) >> Q);
            }
            x[b].assign(o, o + H);
        }
    }
    return out;
}

std::vector<int64_t> CharLM::logits(const LmState& state, const std::vector<int>& cands) const {
    std::vector<int64_t> p;
    matvec(proj_q_, proj_m_, state.h.data() + static_cast<size_t>(layers_ - 1) * H_, E_, H_, p);
    int64_t total = 0, top = 0;
    for (int j = 0; j < E_; ++j) {
        p[j] += proj_b_[j];
        int64_t a = p[j] < 0 ? -p[j] : p[j];
        total += a;
        top = std::max(top, a);
    }
    std::vector<int64_t> out(cands.size());
    if (top <= 32767 && 127 * total < INT32_LIMIT) {  // (the 32-bit sum is exact: usually)
        std::vector<int16_t> p16(E_);
        for (int j = 0; j < E_; ++j) p16[j] = static_cast<int16_t>(p[j]);
        for (size_t k = 0; k < cands.size(); ++k) {
            int c = cands[k];
            out[k] = requant(dot(emb_q_.data() + static_cast<size_t>(c) * E_, p16.data(), E_), emb_m_[c]) + bias_[c];
        }
    } else {
        for (size_t k = 0; k < cands.size(); ++k) {
            int c = cands[k];
            const int8_t* e = emb_q_.data() + static_cast<size_t>(c) * E_;
            int64_t acc = 0;
            for (int j = 0; j < E_; ++j) acc += e[j] * p[j];
            out[k] = requant(acc, emb_m_[c]) + bias_[c];
        }
    }
    return out;
}

std::vector<int64_t> CharLM::scores(const LmState& state, const std::vector<int>& cands) const {
    std::vector<int64_t> d = logits(state, cands);
    if (d.empty()) return d;
    int64_t top = *std::max_element(d.begin(), d.end());
    int64_t s = 0;
    for (auto& v : d) {
        v -= top;  // <= 0
        s += exp_[static_cast<size_t>(std::min(-v, EXP_RANGE))];  // Q16
    }
    int64_t log_s = log_q10(s, log_);  // ln(s / 65536), Q10
    for (auto& v : d) v -= log_s;
    return d;
}

int64_t CharLM::mass_q16(const LmState& state, const std::vector<int>& ids) const {
    if (all_.empty()) {
        all_.resize(static_cast<size_t>(V_));
        for (int i = 0; i < V_; ++i) all_[static_cast<size_t>(i)] = i;
    }
    std::vector<int64_t> lg = logits(state, all_);
    int64_t top = *std::max_element(lg.begin(), lg.end());
    int64_t total = 0;
    for (auto& v : lg) {
        v = exp_[static_cast<size_t>(std::min(top - v, EXP_RANGE))];  // Q16, the best one 1.0
        total += v;
    }
    int64_t part = 0;
    for (int c : ids) part += lg[static_cast<size_t>(c)];
    return (part << 16) / total;
}

std::vector<int64_t> CharLM::unit_scores(const std::vector<const LmState*>& states, const std::vector<int>& ids) const {
    // (logits()'s projection of each state, then lmq::unit_scores: the same numbers as scores())
    std::vector<std::vector<int64_t>> ps(states.size());
    std::vector<const std::vector<int64_t>*> refs;
    for (size_t b = 0; b < states.size(); ++b) {
        matvec(proj_q_, proj_m_, states[b]->h.data() + static_cast<size_t>(layers_ - 1) * H_, E_, H_, ps[b]);
        for (int j = 0; j < E_; ++j) ps[b][j] += proj_b_[j];
        refs.push_back(&ps[b]);
    }
    return lmq::unit_scores(avx2_, emb_q_, emb_m_, bias_, V_, E_, exp_, log_, refs, ids);
}

// ---- LmScorer ----

LmScorer::LmScorer(const LmModel& lm, const Lexicon& lexicon) : lm_(lm), lexicon_(lexicon) {
    // (at load, not on the first key: it walks the whole dictionary, 60 ms)
    std::unordered_map<ustr, std::set<int>> out;
    auto chars = std::make_shared<std::unordered_map<int, char32_t>>();
    for (auto& [reading, lst] : lexicon_.entries()) {
        std::vector<ustr> syls = split(reading, U'-');
        for (auto& [p, _] : lst)
            if (p.size() == syls.size())
                for (size_t k = 0; k < syls.size(); ++k) {
                    int id = lm_.id(p[k]);
                    out[syls[k]].insert(id);
                    chars->emplace(id, p[k]);  // (two characters of one id score the same: either will do)
                }
    }
    auto cands = std::make_shared<std::unordered_map<ustr, std::vector<int>>>();
    for (auto& [s, ids] : out) (*cands)[s] = std::vector<int>(ids.begin(), ids.end());
    cands_ = std::move(cands);
    chars_ = std::move(chars);
    kept_ = std::max<size_t>(64, std::min(LM_STATES_KEPT, LM_STATES_BYTES / std::max<size_t>(1, lm_.state_bytes())));
    start_ = lm_.start();
    clear();
}

LmScorer::LmScorer(const LmScorer& base, const ustr& context)
    : lm_(base.lm_), lexicon_(base.lexicon_), cands_(base.cands_), chars_(base.chars_), context_(context), base_(&base),
      kept_(base.kept_) {
    start_ = lm_.start();
    std::vector<int> ids;
    for (char32_t ch : context_) ids.push_back(lm_.id(ch));
    if (!ids.empty()) start_ = lm_.run(start_, ids).back();  // (together: #41, the first key after a context changed)
    clear();
}

const LmScorer* LmScorer::view(const ustr& context) const {
    // (the last LM_VIEWS_KEPT kept, the one used most lately first; a pointer given out stays valid until a later
    // call drops its view: convert_all uses each view before asking for the next)
    for (auto it = views_.begin(); it != views_.end(); ++it)
        if ((*it)->context_ == context) {
            if (it != views_.begin()) views_.splice(views_.begin(), views_, it);
            return views_.front().get();
        }
    views_.emplace_front(new LmScorer(*this, context));
    if (views_.size() > LM_VIEWS_KEPT) views_.pop_back();
    return views_.front().get();
}

const std::vector<int>* LmScorer::candidates(const ustr& syllable) const {
    auto it = cands_->find(syllable);
    return it == cands_->end() ? nullptr : &it->second;
}

void LmScorer::clear() const {
    states_.clear();
    states_[ustr()] = start_;
    score_cache_.clear();
    mass_cache_.clear();
}

void LmScorer::room() const {
    if (!base_) {
        if (full(0, 0)) clear();
        return;
    }
    if (full(0, 0)) drop();
}

bool LmScorer::full(size_t more_states, size_t more_masses) const {
    if (!base_)
        return states_.size() + more_states >= kept_ || score_cache_.size() >= LM_STATES_KEPT ||
               mass_cache_.size() + more_masses >= LM_STATES_KEPT;
    // (a view: all the views' caches together within the one budget, then all dropped - engine/lm.py Scorer._room)
    size_t states = 0, scores = 0, masses = 0;
    for (auto& v : base_->views_) {
        states += v->states_.size();
        scores += v->score_cache_.size();
        masses += v->mass_cache_.size();
    }
    return states + more_states >= kept_ || scores >= LM_STATES_KEPT || masses + more_masses >= LM_STATES_KEPT;
}

void LmScorer::drop() const {
    if (!base_) return clear();
    bool listed = false;
    for (auto& v : base_->views_) {
        v->clear();
        listed = listed || v.get() == this;
    }
    if (!listed) clear();  // (a view already dropped from the list, still in use by its caller)
}

void LmScorer::prepare(const std::vector<ustr>& texts) const {
    // every beginning of the texts not kept yet but the whole, by length (each one after its parent)
    auto missing = [&] {
        std::vector<std::vector<ustr>> by_len;
        std::unordered_set<ustr> seen;
        for (const ustr& t : texts)
            for (size_t j = 1; j < t.size(); ++j) {  // (a character's score needs the state before it)
                ustr p = t.substr(0, j);
                if (states_.count(p) || !seen.insert(p).second) continue;
                if (by_len.size() < j) by_len.resize(j);
                by_len[j - 1].push_back(std::move(p));
            }
        return by_len;
    };
    auto by_len = missing();
    size_t count = 0;
    for (auto& level : by_len) count += level.size();
    size_t scored = 0;  // (the scores text_score keeps next, at most one per character)
    for (const ustr& t : texts) scored += t.size();
    if (states_.size() + count >= kept_ || score_cache_.size() + scored >= LM_STATES_KEPT) {
        // (now, not part way through text_score: room() would drop the states kept or just
        // stepped, and each would be stepped again alone; the states and scores are the same either way)
        clear();
        by_len = missing();
    }
    for (auto& level : by_len) {
        std::vector<const LmState*> parents;
        std::vector<int> ids;
        for (const ustr& p : level) {
            parents.push_back(&states_.at(p.substr(0, p.size() - 1)));  // (nodes stay put as the map grows)
            ids.push_back(lm_.id(p.back()));
        }
        std::vector<LmState> next = lm_.steps(parents, ids);
        for (size_t k = 0; k < level.size(); ++k) states_[level[k]] = std::move(next[k]);
    }
}

LmState LmScorer::state(const ustr& text) const {
    size_t k = text.size();
    while (!states_.count(text.substr(0, k))) --k;
    LmState st = states_.at(text.substr(0, k));
    std::vector<int> ids;
    for (size_t j = k; j < text.size(); ++j) ids.push_back(lm_.id(text[j]));
    std::vector<LmState> run = lm_.run(st, ids);  // (together, then kept one by one as before)
    for (size_t j = k; j < text.size(); ++j) {
        st = run[j - k];
        room();
        states_[text.substr(0, j + 1)] = st;
    }
    return st;
}

int64_t LmScorer::char_score(const ustr& before, char32_t ch, const ustr& syllable) const {
    const std::vector<int>* cands = candidates(syllable);
    if (!cands) return 0;
    ustr key = before;
    key.push_back(U'\0');
    key += syllable;
    auto it = score_cache_.find(key);
    if (it == score_cache_.end()) {
        std::vector<int64_t> scores = lm_.scores(state(before), *cands);
        room();
        it = score_cache_.emplace(std::move(key), std::move(scores)).first;
    }
    const std::vector<int64_t>& scores = it->second;
    int c = lm_.id(ch);
    auto at = std::lower_bound(cands->begin(), cands->end(), c);
    if (at != cands->end() && *at == c) return scores[at - cands->begin()];
    return *std::min_element(scores.begin(), scores.end());
}

std::vector<char32_t> LmScorer::top_chars(const ustr& before, const ustr& syllable, size_t n) const {
    const std::vector<int>* cands = candidates(syllable);
    if (!cands) return {};
    ustr key = before;
    key.push_back(U'\0');
    key += syllable;
    auto it = score_cache_.find(key);
    if (it == score_cache_.end()) {
        std::vector<int64_t> scores = lm_.scores(state(before), *cands);
        room();
        it = score_cache_.emplace(std::move(key), std::move(scores)).first;
    }
    const std::vector<int64_t>& scores = it->second;
    std::vector<size_t> order;
    for (size_t i = 0; i < cands->size(); ++i)
        if ((*cands)[i] != 0) order.push_back(i);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return scores[a] != scores[b] ? scores[a] > scores[b] : (*cands)[a] < (*cands)[b];
    });
    if (order.size() > n) order.resize(n);
    std::vector<char32_t> out;
    for (size_t i : order) out.push_back(chars_->at((*cands)[i]));
    return out;
}

int64_t LmScorer::among_score(const ustr& before, char32_t ch, const std::vector<ustr>& syllables) const {
    // (#51) ch among the characters of these syllables together (engine/lm.py Scorer.among_score)
    std::vector<int> cands;
    ustr key = before;
    key.push_back(U'\0');
    key.push_back(U'|');  // (never char_score's key, as engine/lm.py)
    bool any = false;
    for (size_t k = 0; k < syllables.size(); ++k) {
        if (k) key.push_back(U'|');
        key += syllables[k];
        if (const std::vector<int>* c = candidates(syllables[k])) {
            cands.insert(cands.end(), c->begin(), c->end());
            any = true;
        }
    }
    if (!any) return 0;
    std::sort(cands.begin(), cands.end());
    cands.erase(std::unique(cands.begin(), cands.end()), cands.end());
    auto it = score_cache_.find(key);
    if (it == score_cache_.end()) {
        std::vector<int64_t> scores = lm_.scores(state(before), cands);
        room();
        it = score_cache_.emplace(std::move(key), std::move(scores)).first;
    }
    const std::vector<int64_t>& scores = it->second;
    int c = lm_.id(ch);
    auto at = std::lower_bound(cands.begin(), cands.end(), c);
    if (at != cands.end() && *at == c) return scores[at - cands.begin()];
    return *std::min_element(scores.begin(), scores.end());
}

int64_t LmScorer::units_logp(const ustr& units) const {
    // (#55) each unit among the whole vocabulary (engine/lm.py Scorer.units_logp); kept with the masses, under a key no
    // syllable's can be (before + U'\0' + U'\x01' + unit)
    if (every_.empty()) {
        every_.resize(static_cast<size_t>(lm_.vocab()));
        for (size_t k = 0; k < every_.size(); ++k) every_[k] = static_cast<int>(k);
    }
    int64_t total = 0;
    for (size_t k = 0; k < units.size(); ++k) {
        ustr before = units.substr(0, k);
        ustr key = before;
        key.push_back(U'\0');
        key.push_back(U'\x01');
        key.push_back(units[k]);
        auto it = mass_cache_.find(key);
        int64_t hit;
        if (it == mass_cache_.end()) {
            std::vector<int64_t> s = lm_.scores(state(before), every_);
            hit = s[static_cast<size_t>(lm_.id(units[k]))];
            room();
            mass_cache_.emplace(std::move(key), hit);
        } else {
            hit = it->second;
        }
        total += hit;
    }
    return total;
}

std::vector<int64_t> LmScorer::units_logp_many(const std::vector<ustr>& texts) const {
    auto mass_key = [](const ustr& units, size_t k) {  // (units_logp's)
        ustr key = units.substr(0, k);
        key.push_back(U'\0');
        key.push_back(U'\x01');
        key.push_back(units[k]);
        return key;
    };
    // the states (beginnings) and the masses of texts[from, to) not kept, each once
    auto missing = [&](size_t from, size_t to, std::unordered_set<ustr>& states, std::unordered_set<ustr>& masses) {
        for (size_t t = from; t < to; ++t)
            for (size_t k = 0; k < texts[t].size(); ++k) {
                if (k && !states_.count(texts[t].substr(0, k))) states.insert(texts[t].substr(0, k));
                ustr key = mass_key(texts[t], k);
                if (!mass_cache_.count(key)) masses.insert(std::move(key));
            }
    };
    std::vector<int64_t> out(texts.size(), 0);
    size_t at = 0;
    while (at < texts.size()) {
        // a group: the texts from at while what they need more stays within the budget (at least one)
        std::unordered_set<ustr> states, masses;
        size_t end = at;
        while (end < texts.size()) {
            // (this text's own, then how many the group gains: no copy of the group's sets per text)
            std::unordered_set<ustr> s1, m1;
            missing(end, end + 1, s1, m1);
            size_t more_s = 0, more_m = 0;
            for (const ustr& p : s1) more_s += !states.count(p);
            for (const ustr& k : m1) more_m += !masses.count(k);
            if (end > at && full(states.size() + more_s, masses.size() + more_m)) break;
            states.insert(s1.begin(), s1.end());
            masses.insert(m1.begin(), m1.end());
            ++end;
        }
        if (full(states.size(), masses.size())) {  // (dropped now, never part way through)
            drop();
            states.clear();
            masses.clear();
            missing(at, end, states, masses);
        }
        // the states, all as one tree (LmModel::tree): shorter first, each after a kept state or an earlier item -
        // a Transformer reads each layer's weights once for all of them, not once a length (a sentence's end that
        // reads a new context stepped 269 states over 32 lengths: 616 ms on max)
        std::vector<ustr> order(states.begin(), states.end());
        std::sort(order.begin(), order.end(), [](const ustr& a, const ustr& b) {
            return a.size() != b.size() ? a.size() < b.size() : a < b;
        });
        std::unordered_map<ustr, int> item;
        std::vector<const LmState*> prev;
        std::vector<int> link, chars;
        for (const ustr& p : order) {
            ustr parent = p.substr(0, p.size() - 1);
            auto it = item.find(parent);
            link.push_back(it == item.end() ? -1 : it->second);
            prev.push_back(it == item.end() ? &states_.at(parent) : nullptr);  // (nodes stay put as the map grows)
            chars.push_back(lm_.id(p.back()));
            item.emplace(p, static_cast<int>(link.size()) - 1);
        }
        if (!order.empty()) {
            std::vector<LmState> made = lm_.tree(prev, link, chars);
            for (size_t k = 0; k < order.size(); ++k) states_[order[k]] = std::move(made[k]);
        }
        // the masses, together
        std::vector<ustr> keys(masses.begin(), masses.end());
        std::sort(keys.begin(), keys.end());
        std::vector<const LmState*> sts;
        std::vector<int> ids;
        for (const ustr& key : keys) {
            sts.push_back(&states_.at(key.substr(0, key.size() - 3)));
            ids.push_back(lm_.id(key.back()));
        }
        std::vector<int64_t> got = lm_.unit_scores(sts, ids);
        for (size_t k = 0; k < keys.size(); ++k) mass_cache_[keys[k]] = got[k];
        for (size_t t = at; t < end; ++t)
            for (size_t k = 0; k < texts[t].size(); ++k) out[t] += mass_cache_.at(mass_key(texts[t], k));
        at = end;
    }
    return out;
}

int64_t LmScorer::syllable_mass_q16(const ustr& before, const ustr& syllable) const {
    ustr key = before;
    key.push_back(U'\0');
    key += syllable;
    auto it = mass_cache_.find(key);
    if (it != mass_cache_.end()) return it->second;
    int64_t mass = 0;
    if (const std::vector<int>* cands = candidates(syllable)) {
        std::vector<int> ids;
        for (int c : *cands)
            if (c > 0) ids.push_back(c);
        if (!ids.empty()) mass = lm_.mass_q16(state(before), ids);
    }
    room();
    mass_cache_.emplace(std::move(key), mass);
    return mass;
}

int64_t LmScorer::text_score(const ustr& text, const std::vector<ustr>& readings) const {
    int64_t sum = 0;
    for (size_t k = 0; k < text.size(); ++k) sum += char_score(text.substr(0, k), text[k], readings[k]);
    return sum;
}

const LmScorer* LanguageModel::get(const std::string& tier) const {
    static const wchar_t* const files[3] = {L"lm.bin", L"lm_plus.bin", L"lm_max.bin"};  // (engine/lm.py TIER_FILES)
    size_t k = tier == "plus" ? 1 : tier == "max" ? 2 : 0;
    Tier& t = tiers_[k];
    if (!t.tried) {
        t.tried = true;
        t.model = LmModel::load(dir_ + L"\\" + files[k]);
        if (t.model) {
            t.model->tier = k == 1 ? "plus" : k == 2 ? "max" : "light";
            t.scorer = std::make_unique<LmScorer>(*t.model, lexicon_);
        }
    }
    if (!t.scorer && k != 0) return get("light");
    return t.scorer.get();
}

}  // namespace sz
