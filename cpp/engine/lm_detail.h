// The fixed point shared by the language models of lm.cpp (version 1) and lm_next.cpp (version 2): the constants,
// requant, the activation tables, the int8 x int16 dot products and reading the file (engine/lm.py, engine/lm_next.py).
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#if defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#include <intrin.h>
#define SZ_LM_X86 1
#endif

namespace sz::lmq {

// the same constants as engine/lm.py (a file must have been made for them)
constexpr int Q = 10;
constexpr int64_t ONE = int64_t(1) << Q;
constexpr int MUL_SHIFT = 24;
constexpr int64_t ACT_RANGE = 8 * ONE;
constexpr int64_t EXP_RANGE = 16 * ONE;
constexpr int LOG_BITS = 10;
constexpr int64_t LOG_SIZE = int64_t(1) << LOG_BITS;
constexpr int64_t LN2_Q = 710;
constexpr int64_t INT32_LIMIT = int64_t(1) << 31;

// int64 accumulators times row multipliers -> Q10 (round half up), engine/lm.py requant
inline int64_t requant(int64_t acc, int64_t m) { return (acc * m + (int64_t(1) << (MUL_SHIFT - 1))) >> MUL_SHIFT; }

inline int64_t act(const std::vector<int32_t>& table, int64_t x) {
    return table[static_cast<size_t>(std::clamp(x, -ACT_RANGE, ACT_RANGE) + ACT_RANGE)];
}

// Python's a // b (towards minus infinity), b > 0
inline int64_t floor_div(int64_t a, int64_t b) {
    int64_t q = a / b;
    return (a % b != 0 && a < 0) ? q - 1 : q;
}

// ln(s / 65536) in Q10 for a sum s >= 1 of Q16 exp() values (engine/lm.py CharLM.scores)
inline int64_t log_q10(int64_t s, const std::vector<int32_t>& log_table) {
    int k = 0;  // 2^k <= s < 2^(k+1)
    while ((s >> (k + 1)) != 0) ++k;
    int64_t frac = ((s - (int64_t(1) << k)) << LOG_BITS) >> k;  // the top LOG_BITS bits under the leading one
    return (k - 16) * LN2_Q + log_table[static_cast<size_t>(frac)];
}

inline int32_t dot_scalar(const int8_t* w, const int16_t* x, int n) {
    int32_t acc = 0;
    for (int j = 0; j < n; ++j) acc += static_cast<int32_t>(w[j]) * static_cast<int32_t>(x[j]);
    return acc;
}

#ifdef SZ_LM_X86
// (the same exact integer sum in another order: identical while nothing overflows)
inline int32_t dot_avx2(const int8_t* w, const int16_t* x, int n) {
    __m256i acc = _mm256_setzero_si256();
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        __m256i w16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + j)));
        __m256i x16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + j));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(w16, x16));
    }
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    int32_t r = _mm_cvtsi128_si32(s);
    for (; j < n; ++j) r += static_cast<int32_t>(w[j]) * static_cast<int32_t>(x[j]);
    return r;
}

inline int32_t hsum(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

// four rows by one vector: each piece of the vector loaded once (the same sums as dot_avx2)
inline void dot4_avx2(const int8_t* w, size_t stride, const int16_t* x, int n, int32_t* out) {
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        __m256i x16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + j));
        auto row = [&](size_t r) {
            return _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + r * stride + j)));
        };
        a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(row(0), x16));
        a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(row(1), x16));
        a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(row(2), x16));
        a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(row(3), x16));
    }
    out[0] = hsum(a0);
    out[1] = hsum(a1);
    out[2] = hsum(a2);
    out[3] = hsum(a3);
    for (; j < n; ++j)
        for (size_t r = 0; r < 4; ++r) out[r] += static_cast<int32_t>(w[r * stride + j]) * static_cast<int32_t>(x[j]);
}
#endif

// the vector path: AVX2 there and not turned off (DUALITYIME_LM_SCALAR=1)
bool use_avx2();

inline int32_t dot(bool avx2, const int8_t* row, const int16_t* x, int n) {
#ifdef SZ_LM_X86
    if (avx2) return dot_avx2(row, x, n);
#endif
    (void)avx2;
    return dot_scalar(row, x, n);
}

struct File {
    FILE* f = nullptr;
    ~File() {
        if (f) std::fclose(f);
    }
    bool read(void* p, size_t n) { return n == 0 || std::fread(p, 1, n, f) == n; }
    bool u32(uint32_t& v) { return read(&v, 4); }
    template <class T> bool array(std::vector<T>& out, uint64_t count) {
        uint32_t n = 0;
        if (!u32(n) || n != count) return false;
        out.resize(n);
        return read(out.data(), static_cast<size_t>(n) * sizeof(T));
    }
};

// the tables a file brings, within what the integer inference needs: gates within 0..1, tanh within -1..1 (states
// within -1..1: int16), exp within 0..1 (Q16) and above 0 at 0 (the best candidate's share; no division by 0)
inline bool tables_ok(const std::vector<int32_t>& sigmoid, const std::vector<int32_t>& tanh,
                      const std::vector<int32_t>& exp) {
    for (int32_t v : sigmoid)
        if (v < 0 || v > ONE) return false;
    for (int32_t v : tanh)
        if (v < -ONE || v > ONE) return false;
    for (int32_t v : exp)
        if (v < 0 || v > 65536) return false;
    return !exp.empty() && exp[0] >= 1;
}

inline bool has_minus_128(const std::vector<int8_t>& w) { return std::find(w.begin(), w.end(), -128) != w.end(); }

}  // namespace sz::lmq
