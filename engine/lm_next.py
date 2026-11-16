"""The integer inference of the next student model (選字模型, #41), read from a version 2 file (b"DLM2"): the same
fixed point and tables as engine/lm.py (Q10 states, int8 weights with a multiplier per row, look-up tables), for the
two prototypes of tools/lm/next/models.py:
    arch 1, TinyTransformer: embedding -> layers of (RMSNorm, causal attention with ALiBi, residual, RMSNorm, ReLU
            feed-forward, residual) -> RMSNorm -> logits on the embedding (tied) + bias
    arch 2, GruAttn: embedding -> GRU layers (engine/lm.py's GRU) -> RMSNorm, causal attention, residual -> RMSNorm ->
            projection to the embedding size -> logits on the embedding + bias
Every operation in a fixed order, so the C++ engine can give exactly the same scores (plan Task 13-15):
    rmsnorm   s = sum x^2 (int64), r = isqrt(s // n) (Q10, at least 1), y = floor(x * g / r)      (g Q10)
    matvec    the input shifted right by k = shift_of(x, clip(n)) bits first (0 for nearly every vector),
              clip(n) = min(32767, (2^31 - 1) // (127 n)): an int32 sum of n int8 x int16 products never overflows;
              then requant_shift(acc, m, k) = (acc * m + 2^(MUL_SHIFT - 1 - k)) >> (MUL_SHIFT - k) (engine/lm.py's
              requant at k = 0) + bias. A large activation loses k low bits instead of being cut (a cut one moved
              every score after it). (arch 2's GRU: its first layer's input, the embedding, shifted so too; the rest is
              engine/lm.py's GRU, states within +-1)
    attention per head (K and V kept clamped to +-(2^31 - 1)): s_j = (q . k_j) >> Q, s_j = (s_j * inv_sqrt) >> 16, s_j -= (slope * (t - j) * ONE) >> 16;
              e_j = exp[min(max s - s_j, EXP_RANGE)] (Q16), w_j = (e_j << 16) // sum e, out = (sum w_j v_j) >> 16
A state is one position: its K and V for every layer, a link to the state before it (never copied), and p, the vector
the logits of the next character are made from. Scores, among a syllable's candidates, as engine/lm.py CharLM.scores.

lm.bin version 2 layout (little endian), made by tools/lm/next/export_next.py:
    b"DLM2", u32 x len(HEADER) (HEADER below)
    u32 byte length + the vocabulary, UTF-8, one character per line ("<en>" at 2: English and letters-and-digits)
    u32 byte length + the order thresholds, UTF-8 "reading<TAB>q16" lines (composer ORDER_LM_THRESHOLDS_Q16 for it)
    the arrays of array_names(arch, layers), each as u32 element count + data (int8 where marked, else int32)
"""
import os
import re
import struct

from .lm import ACT_RANGE, EN_MARK, EXP_RANGE, LN2_Q, LOG_BITS, LOG_SIZE, MUL_SHIFT, ONE, Q, _numpy, act, requant

MAGIC = b"DLM2"
VERSION = 2
ARCH_TF, ARCH_GA = 1, 2
EN_ID = 2      # "<en>": an English or letters-and-digits run (tools/lm/next/text.py)
HEADER = ("version", "arch", "q", "mul_shift", "vocab", "emb", "dim", "layers", "heads", "ffn", "act_range",
          "exp_range", "log_bits", "inv_sqrt_q16", "lm_weight_milli", "context_max")
# emb: the embedding size (= dim for arch 1); dim: the model's width (arch 1: d; arch 2: the GRU's hidden);
# ffn: the feed-forward width (arch 1; 0 for arch 2); inv_sqrt_q16: round(65536 / sqrt(dim // heads))
INT32_MAX = (1 << 31) - 1
# (a position's K and V are kept as int32: clamped to this, never reached by a model - its values are Q10 activations -
# so the C++ engine keeps half the memory and both read the same)

np = None


def _np():
    global np
    if np is None:
        np = _numpy()
    return np


def clip_of(n):
    """The largest |x| a matvec of n columns takes: n products of int8 and int16 then fit an int32 sum."""
    return min(32767, INT32_MAX // (127 * n))


MAX_SHIFT = MUL_SHIFT - 1  # (requant_shift's rounding term needs k <= MUL_SHIFT - 1; past it the input is clipped)


def shift_of(x, c):
    """The least k >= 0 (at most MAX_SHIFT) that brings every x >> k (arithmetic: floor) within +-c."""
    hi, lo = int(x.max()), int(x.min())
    k = 0
    while k < MAX_SHIFT and ((hi >> k) > c or -(lo >> k) > c):
        k += 1
    return k


def requant_shift(acc, m, k):
    """requant of a sum of inputs shifted right by k: the k bits given back (engine/lm.py requant at k = 0)."""
    return (acc * m.astype(np.int64) + (1 << (MUL_SHIFT - 1 - k))) >> (MUL_SHIFT - k)


def shifted(x, n):
    """(x >> k within +-clip(n), k) for a matvec of n columns (clipped only past MAX_SHIFT: no model comes near)."""
    c = clip_of(n)
    k = shift_of(x, c)
    return np.clip(x >> k, -c, c), k


def isqrt(v):
    """floor(sqrt(v)) for an int v >= 0, exactly (the C++ engine: the same Newton steps on uint64)."""
    if v <= 0:
        return 0
    x = 1 << ((v.bit_length() + 1) // 2)  # (>= sqrt(v))
    while True:
        y = (x + v // x) // 2
        if y >= x:
            return x
        x = y


def array_names(arch, layers):
    """The arrays of a version 2 file, in order."""
    names = ["emb_q", "emb_m", "bias"]
    if arch == ARCH_GA:
        for l in range(layers):
            names += [f"w_ih_q{l}", f"w_ih_m{l}", f"w_hh_q{l}", f"w_hh_m{l}", f"b_ih{l}", f"b_hh{l}"]
        names += ["n1_g", "qkv_q", "qkv_m", "qkv_b", "out_q", "out_m", "out_b", "n2_g", "proj_q", "proj_m", "proj_b"]
    else:
        for l in range(layers):
            names += [f"n1_g{l}", f"qkv_q{l}", f"qkv_m{l}", f"qkv_b{l}", f"out_q{l}", f"out_m{l}", f"out_b{l}",
                      f"n2_g{l}", f"up_q{l}", f"up_m{l}", f"up_b{l}", f"down_q{l}", f"down_m{l}", f"down_b{l}"]
        names += ["norm_g"]
    return names + ["slopes", "sigmoid", "tanh", "exp", "log"]


def is_int8(name):
    return name.rstrip("0123456789").endswith("_q")


def array_shapes(h):
    """name -> shape of every array, from the header values."""
    V, E, D, L, H, F = h["vocab"], h["emb"], h["dim"], h["layers"], h["heads"], h["ffn"]
    s = {"emb_q": (V, E), "emb_m": (V,), "bias": (V,), "slopes": (H,), "sigmoid": (2 * ACT_RANGE + 1,),
         "tanh": (2 * ACT_RANGE + 1,), "exp": (EXP_RANGE + 1,), "log": (LOG_SIZE,)}

    def att(sfx):
        s.update({f"qkv_q{sfx}": (3 * D, D), f"qkv_m{sfx}": (3 * D,), f"qkv_b{sfx}": (3 * D,),
                  f"out_q{sfx}": (D, D), f"out_m{sfx}": (D,), f"out_b{sfx}": (D,)})

    if h["arch"] == ARCH_GA:
        for l in range(L):
            n_in = E if l == 0 else D
            s.update({f"w_ih_q{l}": (3 * D, n_in), f"w_ih_m{l}": (3 * D,), f"w_hh_q{l}": (3 * D, D),
                      f"w_hh_m{l}": (3 * D,), f"b_ih{l}": (3 * D,), f"b_hh{l}": (3 * D,)})
        att("")
        s.update({"n1_g": (D,), "n2_g": (D,), "proj_q": (E, D), "proj_m": (E,), "proj_b": (E,)})
    else:
        for l in range(L):
            att(str(l))
            s.update({f"n1_g{l}": (D,), f"n2_g{l}": (D,), f"up_q{l}": (F, D), f"up_m{l}": (F,), f"up_b{l}": (F,),
                      f"down_q{l}": (D, F), f"down_m{l}": (D,), f"down_b{l}": (D,)})
        s["norm_g"] = (D,)
    return s


class State:
    """One position: its K, V for every layer (int64 Q10 [dim] each), the state before it (None at <s>), its index,
    the GRU state (arch 2), and p, what the next character's logits are made from."""
    __slots__ = ("prev", "pos", "k", "v", "gru", "p")

    def __init__(self, prev, pos, k, v, gru, p):
        self.prev, self.pos, self.k, self.v, self.gru, self.p = prev, pos, k, v, gru, p


class NextLM:
    knows_english = True

    def __init__(self, path=None):
        """Reads a version 2 lm.bin; raises ValueError (or OSError) when it cannot. Tests fill it with setup()."""
        _np()
        if path is not None:
            self._read(path)

    # ---- the file -------------------------------------------------------------------------------------------------
    def _read(self, path):
        with open(path, "rb") as f:
            data = f.read()
        head = 4 + 4 * len(HEADER)
        if data[:4] != MAGIC or len(data) < head + 8:
            raise ValueError(f"{path}: not a version 2 language model file")
        h = dict(zip(HEADER, struct.unpack_from("<%dI" % len(HEADER), data, 4)))
        mine = {"version": VERSION, "q": Q, "mul_shift": MUL_SHIFT, "act_range": ACT_RANGE, "exp_range": EXP_RANGE,
                "log_bits": LOG_BITS}
        for k, v in mine.items():
            if h[k] != v:
                raise ValueError(f"{path}: {k} is {h[k]}, expected {v}")
        if h["arch"] not in (ARCH_TF, ARCH_GA) or h["heads"] == 0 or h["dim"] % h["heads"]:
            raise ValueError(f"{path}: arch {h['arch']}, heads {h['heads']}")
        # (sizes no model has, as the C++ engine refuses them before allocating: a file loads in both or neither)
        if not (3 <= h["vocab"] <= 1 << 22 and 1 <= h["emb"] <= 8192 and 1 <= h["dim"] <= 8192 and
                1 <= h["layers"] <= 64) or \
                (h["arch"] == ARCH_TF and (h["emb"] != h["dim"] or not 1 <= h["ffn"] <= 32768)):
            raise ValueError(f"{path}: vocabulary {h['vocab']}, emb {h['emb']}, dim {h['dim']}, layers {h['layers']}, "
                             f"ffn {h['ffn']}")
        pos = head
        texts = []
        for _ in range(2):
            n, = struct.unpack_from("<I", data, pos)
            pos += 4
            texts.append(data[pos:pos + n].decode("utf-8"))
            pos += n
        vocab = texts[0].split("\n")
        if len(vocab) != h["vocab"]:
            raise ValueError(f"{path}: vocabulary size differs from the header")
        thresholds = {}
        for line in texts[1].split("\n") if texts[1] else []:
            reading, q16 = line.split("\t")
            if not re.fullmatch(r"[+-]?[0-9]{1,18}", q16):  # (a plain integer only, as the C++ engine reads it)
                raise ValueError(f"{path}: threshold {q16!r}")
            thresholds[reading] = int(q16)
        shapes = array_shapes(h)
        t = {}
        for name in array_names(h["arch"], h["layers"]):
            n, = struct.unpack_from("<I", data, pos)
            pos += 4
            size = 1
            for x in shapes[name]:
                size *= x
            if n != size:
                raise ValueError(f"{path}: {name} has {n} elements, expected {size}")
            dtype = np.dtype("i1") if is_int8(name) else np.dtype("<i4")
            if pos + n * dtype.itemsize > len(data):
                raise ValueError(f"{path}: cut short")
            a = np.frombuffer(data, dtype=dtype, count=n, offset=pos)
            if is_int8(name) and (a == -128).any():
                raise ValueError(f"{path}: {name} has -128")
            t[name] = a.reshape(shapes[name])
            pos += n * dtype.itemsize
        if pos != len(data):
            raise ValueError(f"{path}: extra bytes at the end")
        # (gates within 0..1, tanh within -1..1: states within -1..1, the C++ engine keeps them in int16; exp within
        # 0..1 and above 0 at 0: no division by 0 - the C++ engine's tables_ok, so a file loads in both or neither)
        if (t["sigmoid"] < 0).any() or (t["sigmoid"] > ONE).any() or (abs(t["tanh"]) > ONE).any() or \
                (t["exp"] < 0).any() or (t["exp"] > 65536).any() or t["exp"][0] < 1:
            raise ValueError(f"{path}: a table out of its range")
        self.setup(h, t, vocab, thresholds)

    def setup(self, h, t, vocab, thresholds):
        self.h, self.t, self.vocab = dict(h), t, vocab
        self.ids = {c: i for i, c in enumerate(vocab)}
        self.ids[EN_MARK] = EN_ID  # (what the engine writes for "<en>" in a context)
        self.arch, self.layers, self.heads = h["arch"], h["layers"], h["heads"]
        self.dim, self.dh = h["dim"], h["dim"] // h["heads"]
        self.lm_weight = h["lm_weight_milli"] / 1000.0
        self.context_max = h["context_max"]
        self.order_thresholds = thresholds
        self.wide = {k: v.astype(np.int64) for k, v in t.items() if is_int8(k) and k != "emb_q"}

    # ---- operations -----------------------------------------------------------------------------------------------
    def matvec(self, name, x, bias):
        w = self.wide[name]
        x, k = shifted(x, w.shape[1])
        return requant_shift(w @ x, self.t[name.replace("_q", "_m")], k) + self.t[bias]

    def rmsnorm(self, x, g):
        s = int((x * x).sum())
        r = max(1, isqrt(s // len(x)))
        return (x * self.t[g].astype(np.int64)) // r

    def embed(self, c):
        t = self.t
        return requant(t["emb_q"][c].astype(np.int64) * ONE, np.full(t["emb_q"].shape[1], t["emb_m"][c]))

    def attention(self, x, sfx, state, k_new_prev):
        """The attention of the position after state reading every position up to it: (its k, its v, the output)."""
        D, dh, H = self.dim, self.dh, self.heads
        qkv = self.matvec(f"qkv_q{sfx}", x, f"qkv_b{sfx}")
        q = qkv[:D]
        k = np.clip(qkv[D:2 * D], -INT32_MAX, INT32_MAX)
        v = np.clip(qkv[2 * D:], -INT32_MAX, INT32_MAX)
        keys, vals = [k], [v]  # (this position first, then back: the distance is the index)
        s = state
        while s is not None:
            keys.append(s.k[k_new_prev])
            vals.append(s.v[k_new_prev])
            s = s.prev
        K = np.stack(keys)                                     # [n, D]
        Vv = np.stack(vals)
        dist = np.arange(len(keys), dtype=np.int64)            # (0 for this position)
        out = np.empty(D, dtype=np.int64)
        inv = self.h["inv_sqrt_q16"]
        e_table = self.t["exp"]
        for hd in range(H):
            a, b = hd * dh, (hd + 1) * dh
            sc = (K[:, a:b] @ q[a:b]) >> Q
            sc = (sc * inv) >> 16
            sc = sc - ((int(self.t["slopes"][hd]) * dist * ONE) >> 16)
            e = e_table[np.minimum(sc.max() - sc, EXP_RANGE)].astype(np.int64)
            w = (e << 16) // int(e.sum())
            out[a:b] = (w @ Vv[:, a:b]) >> 16
        return k, v, self.matvec(f"out_q{sfx}", out, f"out_b{sfx}")

    def gru(self, st, x):
        """engine/lm.py's GRU, its first layer's input (the embedding) shifted as a matvec's: later inputs and the
        states are within +-1 (Q10) already."""
        t, H = self.t, self.dim
        out = np.empty_like(st)
        x, k = shifted(x, self.wide["w_ih_q0"].shape[1])
        for l in range(self.layers):
            h = st[l]
            gi = requant_shift(self.wide[f"w_ih_q{l}"] @ x, t[f"w_ih_m{l}"], k if l == 0 else 0) + t[f"b_ih{l}"]
            gh = requant(self.wide[f"w_hh_q{l}"] @ h, t[f"w_hh_m{l}"]) + t[f"b_hh{l}"]
            r = act(t["sigmoid"], gi[:H] + gh[:H])
            zz = act(t["sigmoid"], gi[H:2 * H] + gh[H:2 * H])
            n = act(t["tanh"], gi[2 * H:] + ((r * gh[2 * H:]) >> Q))
            h = ((ONE - zz) * n + zz * h) >> Q
            out[l] = h
            x = h
        return out

    # ---- states and scores --------------------------------------------------------------------------------------
    def start(self):
        """The state after <s> (id 1)."""
        return self.step(None, 1)

    def step(self, state, c):
        """The state after character id c following state (None: the first position)."""
        pos = 0 if state is None else state.pos + 1
        x = self.embed(c)
        if self.arch == ARCH_GA:
            g0 = np.zeros((self.layers, self.dim), dtype=np.int64) if state is None else state.gru
            g = self.gru(g0, x)
            y = g[-1]
            k, v, a = self.attention(self.rmsnorm(y, "n1_g"), "", state, 0)
            y = y + a
            n2 = self.rmsnorm(y, "n2_g")
            p = self.matvec("proj_q", n2, "proj_b")
            return State(state, pos, [k], [v], g, p)
        ks, vs = [], []
        for l in range(self.layers):
            k, v, a = self.attention(self.rmsnorm(x, f"n1_g{l}"), str(l), state, l)
            ks.append(k)
            vs.append(v)
            x = x + a
            up = np.maximum(self.matvec(f"up_q{l}", self.rmsnorm(x, f"n2_g{l}"), f"up_b{l}"), 0)
            x = x + self.matvec(f"down_q{l}", up, f"down_b{l}")
        return State(state, pos, ks, vs, None, self.rmsnorm(x, "norm_g"))

    def logits(self, state, cands):
        t = self.t
        e = t["emb_q"][cands].astype(np.int64)
        return requant(e @ state.p, t["emb_m"][cands]) + t["bias"][cands]

    def scores(self, state, cands):
        """Q10 natural log probability of each candidate, among the candidates alone (<= 0): engine/lm.py's way."""
        t = self.t
        lg = self.logits(state, cands)
        d = lg - lg.max()
        e = t["exp"][np.minimum(-d, EXP_RANGE)]
        s = int(e.sum())
        k = s.bit_length() - 1
        frac = ((s - (1 << k)) << LOG_BITS) >> k
        log_s = (k - 16) * LN2_Q + int(t["log"][frac])
        return d - log_s

    def mass_q16(self, state, ids):
        """Q16 probability that the next character is one of ids, over the whole vocabulary (engine/lm.py's way)."""
        t = self.t
        if getattr(self, "_emb_wide", None) is None:
            self._emb_wide = t["emb_q"].astype(np.int64)
        lg = requant(self._emb_wide @ state.p, t["emb_m"]) + t["bias"]
        e = t["exp"][np.minimum(lg.max() - lg, EXP_RANGE)].astype(np.int64)
        return (int(e[ids].sum()) << 16) // int(e.sum())


def load(path):
    """The version 2 model at path, or None (no file, another format, no numpy). Never raises."""
    try:
        if not os.path.isfile(path):
            return None
        return NextLM(path)
    except Exception:
        return None
