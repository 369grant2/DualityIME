"""tools/lm/next: the next student model of 長句選字加強 (#41) - its training text with English kept as one mark
(docs/superpowers/plans/2026-11-15-next-gen-student.md)."""
import importlib.util
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEXT = os.path.join(ROOT, "tools", "lm", "next")


def load(name):
    # (by its path: tools/lm on sys.path would shadow modules of other tests)
    spec = importlib.util.spec_from_file_location(f"lm_next_{name}", os.path.join(NEXT, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TextUnitsTest(unittest.TestCase):
    def setUp(self):
        self.text = load("text")

    def test_english_and_digits_one_mark(self):
        self.assertEqual(self.text.to_units("我在用Qwen3-1.7B做蒸餾"), ["我", "在", "用", "<en>", "做", "蒸", "餾"])

    def test_full_width_folded(self):
        self.assertEqual(self.text.to_units("Ｇｉｔｈｕｂ，辨識"), ["<en>", "辨", "識"])

    def test_number_with_point(self):
        self.assertEqual(self.text.to_units("1.5倍"), ["<en>", "倍"])

    def test_space_makes_two(self):
        self.assertEqual(self.text.to_units("A - B"), ["<en>", "<en>"])

    def test_punctuation_dropped(self):
        self.assertEqual(self.text.to_units("。，"), [])
        self.assertEqual(self.text.to_units("你好，世界"), ["你", "好", "世", "界"])

    def test_mark_constant(self):
        self.assertEqual(self.text.EN, "<en>")


class PrepareNextTest(unittest.TestCase):
    def test_encode_matches_units(self):
        # (prepare_next.encode_text works on whole texts with numpy: the same units as text.to_units, per paragraph)
        import numpy as np
        prep, text = load("prepare_next"), load("text")
        vocab = ["<unk>", "<s>", "<en>"] + list("我在用做蒸餾你好世界倍")
        lut = np.zeros(0x110000, dtype=np.uint16)
        for i, ch in enumerate(vocab[3:], 3):
            lut[ord(ch)] = i
        lut[ord(prep.MARK)] = prep.EN_ID
        doc = "我在用Qwen3-1.7B做蒸餾\n你\n\n你好，世界 1.5倍\n。"
        ids = prep.encode_text(doc, lut)
        want = []
        for para in doc.split("\n"):
            u = text.to_units(para)
            if len(u) >= 2:
                want += ["<s>"] + u
        self.assertEqual([vocab[i] for i in ids], want)
        self.assertEqual(want.count("<s>"), 2)  # (你 alone: under 2 units)


class ModelsTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("no torch")
        self.m = load("models")

    def test_causal(self):
        # (changing the character at 5 changes nothing before it, in both prototypes)
        import torch
        torch.manual_seed(0)
        for model in (self.m.TinyTransformer(50, d=32, layers=2, heads=4, ffn=64),
                      self.m.GruAttn(50, emb=16, hidden=32, layers=2, heads=4)):
            model.eval()
            a = torch.randint(3, 50, (1, 12))
            b = a.clone()
            b[0, 5] = (a[0, 5] + 1 - 3) % 47 + 3
            with torch.no_grad():
                la, lb = model(a), model(b)
            self.assertTrue(torch.allclose(la[0, :5], lb[0, :5], atol=1e-5), type(model).__name__)
            self.assertFalse(torch.allclose(la[0, 5:], lb[0, 5:]), type(model).__name__)

    def test_light_budget(self):
        # light: int8 weights <= 12M, cost per character <= 1.10x the shipped XL GRU's (plan, Global Constraints)
        spec = importlib.util.spec_from_file_location("lm_train_budget", os.path.join(ROOT, "tools", "lm", "train.py"))
        train = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train)
        xl = train.CharLM(12263, 384, 768, 2)  # (the vocabulary of today's model: 12263)
        limit = 1.10 * self.m.macs_per_step(xl, 72)
        for arch in ("tf", "ga"):
            model = self.m.build(arch, "light", 12264)
            self.assertLessEqual(self.m.count_int8_weights(model), 12_000_000, arch)
            self.assertLessEqual(self.m.macs_per_step(model, 72), limit, arch)


if __name__ == "__main__":
    unittest.main()
