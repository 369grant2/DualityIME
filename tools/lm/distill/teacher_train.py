"""Teacher scores (log10 P(text)) for every candidate of the students' training data.

    py teacher_train.py <hf model dir> <name> [--limit N]

Reads  <WORK>\\teacher_rerank\\distill\\train_nbest.jsonl (gen_train.py)
Writes <WORK>\\teacher_rerank\\distill\\teacher_<name>.json  {text: log10 P}
Same scoring as teacher_rerank.teacher (<|endoftext|> before the text), but batches of texts with the
same token length (no padding) up to TOKENS tokens, for speed.
"""
import json
import math
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tools/lm
from common import WORK  # noqa: E402  (DUALITYIME_LM_WORK moves it)
DIR = os.environ.get("TT_DIR", os.path.join(WORK, "teacher_rerank", "distill"))
TOKENS = 6144


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id, name = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    data = sys.argv[sys.argv.index("--data") + 1] if "--data" in sys.argv else "train_nbest.jsonl"
    reuse = sys.argv[sys.argv.index("--reuse") + 1] if "--reuse" in sys.argv else None
    old = json.load(open(os.path.join(DIR, reuse), encoding="utf-8")) if reuse else {}
    texts = set()
    with open(os.path.join(DIR, data), encoding="utf-8") as f:
        for line in f:
            for h in json.loads(line)["hyps"]:
                if h[0] not in old:
                    texts.add(h[0])
    texts = sorted(texts)[:limit] if limit else sorted(texts)
    print("reused", len(old), flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda().eval()
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.convert_tokens_to_ids("<|endoftext|>")
    by_len = defaultdict(list)
    for t in texts:
        ids = [bos] + tok(t, add_special_tokens=False)["input_ids"]
        by_len[len(ids)].append((t, ids))
    print(model_id, len(texts), "texts", flush=True)
    scores = dict(old)
    done, start = 0, time.time()
    for n, items in sorted(by_len.items()):
        rows = max(1, TOKENS // n)
        for i in range(0, len(items), rows):
            chunk = items[i:i + rows]
            x = torch.tensor([ids for _, ids in chunk], dtype=torch.long).cuda()
            with torch.no_grad():
                logits = model(input_ids=x).logits[:, :-1]
                picked = logits.gather(2, x[:, 1:, None])[..., 0].float()
                norm = torch.cat([torch.logsumexp(logits[j:j + 16].float(), -1) for j in range(0, len(chunk), 16)])
                del logits
            logp = (picked - norm).sum(1) / math.log(10)
            for (t, _), v in zip(chunk, logp.tolist()):
                scores[t] = v
            done += len(chunk)
        print(f"  len {n}: {done}/{len(texts)}  {time.time() - start:.0f}s", flush=True)
    path = os.path.join(DIR, f"teacher_{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False)
    print("saved", path, f"{time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
