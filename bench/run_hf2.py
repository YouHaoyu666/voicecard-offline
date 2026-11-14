"""Eval for models that need a newer transformers (separate virtual environment): GLM-ASR-Nano, MERaLiON-3-3B-ASR.
Long audio is chunked by data/eval/chunks.json (same as for the other models), transcribed per chunk and joined; all chunks are sorted by length and run in batches.

Usage: run_hf2.py <glm|meralion> <tag> [batch_size]
"""
import json
import os
import sys
import time

import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
SR = 16000


class GLM:
    mid = "zai-org/GLM-ASR-Nano-2512"

    def __init__(self):
        from transformers import AutoProcessor, GlmAsrForConditionalGeneration
        self.p = AutoProcessor.from_pretrained(self.mid)
        self.m = GlmAsrForConditionalGeneration.from_pretrained(self.mid, dtype=torch.bfloat16, device_map="cuda:0")

    def __call__(self, arrays):
        inputs = self.p.apply_transcription_request(list(arrays)).to(self.m.device, dtype=self.m.dtype)
        out = self.m.generate(**inputs, do_sample=False, max_new_tokens=500)
        return self.p.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)


class MERaLiON:
    mid = "MERaLiON/MERaLiON-3-3B-ASR"

    def __init__(self):
        from meralion_3_asr import Meralion3ASR
        # 3B bf16 is about 6.6 GB, so an 8 GB card only fits small batches
        self.m = Meralion3ASR.from_pretrained(self.mid, backend="transformers", batch_size=2)

    def __call__(self, arrays):
        return self.m.transcribe_batch([(a, SR) for a in arrays])


def main(kind, tag, bs="4"):
    bs = int(bs)
    items = [json.loads(line) for line in open(os.path.join(ROOT, "data", "eval", "manifest.jsonl"), encoding="utf-8")]
    chunks = json.load(open(os.path.join(ROOT, "data", "eval", "chunks.json"), encoding="utf-8"))
    segs = []  # (item_idx, seg_idx, array)
    for i, it in enumerate(items):
        audio, sr = sf.read(it["wav"], dtype="float32")
        assert sr == SR
        spans = chunks.get(it["id"]) or [[0, len(audio) / SR]]
        for j, (s, e) in enumerate(spans):
            segs.append((i, j, audio[int(s * SR):int(e * SR)]))
    t = time.time()
    model = {"glm": GLM, "meralion": MERaLiON}[kind]()
    print(f"load {time.time() - t:.1f}s  segments={len(segs)}", flush=True)
    model([segs[0][2]])  # warm-up
    order = sorted(range(len(segs)), key=lambda k: len(segs[k][2]))
    texts, sec = {}, {}
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        t = time.time()
        res = model([segs[k][2] for k in idx])
        el = time.time() - t
        tot = sum(len(segs[k][2]) for k in idx)
        for k, r in zip(idx, res):
            i, j, a = segs[k]
            texts.setdefault(i, {})[j] = (r or "").strip()
            sec[i] = sec.get(i, 0.0) + el * len(a) / tot
        if (b // bs) % 20 == 0:
            print(f"{b}/{len(order)}  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
    out = os.path.join(ROOT, "results", f"{tag}.jsonl")
    with open(out, "w", encoding="utf-8") as fo:
        for i, it in enumerate(items):
            hyp = " ".join(texts[i][j] for j in sorted(texts[i]))
            fo.write(json.dumps({"id": it["id"], "set": it["set"], "hyp": hyp, "sec": sec[i]}, ensure_ascii=False) + "\n")
    print("done", out, f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


if __name__ == "__main__":
    main(*sys.argv[1:])
