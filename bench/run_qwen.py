"""Run the eval set with Qwen3-ASR-architecture models (in the bench virtual environment): Qwen/Qwen3-ASR-1.7B, knoveleng/polyglot-lion-1.7b, etc.

Usage: run_qwen.py <hf_model_id> <tag> [batch_size] [context_file] [sets]
Clips are sorted by duration and run in batches; the batch time is split across clips by duration (sec field).
context_file: context (vocabulary) given to the model, empty if omitted; sets: comma-separated, run only these sets.
"""
import json
import os
import sys
import time

import torch
from qwen_asr import Qwen3ASRModel

ROOT = os.path.dirname(os.path.abspath(__file__))


def vram():
    return f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB"


def main(model_id, tag, bs="8", context_file="", sets=""):
    bs = int(bs)
    context = open(context_file, encoding="utf-8").read().strip() if context_file else ""
    t = time.time()
    model = Qwen3ASRModel.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0",
                                          max_inference_batch_size=bs, max_new_tokens=512)
    print(f"load {time.time() - t:.1f}s  peak {vram()}", flush=True)
    with open(os.path.join(ROOT, "data", "eval", "manifest.jsonl"), encoding="utf-8") as f:
        items = sorted((json.loads(line) for line in f), key=lambda r: r["dur"])
    if sets:
        items = [it for it in items if it["set"] in sets.split(",")]
    model.transcribe(audio=[items[0]["wav"]], language=None)  # warm-up
    out = os.path.join(ROOT, "results", f"{tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fo:
        for i in range(0, len(items), bs):
            batch = items[i:i + bs]
            t = time.time()
            res = model.transcribe(audio=[it["wav"] for it in batch], context=[context] * len(batch), language=None)
            el = time.time() - t
            tot = sum(it["dur"] for it in batch)
            for it, r in zip(batch, res):
                fo.write(json.dumps({"id": it["id"], "set": it["set"], "hyp": r.text, "lang": r.language,
                                     "sec": el * it["dur"] / tot}, ensure_ascii=False) + "\n")
            if (i // bs) % 10 == 0:
                print(f"{i}/{len(items)}  peak {vram()}", flush=True)
    print("done", out, "peak", vram())


if __name__ == "__main__":
    main(*sys.argv[1:])
