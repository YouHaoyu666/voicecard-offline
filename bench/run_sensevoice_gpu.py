"""Run the eval set with SenseVoice-Small on the official FunASR (PyTorch, GPU), to compare speed with Qwen3-ASR on the same GPU.
Needs funasr in the bench virtual environment.

Runs in batches over data/eval/segments.jsonl (uniformly chunked wavs); sec is split across clips by duration; tags such as <|zh|><|NEUTRAL|> are stripped from the output.
Usage: run_sensevoice_gpu.py <tag> [batch_size] [language] [model_dir] [itn]
  model_dir: fine-tuned FunASR output directory (default: official FunAudioLLM/SenseVoiceSmall); itn=1 outputs punctuation, case and Arabic digits
"""
import json
import os
import re
import sys
import time

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from paths import EVAL, RESULTS  # noqa: E402  EVAL_SUITE=en switches to the English multi-accent set
TAG = re.compile(r"<\|[^|]*\|>")


def main(tag, bs="16", language="auto", model_dir="", itn="0"):
    from funasr import AutoModel
    bs = int(bs)
    t = time.time()
    use_itn = itn == "1"
    if model_dir:  # fine-tuned output directory (uses its model.pt), or a checkpoint file directly
        ckpt = model_dir if os.path.isfile(model_dir) else os.path.join(model_dir, "model.pt")
        model = AutoModel(model=os.path.dirname(ckpt), init_param=ckpt, device="cuda:0", disable_update=True)
    else:
        model = AutoModel(model="FunAudioLLM/SenseVoiceSmall", hub="hf", device="cuda:0", disable_update=True)
    print(f"load {time.time() - t:.1f}s", flush=True)
    segs = [json.loads(line) for line in open(os.path.join(EVAL, "segments.jsonl"), encoding="utf-8")]
    model.generate(input=[segs[0]["wav"]], language=language, use_itn=use_itn, batch_size=1, disable_pbar=True)  # warm-up
    order = sorted(range(len(segs)), key=lambda k: segs[k]["dur"])
    texts, sec, meta = {}, {}, {}
    torch.cuda.synchronize()
    t_all = time.time()
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        t = time.time()
        res = model.generate(input=[segs[k]["wav"] for k in idx], language=language, use_itn=use_itn,
                             batch_size=len(idx), disable_pbar=True)
        torch.cuda.synchronize()
        el = time.time() - t
        tot = sum(segs[k]["dur"] for k in idx) or 1
        for k, r in zip(idx, res):
            s = segs[k]
            texts.setdefault(s["id"], {})[s["j"]] = TAG.sub("", r.get("text", "")).strip()
            sec[s["id"]] = sec.get(s["id"], 0.0) + el * s["dur"] / tot
            meta[s["id"]] = s["set"]
    total = time.time() - t_all
    out = os.path.join(RESULTS, f"{tag}.jsonl")
    with open(out, "w", encoding="utf-8") as fo:
        for uid, parts in texts.items():
            fo.write(json.dumps({"id": uid, "set": meta[uid], "hyp": " ".join(parts[j] for j in sorted(parts)),
                                 "sec": sec[uid]}, ensure_ascii=False) + "\n")
    audio = sum(s["dur"] for s in segs)
    print(f"done {out}  audio {audio / 60:.1f} min  wall {total:.1f} s  = {audio / total:.0f}x realtime"
          f"  peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main(*sys.argv[1:])
