"""Teacher candidate: IBM Granite Speech 3.3 (Apache-2.0) on the English accent suite (data/eval_en), batch 1, greedy.
Usage follows the model card (chat template with <|audio|>, "can you transcribe the speech into a written format?").
Run in the bench virtual environment (transformers 4.57).

  run_granite.py <2b|8b> <tag>      -> results_en/<tag>.jsonl
"""
import json
import os
import sys
import time

import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(ROOT, "data", "eval_en")
SYSTEM = ("Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\n"
          "You are Granite, developed by IBM. You are a helpful AI assistant")
USER = "<|audio|>can you transcribe the speech into a written format?"


def main(size, tag):
    mid = f"ibm-granite/granite-speech-3.3-{size}"
    proc = AutoProcessor.from_pretrained(mid)
    tok = proc.tokenizer
    model = AutoModelForSpeechSeq2Seq.from_pretrained(mid, device_map="cuda:0", torch_dtype=torch.bfloat16)
    prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
                                     tokenize=False, add_generation_prompt=True)
    items = [json.loads(l) for l in open(os.path.join(EVAL, "manifest.jsonl"), encoding="utf-8")]
    os.makedirs(os.path.join(ROOT, "results_en"), exist_ok=True)
    out = os.path.join(ROOT, "results_en", f"{tag}.jsonl")
    with open(out, "w", encoding="utf-8") as fo:
        for k, it in enumerate(items):
            wav = os.path.join(EVAL, it["set"], f"{it['id']}.wav")  # rebuild the path from the set/id instead of using the manifest's absolute path
            a, sr = sf.read(wav, dtype="float32")
            assert sr == 16000
            t = time.time()
            inputs = proc(prompt, torch.from_numpy(a)[None], device="cuda:0", return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                o = model.generate(**inputs, max_new_tokens=int(12 * it["dur"]) + 32, do_sample=False, num_beams=1)
            text = tok.batch_decode(o[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0].strip()
            fo.write(json.dumps({"id": it["id"], "set": it["set"], "hyp": text, "sec": time.time() - t},
                                ensure_ascii=False) + "\n")
            if k % 50 == 0:
                print(k, len(items), text[:80], flush=True)
    print("done", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
