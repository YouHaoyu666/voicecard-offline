"""Qwen3-ASR eval with repetition guard, and language restricted to Chinese/English (in the bench virtual environment).

Small models make two catastrophic kinds of error on Singapore code-switched speech; a single clip can raise a set's MER by several points:
- repetition loops: the same sentence is output over and over until max_new_tokens;
- language misdetection: Singapore English is detected as Malay/Thai/Cantonese and the model outputs a "translation" in that language.
Approach:
1. max_new_tokens per batch is capped by duration (12 tokens/s + 32), so a runaway clip does not burn 512 tokens;
2. output that looks like a loop (UTF-8 zlib compression ratio > 4, or more than 8 Chinese characters/English words per second)
   is re-decoded alone with no_repeat_ngram_size=8;
3. a detected language outside Chinese/English is re-decoded alone with the language forced (Chinese if more than 30% of
   the first-pass output is Chinese characters, otherwise English).
The loop thresholds were calibrated on 410 local clips: looped outputs all had compression ratio >= 4.5 and >= 8.3 words/s;
normal outputs <= 3.2 and <= 6.7 words/s.
Note: Qwen3ASRForConditionalGeneration.generate forwards its arguments to the thinker, so the generation config to change is model.thinker.generation_config.

Usage: run_qwen_guard.py <hf_model_id> <tag> [batch_size] [fixed_lang, e.g. English]; EVAL_SUITE=en runs the English multi-accent set
"""
import json
import os
import re
import sys
import time
import zlib

import torch
from qwen_asr import Qwen3ASRModel

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from paths import EVAL, RESULTS  # noqa: E402  EVAL_SUITE=en switches to the English multi-accent set
WORD = re.compile(r"[一-鿿]|[A-Za-z]+|\d+")
CJK = re.compile(r"[一-鿿]")
ALLOWED = {"Chinese", "English"}


def looks_looped(text, dur):
    b = text.encode("utf-8")
    ratio = len(b) / max(1, len(zlib.compress(b)))
    return ratio > 4.0 or len(WORD.findall(text)) / max(dur, 1.0) > 8.0


def off_language(lang):
    return bool(lang) and not set(lang.split(",")) <= ALLOWED


def pick_language(text):
    words = WORD.findall(text)
    return "Chinese" if words and sum(bool(CJK.match(w)) for w in words) / len(words) > 0.3 else "English"


def cap(dur):
    return int(dur * 12) + 32


def main(model_id, tag, bs="8", fixed_lang=""):
    """fixed_lang: if given (e.g. English), force this language for every batch (the product's "English mode") and keep only the repetition guard."""
    bs = int(bs)
    fixed_lang = fixed_lang or None
    model = Qwen3ASRModel.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0",
                                          max_inference_batch_size=bs, max_new_tokens=512)
    gen = model.model.thinker.generation_config
    default_ngram = gen.no_repeat_ngram_size
    with open(os.path.join(EVAL, "manifest.jsonl"), encoding="utf-8") as f:
        items = sorted((json.loads(line) for line in f), key=lambda r: r["dur"])
    model.transcribe(audio=[items[0]["wav"]], language=None)  # warm-up
    out = os.path.join(RESULTS, f"{tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    loops, langs_fixed = [], []
    with open(out, "w", encoding="utf-8") as fo:
        for i in range(0, len(items), bs):
            batch = items[i:i + bs]
            t = time.time()
            model.max_new_tokens = cap(max(it["dur"] for it in batch))
            res = model.transcribe(audio=[it["wav"] for it in batch], language=fixed_lang)
            texts, langs = [r.text for r in res], [r.language for r in res]
            for j, it in enumerate(batch):
                looped, off = looks_looped(texts[j], it["dur"]), not fixed_lang and off_language(langs[j])
                if not (looped or off):
                    continue
                force = pick_language(texts[j]) if off else fixed_lang
                (langs_fixed if off else loops).append(f"{it['id']}:{langs[j]}->{force or 'auto'}")
                gen.no_repeat_ngram_size = 8 if looped else default_ngram
                model.max_new_tokens = cap(it["dur"])
                r2 = model.transcribe(audio=[it["wav"]], language=force)[0]
                gen.no_repeat_ngram_size = default_ngram
                texts[j], langs[j] = r2.text, r2.language
            el = time.time() - t
            tot = sum(it["dur"] for it in batch)
            for it, txt, lang in zip(batch, texts, langs):
                fo.write(json.dumps({"id": it["id"], "set": it["set"], "hyp": txt, "lang": lang,
                                     "sec": el * it["dur"] / tot}, ensure_ascii=False) + "\n")
            if (i // bs) % 10 == 0:
                print(f"{i}/{len(items)}", flush=True)
    print("done", out)
    print(f"loop retries {len(loops)}:", " ".join(loops))
    print(f"language retries {len(langs_fixed)}:", " ".join(langs_fixed))


if __name__ == "__main__":
    main(*sys.argv[1:])
