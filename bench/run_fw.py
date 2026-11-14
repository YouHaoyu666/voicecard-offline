"""Run the eval set with faster-whisper.

Usage: run_fw.py <model> <tag> [pipeline|none|zhen|en]
  pipeline = v0.1 pipeline settings (beam5 + VAD + auto language + Chinese initial_prompt)
  none     = same but without initial_prompt
  zhen     = v0.2 pipeline: language chosen only between zh/en, no prompt
  en       = product "English mode": language fixed to English, no prompt
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from paths import EVAL, RESULTS  # noqa: E402  EVAL_SUITE=en switches to the English multi-accent set
ASR_LANGS = ("zh", "en")  # languages the recorder transcribes


def _add_cuda_dlls():
    """Windows: put the pip-installed CUDA libraries (nvidia-cublas, cudnn, nvrtc) on the DLL search path."""
    import site
    for sp in site.getsitepackages():
        for sub in ("cublas", "cudnn", "cuda_nvrtc"):
            d = os.path.join(sp, "nvidia", sub, "bin")
            if os.path.isdir(d):
                os.add_dll_directory(d)
                os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]


_add_cuda_dlls()
from faster_whisper import WhisperModel, decode_audio  # noqa: E402


# The original pipeline's Chinese initial prompt (it names the project and parts); kept verbatim because the
# Singapore result in docs/singapore-asr.md shows what this kind of prompt does to English.
V01_PROMPT = "以下是普通话和英语混合的对话，使用简体中文和标点符号。可能出现：录音豆、Seeed、XIAO、ESP32-S3、nRF5340、Claude、纪要、PCB。"


def main(model_name, tag, mode="pipeline"):
    prompt = V01_PROMPT if mode == "pipeline" else None
    t = time.time()
    m = WhisperModel(model_name, device="cuda", compute_type="float16")
    print(f"load {time.time() - t:.1f}s", flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"{tag}.jsonl")
    with open(os.path.join(EVAL, "manifest.jsonl"), encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    with open(out, "w", encoding="utf-8") as fo:
        for i, it in enumerate(items):
            t = time.time()
            lang = None
            if mode == "en":  # product "English mode": language fixed to English
                lang = "en"
            elif mode == "zhen":
                audio = decode_audio(it["wav"])
                probs = dict(m.detect_language(audio=audio, vad_filter=True, language_detection_segments=3)[2])
                lang = max(ASR_LANGS, key=lambda k: probs.get(k, 0.0))
            segs, info = m.transcribe(it["wav"], language=lang, beam_size=5, vad_filter=True,
                                      initial_prompt=prompt, condition_on_previous_text=False)
            hyp = "".join(s.text for s in segs)
            fo.write(json.dumps({"id": it["id"], "set": it["set"], "hyp": hyp, "sec": time.time() - t,
                                 "lang": info.language}, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                print(f"{i}/{len(items)}", flush=True)
    print("done", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
