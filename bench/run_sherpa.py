"""Run on-device candidates with sherpa-onnx on CPU (closest to the dev board), transcribing data/eval/segments.jsonl segment by segment.

Usage: run_sherpa.py <qwen3-int8|xasr-1920|xasr-480|parakeet-v2|parakeet-v3|parakeet-110m|moonshine-base|moonshine-tiny|canary-180m> <tag> [num_threads]; EVAL_SUITE=en runs the English multi-accent set
  qwen3-int8: sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25 (offline)
  xasr-*: GilgameshWind/X-ASR-zh-en streaming Zipformer (0.16B, fp32 onnx), one variant per chunk length
Records the CPU wall-clock time per clip (sec); RTFx in score.py is then the CPU real-time factor.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import sherpa_onnx
import soundfile as sf

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from paths import EVAL, RESULTS  # noqa: E402  EVAL_SUITE=en switches to the English multi-accent set
MS = os.path.join(ROOT, "models_sherpa")
HF = os.path.join(ROOT, "hf", "hub")


def xasr_dir(chunk):
    d = glob.glob(os.path.join(HF, "models--GilgameshWind--X-ASR-zh-en", "snapshots", "*", "deployment", "models",
                               f"chunk-{chunk}ms-model"))
    return d[0]


def offline(rec):
    def run(audio):
        s = rec.create_stream()
        s.accept_waveform(16000, audio)
        rec.decode_stream(s)
        return s.result.text
    return run


def make(kind, threads):
    # small English models (sherpa-onnx int8 packages, see models_sherpa/)
    p = lambda name, *f: os.path.join(MS, name, *f)  # noqa: E731
    if kind in ("parakeet-v2", "parakeet-v3"):
        d = f"sherpa-onnx-nemo-parakeet-tdt-0.6b-{kind[-2:]}-int8"
        return offline(sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=p(d, "encoder.int8.onnx"), decoder=p(d, "decoder.int8.onnx"), joiner=p(d, "joiner.int8.onnx"),
            tokens=p(d, "tokens.txt"), num_threads=threads, model_type="nemo_transducer"))
    if kind == "parakeet-110m":
        d = "sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8"
        return offline(sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=p(d, "model.int8.onnx"), tokens=p(d, "tokens.txt"), num_threads=threads))
    if kind in ("moonshine-base", "moonshine-tiny"):
        d = f"sherpa-onnx-{kind}-en-int8"
        return offline(sherpa_onnx.OfflineRecognizer.from_moonshine(
            preprocessor=p(d, "preprocess.onnx"), encoder=p(d, "encode.int8.onnx"),
            uncached_decoder=p(d, "uncached_decode.int8.onnx"), cached_decoder=p(d, "cached_decode.int8.onnx"),
            tokens=p(d, "tokens.txt"), num_threads=threads))
    if kind == "canary-180m":
        d = "sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8"
        return offline(sherpa_onnx.OfflineRecognizer.from_nemo_canary(
            encoder=p(d, "encoder.int8.onnx"), decoder=p(d, "decoder.int8.onnx"), tokens=p(d, "tokens.txt"),
            src_lang="en", tgt_lang="en", num_threads=threads))
    if kind == "qwen3-int8":
        d = glob.glob(os.path.join(MS, "sherpa-onnx-qwen3-asr-0.6B-int8-*"))[0]
        rec = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=os.path.join(d, "conv_frontend.onnx"), encoder=os.path.join(d, "encoder.int8.onnx"),
            decoder=os.path.join(d, "decoder.int8.onnx"), tokenizer=os.path.join(d, "tokenizer"),
            num_threads=threads, max_total_len=1024, max_new_tokens=512)

        def run(audio):
            s = rec.create_stream()
            s.accept_waveform(16000, audio)
            rec.decode_stream(s)
            return s.result.text
        return run
    chunk = kind.split("-")[1]
    d = xasr_dir(chunk)
    rec = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=os.path.join(d, "tokens.txt"), encoder=os.path.join(d, f"encoder-{chunk}ms.onnx"),
        decoder=os.path.join(d, f"decoder-{chunk}ms.onnx"), joiner=os.path.join(d, f"joiner-{chunk}ms.onnx"),
        num_threads=threads, sample_rate=16000, feature_dim=80, decoding_method="greedy_search")

    tail = np.zeros(int((int(chunk) / 1000 + 1.0) * 16000), dtype=np.float32)

    def run(audio):
        s = rec.create_stream()
        s.accept_waveform(16000, audio)
        # pad the tail with "chunk length + 1 s" of silence to flush the last chunk (with only 0.66 s the 1920 ms variant cuts off the end of short sentences)
        s.accept_waveform(16000, tail)
        s.input_finished()
        while rec.is_ready(s):
            rec.decode_stream(s)
        return rec.get_result(s)
    return run


def main(kind, tag, threads="4"):
    run = make(kind, int(threads))
    segs = [json.loads(line) for line in open(os.path.join(EVAL, "segments.jsonl"), encoding="utf-8")]
    only = os.environ.get("EVAL_SETS")  # e.g. EVAL_SETS=nsc_p3,ascend_en runs only these sets (English models skip the Chinese sets)
    if only:
        segs = [s for s in segs if s["set"] in only.split(",")]
    texts, sec, meta = {}, {}, {}
    for k, s in enumerate(segs):
        audio, sr = sf.read(s["wav"], dtype="float32")
        t = time.time()
        txt = run(audio)
        el = time.time() - t
        texts.setdefault(s["id"], {})[s["j"]] = (txt if isinstance(txt, str) else str(txt)).strip()
        sec[s["id"]] = sec.get(s["id"], 0.0) + el
        meta[s["id"]] = s["set"]
        if k % 100 == 0:
            print(f"{k}/{len(segs)}", flush=True)
    out = os.path.join(RESULTS, f"{tag}.jsonl")
    with open(out, "w", encoding="utf-8") as fo:
        for uid, parts in texts.items():
            fo.write(json.dumps({"id": uid, "set": meta[uid], "hyp": " ".join(parts[j] for j in sorted(parts)),
                                 "sec": sec[uid]}, ensure_ascii=False) + "\n")
    print("done", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
