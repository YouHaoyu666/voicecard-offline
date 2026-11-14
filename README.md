# VoiceCard Offline

An offline AI meeting recorder: speech recognition and meeting notes run on a small NPU, with no cloud, no app and no subscription. The goal is a card-sized device that records all day, understands Singapore-style Chinese-English code-switching, and never sends audio anywhere.

This repository is built in public. One engineer defines the product, the architecture and the acceptance tests; AI agents do most of the building. Every result here comes with the script that produced it, including the ones that failed.

## Status (November 2026)

| Part | State | Numbers |
|---|---|---|
| Speech recognition on the NPU | Works | SenseVoice-Small (~230M params) fine-tuned for accents: Singapore code-switching error 55% → 27% (NSC test set, same corpus as part of the training data); on the board, LibriSpeech test-clean 3.8% WER |
| Meeting notes on the NPU | Works, quality still well below a large model | 2B model, blind review 4.55 / 10 vs 7.15 for the large teacher model |
| End to end on a dev board | Works on recorded audio | 39-minute meeting → notes in 20 minutes, fully offline, Orange Pi 5 (RK3588S, 4 GB) |
| Live microphone | Dev-board microphone not usable at meeting distance | Product design uses two digital MEMS microphones |
| Custom board, enclosure, battery | Not started | — |

## Write-ups

Published in batches, each with its code:

1. **Accent tests** (this release)
   - [Singapore English: Whisper outputs Malay](docs/singapore-asr.md)
   - [English accent test: 12 open speech-recognition models](docs/accent-test.md)
2. Distilling accent coverage into a 230M model (7 versions, what helped and what made English worse)
3. Running it on an RK3588 board: fp16 overflow behind empty transcripts, and the end-to-end demo
4. Teaching a 2B model to write meeting notes: 15 versions, the failures, and the rules that came out of them
5. Custom board and enclosure

## Layout

- `bench/` scripts: build test sets from public datasets, run models, score
- `docs/` write-ups

Datasets are not redistributed. Each script downloads its data from the original source, under that dataset's licence.

## Running the accent test

```
cd bench
python -m venv .venv
.venv/Scripts/pip install faster-whisper pyarrow soundfile huggingface_hub whisper-normalizer sherpa-onnx
python prep_en.py                       # samples the 474-utterance test set into data/eval_en/
```

Then run one model family per script, with `EVAL_SUITE=en` set so results go to `results_en/`:

- `run_sherpa.py <model> <tag> [threads]`: sherpa-onnx int8 on CPU (Parakeet, Moonshine, Canary, X-ASR, Qwen3-ASR int8). Download the matching sherpa-onnx model into `bench/models_sherpa/` first; the folder names are in the script.
- `run_fw.py <model> <tag> en`: faster-whisper on GPU.
- `run_qwen.py`, `run_qwen_guard.py`, `run_sensevoice_gpu.py`, `run_granite.py`, `run_hf2.py`: Hugging Face models on GPU.

Score: `python score_en.py --region results_en/*.jsonl`

## Licence

Code: Apache-2.0 (see `LICENSE`). Model weights, when released, keep the licence of the model they were fine-tuned from (SenseVoice: FunASR model licence; Qwen: Apache-2.0).
