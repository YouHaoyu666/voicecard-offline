# English accent test: 12 open speech-recognition models

Measured 2026-10-19. Question: is a model in the 0.2B class (small enough for a card-sized device) good enough for English across regional accents, and which model is the best base to build on?

Scripts: `bench/prep_en.py` builds the test set from public datasets, `bench/run_*.py` run the models, `bench/score_en.py` scores (`--region` for the per-region table).

## Test set

474 utterances, about 49 minutes, 1.5 to 25 s each.

| Subset | Utterances | Content |
|---|---|---|
| EdAcc (Edinburgh International Accents of English, test) | 183 | Free conversation, 23 accents, 8 each |
| SD-QA (test) | 132 | Read questions, 11 regions × 12: US, UK, Ireland, Australia, New Zealand, India north and south, Kenya, Nigeria, South Africa, Philippines |
| Common Voice accent subset (DTU54DL/common-accent) | 39 | Read speech, single-accent labels only |
| AMI (single distant microphone) | 40 | Meeting room, far field |
| Earnings-22 | 40 | Earnings calls |
| LibriSpeech test-other | 40 | US read speech, as a control |

- Regions (utterances): North America 60, UK/Ireland 48, Australia/NZ 24, South Asia 32, Africa 60, Southeast Asia 51, East Asia 18, Europe (non-native) 77, Middle East 8, Caribbean 8, Latin America 8. The last three are small; treat them as indicative only.
- Metric: WER. References and hypotheses both go through Whisper's `EnglishTextNormalizer` (the Open ASR Leaderboard convention), then tags such as `<laugh>` and `<|en|>` are removed.
- Two utterances with wrong references are excluded: `edacc_0033` is marked `IGNORE_TIME_SEGMENT_IN_SCORING` in the EdAcc labels, and the reference of `edacc_0058` covers only the second half (all 15 systems output about 25 extra words at the start). Excluding them lowers every model by 1.2 to 1.4 points and does not change the ranking.

## Results (WER %, lower is better)

### By subset

| Model | Params | Architecture | Overall | Conversation | Read questions | CV accents | Meeting far field | Earnings calls | Libri | CPU speed | GPU speed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-ASR-1.7B (English forced) | 2.35B | LLM | **9.8** | **10.7** | 6.2 | 5.8 | **23.3** | 8.0 | 2.5 | — | 42× |
| Qwen3-ASR-0.6B (English forced) | 0.94B | LLM | 11.3 | 12.3 | 8.1 | 7.6 | 24.6 | **7.9** | 4.3 | 2× ¹ | 45× |
| Parakeet-TDT-0.6B v2 | 0.6B | TDT | 11.8 | 13.7 | 8.3 | **4.1** | 25.0 | 9.6 | **1.6** | 4× | — |
| Parakeet-TDT-0.6B v3 | 0.6B | TDT | 13.9 | 14.8 | 9.2 | 11.4 | 29.3 | 14.0 | 3.8 | 3× | — |
| Whisper large-v3-turbo | 0.8B | AED | 14.0 | 16.0 | **6.0** | 5.3 | 33.6 | 12.5 | 3.6 | — | 10× |
| X-ASR-zh-en, streaming 1920 ms | 0.16B | Streaming Zipformer | 15.0 | 14.8 | 17.4 | 17.5 | 25.4 | 11.1 | 6.3 | 3× ² | — |
| Canary-180M-flash | 0.18B | AED | 16.1 | 17.3 | 11.6 | 6.8 | 31.9 | 21.1 | 3.2 | 3× | — |
| SenseVoice-Small (English forced) | 0.23B | CTC | 16.8 | 18.4 | 16.8 | 12.7 | 28.2 | 12.3 | 5.7 | 12× (int8: 17.2) | **726×** |
| Whisper small | 0.24B | AED | 17.0 | 18.8 | 9.9 | 11.1 | 37.9 | 12.9 | 6.0 | — | 13× |
| Parakeet-TDT-CTC-110M | 0.11B | CTC | 17.2 | 18.3 | 15.6 | 11.4 | 35.3 | 12.3 | 6.3 | **12×** | — |
| Moonshine base | 61M | AED | 23.6 | 24.9 | 23.1 | 20.8 | 47.4 | 16.0 | 6.6 | 5× | — |
| Moonshine tiny | 27M | AED | 29.3 | 31.7 | 27.4 | 21.3 | 56.3 | 19.4 | 10.1 | 9× | — |

- Speeds are × real time. CPU: Intel i5-13500HX (laptop), sherpa-onnx int8, 4 threads, the closest setup to an embedded board CPU. GPU: RTX 5090 Laptop for Qwen3-ASR and SenseVoice, RTX 4060 Laptop for Whisper; batch sizes and frameworks differ, so only compare GPU speeds roughly.
- ¹ Measured on a Chinese test set.
- ² X-ASR needs about 2.9 s of trailing silence per utterance to flush the last streaming chunk, which slows short utterances; on continuous audio it runs at about 4× (measured on a Chinese test set).
- SenseVoice without a forced language: 17.2. Qwen3-ASR with a guard (language limited to zh/en, repetition guard): 0.6B 11.4, 1.7B 9.9, about the same as English forced.

### By region (EdAcc + SD-QA + Common Voice)

| Model | North America | UK/Ireland | Australia/NZ | South Asia | Africa | Southeast Asia | East Asia | Europe (non-native) | Middle East | Caribbean | Latin America |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | 4.7 | **7.1** | 6.6 | **9.9** | **10.1** | **10.1** | **11.1** | **9.0** | **10.0** | 21.8 | **7.1** |
| Qwen3-0.6B | 5.9 | 9.2 | 7.8 | 11.8 | 12.6 | 10.9 | 13.7 | 10.0 | 11.6 | 23.9 | 12.5 |
| Parakeet v2 | **4.5** | 10.0 | 6.6 | 12.9 | 14.3 | 12.8 | 13.1 | 11.3 | 11.1 | **20.4** | 8.9 |
| Parakeet v3 | 6.0 | 14.0 | **3.6** | 13.2 | 16.5 | 15.8 | 15.3 | 11.2 | 13.7 | 24.6 | 10.7 |
| Whisper turbo | 6.2 | 10.8 | 5.4 | 11.0 | 14.7 | 15.4 | 13.7 | 13.3 | 11.1 | 34.5 | 9.5 |
| X-ASR 0.16B | 7.9 | 15.4 | 11.4 | 20.2 | 21.2 | 18.9 | 16.6 | 11.4 | 13.7 | 23.9 | 10.7 |
| Canary 180M | 6.5 | 13.9 | 10.8 | 17.3 | 18.8 | 14.5 | 16.2 | 13.3 | 14.7 | 39.4 | 10.7 |
| SenseVoice | 7.8 | 14.1 | 13.9 | 21.3 | 22.1 | 19.4 | 18.2 | 16.5 | 15.3 | 28.2 | 14.9 |
| Whisper small | 8.8 | 14.4 | 10.8 | 14.3 | 18.2 | 18.9 | 18.5 | 15.3 | 22.6 | 31.0 | 11.9 |
| Parakeet 110M | 8.6 | 14.1 | 12.7 | 20.6 | 21.5 | 17.3 | 20.1 | 15.6 | 18.4 | 30.3 | 13.1 |
| Moonshine base | 11.3 | 18.9 | 23.5 | 28.3 | 24.0 | 22.5 | 57.0 | 20.6 | 16.8 | 38.0 | 17.9 |
| Moonshine tiny | 14.9 | 28.2 | 24.7 | 37.1 | 31.7 | 31.2 | 29.6 | 29.4 | 25.3 | 45.8 | 24.4 |

## What the numbers say

1. **Three tiers by size.** 0.6B and up (Qwen3-ASR, Parakeet v2): 11–12%. 0.1–0.25B (X-ASR, Canary, SenseVoice, Whisper small, Parakeet 110M): 15–17%. Moonshine (27–61M): 24–29%.
2. **The gap is in the accents.** On North American English the small models are only 1–4 points behind. On South Asian, African and Southeast Asian English, the small models are mostly at 14–22% and the 0.6B tier at 11–14%. This looks like training-data coverage: at the same 0.6B size, Parakeet v2 (mostly US/UK training data) is about 2 points behind Qwen3-0.6B on African and Southeast Asian accents.
3. **Names and places hurt small models.** Read questions and Common Voice are full of proper nouns; SenseVoice and X-ASR are at 16–17% there, the 0.6B tier at 4–8%. A hotword list can patch common names on device.
4. **Frame-aligned architectures never failed catastrophically.** CTC, TDT and streaming transducers produce output bounded by the number of audio frames: no repetition loops, no language confusion. Small encoder-decoder (AED) models did fail: Canary 180M produced one empty output, one output of just "1", and one that repeated "never" to the end; Moonshine base produced 14 empty outputs and one loop. Parakeet v3 (TDT, multilingual) dropped 5 whole utterances. **On a device, prefer CTC, TDT or transducer models.**
5. **SenseVoice-Small** loses little to int8 quantization (17.2 vs 16.8), runs at 12× real time on the same CPU (tied fastest), and has the most mature NPU support. Its weak spot is accents (South Asia 21, Africa 22).

## What we did with it

Architecture is not the bottleneck: models of very different architectures land at 15–17% in the same size class. Data is. So instead of switching models we kept SenseVoice-Small (fast, mature NPU support, good Chinese) and distilled accent coverage into it from larger teachers. That is the next write-up.

## Sources

- Parakeet-TDT-0.6B v2 model card: https://build.nvidia.com/nvidia/parakeet-tdt-0_6b-v2/modelcard
- Parakeet-TDT-CTC-110M: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/parakeet-tdt_ctc-110m
- X-ASR-zh-en: https://huggingface.co/GilgameshWind/X-ASR-zh-en
- SenseVoice: https://github.com/FunAudioLLM/SenseVoice
- sherpa-onnx: https://github.com/k2-fsa/sherpa-onnx
- Datasets: EdAcc, SD-QA, DTU54DL/common-accent, AMI, Earnings-22, LibriSpeech (each under its own licence; this repo ships the sampling script, not the audio).
