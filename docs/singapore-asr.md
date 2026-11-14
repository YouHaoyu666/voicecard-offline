# Singapore English and code-switching: Whisper outputs Malay

Measured 2026-10-18, before any training. 410 real utterances, about 90 minutes. The scripts for this test (sampling the ASCEND and MNSC test sets, the Whisper pipeline variants) are added with the next batch.

## Test set

| Set | Source | Utterances | Duration | Content |
|---|---|---|---|---|
| ascend_mixed | ASCEND test (CAiRE/ASCEND, CC BY-SA 4.0) | 150 | 11.4 min | Hong Kong students, Chinese-English code-switching, short |
| ascend_zh / ascend_en | same | 50 / 50 | 3.2 / 3.3 min | Chinese only / English only |
| nsc_p3 | MNSC ASR-PART3 test (IMDA National Speech Corpus) | 60 | 28.0 min | Singapore English free conversation, about 28 s each |
| nsc_p4_zh | MNSC ASR-PART4 test, utterances containing Chinese | 100 | 44.4 min | Singapore Chinese-English code-switching, about 27 s each |

Metric: MER (mixed error rate: characters for Chinese, words for English). References and outputs share one normalisation: NSC speaker labels, fillers like `(uh)` and particles like `[lah]` removed, traditional to simplified Chinese, Chinese numerals to digits, lower case. Spelling variants such as ya/yeah still count as errors, so absolute numbers on free conversation are high; compare rows, not absolutes.

## Results (MER %, lower is better)

| Model / setup | ascend_mixed | ascend_zh | ascend_en | nsc_p3 | nsc_p4_zh |
|---|---|---|---|---|---|
| Whisper large-v3, automatic language detection, Chinese prompt | 27.3 | 18.8 | 29.6 | **86.6** | **89.2** |
| Whisper large-v3-turbo, Chinese prompt | 16.9 | 12.8 | 33.7 | 80.7 | 72.5 |
| Whisper large-v3-turbo, no prompt | 15.3 | 12.0 | 23.6 | 41.1 | 52.6 |
| Whisper large-v3-turbo, language limited to zh/en, no prompt | 15.3 | 12.0 | 23.6 | 27.5 | 48.4 |
| SenseVoice-Small int8 (CPU, sherpa-onnx) | 11.2 | 6.0 | 19.5 | 24.9 | 53.8 |
| Polyglot-Lion-1.7B (Singapore fine-tune of Qwen3-ASR) | 10.7 | 6.1 | 16.6 | 25.0 | 41.3 |
| **Qwen3-ASR-1.7B** (bf16, transformers) | **9.3** | **5.6** | **15.9** | **22.4** | **39.7** |
| Qwen3-ASR-1.7B, then an LLM corrects the text (no audio) | 9.4 | – | – | 22.1 | 39.6 |

## Findings

1. **Whisper large-v3 with automatic language detection is unusable for Singapore speech.** It labelled 51 of 60 nsc_p3 utterances and 84 of 100 nsc_p4_zh utterances as Malay, and **output a Malay translation** instead of a transcript. A Chinese initial prompt also drags English toward Chinese (turbo: 41% → 81% on nsc_p3). Limiting the language to zh/en and dropping the prompt fixes most of it.
2. **Whisper translates the other language away in code-switched speech.** For example, "你买 like 一张两张" came out as "you buy like one, two".
3. **Qwen3-ASR-1.7B was best on all five sets** and handles code-switching natively ("都是很entry level的东西" word for word).
4. **An LLM correcting the text without the audio did not help** (39.7 → 39.6). It cannot recover words the recogniser got wrong; accuracy has to come from the speech model.
5. **Singapore code-switched free conversation is the hardest case** (best 39.7%): fast, colloquial, full of names. The Singapore accent decides the ranking far more than model size does.

## Caveat

The MNSC test sets come from the same corpus as MNSC training data, so models trained on MNSC have a home advantage on nsc_p3 and nsc_p4_zh. Later results in this repo are also checked on held-out recordings and public test sets.
