"""English multi-accent eval set: sample from public, no-login datasets and write 16 kHz mono wav + data/eval_en/manifest.jsonl.
Run in the bench virtual environment.

Sets (all <= 25 s, no chunking needed):
  en_edacc   EdAcc test (Edinburgh, spontaneous conversation, 24 accents): 8 clips per accent
  en_sdqa    SD-QA test (same questions read by speakers from 11 English regions): 12 questions x 11 regions
  en_cv      common-accent test (Common Voice accent subset, read speech): up to 8 clips per accent
  en_ami     AMI meetings SDM (single far-field tabletop mic): 40 clips
  en_earn    Earnings-22 earnings calls (speakers from around the world): 40 clips
  en_libri   LibriSpeech test-other (American read speech, baseline): 40 clips
Each clip carries accent (original label) and region (our grouping); score_en.py aggregates by region.
Raw parquet files are in data/raw_en/.
"""
import io
import json
import os
import random
import re

import pyarrow.parquet as pq
import soundfile as sf
from faster_whisper import decode_audio
from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw_en")
OUT = os.path.join(ROOT, "data", "eval_en")
MAX_S, MIN_S = 25.0, 1.5

NA, UK, ANZ, SA = "North America", "UK/Ireland", "Australia/NZ", "South Asia"
AF, SEA, EA, EU = "Africa", "Southeast Asia", "East Asia", "Europe (non-native)"
ME, CAR, LATAM, OTHER = "Middle East", "Caribbean", "Latin America", "Other"

REGION = {
    # EdAcc accent labels
    "Mainstream US English": NA, "Southern British English": UK, "Scottish English": UK,
    "Irish English": UK, "Nigerian English": AF, "Kenyan English": AF, "Ghanain English": AF,
    "Jamaican English": CAR, "Indian English": SA, "Vietnamese": SEA, "Indonesian": SEA,
    "Chinese": EA, "Israeli": ME, "Latin American": LATAM, "Spanish": EU,
    "Lithuanian": EU, "Italian": EU, "Romanian": EU, "Catalan": EU,
    "French": EU, "Eastern European": EU, "European": EU, "Bulgarian": EU,
    # SD-QA dialect columns
    "usa": NA, "gbr": UK, "irl": UK, "aus": ANZ, "nzl": ANZ, "ind_n": SA,
    "ind_s": SA, "kenya": AF, "nga": AF, "zaf": AF, "phl": SEA,
    # common-accent labels (Common Voice)
    "us": NA, "canada": NA, "england": UK, "scotland": UK, "ireland": UK,
    "wales": UK, "australia": ANZ, "newzealand": ANZ, "indian": SA, "african": AF,
    "philippines": SEA, "malaysia": SEA, "singapore": SEA, "hongkong": EA, "bermuda": CAR,
    "southatlandtic": OTHER,
    # scenario sets are not split by region; each forms its own group
    "Meeting far-field (AMI)": "Meeting far-field", "Earnings call (Earnings-22)": "Earnings call",
}


CV_KEYWORDS = [("United States", NA), ("Canad", NA), ("England", UK), ("Scot", UK),
               ("Irish", UK), ("Australia", ANZ), ("New Zealand", ANZ), ("India", SA),
               ("Southern African", AF), ("Kenya", AF), ("Nigeria", AF), ("West Indies", CAR),
               ("Filipino", SEA), ("Malaysia", SEA), ("Singapor", SEA), ("Hong Kong", EA),
               ("Japanese", EA), ("Chinese", EA), ("Korean", EA), ("German", EU),
               ("French", EU), ("Dutch", EU), ("European", EU)]


def region_of(accent):
    if accent in REGION:
        return REGION[accent]
    for key, reg in CV_KEYWORDS:
        if key in accent:
            return reg
    return OTHER


def raw(repo, path):
    return hf_hub_download(repo, path, repo_type="dataset", local_dir=os.path.join(RAW, repo.replace("/", "__")))


def save(set_name, uid, audio_bytes, ref, accent, items):
    audio = decode_audio(io.BytesIO(audio_bytes))
    dur = len(audio) / 16000
    if not (MIN_S <= dur <= MAX_S) or len(re.findall(r"[A-Za-z]+", ref)) < 3:
        return False
    os.makedirs(os.path.join(OUT, set_name), exist_ok=True)
    wav = os.path.join(OUT, set_name, f"{uid}.wav")
    sf.write(wav, audio, 16000, subtype="PCM_16")
    items.append({"set": set_name, "id": uid, "wav": wav, "dur": round(dur, 2), "ref": ref,
                  "accent": accent, "region": region_of(accent)})
    return True


def take(rows, n, keep):
    """rows are already shuffled; try each in turn, count the ones keep() accepts, stop at n."""
    k = 0
    for r in rows:
        if k >= n:
            break
        k += bool(keep(r))


def main():
    rng = random.Random(1019)
    items = []

    # EdAcc: 8 clips per accent (skip "Don't know"). 10 shards, 3.6 GB total; take rows from Arrow tables
    # instead of converting everything to Python objects
    d = os.path.join(RAW, "edinburghcstr__edacc", "data")
    tabs = [pq.read_table(os.path.join(d, n)) for n in sorted(os.listdir(d)) if n.startswith("test-")]
    by = {}
    for ti, t in enumerate(tabs):
        for ri, acc in enumerate(t.column("accent").to_pylist()):
            by.setdefault(acc, []).append((ti, ri))
    for acc, refs in sorted(by.items()):
        if acc == "Don't know":
            continue
        rng.shuffle(refs)
        rows = (tabs[ti].slice(ri, 1).to_pylist()[0] for ti, ri in refs)
        take(rows, 8, lambda r: save("en_edacc", f"edacc_{len(items):04d}", r["audio"]["bytes"], r["text"], acc, items))
    del tabs

    # SD-QA: 12 questions x 11 regions (same question, different accents; best for comparison)
    t = pq.read_table(raw("WillHeld/SD-QA", "data/test-00000-of-00004.parquet")).to_pylist()
    dialects = ["usa", "gbr", "irl", "aus", "nzl", "ind_n", "ind_s", "kenya", "nga", "zaf", "phl"]
    rng.shuffle(t)
    q = 0
    for r in t:
        if q >= 12:
            break
        if not all(r.get(d) and r[d].get("bytes") for d in dialects) or len((r.get("question") or "").split()) < 5:
            continue
        q += 1
        for d in dialects:
            save("en_sdqa", f"sdqa_{q:02d}_{d}", r[d]["bytes"], r["question"], d, items)

    # common-accent: up to 8 clips per accent; only single-accent labels
    # (multi-label values like "US English,India..." cannot be assigned to a region)
    t = pq.read_table(raw("DTU54DL/common-accent", "data/test-00000-of-00001-53b88232efc0bf7e.parquet")).to_pylist()
    by = {}
    for r in t:
        if r["accent"] and "," not in r["accent"]:
            by.setdefault(r["accent"], []).append(r)
    for acc, rs in sorted(by.items()):
        rng.shuffle(rs)
        take(rs, 8, lambda r: save("en_cv", f"cv_{len(items):04d}", r["audio"]["bytes"], r["sentence"], acc, items))

    # AMI SDM, Earnings-22, LibriSpeech test-other
    for set_name, repo, path, text_col, acc, n in [
        ("en_ami", "edinburghcstr/ami", "sdm/test-00000-of-00004.parquet", "text", "Meeting far-field (AMI)", 40),
        ("en_earn", "distil-whisper/earnings22", "chunked/test-00000-of-00038-6f4f182bdbb6f186.parquet",
         "transcription", "Earnings call (Earnings-22)", 40),
        ("en_libri", "openslr/librispeech_asr", "all/test.other/0000.parquet", "text", "us", 40)]:
        rs = pq.read_table(raw(repo, path)).to_pylist()
        rng.shuffle(rs)
        take(rs, n, lambda r: save(set_name, f"{set_name}_{len(items):04d}", r["audio"]["bytes"], r[text_col], acc, items))

    with open(os.path.join(OUT, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    # all clips <= 25 s, so the segment list is just the whole clip (scripts such as GPU SenseVoice read segments.jsonl)
    with open(os.path.join(OUT, "segments.jsonl"), "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({"id": it["id"], "set": it["set"], "j": 0, "wav": it["wav"], "dur": it["dur"]},
                               ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)
    tot = {}
    for it in items:
        a = tot.setdefault(it["set"], [0, 0.0])
        a[0] += 1
        a[1] += it["dur"]
    for s, (n, d) in tot.items():
        print(f"{s}: {n} clips, {d / 60:.1f} min")
    reg = {}
    for it in items:
        reg[it["region"]] = reg.get(it["region"], 0) + 1
    print("Regions:", reg)


if __name__ == "__main__":
    main()
