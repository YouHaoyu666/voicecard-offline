"""English multi-accent scoring: WER, with reference and hypothesis both passed through Whisper's EnglishTextNormalizer
(same convention as the Open ASR Leaderboard: numbers, spelling, contractions, fillers unified). Aggregates by set and by region.
Needs whisper-normalizer.

Usage: score_en.py results_en/*.jsonl            (by set)
       score_en.py --region results_en/*.jsonl   (by region: the three sets with accent labels: EdAcc, SD-QA, Common Voice)
       add --skip-cv: leave out the Common Voice subset (if a model was trained on the common-accent training set,
       speakers may overlap with the eval)
"""
import json
import os
import re
import sys
from collections import defaultdict

from whisper_normalizer.english import EnglishTextNormalizer

from score import edit

ROOT = os.path.dirname(os.path.abspath(__file__))
NORM = EnglishTextNormalizer()
TAGS = re.compile(r"<\|[^|]*\|>|<[^>]*>|\[[^\]]*\]|\([^)]*\)")  # SenseVoice tags, EdAcc annotations such as <laugh>
SETS = ["en_edacc", "en_sdqa", "en_cv", "en_ami", "en_earn", "en_libri"]
SET_NAMES = {"en_edacc": "EdAcc conversation", "en_sdqa": "SD-QA read", "en_cv": "CV accents",
             "en_ami": "Meeting far-field", "en_earn": "Earnings call", "en_libri": "Libri US"}
REGIONS = ["North America", "UK/Ireland", "Australia/NZ", "South Asia", "Africa", "Southeast Asia", "East Asia",
           "Europe (non-native)", "Middle East", "Caribbean", "Latin America"]
# Manifests written by an older prep_en.py carry Chinese region labels; map them to the English ones when reading
OLD_REGIONS = {"北美": "North America", "英国/爱尔兰": "UK/Ireland", "澳新": "Australia/NZ", "南亚": "South Asia",
               "非洲": "Africa", "东南亚": "Southeast Asia", "东亚": "East Asia", "欧洲（非母语）": "Europe (non-native)",
               "中东": "Middle East", "加勒比": "Caribbean", "拉美": "Latin America", "其他": "Other",
               "会议远场": "Meeting far-field", "电话会": "Earnings call"}
# Entries with a faulty reference are not scored (excluded here rather than changing the prep_en.py sampling,
# which would force every model to be re-run)
EXCLUDE = {
    "edacc_0033": "EdAcc's own annotation is IGNORE_TIME_SEGMENT_IN_SCORING (dataset convention: not scored)",
    "edacc_0058": "reference is incomplete: ~25 words at the start of the audio are missing from it; all 15 systems output this extra text",
}


def words(s):
    return NORM(TAGS.sub(" ", s or "")).split()


def main(args):
    by_region = "--region" in args
    skip_cv = "--skip-cv" in args
    paths = [a for a in args if not a.startswith("--")]
    man = {}
    for line in open(os.path.join(ROOT, "data", "eval_en", "manifest.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        man[r["id"]] = r
    cols = REGIONS if by_region else SETS
    head = cols if by_region else [SET_NAMES[s] for s in SETS]
    print("| Model | Overall | " + " | ".join(head) + " | Speed |")
    print("|---|---|" + "---|" * len(cols) + "---|")
    for p in paths:
        agg = defaultdict(lambda: [0, 0])
        audio = infer = 0.0
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["id"] in EXCLUDE:
                continue
            m = man[r["id"]]
            if skip_cv and m["set"] == "en_cv":
                continue
            ref, hyp = words(m["ref"]), words(r["hyp"])
            e = edit(ref, hyp)
            key = OLD_REGIONS.get(m["region"], m["region"]) if by_region else m["set"]
            agg[key][0] += e
            agg[key][1] += len(ref)
            agg["_all"][0] += e
            agg["_all"][1] += len(ref)
            audio += m["dur"]
            infer += r.get("sec", 0.0)

        def f(k):
            return f"{100 * agg[k][0] / agg[k][1]:.1f}" if agg[k][1] else "–"
        speed = f"{audio / infer:.0f}×" if infer else "–"
        print(f"| {os.path.splitext(os.path.basename(p))[0]} | {f('_all')} | " + " | ".join(f(c) for c in cols)
              + f" | {speed} |")


if __name__ == "__main__":
    main(sys.argv[1:])
