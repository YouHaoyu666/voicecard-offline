"""Per-set MER (mixed error rate: Chinese by character, English by word) + speed.

Usage: score.py results/*.jsonl
Each line of a results file: {"id", "set", "hyp", "sec"} (sec = inference time for that clip, optional)

Normalisation (same for reference and hypothesis):
- strip MNSC annotations: <SpeakerN>:, (filler), [interjection], <unk>, half-words marked with ~; for #word# and !word! only the symbols are removed
- Traditional to Simplified Chinese (opencc t2s), Chinese numerals to Arabic digits (cn2an), lower-case
- drop fillers/interjections on both sides: English list in FILLERS_EN, Chinese characters in FILLERS_ZH
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import cn2an
import opencc

ROOT = os.path.dirname(os.path.abspath(__file__))
T2S = opencc.OpenCC("t2s")
FILLERS_EN = set("uh um umm mm mmm hmm er erm ah eh oh orh lah leh lor meh mah sia hor huh".split())
FILLERS_ZH = set("呃嗯啊哦噢诶欸哎嘛啦呀哈")
TOKEN = re.compile(r"[一-鿿]|[a-z]+|\d+")


def clean_ref(s):
    s = re.sub(r"<speaker\d+>:", " ", s, flags=re.I)
    s = re.sub(r"\([^)]*\)|\[[^\]]*\]|<[^>]*>", " ", s)
    s = re.sub(r"\S*~\S*", " ", s)
    return s.replace("#", " ").replace("!", " ")


def tokens(s):
    s = T2S.convert(s).lower().replace("'", "")
    try:
        s = cn2an.transform(s, "cn2an")
    except Exception:
        pass
    toks = TOKEN.findall(s)
    return [t for t in toks if t not in FILLERS_EN and t not in FILLERS_ZH]


def edit(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def main(paths):
    man = {}
    with open(os.path.join(ROOT, "data", "eval", "manifest.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            man[r["id"]] = r
    rows = []
    speed = defaultdict(lambda: [0.0, 0.0])  # model -> audio_sec, infer_sec
    for p in paths:
        model = os.path.splitext(os.path.basename(p))[0]
        agg = defaultdict(lambda: [0, 0, 0.0, 0.0])  # errs, ref_len, audio_sec, infer_sec
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                m = man[r["id"]]
                ref, hyp = tokens(clean_ref(m["ref"])), tokens(r["hyp"])
                a = agg[m["set"]]
                a[0] += edit(ref, hyp)
                a[1] += len(ref)
                a[2] += m["dur"]
                a[3] += r.get("sec", 0.0)
        for s, (e, n, au, inf) in sorted(agg.items()):
            rows.append((model, s, e / max(n, 1), au / inf if inf else 0.0))
            if inf:
                speed[model][0] += au
                speed[model][1] += inf
    sets = sorted({s for _, s, _, _ in rows})
    models = list(dict.fromkeys(m for m, _, _, _ in rows))
    print("| Model | " +" | ".join(sets) + " | RTFx |")
    print("|---|" + "---|" * (len(sets) + 1))
    for m in models:
        cells = []
        for s in sets:
            hit = [mer for mm, ss, mer, _ in rows if mm == m and ss == s]
            cells.append(f"{hit[0]:.1%}" if hit else "–")
        au, inf = speed[m]
        rtfx = f"{au / inf:.0f}×" if inf else "–"
        print(f"| {m} | " + " | ".join(cells) + f" | {rtfx} |")


if __name__ == "__main__":
    main([p for a in sys.argv[1:] for p in glob.glob(a)])
