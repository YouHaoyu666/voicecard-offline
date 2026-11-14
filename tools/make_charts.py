"""Draw the README charts from results/*.csv into docs/img/.

Monochrome line style (black on white, serif type), matching docs/img/pipeline.png.
Usage: python tools/make_charts.py   (needs matplotlib)
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "img")

INK = "#000000"
MUTED = "#555555"
GRID = "#dddddd"
# bar fills, light to dark: reference / baseline / ours
F_REF = "#ffffff"
F_MID = "#bbbbbb"
F_OURS = "#000000"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "font.size": 11, "axes.edgecolor": INK, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "hatch.linewidth": 0.5,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 170,
})
BAR = dict(edgecolor=INK, linewidth=0.6)


def rows(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def title(ax, text):
    ax.set_title(text, loc="left", color=INK, fontsize=14, pad=12)


def asr_progress():
    r = rows("asr_versions.csv")
    x = list(range(len(r)))
    labels = [d["version"] for d in r]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    series = (("sg_codeswitch", "Singapore Chinese-English code-switching", "-", "o", INK),
              ("english_overall", "English, 474 utterances across accents", "--", "s", INK),
              ("chinese", "Chinese", ":", "^", MUTED))
    for key, name, ls, mk, color in series:
        y = [float(d[key]) for d in r]
        ax.plot(x, y, ls=ls, marker=mk, ms=5, color=color, lw=1.1, label=name, mfc="white", mew=1)
        ax.annotate(f"{y[-1]:.1f}", (x[-1], y[-1]), xytext=(7, 0), textcoords="offset points",
                    va="center", color=color, fontsize=10)
    sg0, sg5 = float(r[0]["sg_codeswitch"]), float(r[7]["sg_codeswitch"])
    ax.annotate(f"{sg0:.1f} → {sg5:.1f}", (7, sg5), xytext=(7, sg5 + 14), ha="center", color=INK,
                fontsize=12, arrowprops=dict(arrowstyle="->", color=INK, lw=0.7))
    ax.axvspan(0.5, 2.5, facecolor="none", edgecolor=GRID, hatch="///", lw=0)
    ax.text(1.5, 58, "read speech only:\nworse", ha="center", va="top", color=MUTED, fontsize=9.5,
            style="italic")
    ax.axvline(7, color=INK, lw=0.5, ls=(0, (2, 3)))
    ax.text(7, 2, "shipped", ha="center", color=MUTED, fontsize=9.5, style="italic")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 60)
    ax.set_ylabel("error rate %  (lower is better)")
    title(ax, "SenseVoice-Small (230M) fine-tuning, version by version")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    save(fig, "asr_progress.png")


def asr_board():
    r = rows("asr_board.csv")
    cols = [("librispeech_clean", "LibriSpeech\nclean"), ("librispeech_other", "LibriSpeech\nother"),
            ("fleurs_en", "FLEURS\nEnglish"), ("fleurs_zh", "FLEURS\nChinese"), ("aishell1", "AISHELL-1")]
    fills = [F_REF, F_MID, F_OURS]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.26
    for i, d in enumerate(r):
        xs = [j + (i - 1) * w for j in range(len(cols))]
        ys = [float(d[k]) for k, _ in cols]
        ax.bar(xs, ys, w, color=fills[i], label=d["config"], **BAR)
        if i == 2:
            for xx, yy in zip(xs, ys):
                ax.text(xx, yy + 0.3, f"{yy:.1f}", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(range(len(cols)), [c for _, c in cols])
    ax.set_ylabel("error rate %  (lower is better)")
    title(ax, "On the RK3588 NPU: fine-tune, then fix an fp16 overflow")
    ax.set_ylim(0, 15.5)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    save(fig, "asr_board.png")


def notes_finetune():
    r = rows("notes_finetune_effect.csv")
    r = [d for d in r if "thinking" not in d["setup"]]
    r.sort(key=lambda d: float(d["overall_en"]))
    names = [d["model"] + ", " + d["setup"] for d in r]
    en = [float(d["overall_en"]) for d in r]
    zh = [float(d["overall_zh"]) for d in r]
    y = list(range(len(r)))
    fig, ax = plt.subplots(figsize=(8, 3.9))
    h = 0.36
    ours = ["fine-tuned" in d["setup"] for d in r]
    ax.barh([i + h / 2 for i in y], en, h, color=[F_OURS if o else F_MID for o in ours], **BAR)
    ax.barh([i - h / 2 for i in y], zh, h, color="white", hatch=["////" if o else "" for o in ours], **BAR)
    for i in y:
        ax.text(en[i] + 0.08, i + h / 2, f"{en[i]:.1f}", va="center", fontsize=9.5)
        ax.text(zh[i] + 0.08, i - h / 2, f"{zh[i]:.1f}", va="center", fontsize=9.5, color=MUTED)
    ax.set_yticks(y, names, fontsize=10)
    ax.set_xlim(0, 9)
    ax.set_xlabel("blind-judged overall score, 1–10   (filled: English, outline: Chinese)")
    title(ax, "Fine-tuning moves a 2B model past a zero-shot 4B")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    save(fig, "notes_finetune.png")


def notes_round24():
    r = rows("notes_round24.csv")
    names = [d["system"] for d in r]
    overall = [float(d["overall"]) for d in r]
    cov = [float(d["topic_coverage"]) * 100 for d in r]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    col = [F_REF] + [F_MID] * (len(r) - 2) + [F_OURS]
    y = list(range(len(r)))[::-1]
    a1.barh(y, overall, color=col, height=0.6, **BAR)
    a2.barh(y, cov, color=col, height=0.6, **BAR)
    for i, v in zip(y, overall):
        a1.text(v + 0.12, i, f"{v:.2f}", va="center", fontsize=9.5)
    for i, v in zip(y, cov):
        a2.text(v + 1.2, i, f"{v:.0f}%", va="center", fontsize=9.5)
    a1.set_yticks(y, names, fontsize=10)
    a2.set_yticks(y, [""] * len(y))
    a1.set_xlim(0, 10)
    a2.set_xlim(0, 100)
    a1.set_title("Overall (1–10)", loc="left", fontsize=11, color=INK)
    a2.set_title("Topics covered", loc="left", fontsize=11, color=INK)
    for a in (a1, a2):
        a.grid(axis="x", color=GRID, lw=0.5)
        a.set_axisbelow(True)
    fig.suptitle("Meeting notes, 20 English meetings, 5 blind judges", x=0.02, y=1.05, ha="left", color=INK,
                 fontsize=14)
    save(fig, "notes_round24.png")


if __name__ == "__main__":
    asr_progress()
    asr_board()
    notes_finetune()
    notes_round24()
