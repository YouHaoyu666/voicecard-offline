"""Draw the README charts from results/*.csv into docs/img/.

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

INK = "#1f2933"
GRID = "#d9dee3"
MUTED = "#7b8794"
C1 = "#c2410c"   # headline series
C2 = "#2563eb"
C3 = "#0f766e"
TEACHER = "#94a3b8"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 160,
})


def rows(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def asr_progress():
    r = rows("asr_versions.csv")
    x = list(range(len(r)))
    labels = [d["version"] for d in r]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for key, name, color in (("sg_codeswitch", "Singapore Chinese-English code-switching", C1),
                             ("english_overall", "English, 474 utterances across accents", C2),
                             ("chinese", "Chinese", C3)):
        y = [float(d[key]) for d in r]
        ax.plot(x, y, marker="o", color=color, lw=2, label=name)
        ax.annotate(f"{y[-1]:.1f}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=color, fontsize=9)
    sg0, sg5 = float(r[0]["sg_codeswitch"]), float(r[7]["sg_codeswitch"])
    ax.annotate(f"{sg0:.1f} -> {sg5:.1f}", (7, sg5), xytext=(7, sg5 + 14), ha="center", color=C1,
                fontsize=10, fontweight="bold", arrowprops=dict(arrowstyle="->", color=C1))
    ax.axvspan(0.5, 2.5, color=GRID, alpha=0.35, lw=0)
    ax.text(1.5, 58, "read speech only:\nworse", ha="center", va="top", color=MUTED, fontsize=8)
    ax.axvline(7, color=MUTED, lw=0.8, ls=":")
    ax.text(7, 2, "shipped", ha="center", color=MUTED, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 60)
    ax.set_ylabel("error rate % (lower is better)")
    ax.set_title("SenseVoice-Small (230M) fine-tuning, version by version", loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    save(fig, "asr_progress.png")


def asr_board():
    r = rows("asr_board.csv")
    cols = [("librispeech_clean", "LibriSpeech\nclean"), ("librispeech_other", "LibriSpeech\nother"),
            ("fleurs_en", "FLEURS\nEnglish"), ("fleurs_zh", "FLEURS\nChinese"), ("aishell1", "AISHELL-1")]
    colors = [TEACHER, C2, C1]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.26
    for i, d in enumerate(r):
        xs = [j + (i - 1) * w for j in range(len(cols))]
        ys = [float(d[k]) for k, _ in cols]
        ax.bar(xs, ys, w, color=colors[i], label=d["config"])
        if i == 2:
            for xx, yy in zip(xs, ys):
                ax.text(xx, yy + 0.3, f"{yy:.1f}", ha="center", fontsize=8, color=C1)
    ax.set_xticks(range(len(cols)), [c for _, c in cols])
    ax.set_ylabel("error rate % (lower is better)")
    ax.set_title("On the RK3588 NPU: fine-tune, then fix an fp16 overflow", loc="left", color=INK)
    ax.set_ylim(0, 15.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    save(fig, "asr_board.png")


def notes_finetune():
    r = rows("notes_finetune_effect.csv")
    r = [d for d in r if "thinking" not in d["setup"]]
    r.sort(key=lambda d: float(d["overall_en"]))
    names = [d["model"] + ", " + d["setup"] for d in r]
    en = [float(d["overall_en"]) for d in r]
    zh = [float(d["overall_zh"]) for d in r]
    y = list(range(len(r)))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    h = 0.38
    col = [C1 if "fine-tuned" in d["setup"] else (TEACHER if "teacher" in d["setup"] else C2) for d in r]
    ax.barh([i + h / 2 for i in y], en, h, color=col, label="English")
    ax.barh([i - h / 2 for i in y], zh, h, color=col, alpha=0.45, label="Chinese")
    for i in y:
        ax.text(en[i] + 0.08, i + h / 2, f"{en[i]:.1f}", va="center", fontsize=8)
        ax.text(zh[i] + 0.08, i - h / 2, f"{zh[i]:.1f}", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(y, names, fontsize=8.5)
    ax.set_xlim(0, 9)
    ax.set_xlabel("blind-judged overall score, 1-10 (solid: English, light: Chinese)")
    ax.set_title("Fine-tuning moves a 2B model past a zero-shot 4B", loc="left", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)
    save(fig, "notes_finetune.png")


def notes_round24():
    r = rows("notes_round24.csv")
    names = [d["system"] for d in r]
    overall = [float(d["overall"]) for d in r]
    cov = [float(d["topic_coverage"]) * 100 for d in r]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    col = [TEACHER] + [C2] * (len(r) - 2) + [C1]
    y = list(range(len(r)))[::-1]
    a1.barh(y, overall, color=col)
    a2.barh(y, cov, color=col)
    for i, v in zip(y, overall):
        a1.text(v + 0.1, i, f"{v:.2f}", va="center", fontsize=8.5)
    for i, v in zip(y, cov):
        a2.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=8.5)
    a1.set_yticks(y, names, fontsize=8.5)
    a2.set_yticks(y, [""] * len(y))
    a1.set_xlim(0, 10)
    a2.set_xlim(0, 100)
    a1.set_title("Overall (1-10)", loc="left", fontsize=10, color=INK)
    a2.set_title("Topics covered", loc="left", fontsize=10, color=INK)
    for a in (a1, a2):
        a.grid(axis="x", color=GRID, lw=0.6)
    fig.suptitle("Meeting notes, 20 English meetings, 5 blind judges", x=0.02, y=1.04, ha="left", color=INK)
    save(fig, "notes_round24.png")


if __name__ == "__main__":
    asr_progress()
    asr_board()
    notes_finetune()
    notes_round24()
