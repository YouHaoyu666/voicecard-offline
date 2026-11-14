"""Draw the training pipeline diagram (docs/img/pipeline.png).

Usage: python tools/make_flow.py   (needs matplotlib)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "img", "pipeline.png")

# Monochrome line style: black strokes on white, serif type.
INK = "#000000"
MUTED = "#444444"
LINE = "#000000"
DATA = TEACH = TRAIN = TEST = BOARD = ("#ffffff", "#000000")  # fill, edge
TEACHER_DASH = (0, (4, 2.5))

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
                     "figure.facecolor": "white", "savefig.dpi": 170})

W, H = 2.2, 1.15  # box size


def box(ax, x, y, title, body, style, w=W, h=H, dashed=False):
    fill, edge = style
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="square,pad=0.02",
                                fc=fill, ec=edge, lw=0.9, ls=TEACHER_DASH if dashed else "-"))
    ax.text(x, y + h / 2 - 0.22, title, ha="center", va="center", fontsize=11, color=INK)
    ax.text(x, y - 0.15, body, ha="center", va="center", fontsize=8.6, color=MUTED, linespacing=1.3,
            style="italic")


def arrow(ax, p, q, text=None, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=0.8, color=LINE,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.13, text, ha="center", fontsize=8, color=MUTED,
                style="italic")


def lane(ax, y, h, label):
    ax.plot([0.15, 17.75], [y + h / 2, y + h / 2], color=INK, lw=0.5)
    ax.text(0.45, y, label, rotation=90, ha="center", va="center", fontsize=11.5, color=INK)


def main():
    fig, ax = plt.subplots(figsize=(14, 7.2))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9.0)
    ax.axis("off")

    ax.text(0.15, 8.95, "How the two on-device models were trained", fontsize=17, color=INK,
            va="top")
    ax.text(0.15, 8.5, "Teachers are open-weight models on a home RTX 5090, or the DeepSeek API (its terms allow distillation).\n"
            "Claude writes the code and judges test sets; it never writes training data.",
            fontsize=10, color=MUTED, va="top", style="italic")

    # speech lane
    ys = 6.35
    lane(ax, ys, 2.0, "Speech\n230M")
    xs = [2.0, 4.85, 7.7, 10.55, 13.4, 16.25]
    box(ax, xs[0], ys, "Public audio", "YouTube CC-BY talk\nSingapore NSC\nCommon Voice accents", DATA)
    box(ax, xs[1], ys, "Teacher labels", "Qwen3-ASR-1.7B, local\nkept if WER <= 0.3\nvs human captions", TEACH, dashed=True)
    box(ax, xs[2], ys, "Fine-tune", "SenseVoice-Small\n277 h, 2 epochs\n~2.8 h on RTX 5090", TRAIN)
    box(ax, xs[3], ys, "Test", "12-accent English\nSingapore code-switch\nChinese, far field", TEST)
    box(ax, xs[4], ys, "Pick version", "one change per run\nv1 ... v7\nshipped: v5", TEST)
    box(ax, xs[5], ys, "On the NPU", "RKNN fp16\n+ overflow rescale\nRK3588, 4 GB", BOARD)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + W / 2, ys), (b - W / 2, ys))
    ax.text(xs[1], ys - 0.78, "Singapore speech uses human transcripts", ha="center", fontsize=8.4,
            color=MUTED, style="italic")

    # notes lane
    yn = 3.2
    lane(ax, yn, 3.6, "Meeting notes\n2B")
    yt, yb = yn + 0.85, yn - 0.85
    box(ax, 2.0, yt, "Public meetings", "AMI, QMSum\nMeetingBank, ELITR", DATA)
    box(ax, 4.85, yt, "Teacher notes", "DeepSeek API\nwrites notes per\n15-min part + merge", TEACH, dashed=True)
    box(ax, 7.7, yt, "SFT", "Qwen3.5-2B\n~3,000 examples\n2 epochs, RTX 5090", TRAIN)
    box(ax, 7.7, yb, "Student samples", "2-4 answers\nper training prompt", TRAIN)
    box(ax, 10.55, yb, "Grade + pair", "DeepSeek grades vs\nsource; graders\naudited first", TEACH, dashed=True)
    box(ax, 13.4, yb, "Preference (DPO)", "good vs bad answer\nfor the same prompt", TRAIN)
    box(ax, 10.55, yt, "Blind judging", "5 judges, 20 meetings\nmust quote last line\nbar set before run", TEST)
    box(ax, 13.4, yt, "Freeze version", "v1 ... v15\nshipped: v15", TEST)
    box(ax, 16.25, yn, "On the NPU", "RKLLM w8a8\nRK3588, 4 GB", BOARD)

    arrow(ax, (2.0 + W / 2, yt), (4.85 - W / 2, yt))
    arrow(ax, (4.85 + W / 2, yt), (7.7 - W / 2, yt))
    arrow(ax, (7.7 + W / 2, yt), (10.55 - W / 2, yt))
    arrow(ax, (7.7, yt - H / 2), (7.7, yb + H / 2))
    arrow(ax, (7.7 + W / 2, yb), (10.55 - W / 2, yb))
    arrow(ax, (10.55 + W / 2, yb), (13.4 - W / 2, yb))
    arrow(ax, (13.4 - W / 2 + 0.15, yb + H / 2), (10.55 + W / 2 - 0.15, yt - H / 2))
    ax.text(12.45, yn, "retrained\nmodel", ha="left", va="center", fontsize=8, color=MUTED, style="italic")
    arrow(ax, (10.55 + W / 2, yt), (13.4 - W / 2, yt), "passes")
    arrow(ax, (13.4 + W / 2, yt), (16.25 - W / 2, yn + 0.2))

    # roles
    yr = 0.55
    ax.plot([0.15, 17.75], [yr + 0.5, yr + 0.5], color=INK, lw=0.5)
    ax.text(0.15, yr, "Human — product goals, hardware budget, acceptance bars, final calls", fontsize=10.5,
            va="center", color=INK)
    ax.text(9.15, yr, "Claude Code — model survey, data, training and test code, runs, reports", fontsize=10.5,
            va="center", color=INK)
    ax.text(17.75, 8.95, "dashed box = teacher model", ha="right", va="top", fontsize=9, color=MUTED,
            style="italic")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
