"""Draws the summary figure for the README from results/results.json.

    python src/make_figures.py    ->  figures/results_summary.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is set)

ROOT = Path(__file__).resolve().parent.parent
BLUE, ORANGE = "#2a78d6", "#eb6834"  # validated categorical slots 1 and 2
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d6d5d0")
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color="#ebeae5", lw=0.8)
    ax.set_axisbelow(True)


def label_bars(ax, bars, fmt="{:.3f}"):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                fmt.format(bar.get_height()), ha="center", va="bottom", fontsize=9, color=INK)


def main():
    results = json.loads((ROOT / "results" / "results.json").read_text())
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.1), facecolor=SURFACE)

    # Left: cross-validated grading accuracy, one bar per model.
    models = ["ABMIL", "MeanPool"]
    names = ["Attention MIL", "Mean pooling"]
    means = [results["cv_summary"][m]["qwk_mean"] for m in models]
    sds = [results["cv_summary"][m]["qwk_std"] for m in models]
    bars = left.bar(names, means, width=0.5, color=[BLUE, ORANGE], zorder=3)
    left.errorbar(names, means, yerr=sds, fmt="none", ecolor=INK, capsize=5, lw=1.2, zorder=4)
    label_bars(left, bars)
    left.set_ylim(0, 1.0)
    left.set_ylabel("quadratic weighted kappa", color=MUTED, fontsize=10)
    left.set_title("Grading accuracy (5-fold CV, ±1 sd)", color=INK, fontsize=11, pad=12)
    style(left)

    # Right: the same model trained on one hospital, tested on both.
    shift = {(r["train"], r["test"]): r["qwk"] for r in results["cross_centre"]}
    groups = ["radboud", "karolinska"]
    same = [shift[(g, "in-centre")] for g in groups]
    other = [shift[(g, "other centre")] for g in groups]
    positions = range(len(groups))
    width = 0.34
    gap = 0.012  # 2px surface gap between adjacent fills at this figure size
    bars_same = right.bar([p - width / 2 - gap for p in positions], same, width, label="same hospital",
                          color=BLUE, zorder=3)
    bars_other = right.bar([p + width / 2 + gap for p in positions], other, width, label="other hospital",
                           color=ORANGE, zorder=3)
    label_bars(right, bars_same)
    label_bars(right, bars_other)
    right.set_xticks(list(positions))
    right.set_xticklabels([f"trained on\n{g.capitalize()}" for g in groups])
    right.set_ylim(0, 1.0)
    right.set_title("What a single-hospital model transfers", color=INK, fontsize=11, pad=12)
    legend = right.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(MUTED)
    style(right)

    fig.tight_layout()
    out = ROOT / "figures" / "results_summary.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
