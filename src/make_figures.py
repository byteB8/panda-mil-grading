"""Draws the summary figure for the README from results/results.json.

    python src/make_figures.py    ->  figures/results_summary.png
"""

import json
from pathlib import Path

import pandas as pd

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


def label_bars(ax, bars, errors=None, fmt="{:.3f}"):
    """Values above each bar, clearing the error bar when there is one."""
    for i, bar in enumerate(bars):
        lift = 0.015 + (errors[i] if errors is not None else 0)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + lift,
                fmt.format(bar.get_height()), ha="center", va="bottom", fontsize=9, color=INK)


METHOD_LABELS = {"none": "none\n(baseline)", "per-centre": "per-centre\nstandardise",
                 "coral": "CORAL", "dann": "DANN"}


def adaptation_figure():
    """figures/adaptation.png — how far each feature-level fix closes the cross-hospital gap."""
    runs = []
    for folder in sorted(ROOT.glob("results-adapt*")):
        table = pd.read_csv(folder / "adaptation.csv")
        table["seed"] = json.loads((folder / "adaptation.json").read_text())["config"]["seed"]
        runs.append(table)
    if not runs:
        print("no results-adapt* folders; skipping the adaptation figure")
        return
    runs = pd.concat(runs, ignore_index=True)
    stats = runs.groupby(["method", "train", "test"]).qwk.agg(["mean", "std", "count"]).reset_index()
    print(stats.round(3).to_string(index=False))

    order = [m for m in METHOD_LABELS if m in set(stats.method)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), facecolor=SURFACE, sharey=True)
    for ax, source in zip(axes, ["radboud", "karolinska"]):
        target = "karolinska" if source == "radboud" else "radboud"
        here = stats[(stats.train == source) & (stats.test == "other centre")].set_index("method")
        home = stats[(stats.train == source) & (stats.test == "in-centre")].set_index("method")
        means = [here.loc[m, "mean"] for m in order]
        errors = [here.loc[m, "std"] for m in order]
        colours = [ORANGE if m == "none" else BLUE for m in order]
        bars = ax.bar([METHOD_LABELS[m] for m in order], means, width=0.55, color=colours, zorder=3)
        ax.errorbar(range(len(order)), means, yerr=errors, fmt="none", ecolor=INK, capsize=4, lw=1.1, zorder=4)
        label_bars(ax, bars, errors=errors)
        ax.axhline(home.loc["none", "mean"], color=MUTED, ls="--", lw=1, zorder=2)
        ax.text(len(order) - 0.45, home.loc["none", "mean"] + 0.02, "same hospital", ha="right",
                fontsize=8, color=MUTED)
        ax.set_title(f"trained on {source.capitalize()}, tested on {target.capitalize()}",
                     color=INK, fontsize=11, pad=12)
        ax.set_ylim(0, 1.0)
        style(ax)
    axes[0].set_ylabel("quadratic weighted kappa", color=MUTED, fontsize=10)
    seeds = runs.seed.nunique()
    fig.suptitle(f"Closing the cross-hospital gap without touching the images "
                 f"(mean of {seeds} seed{'s' if seeds > 1 else ''}, ±1 sd)", color=INK, fontsize=11)
    fig.tight_layout()
    out = ROOT / "figures" / "adaptation.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("wrote", out)


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
    adaptation_figure()


if __name__ == "__main__":
    main()
