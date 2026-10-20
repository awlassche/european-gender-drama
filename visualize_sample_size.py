# %%
"""
Accuracy per language as a function of the number of speech chunks per
gender, for TF-IDF and bge-m3, from the train_classifiers_europe_noreplace.py
runs (sampling without replacement, TF-IDF fitted on the training split,
held-out plays). Error bars are ±1 std over the 50 iterations.

Each language is plotted at the sample size actually used: in the first
version Italian had only 1,130 female chunks, so its 1,250 and 1,500 runs
both used 1,130 and the duplicate was dropped. In the current data every
language has at least 1,500 chunks per gender.
"""

import os
import re

import matplotlib.pyplot as plt

from visualize_method_comparison import (
    CHRred, CHRyellow, CORPORA, LANGUAGE_NAMES, _REPO_DIR, parse_results,
)

# current results: the 2026-10-15 DraCor extraction (bge-m3 embeddings v2)
RESULTS_PREFIX = os.path.join(
    _REPO_DIR, "results/evaluation_results_v2_noreplace_tfidf-train_heldout-plays"
)
RESULTS_FILES = {n: f"{RESULTS_PREFIX}_n{n}.txt" for n in (500, 750, 1000, 1250, 1500)}
OUTPUT_FILE = "accuracy_by_sample_size_v2.pdf"

# the first version, on the original extraction:
# RESULTS_PREFIX = os.path.join(
#     _REPO_DIR, "results/evaluation_results_stratified_noreplace_tfidf-train_heldout-plays"
# )
# RESULTS_FILES = {n: f"{RESULTS_PREFIX}_n{n}.txt" for n in (500, 750, 1000, 1250)}
# RESULTS_FILES[1500] = f"{RESULTS_PREFIX}.txt"
# OUTPUT_FILE = "accuracy_by_sample_size.pdf"

# larger text than the shared settings in visualize_method_comparison.py,
# so the six small panels stay readable when the figure is scaled down
plt.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 14,
})

METHODS = ("TF-IDF", "bge-m3")
COLORS = {"TF-IDF": CHRyellow, "bge-m3": CHRred}
MARKERS = {"TF-IDF": "o", "bge-m3": "s"}

# ── parse ─────────────────────────────────────────────────────────────────────
# %%


def n_per_class(path):
    """{language: sample size per class actually used} from the file headers."""
    header_re = re.compile(r"Language:\s*(\w+)\s*\(n per class:\s*(\d+)")
    with open(path) as f:
        return {m.group(1): int(m.group(2)) for m in map(header_re.search, f) if m}


def collect(results_files):
    """{language: {method: [(n, mean, std), ...]}}, sorted by n, one point per
    distinct sample size."""
    series = {}
    for path in results_files.values():
        sizes = n_per_class(path)
        for lang in parse_results(path):
            code = lang["language"]
            for method in METHODS:
                points = series.setdefault(code, {}).setdefault(method, {})
                mean, std = lang["methods"][method]["Accuracy"]
                points.setdefault(sizes[code], (mean, std))
    return {
        code: {method: sorted((n, m, s) for n, (m, s) in points.items())
               for method, points in methods.items()}
        for code, methods in series.items()
    }


# ── accuracy by sample size, one panel per language ──────────────────────────
# %%


def plot_accuracy_by_sample_size(series):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, sharey=True)
    ticks = sorted(RESULTS_FILES)

    for ax, code in zip(axes.flat, CORPORA):
        for method in METHODS:
            ns, means, stds = zip(*series[code][method])
            ax.errorbar(
                ns, means, yerr=stds,
                label=method,
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=6,
                capsize=3,
                linewidth=1.8,
                alpha=0.9,
            )

        ax.set_title(LANGUAGE_NAMES[code], fontweight="semibold")
        ax.set_xticks(ticks)
        ax.set_ylim(0.5, 0.8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[1]:
        ax.set_xlabel("Speech chunks per gender", fontweight="semibold")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy", fontweight="semibold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, framealpha=0.7,
               bbox_to_anchor=(0.5, -0.09))

    fig.tight_layout()
    plt.savefig(os.path.join(_REPO_DIR, "figs", OUTPUT_FILE), bbox_inches="tight")
    plt.show()


# %%

if __name__ == "__main__":
    plot_accuracy_by_sample_size(collect(RESULTS_FILES))

# %%
