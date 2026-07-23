# %%

import re

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

for _font_path in [
    "/Users/alielassche/Library/Fonts/RobotoCondensed-Regular.ttf",
    "/Users/alielassche/Library/Fonts/RobotoCondensed-Bold.ttf",
]:
    fm.fontManager.addfont(_font_path)

plt.rcParams.update({
    "font.family": "Roboto Condensed",
    "font.sans-serif": ["Roboto Condensed"],
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Roboto Condensed",
    "mathtext.bf": "Roboto Condensed:bold",
})

RESULTS_FILE = "results/evaluation_results_2605012_stratified.txt"

CHRgreen = '#1B4332'
CHRred = '#C0392B'
CHRyellow = '#E8A838'
CHRbeige = '#D4C5B0'
CHRcream = '#F5F0E8'

LANGUAGE_NAMES = {
    "fre": "French",
    "eng": "English",
    "ger": "German",
    "cal": "Spanish",
    "dutch": "Dutch",
    "ita": "Italian",
}

# ── parse ─────────────────────────────────────────────────────────────────────
# %%


def parse_results(path):
    languages = []
    current = None

    lang_re   = re.compile(r"Language:\s*(\w+)")
    method_re = re.compile(r"^(TF-IDF|BGE embeddings|Jina embeddings):")
    class_re  = re.compile(r"(MALE|FEMALE)\s*→\s*P:\s*([\d.]+)±([\d.]+),\s*R:\s*([\d.]+)±([\d.]+),\s*F1:\s*([\d.]+)±([\d.]+)")
    acc_re    = re.compile(r"Accuracy:\s*([\d.]+)±([\d.]+)")

    method_map = {"TF-IDF": "TF-IDF", "BGE embeddings": "bge-m3", "Jina embeddings": "Jina"}
    current_method = None

    with open(path) as f:
        for line in f:
            line = line.rstrip()

            m = lang_re.search(line)
            if m:
                current = {"language": m.group(1), "methods": {}}
                languages.append(current)
                continue

            m = method_re.match(line.strip())
            if m and current is not None:
                current_method = method_map[m.group(1)]
                current["methods"][current_method] = {}
                continue

            m = class_re.search(line)
            if m and current_method:
                cls = m.group(1)
                current["methods"][current_method][cls] = {
                    "P":  (float(m.group(2)), float(m.group(3))),
                    "R":  (float(m.group(4)), float(m.group(5))),
                    "F1": (float(m.group(6)), float(m.group(7))),
                }
                continue

            m = acc_re.search(line)
            if m and current_method:
                current["methods"][current_method]["Accuracy"] = (
                    float(m.group(1)), float(m.group(2))
                )

    return languages


# ── accuracy by language, grouped by method ──────────────────────────────────
# %%


def plot_accuracy_by_language(languages, methods=("TF-IDF", "bge-m3")):
    colors = {"TF-IDF": CHRyellow, "bge-m3": CHRred, "Jina": CHRgreen}

    languages_sorted = sorted(
        languages, key=lambda l: l["methods"][methods[0]]["Accuracy"][0]
    )
    lang_codes = [l["language"] for l in languages_sorted]
    y = np.arange(len(lang_codes))

    height = 0.8 / len(methods)
    offsets = np.linspace(
        -height * (len(methods) - 1) / 2, height * (len(methods) - 1) / 2, len(methods)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for method, offset in zip(methods, offsets):
        vals = [l["methods"][method]["Accuracy"][0] for l in languages_sorted]
        errs = [l["methods"][method]["Accuracy"][1] for l in languages_sorted]
        ax.barh(
            y + offset, vals, height=height, xerr=errs,
            label=method, color=colors[method], capsize=3, alpha=0.9,
        )

    ax.set_yticks(y, labels=[LANGUAGE_NAMES[c] for c in lang_codes])
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Accuracy", fontweight="semibold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", framealpha=0.7)

    fig.tight_layout()
    plt.savefig("figs/accuracy_by_language.pdf", bbox_inches="tight")
    plt.show()


# ── F1 by language and gender, faceted by method ─────────────────────────────
# %%


def plot_metrics_grid(languages, methods=("TF-IDF", "bge-m3")):
    metrics    = ["P", "R", "F1", "Accuracy"]
    row_labels = {"P": "Precision", "R": "Recall", "F1": "F1-score", "Accuracy": "Accuracy"}
    colors     = {"MALE": CHRyellow, "FEMALE": CHRred}

    lang_codes = [l["language"] for l in languages]
    x = np.arange(len(lang_codes))
    width = 0.35

    fig, axes = plt.subplots(
        len(metrics), len(methods),
        figsize=(4 * len(methods), 2.5 * len(metrics)),
        sharex=True, sharey=True,
    )

    for row, metric in enumerate(metrics):
        for col, method in enumerate(methods):
            ax = axes[row, col]

            if metric == "Accuracy":
                vals = [l["methods"][method]["Accuracy"][0] for l in languages]
                errs = [l["methods"][method]["Accuracy"][1] for l in languages]
                ax.bar(
                    x, vals, width=0.6, yerr=errs,
                    label="Accuracy", color="grey", capsize=3, alpha=0.7,
                )
            else:
                for i, cls in enumerate(["MALE", "FEMALE"]):
                    vals = [l["methods"][method][cls][metric][0] for l in languages]
                    errs = [l["methods"][method][cls][metric][1] for l in languages]
                    offset = (i - 0.5) * width
                    ax.bar(
                        x + offset, vals, width=width, yerr=errs,
                        label=cls, color=colors[cls], capsize=3, alpha=0.8,
                    )

            if row == 0:
                ax.set_title(method, fontweight="bold")
            if col == 0:
                ax.set_ylabel(row_labels[metric], fontweight="semibold")

            ax.set_xticks(x)
            if row == len(metrics) - 1:
                ax.set_xticklabels([LANGUAGE_NAMES[c] for c in lang_codes], rotation=0, ha="center")
            else:
                ax.set_xticklabels([])

            ax.set_ylim(0.58, 1.0)
            ax.yaxis.grid(True, linestyle="--", alpha=0.4)
            ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, framealpha=0.7, bbox_to_anchor=(0.55, -0.03))

    fig.tight_layout()
    plt.savefig("figs/metrics_grid_by_gender_method.pdf", bbox_inches="tight")
    plt.show()


# %%

if __name__ == "__main__":
    languages = parse_results(RESULTS_FILE)
    plot_accuracy_by_language(languages)
    plot_metrics_grid(languages)

# %%
