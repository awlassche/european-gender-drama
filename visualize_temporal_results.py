# %%

import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.font_manager as fm

for _font_path in [
    "/Users/alielassche/Library/Fonts/RobotoCondensed-Regular.ttf",
    "/Users/alielassche/Library/Fonts/RobotoCondensed-Bold.ttf",
]:
    fm.fontManager.addfont(_font_path)

plt.rcParams.update({
    "font.family": "Roboto Condensed",
    "font.sans-serif": ["Roboto Condensed"],
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})

RESULTS_FILE = "results/temporal_fre_results_stratified_1000.txt"

CHRgreen = '#1B4332' 
CHRred = '#C0392B'
CHRyellow = '#E8A838' 
CHRbeige = '#D4C5B0' 
CHRcream = '#F5F0E8' 

# ── parse ─────────────────────────────────────────────────────────────────────

def parse_results(path):
    windows = []
    current = None

    window_re  = re.compile(r"Window:\s*(\d{4})–(\d{4})")
    method_re  = re.compile(r"^(TF-IDF|BGE embeddings):")
    class_re   = re.compile(r"(MALE|FEMALE)\s*→\s*P:\s*([\d.]+)±([\d.]+),\s*R:\s*([\d.]+)±([\d.]+),\s*F1:\s*([\d.]+)±([\d.]+)")
    acc_re     = re.compile(r"Accuracy:\s*([\d.]+)±([\d.]+)")

    method_map = {"TF-IDF": "TF-IDF", "BGE embeddings": "BGE"}
    current_method = None

    with open(path) as f:
        for line in f:
            line = line.rstrip()

            m = window_re.search(line)
            if m:
                current = {"window": f"{m.group(1)}–{m.group(2)}", "methods": {}}
                windows.append(current)
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

    return windows


# ── plot ──────────────────────────────────────────────────────────────────────
#%%

def plot(windows):
    methods    = ["TF-IDF", "BGE"]
    classes    = ["MALE", "FEMALE"]
    metrics    = ["P", "R", "F1"]

    col_titles = {"TF-IDF": "TF-IDF", "BGE": "bge-m3"}
    row_labels = {"P": "Precision", "R": "Recall", "F1": "F1-score"}
    colors     = {"MALE": CHRyellow, "FEMALE": CHRred}
    styles     = {"MALE": "-", "FEMALE": "-"}
    markers    = {"MALE": "o", "FEMALE": "s"}

    x = np.arange(len(windows))
    win_labels = [w["window"] for w in windows]

    fig = plt.figure(figsize=(12, 8))
    #fig.suptitle(
    #    "Gender classification performance across time windows\n"
    #    "(stratified, min 1 000 samples per class)",
    #    fontsize=14, fontweight="bold", y=1.01,
    #)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.12, wspace=0.28)

    first_ax = None
    for row, metric in enumerate(metrics):
        for col, method in enumerate(methods):
            ax = fig.add_subplot(gs[row, col], sharey=first_ax)
            if first_ax is None:
                first_ax = ax

            for cls in classes:
                vals  = [w["methods"][method][cls][metric][0] for w in windows]
                errs  = [w["methods"][method][cls][metric][1] for w in windows]
                ax.errorbar(
                    x, vals, yerr=errs,
                    label=cls,
                    color=colors[cls],
                    linestyle=styles[cls],
                    marker=markers[cls],
                    markersize=5,
                    capsize=3,
                    linewidth=1.6,
                    alpha=0.9,
                )

            acc_vals = [w["methods"][method]["Accuracy"][0] for w in windows]
            acc_errs = [w["methods"][method]["Accuracy"][1] for w in windows]
            ax.errorbar(
                x, acc_vals, yerr=acc_errs,
                label="Accuracy",
                color="grey",
                linestyle=":",
                marker="D",
                markersize=4,
                capsize=3,
                linewidth=1.2,
                alpha=0.7,
            )

            if row == 0:
                ax.set_title(col_titles[method], fontweight="semibold")
            if col == 0:
                ax.set_ylabel(row_labels[metric], fontweight="semibold")

            ax.set_xticks(x)
            if row == 2:
                ax.set_xticklabels(win_labels, rotation=45, ha="center")
            else:
                ax.set_xticklabels([])

            ax.set_ylim(0.57, 1)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.spines[["top", "right"]].set_visible(False)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=3,
        framealpha=0.7,
        bbox_to_anchor=(0.40, -0.02),
    )

    out = "figs/temporal_fre_results_stratified_1000_2.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    windows = parse_results(RESULTS_FILE)
    plot(windows)

# %%
