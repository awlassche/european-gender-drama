# %%

import io
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

for _font_path in [
    os.path.expanduser("~/Library/Fonts/RobotoCondensed-Regular.ttf"),
    os.path.expanduser("~/Library/Fonts/RobotoCondensed-Bold.ttf"),
]:
    fm.fontManager.addfont(_font_path)

plt.rcParams.update({
    "font.family": "Roboto Condensed",
    "font.sans-serif": ["Roboto Condensed"],
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Roboto Condensed",
    "mathtext.bf": "Roboto Condensed:bold",
})

CHRgreen = '#1B4332'
CHRred = '#C0392B'
CHRyellow = '#E8A838'
CHRbeige = '#D4C5B0'
CHRcream = '#F5F0E8'

CREDENTIALS_FILE = os.environ.get("GOOGLE_DRIVE_CREDENTIALS", "../credentials.json")
DRIVE_FOLDER_NAME = "DutchDraCor_gender"
CACHE_DIR = "data/corpus_stats"

CORPORA = ["fre", "eng", "ger", "cal", "dutch", "ita"]
CORPUS_NAMES = {
    "fre": "French",
    "eng": "English",
    "ger": "German",
    "cal": "Spanish",
    "dutch": "Dutch",
    "ita": "Italian",
}

# ── fetch CSVs from Google Drive (service account) ────────────────────────────
# %%


def find_subfolder(service, parent_id, name):
    resp = service.files().list(
        q=f"'{parent_id}' in parents and name = '{name}' and trashed = false",
        fields="files(id, name)",
        pageSize=10,
    ).execute()
    files = resp.get("files", [])
    if not files:
        raise FileNotFoundError(f"Could not find '{name}' under parent {parent_id}")
    return files[0]["id"]


def find_root_folder(service, name):
    resp = service.files().list(
        q=f"name = '{name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false",
        fields="files(id, name)",
        pageSize=10,
    ).execute()
    files = resp.get("files", [])
    if not files:
        raise FileNotFoundError(f"Could not find folder '{name}'")
    return files[0]["id"]


def download_file(service, file_id, dest_path):
    request = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    with open(dest_path, "wb") as f:
        f.write(buf.getvalue())


def fetch_corpus_csvs():
    os.makedirs(CACHE_DIR, exist_ok=True)

    needed = (
        [f"play_data/plays_df_{c}.csv" for c in CORPORA]
        + [f"character_data/character_df_{c}.csv" for c in CORPORA]
    )
    missing = [n for n in needed if not os.path.exists(os.path.join(CACHE_DIR, os.path.basename(n)))]
    if not missing:
        return

    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    service = build("drive", "v3", credentials=creds)

    root_id = find_root_folder(service, DRIVE_FOLDER_NAME)
    data_id = find_subfolder(service, root_id, "data")
    play_data_id = find_subfolder(service, data_id, "play_data")
    character_data_id = find_subfolder(service, data_id, "character_data")

    for sub_id, csv_name in [(play_data_id, f"plays_df_{c}.csv") for c in CORPORA]:
        dest = os.path.join(CACHE_DIR, csv_name)
        if os.path.exists(dest):
            continue
        resp = service.files().list(
            q=f"'{sub_id}' in parents and name = '{csv_name}' and trashed = false",
            fields="files(id, name)",
        ).execute()
        files = resp.get("files", [])
        if not files:
            raise FileNotFoundError(f"Could not find '{csv_name}' in play_data")
        download_file(service, files[0]["id"], dest)
        print(f"Downloaded {csv_name}")

    for csv_name in [f"character_df_{c}.csv" for c in CORPORA]:
        dest = os.path.join(CACHE_DIR, csv_name)
        if os.path.exists(dest):
            continue
        resp = service.files().list(
            q=f"'{character_data_id}' in parents and name = '{csv_name}' and trashed = false",
            fields="files(id, name)",
        ).execute()
        files = resp.get("files", [])
        if not files:
            raise FileNotFoundError(f"Could not find '{csv_name}' in character_data")
        download_file(service, files[0]["id"], dest)
        print(f"Downloaded {csv_name}")


fetch_corpus_csvs()

# ── load ────────────────────────────────────────────────────────────────────
# %%

play_data = {
    c: pd.read_csv(os.path.join(CACHE_DIR, f"plays_df_{c}.csv"), sep=";")
    for c in CORPORA
}
character_data = {
    c: pd.read_csv(os.path.join(CACHE_DIR, f"character_df_{c}.csv"), sep=";")
    for c in CORPORA
}

merged_play_df = pd.concat(play_data.values())
merged_character_df = pd.concat(character_data.values())

# ── plays per decade ──────────────────────────────────────────────────────────
# %%


def plot_plays_per_decade():
    decades = [str(decade) + "0" for decade in range(150, 180)]

    fig, axes = plt.subplots(3, 2, figsize=(7.5, 5), sharex=True)

    for ax, corpus in zip(axes.flatten(), CORPORA):
        df = play_data[corpus]
        histogram = {decade: 0 for decade in decades}
        years = pd.to_numeric(df["year"], errors="coerce").dropna()
        for year in years:
            decade = str(int(year))[0:3] + "0"
            if decade in histogram:
                histogram[decade] += 1

        ax.bar(
            list(histogram.keys()), list(histogram.values()),
            width=0.8, align="center", color=CHRyellow,
        )
        ax.set_title(r"$\bf{" + CORPUS_NAMES[corpus] + "}$" + f" (N={len(df)})")
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-0.5, len(decades) - 0.5)

    fig.tight_layout()
    plt.savefig("figs/plays_per_decade_columns.pdf", bbox_inches="tight")


plot_plays_per_decade()

# ── % female characters per corpus ───────────────────────────────────────────
# %%


def plot_female_chars_per_corpus():
    independent_var = "%_female_chars"

    means = [merged_play_df[merged_play_df.corpus == c][independent_var].mean() for c in CORPORA]
    stdvs = [merged_play_df[merged_play_df.corpus == c][independent_var].std() for c in CORPORA]
    counts = [len(merged_play_df[merged_play_df.corpus == c]) for c in CORPORA]

    fig, ax = plt.subplots(figsize=(8.5, 4))

    parts = ax.violinplot(
        dataset=[merged_play_df[merged_play_df.corpus == c][independent_var].values for c in CORPORA],
        showmedians=True,
    )
    for body in parts["bodies"]:
        body.set_facecolor(CHRred)
        body.set_edgecolor(CHRgreen)
        body.set_alpha(0.8)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color(CHRgreen)

    ax.set_xticks(range(1, len(CORPORA) + 1), labels=[""] * len(CORPORA))
    ax.set_ylim(0, 1)
    #ax.set_xlabel("Corpus", fontweight="semibold")
    ax.set_ylabel("Share of female characters (%)", fontweight="semibold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    trans = ax.get_xaxis_transform()
    for i, c in enumerate(CORPORA):
        ax.text(i + 1, -0.02, CORPUS_NAMES[c], transform=trans,
                 ha="center", va="top", fontweight="bold")
        ax.text(i + 1, -0.065, f"(N={counts[i]}, M={means[i]:.2f}, SD={stdvs[i]:.2f})",
                 transform=trans, ha="center", va="top", fontsize=8)

    fig.tight_layout()
    plt.savefig("figs/pct_female_chars_per_corpus.pdf", bbox_inches="tight")


plot_female_chars_per_corpus()

# ── % female lines per corpus (hearability) ──────────────────────────────────
# %%


def plot_female_lines_per_corpus():
    independent_var = "%_female_lines"

    means = [merged_play_df[merged_play_df.corpus == c][independent_var].mean() for c in CORPORA]
    stdvs = [merged_play_df[merged_play_df.corpus == c][independent_var].std() for c in CORPORA]
    counts = [len(merged_play_df[merged_play_df.corpus == c]) for c in CORPORA]

    fig, ax = plt.subplots(figsize=(8.5, 4))

    parts = ax.violinplot(
        dataset=[merged_play_df[merged_play_df.corpus == c][independent_var].values for c in CORPORA],
        showmedians=True,
    )
    for body in parts["bodies"]:
        body.set_facecolor(CHRred)
        body.set_edgecolor(CHRgreen)
        body.set_alpha(0.8)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color(CHRgreen)

    ax.set_xticks(range(1, len(CORPORA) + 1), labels=[""] * len(CORPORA))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of female lines (%)", fontweight="semibold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    trans = ax.get_xaxis_transform()
    for i, c in enumerate(CORPORA):
        ax.text(i + 1, -0.02, CORPUS_NAMES[c], transform=trans,
                 ha="center", va="top", fontweight="bold")
        ax.text(i + 1, -0.065, f"(N={counts[i]}, M={means[i]:.2f}, SD={stdvs[i]:.2f})",
                 transform=trans, ha="center", va="top", fontsize=8)

    fig.tight_layout()
    plt.savefig("figs/pct_female_lines_per_corpus.pdf", bbox_inches="tight")


plot_female_lines_per_corpus()

# ── centrality by gender, per corpus ─────────────────────────────────────────
# %%


def plot_centrality_by_gender():
    independent_var = "degree (rel)"
    colors = {"MALE": CHRyellow, "FEMALE": CHRred}

    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5))

    for ax, corpus in zip(axes.flatten(), CORPORA):
        df = character_data[corpus]
        male_vals = df[df.gender == "MALE"][independent_var].dropna().values
        female_vals = df[df.gender == "FEMALE"][independent_var].dropna().values

        parts = ax.violinplot(dataset=[male_vals, female_vals], showmeans=True, widths=0.7)
        for body, cls in zip(parts["bodies"], ["MALE", "FEMALE"]):
            body.set_facecolor(colors[cls])
            body.set_edgecolor(CHRgreen)
            body.set_alpha(0.8)
        for key in ("cmeans", "cmins", "cmaxes", "cbars"):
            parts[key].set_color(CHRgreen)

        labels = [
            f"MALE\n(N={len(male_vals)}, M={male_vals.mean():.2f}, SD={male_vals.std():.2f})",
            f"FEMALE\n(N={len(female_vals)}, M={female_vals.mean():.2f}, SD={female_vals.std():.2f})",
        ]
        ax.set_xticks([1, 2], labels=labels, fontsize=8)
        ax.set_title(CORPUS_NAMES[corpus], fontweight="semibold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    #fig.suptitle("Degree centrality by gender", fontweight="bold")
    fig.tight_layout()
    plt.savefig("figs/centrality_by_gender.pdf", bbox_inches="tight")


plot_centrality_by_gender()

# ── mean speech length over time ─────────────────────────────────────────────
# %%


def plot_mean_speech_length_over_time():
    variable = "mean_sp_length"

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [CHRgreen, CHRred, CHRyellow, CHRbeige, "#4A7C93", "#7D5BA6"]
    for corpus, color in zip(CORPORA, colors):
        df = play_data[corpus].copy()
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df.dropna(subset=["year", variable]).sort_values("year")
        df = df[(df.year >= 1500) & (df.year <= 1800)]
        ax.plot(
            df["year"], df[variable].rolling(10, min_periods=1).mean(),
            label=CORPUS_NAMES[corpus], color=color, linewidth=1.8,
        )

    ax.set_xlim(1500, 1800)
    ax.set_ylim(0, 20)
    ax.set_xlabel("Year", fontweight="semibold")
    ax.set_ylabel("Mean speech length (10-play rolling mean)", fontweight="semibold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", ncol=2, framealpha=0.7)

    fig.tight_layout()
    plt.savefig("figs/mean_speech_length_over_time.pdf", bbox_inches="tight")


plot_mean_speech_length_over_time()

# ── % female lines & characters over time, French, 25-yr windows ─────────────
# %%

# Same windowing as train_temporal_fre.py, so the two plots line up.
WINDOW_SIZE = 25
START_YEAR = 1625
END_YEAR = 1800


def plot_female_pct_over_time_fre():
    df = play_data["fre"].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year", "%_female_chars", "%_female_lines"])

    windows, chars_mean, chars_std, lines_mean, lines_std, counts = [], [], [], [], [], []
    for start in range(START_YEAR, END_YEAR, WINDOW_SIZE):
        end = start + WINDOW_SIZE
        w = df[(df.year >= start) & (df.year < end)]
        if w.empty:
            continue
        windows.append(f"{start}–{end - 1}")
        chars_mean.append(w["%_female_chars"].mean())
        chars_std.append(w["%_female_chars"].std())
        lines_mean.append(w["%_female_lines"].mean())
        lines_std.append(w["%_female_lines"].std())
        counts.append(len(w))

    print("French, % female chars/lines per 25-yr window:")
    for label, n, cm, lm in zip(windows, counts, chars_mean, lines_mean):
        print(f"  {label} (N={n}): chars={cm:.2f}, lines={lm:.2f}")

    x = np.arange(len(windows))

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.errorbar(
        x, chars_mean, yerr=chars_std,
        label="FEMALE CHARACTERS", color=CHRred,
        marker="o", markersize=5, capsize=3, linewidth=1.8, alpha=0.9,
    )
    ax.errorbar(
        x, lines_mean, yerr=lines_std,
        label="FEMALE LINES", color='grey',
        marker="s", markersize=5, capsize=3, linewidth=1.8, alpha=0.9,
    )

    ax.set_xticks(x, labels=windows, ha="center")
    ax.set_ylim(0, 0.6)
    #ax.set_xlabel("Year (25-year windows)", fontweight="semibold")
    ax.set_ylabel("Share (%)", fontweight="semibold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, framealpha=0.7)

    fig.tight_layout()
    plt.savefig("figs/female_pct_over_time_fre.pdf", bbox_inches="tight")


plot_female_pct_over_time_fre()

# %%
