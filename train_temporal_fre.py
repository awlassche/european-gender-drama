"""
Temporal gender classification within one corpus (French by default).
Trains classifiers on 25-year subcorpora to see how well male vs. female
speech can be distinguished across time.

  French:  1625-1799, 800 chunks per gender per window (the defaults)
  English: 1575-1699, 650 chunks per gender per window. 1550-1574 is left
           out (22 plays, 193 female chunks), and so are the ten plays dated
           1700. The smallest window, 1575-1599, has 687 female chunks.

Uses the sampling and evaluation of train_classifiers_europe_noreplace.py,
for TF-IDF and the BGE (bge-m3) embeddings:
  - balanced samples drawn without replacement, as evenly as possible per
    speaker-play combination (by DraCor character id);
  - the TF-IDF vectorizer fitted on the training split only (--tfidf_fit);
  - whole plays held out for testing (--split, default "play").
A window uses min(--min_size, the smaller class) chunks per class.

--equal_speakers N switches to a design with the same number of distinct
speakers in every window: per gender, N speaker-play combinations are drawn
at random (from those with at least --chunks_per_speaker chunks), with
exactly --chunks_per_speaker chunks each. This separates a change over time
from a change in how many different speakers a window has (smaller later
pools mean fewer distinct female speakers). A window without N such speakers
for both genders is skipped.

python train_temporal_fre.py \
  --bge_dataset_id awlassche/european-gender-drama-bge-embeddings-v2 \
  --output results/temporal_fre_results_v2_noreplace_tfidf-train_heldout-plays_800.txt

python train_temporal_fre.py \
  --bge_dataset_id awlassche/european-gender-drama-bge-embeddings-v2 \
  --language eng --start_year 1575 --end_year 1700 --min_size 650 \
  --output results/temporal_eng_results_v2_noreplace_tfidf-train_heldout-plays_650.txt

python train_temporal_fre.py \
  --bge_dataset_id awlassche/european-gender-drama-bge-embeddings-v2 \
  --equal_speakers 300 --chunks_per_speaker 1 \
  --output results/temporal_fre_results_v2_equal-speakers_300x1.txt
"""

import argparse
import os
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

from train_classifiers_europe_noreplace import _run_iterations, clean_text, sample_equal_speakers


def evaluate_window(window_label, df_bge, n_iterations, min_size, tfidf_fit, split, sampler=None,
                    predictions=None):
    df_bge = df_bge[df_bge['gender'].isin(['MALE', 'FEMALE'])].copy()

    def get_tfidf(balanced, train_idx, test_idx):
        vec = TfidfVectorizer(ngram_range=(2, 3), min_df=3, max_df=0.9, preprocessor=clean_text)
        texts = balanced['speech_chunk']
        if tfidf_fit == "train":
            return vec.fit_transform(texts.iloc[train_idx]), vec.transform(texts.iloc[test_idx])
        X = vec.fit_transform(texts)
        return X[train_idx], X[test_idx]

    def get_embeddings(balanced, train_idx, test_idx):
        X = np.vstack(balanced['embedding'].to_numpy())
        return X[train_idx], X[test_idx]

    tfidf_p, bge_p = ([], []) if predictions is not None else (None, None)
    tfidf_r = _run_iterations(get_tfidf,      df_bge,  n_iterations, min_size, f"TF-IDF [{window_label}]", split,
                              sampler, tfidf_p)
    bge_r   = _run_iterations(get_embeddings, df_bge,  n_iterations, min_size, f"BGE    [{window_label}]", split,
                              sampler, bge_p)
    if predictions is not None:
        for method, records in (("TF-IDF", tfidf_p), ("bge-m3", bge_p)):
            predictions.extend({**rec, 'method': method, 'window': window_label} for rec in records)

    def avg(r, key, sub=None):
        vals = r[key][sub] if sub else r[key]
        return np.mean(vals), np.std(vals)

    row = {
        'window':   window_label,
        'n_male':   len(df_bge[df_bge['gender'] == 'MALE']),
        'n_female': len(df_bge[df_bge['gender'] == 'FEMALE']),
        'min_size': min_size,
    }
    for prefix, r in [('tfidf', tfidf_r), ('bge', bge_r)]:
        for gender in ['MALE', 'FEMALE']:
            for metric, sub in [('precision', 'precision'), ('recall', 'recall'), ('f1', 'f1')]:
                row[f'{prefix}_{gender}_{metric}'], row[f'{prefix}_{gender}_{metric}_std'] = avg(r, gender, sub)
        row[f'{prefix}_accuracy'], row[f'{prefix}_accuracy_std'] = avg(r, 'accuracy')

    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bge_dataset_id",  type=str, required=True)
    parser.add_argument("--language",        type=str, default="fre",
                        help="Corpus to analyse (fre, eng, ger, dutch, ita, cal). Default: fre")
    parser.add_argument("--n_iterations",    type=int, default=50)
    parser.add_argument("--min_size",        type=int, default=800,
                        help="Samples per gender per window; auto-capped at available data. The default, "
                             "800, fits every 1625-1799 window of the current French data (for English "
                             "1575-1699, use 650), so all "
                             "windows use the same sample size.")
    parser.add_argument("--min_threshold",   type=int, default=50,
                        help="Skip window if either gender has fewer than this many samples (default: 50)")
    parser.add_argument("--window_size",     type=int, default=25)
    parser.add_argument("--start_year",      type=int, default=1625)
    parser.add_argument("--end_year",        type=int, default=1800)
    parser.add_argument("--output",          type=str, default="results/temporal_fre_results.txt")
    parser.add_argument("--tfidf_fit",       choices=["train", "all"], default="train",
                        help="Fit the TF-IDF vectorizer on the training split only (default) or on "
                             "the whole balanced sample.")
    parser.add_argument("--equal_speakers",  type=int, default=None,
                        help="Draw this many distinct speakers per gender in every window "
                             "(see the module docstring). Default: off.")
    parser.add_argument("--chunks_per_speaker", type=int, default=1,
                        help="With --equal_speakers: chunks per speaker. Default: 1")
    parser.add_argument("--predictions_out", type=str, default=None,
                        help="Also save every test prediction (method, window, iteration, chunk, "
                             "play, year, gender, predicted) to this CSV, for test_temporal_trend.py.")
    parser.add_argument("--split",           choices=["passage", "play"], default="play",
                        help="Hold out whole plays (default) or random passages (chunks).")
    args = parser.parse_args()

    print("Loading BGE dataset...")
    bge_df = load_dataset(args.bge_dataset_id, split="train").to_pandas()
    bge_df = bge_df[(bge_df['language'] == args.language) & bge_df['year'].notna()].copy()
    bge_df['year'] = bge_df['year'].astype(int)


    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    all_results = []
    all_predictions = [] if args.predictions_out else None
    for start in range(args.start_year, args.end_year, args.window_size):
        end   = start + args.window_size
        label = f"{start}–{end - 1}"

        w_bge = bge_df[(bge_df['year'] >= start) & (bge_df['year'] < end)]

        n_male   = len(w_bge[w_bge['gender'] == 'MALE'])
        n_female = len(w_bge[w_bge['gender'] == 'FEMALE'])

        if n_male < args.min_threshold or n_female < args.min_threshold:
            print(f"\nSkipping {label}: MALE={n_male}, FEMALE={n_female} (below threshold {args.min_threshold})")
            continue

        # without replacement, a class cannot provide more chunks than it has
        effective_min = min(args.min_size, n_male, n_female)
        sampler = None
        if args.equal_speakers:
            c = args.chunks_per_speaker
            available = {g: (w_bge[w_bge['gender'] == g].groupby(['speaker_id', 'play']).size() >= c).sum()
                         for g in ('MALE', 'FEMALE')}
            if min(available.values()) < args.equal_speakers:
                print(f"\nSkipping {label}: speakers with >= {c} chunks: {available} "
                      f"(fewer than {args.equal_speakers})")
                continue
            effective_min = args.equal_speakers * c
            sampler = lambda class_df, random_state: sample_equal_speakers(
                class_df, args.equal_speakers, c, random_state)

        speaker_col   = 'speaker_id' if 'speaker_id' in w_bge.columns else 'speaker'
        male_combos   = w_bge[w_bge['gender'] == 'MALE'].groupby([speaker_col, 'play']).ngroups
        female_combos = w_bge[w_bge['gender'] == 'FEMALE'].groupby([speaker_col, 'play']).ngroups
        print(f"\n--- {label} | chunks: MALE={n_male}, FEMALE={n_female} | "
              f"speaker-play combos: MALE={male_combos}, FEMALE={female_combos} | "
              f"plays: {w_bge['play'].nunique()} | min_size={effective_min} ---")

        result = evaluate_window(label, w_bge, args.n_iterations, effective_min,
                                 args.tfidf_fit, args.split, sampler, all_predictions)
        all_results.append(result)

    with open(args.output, "w") as f:
        for r in all_results:
            design = (f"equal speakers: {args.equal_speakers} x {args.chunks_per_speaker} chunks"
                      if args.equal_speakers else "sampled without replacement")
            f.write(f"Window: {r['window']} (MALE={r['n_male']}, FEMALE={r['n_female']}, min_size={r['min_size']}, "
                    f"{design}, TF-IDF fitted on: {args.tfidf_fit}, held out: {args.split}s)\n")
            for label, prefix in [("TF-IDF", "tfidf"), ("BGE embeddings", "bge")]:
                f.write(f"{label}:\n")
                for gender in ['MALE', 'FEMALE']:
                    p,  ps  = r[f'{prefix}_{gender}_precision'],     r[f'{prefix}_{gender}_precision_std']
                    rc, rcs = r[f'{prefix}_{gender}_recall'],        r[f'{prefix}_{gender}_recall_std']
                    f1, f1s = r[f'{prefix}_{gender}_f1'],            r[f'{prefix}_{gender}_f1_std']
                    f.write(f"  {gender:<6} → P: {p:.4f}±{ps:.4f}, R: {rc:.4f}±{rcs:.4f}, F1: {f1:.4f}±{f1s:.4f}\n")
                acc, accs = r[f'{prefix}_accuracy'], r[f'{prefix}_accuracy_std']
                f.write(f"  Accuracy: {acc:.4f}±{accs:.4f}\n")
            f.write("\n")

    print(f"\nResults saved to {args.output}")

    if args.predictions_out:
        import pandas as pd
        os.makedirs(os.path.dirname(args.predictions_out) or ".", exist_ok=True)
        pd.DataFrame(all_predictions).to_csv(args.predictions_out, index=False)
        print(f"Predictions saved to {args.predictions_out}")
