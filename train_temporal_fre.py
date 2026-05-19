"""
Temporal gender classification on the French corpus.
Trains classifiers on 25-year subcorpora (1625–1800) to see how well
male vs. female speech can be distinguished across time.

python train_temporal_fre.py \
  --bge_dataset_id awlassche/european-gender-drama-bge-embeddings \
  --jina_dataset_id awlassche/european-gender-drama-jina-embeddings \
  --output results/temporal_fre_results_stratified.txt
"""

import argparse
import os
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report


def clean_text(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))


def stratified_resample(df, n_samples, random_state):
    """Sample n_samples from df with equal contribution per speaker-play combination."""
    groups = df.groupby(['speaker', 'play'])
    per_speaker = max(1, n_samples // len(groups))
    frames = [
        g.sample(n=per_speaker, replace=True, random_state=random_state)
        for _, g in groups
    ]
    sampled = pd.concat(frames)
    if len(sampled) > n_samples:
        sampled = sampled.sample(n=n_samples, random_state=random_state)
    elif len(sampled) < n_samples:
        extra = df.sample(n=n_samples - len(sampled), replace=True, random_state=random_state)
        sampled = pd.concat([sampled, extra])
    return sampled.reset_index(drop=True)


def _run_iterations(get_X, df, n_iterations, min_size, desc):
    male_df   = df[df['gender'] == 'MALE']
    female_df = df[df['gender'] == 'FEMALE']
    results   = {'MALE':   {'precision': [], 'recall': [], 'f1': []},
                 'FEMALE': {'precision': [], 'recall': [], 'f1': []},
                 'accuracy': []}

    for i in tqdm(range(n_iterations), desc=desc):
        male_sample   = stratified_resample(male_df,   n_samples=min_size, random_state=i)
        female_sample = stratified_resample(female_df, n_samples=min_size, random_state=i)
        balanced = (pd.concat([male_sample, female_sample])
                      .sample(frac=1, random_state=i)
                      .reset_index(drop=True))

        X = get_X(balanced, i)
        y = balanced['gender'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i, stratify=y)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        report = classification_report(y_test, clf.predict(X_test), output_dict=True)

        for gender in ['MALE', 'FEMALE']:
            results[gender]['precision'].append(report[gender]['precision'])
            results[gender]['recall'].append(report[gender]['recall'])
            results[gender]['f1'].append(report[gender]['f1-score'])
        results['accuracy'].append(report['accuracy'])

    return results


def evaluate_window(window_label, df_bge, df_jina, n_iterations, min_size):
    df_bge  = df_bge[df_bge['gender'].isin(['MALE', 'FEMALE'])].copy()
    df_jina = df_jina[df_jina['gender'].isin(['MALE', 'FEMALE'])].copy()

    def get_tfidf(balanced, i):
        vec = TfidfVectorizer(ngram_range=(2, 3), min_df=3, max_df=0.9, preprocessor=clean_text)
        return vec.fit_transform(balanced['speech_chunk'])

    def get_embeddings(balanced, i):
        return np.vstack(balanced['embedding'].values)

    tfidf_r = _run_iterations(get_tfidf,      df_bge,  n_iterations, min_size, f"TF-IDF [{window_label}]")
    bge_r   = _run_iterations(get_embeddings, df_bge,  n_iterations, min_size, f"BGE    [{window_label}]")
    jina_r  = _run_iterations(get_embeddings, df_jina, n_iterations, min_size, f"Jina   [{window_label}]")

    def avg(r, key, sub=None):
        vals = r[key][sub] if sub else r[key]
        return np.mean(vals), np.std(vals)

    row = {
        'window':   window_label,
        'n_male':   len(df_bge[df_bge['gender'] == 'MALE']),
        'n_female': len(df_bge[df_bge['gender'] == 'FEMALE']),
        'min_size': min_size,
    }
    for prefix, r in [('tfidf', tfidf_r), ('bge', bge_r), ('jina', jina_r)]:
        for gender in ['MALE', 'FEMALE']:
            for metric, sub in [('precision', 'precision'), ('recall', 'recall'), ('f1', 'f1')]:
                row[f'{prefix}_{gender}_{metric}'], row[f'{prefix}_{gender}_{metric}_std'] = avg(r, gender, sub)
        row[f'{prefix}_accuracy'], row[f'{prefix}_accuracy_std'] = avg(r, 'accuracy')

    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bge_dataset_id",  type=str, required=True)
    parser.add_argument("--jina_dataset_id", type=str, required=True)
    parser.add_argument("--n_iterations",    type=int, default=50)
    parser.add_argument("--min_size",        type=int, default=1000,
                        help="Samples per gender per window; auto-capped at available data (default: 1000)")
    parser.add_argument("--min_threshold",   type=int, default=50,
                        help="Skip window if either gender has fewer than this many samples (default: 50)")
    parser.add_argument("--window_size",     type=int, default=25)
    parser.add_argument("--start_year",      type=int, default=1625)
    parser.add_argument("--end_year",        type=int, default=1800)
    parser.add_argument("--output",          type=str, default="results/temporal_fre_results.txt")
    args = parser.parse_args()

    print("Loading BGE dataset...")
    bge_df = load_dataset(args.bge_dataset_id, split="train").to_pandas()
    bge_df = bge_df[(bge_df['language'] == 'fre') & bge_df['year'].notna()].copy()
    bge_df['year'] = bge_df['year'].astype(int)

    print("Loading Jina dataset...")
    jina_df = load_dataset(args.jina_dataset_id, split="train").to_pandas()
    jina_df = jina_df[(jina_df['language'] == 'fre') & jina_df['year'].notna()].copy()
    jina_df['year'] = jina_df['year'].astype(int)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    all_results = []
    for start in range(args.start_year, args.end_year, args.window_size):
        end   = start + args.window_size
        label = f"{start}–{end - 1}"

        w_bge  = bge_df[(bge_df['year']   >= start) & (bge_df['year']   < end)]
        w_jina = jina_df[(jina_df['year'] >= start) & (jina_df['year']  < end)]

        n_male   = len(w_bge[w_bge['gender'] == 'MALE'])
        n_female = len(w_bge[w_bge['gender'] == 'FEMALE'])

        if n_male < args.min_threshold or n_female < args.min_threshold:
            print(f"\nSkipping {label}: MALE={n_male}, FEMALE={n_female} (below threshold {args.min_threshold})")
            continue

        effective_min = min(args.min_size, n_male, n_female)

        male_combos   = w_bge[w_bge['gender'] == 'MALE'].groupby(['speaker', 'play']).ngroups
        female_combos = w_bge[w_bge['gender'] == 'FEMALE'].groupby(['speaker', 'play']).ngroups
        per_speaker_m = max(1, effective_min // male_combos)
        per_speaker_f = max(1, effective_min // female_combos)
        print(f"\n--- {label} | chunks: MALE={n_male}, FEMALE={n_female} | "
              f"speaker-play combos: MALE={male_combos}, FEMALE={female_combos} | "
              f"per_speaker: MALE={per_speaker_m}, FEMALE={per_speaker_f} | "
              f"min_size={effective_min} ---")

        result = evaluate_window(label, w_bge, w_jina, args.n_iterations, effective_min)
        all_results.append(result)

    with open(args.output, "w") as f:
        for r in all_results:
            f.write(f"Window: {r['window']} (MALE={r['n_male']}, FEMALE={r['n_female']}, min_size={r['min_size']})\n")
            for label, prefix in [("TF-IDF", "tfidf"), ("BGE embeddings", "bge"), ("Jina embeddings", "jina")]:
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
