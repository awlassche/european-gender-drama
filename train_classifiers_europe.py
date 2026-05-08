"""
python train_classifiers_europe.py \
  --bge_dataset_id awlassche/european-gender-drama-bge-embeddings \
  --jina_dataset_id awlassche/european-gender-drama-jina-embeddings \
  --output results/evaluation_results_260508.txt
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
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report


def clean_text(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))


def _run_iterations(get_X, df, n_iterations, min_size, desc):
    """Shared resampling + logistic regression loop. get_X(balanced_df) → feature matrix."""
    male_df   = df[df['gender'] == 'MALE']
    female_df = df[df['gender'] == 'FEMALE']
    results   = {'MALE': {'precision': [], 'recall': [], 'f1': []},
                 'FEMALE': {'precision': [], 'recall': [], 'f1': []},
                 'accuracy': []}

    for i in tqdm(range(n_iterations), desc=desc):
        male_sample   = resample(male_df,   n_samples=min_size, random_state=i)
        female_sample = resample(female_df, n_samples=min_size, random_state=i)
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


def evaluate_language(language, df_bge, df_jina, n_iterations=50, min_size=1500):
    df_bge  = df_bge[df_bge['gender'].isin(['MALE', 'FEMALE'])].copy()
    df_jina = df_jina[df_jina['gender'].isin(['MALE', 'FEMALE'])].copy()

    def get_tfidf(balanced, i):
        vec = TfidfVectorizer(ngram_range=(2, 3), min_df=3, max_df=0.9, preprocessor=clean_text)
        return vec.fit_transform(balanced['speech_chunk'])

    def get_embeddings(balanced, i):
        return np.vstack(balanced['embedding'].values)

    tfidf_r = _run_iterations(get_tfidf,      df_bge,  n_iterations, min_size, f"TF-IDF [{language}]")
    bge_r   = _run_iterations(get_embeddings, df_bge,  n_iterations, min_size, f"BGE    [{language}]")
    jina_r  = _run_iterations(get_embeddings, df_jina, n_iterations, min_size, f"Jina   [{language}]")

    def avg(r, key, sub=None):
        vals = r[key][sub] if sub else r[key]
        return np.mean(vals), np.std(vals)

    row = {'language': language}
    for prefix, r in [('tfidf', tfidf_r), ('bge', bge_r), ('jina', jina_r)]:
        for gender in ['MALE', 'FEMALE']:
            for metric, sub in [('precision', 'precision'), ('recall', 'recall'), ('f1', 'f1')]:
                row[f'{prefix}_{gender}_{metric}'], row[f'{prefix}_{gender}_{metric}_std'] = avg(r, gender, sub)
        row[f'{prefix}_accuracy'], row[f'{prefix}_accuracy_std'] = avg(r, 'accuracy')

    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bge_dataset_id",  type=str, required=True,
                        help="HuggingFace dataset ID for BGE embeddings")
    parser.add_argument("--jina_dataset_id", type=str, required=True,
                        help="HuggingFace dataset ID for Jina embeddings")
    parser.add_argument("--n_iterations", type=int, default=50)
    parser.add_argument("--min_size",     type=int, default=1500)
    parser.add_argument("--output",       type=str, default="results/evaluation_results.txt")
    args = parser.parse_args()

    print("Loading BGE dataset...")
    bge_df  = load_dataset(args.bge_dataset_id,  split="train").to_pandas()
    print("Loading Jina dataset...")
    jina_df = load_dataset(args.jina_dataset_id, split="train").to_pandas()

    languages = sorted(bge_df['language'].unique())
    print(f"Languages found: {languages}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    all_results = []
    for language in languages:
        print(f"\n--- Language: {language} ---")
        result = evaluate_language(
            language    = language,
            df_bge      = bge_df[bge_df['language']   == language].copy(),
            df_jina     = jina_df[jina_df['language'] == language].copy(),
            n_iterations = args.n_iterations,
            min_size     = args.min_size,
        )
        all_results.append(result)

    with open(args.output, "w") as f:
        for r in all_results:
            f.write(f"Language: {r['language']}\n")
            for label, prefix in [("TF-IDF", "tfidf"), ("BGE embeddings", "bge"), ("Jina embeddings", "jina")]:
                f.write(f"{label}:\n")
                for gender in ['MALE', 'FEMALE']:
                    p,  ps  = r[f'{prefix}_{gender}_precision'],    r[f'{prefix}_{gender}_precision_std']
                    rc, rcs = r[f'{prefix}_{gender}_recall'],       r[f'{prefix}_{gender}_recall_std']
                    f1, f1s = r[f'{prefix}_{gender}_f1'],           r[f'{prefix}_{gender}_f1_std']
                    f.write(f"  {gender:<6} → P: {p:.4f}±{ps:.4f}, R: {rc:.4f}±{rcs:.4f}, F1: {f1:.4f}±{f1s:.4f}\n")
                acc, accs = r[f'{prefix}_accuracy'], r[f'{prefix}_accuracy_std']
                f.write(f"  Accuracy: {acc:.4f}±{accs:.4f}\n")
            f.write("\n")

    print(f"\nResults saved to {args.output}")
