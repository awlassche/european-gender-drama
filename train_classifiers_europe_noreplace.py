"""
Same experiment as train_classifiers_europe.py, but every balanced sample
is drawn WITHOUT replacement, so no speech chunk occurs twice in a sample
(and therefore never in both the training and the test split).

What changes compared to train_classifiers_europe.py:
  - Speakers are identified by their DraCor character id (speaker_id), not
    by name, since two characters in one play can share a name. This needs
    embedding datasets made with the current get_embeddings_europe.py.
  - Per speaker-play group, chunks are drawn without replacement. A group
    can contribute at most as many chunks as it has.
  - The quota per group is spread as evenly as possible: every group gets
    the same share, a group smaller than that share gives all its chunks,
    and what it could not give is shared out over the larger groups
    (repeated until n_samples is reached). There is no unstratified
    top-up from the whole class any more.
  - The TF-IDF vectorizer is fitted on the training split only (vocabulary,
    min_df/max_df filtering and IDF weights), and the test split is only
    transformed. The original fits it on the whole balanced sample before
    the split, so the test chunks help decide which n-grams are kept and
    how they are weighted. --tfidf_fit all restores the original behaviour
    for comparison. Embeddings need no fitting and are unaffected.
  - --split play holds out whole plays instead of passages: the test set
    is one fold of a StratifiedGroupKFold (5 folds, grouped by play,
    stratified by gender), so ~20% of the plays in the sample, and no
    chunk, character or play occurs on both sides. The test set is then
    only approximately gender-balanced. --split passage (default) keeps the
    original random 80/20 split of chunks, stratified by gender.
  - A class can only provide as many chunks as it has. If the smaller
    class of a language has fewer than --min_size chunks, that language
    uses the smaller class size for both classes, and the output file
    reports the sample size used per language.

  - --equal_speakers N (robustness check): instead of a fixed number of
    chunks, every language gets the same number of distinct speakers per
    gender: N speaker-play combinations drawn at random (from those with at
    least --chunks_per_speaker chunks), with exactly --chunks_per_speaker
    chunks each. With a fixed number of chunks, small corpora (Italian,
    Dutch) fill their samples with a few hundred speakers who each give
    several chunks, while English and French give one chunk per speaker;
    this design removes that difference.

Everything else (seeds, train/test split, TF-IDF settings, classifier,
output format) is identical, so the two result files can be compared
directly - except that only TF-IDF and the BGE (bge-m3) embeddings are
evaluated; the Jina embeddings are no longer part of the study.

python train_classifiers_europe_noreplace.py \
  --bge_dataset_id awlassche/european-gender-drama-bge-embeddings-v2 \
  --output results/evaluation_results_stratified_noreplace.txt

python train_classifiers_europe_noreplace.py \
  --bge_dataset_id awlassche/european-gender-drama-bge-embeddings-v2 \
  --tfidf_fit train --split play --equal_speakers 350 --chunks_per_speaker 1 \
  --output results/evaluation_results_v2_noreplace_tfidf-train_heldout-plays_equal-speakers_350x1.txt
"""

import argparse
import os
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report


def clean_text(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))


def group_quotas(group_sizes, n_samples, rng):
    """How many chunks to draw from each group: as equal as possible,
    never more than a group has, summing to n_samples."""
    caps = np.asarray(group_sizes)
    quota = np.zeros(len(caps), dtype=int)
    remaining = n_samples
    while remaining > 0:
        open_groups = np.flatnonzero(quota < caps)
        share = remaining // len(open_groups)
        if share == 0:
            # fewer chunks left to hand out than groups that can take one:
            # give one each to a random selection of those groups
            quota[rng.choice(open_groups, size=remaining, replace=False)] += 1
            break
        added = np.minimum(share, caps[open_groups] - quota[open_groups])
        quota[open_groups] += added
        remaining -= added.sum()
    return quota


def stratified_sample_without_replacement(df, n_samples, random_state):
    """Sample n_samples rows from df without replacement, with contributions
    per speaker-play combination as equal as the group sizes allow."""
    if n_samples > len(df):
        raise ValueError(f"Cannot draw {n_samples} rows without replacement from {len(df)}.")
    rng = np.random.default_rng(random_state)
    # datasets made before speaker_id was added only have the name
    speaker_col = 'speaker_id' if 'speaker_id' in df.columns else 'speaker'
    groups = [g for _, g in df.groupby([speaker_col, 'play'])]
    quotas = group_quotas([len(g) for g in groups], n_samples, rng)
    frames = [
        g.sample(n=q, replace=False, random_state=random_state)
        for g, q in zip(groups, quotas) if q > 0
    ]
    return pd.concat(frames).reset_index(drop=True)


def split_rows(balanced, y, split, random_state):
    """(train_idx, test_idx) row positions for one iteration."""
    if split == "play":
        folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
        return next(folds.split(np.zeros(len(y)), y, groups=balanced['play'].to_numpy()))
    # "passage": the same partition train_test_split gives on the feature matrix
    return train_test_split(np.arange(len(balanced)), test_size=0.2,
                            random_state=random_state, stratify=y)


def sample_equal_speakers(df, n_speakers, chunks_per_speaker, random_state):
    """n_speakers speaker-play combinations drawn at random from those with at
    least chunks_per_speaker chunks, and exactly chunks_per_speaker chunks from
    each (without replacement): every sample has the same number of distinct
    speakers, each weighing equally."""
    speaker_col = 'speaker_id' if 'speaker_id' in df.columns else 'speaker'
    groups = [g for _, g in df.groupby([speaker_col, 'play']) if len(g) >= chunks_per_speaker]
    if len(groups) < n_speakers:
        raise ValueError(f"Only {len(groups)} speakers with at least {chunks_per_speaker} chunks; "
                         f"{n_speakers} requested.")
    rng = np.random.default_rng(random_state)
    chosen = rng.choice(len(groups), size=n_speakers, replace=False)
    frames = [groups[k].sample(n=chunks_per_speaker, replace=False, random_state=random_state)
              for k in sorted(chosen)]
    return pd.concat(frames).reset_index(drop=True)


def _run_iterations(get_X, df, n_iterations, min_size, desc, split="passage", sampler=None,
                    predictions=None):
    """Shared resampling + logistic regression loop.
    get_X(balanced_df, train_idx, test_idx) → (X_train, X_test).
    sampler(class_df, random_state) → sample of one class; by default
    stratified_sample_without_replacement with min_size chunks.
    predictions: if a list is given, one dict per test chunk is appended to
    it (iteration, unique_id, play, year, gender, predicted), e.g. for a
    trend test at the level of plays."""
    if sampler is None:
        def sampler(class_df, random_state):
            return stratified_sample_without_replacement(class_df, n_samples=min_size, random_state=random_state)
    male_df   = df[df['gender'] == 'MALE']
    female_df = df[df['gender'] == 'FEMALE']
    results   = {'MALE': {'precision': [], 'recall': [], 'f1': []},
                 'FEMALE': {'precision': [], 'recall': [], 'f1': []},
                 'accuracy': []}

    for i in tqdm(range(n_iterations), desc=desc):
        male_sample   = sampler(male_df,   random_state=i)
        female_sample = sampler(female_df, random_state=i)
        balanced = (pd.concat([male_sample, female_sample])
                      .sample(frac=1, random_state=i)
                      .reset_index(drop=True))

        y = balanced['gender'].to_numpy()  # .values is an Arrow array with pandas 3, which sklearn cannot index

        # split row positions first, so features can be fitted on the
        # training rows only
        train_idx, test_idx = split_rows(balanced, y, split, random_state=i)
        X_train, X_test = get_X(balanced, train_idx, test_idx)
        y_train, y_test = y[train_idx], y[test_idx]
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        if predictions is not None:
            test_rows = balanced.iloc[test_idx]
            for row, gender, predicted in zip(test_rows.itertuples(), y_test, y_pred):
                predictions.append({
                    'iteration': i,
                    'unique_id': getattr(row, 'unique_id', None),
                    'play': row.play,
                    'year': getattr(row, 'year', None),
                    'gender': gender,
                    'predicted': predicted,
                })

        for gender in ['MALE', 'FEMALE']:
            results[gender]['precision'].append(report[gender]['precision'])
            results[gender]['recall'].append(report[gender]['recall'])
            results[gender]['f1'].append(report[gender]['f1-score'])
        results['accuracy'].append(report['accuracy'])

    return results


def evaluate_language(language, df_bge, n_iterations=50, min_size=1500, tfidf_fit="train",
                      split="passage", equal_speakers=None, chunks_per_speaker=1):
    df_bge = df_bge[df_bge['gender'].isin(['MALE', 'FEMALE'])].copy()

    sampler = None
    if equal_speakers:
        min_size = equal_speakers * chunks_per_speaker

        def sampler(class_df, random_state):
            return sample_equal_speakers(class_df, equal_speakers, chunks_per_speaker, random_state)

    # without replacement, a class cannot provide more chunks than it has
    smallest_class = min((df_bge['gender'] == g).sum() for g in ('MALE', 'FEMALE'))
    if sampler is None and smallest_class < min_size:
        print(f"  {language}: smallest class has {smallest_class} chunks, "
              f"sampling {smallest_class} per class instead of {min_size}")
        min_size = smallest_class

    def get_tfidf(balanced, train_idx, test_idx):
        vec = TfidfVectorizer(ngram_range=(2, 3), min_df=3, max_df=0.9, preprocessor=clean_text)
        texts = balanced['speech_chunk']
        if tfidf_fit == "train":
            return vec.fit_transform(texts.iloc[train_idx]), vec.transform(texts.iloc[test_idx])
        X = vec.fit_transform(texts)  # "all": as in train_classifiers_europe.py
        return X[train_idx], X[test_idx]

    def get_embeddings(balanced, train_idx, test_idx):
        X = np.vstack(balanced['embedding'].to_numpy())
        return X[train_idx], X[test_idx]

    tfidf_r = _run_iterations(get_tfidf,      df_bge,  n_iterations, min_size, f"TF-IDF [{language}]", split, sampler)
    bge_r   = _run_iterations(get_embeddings, df_bge,  n_iterations, min_size, f"BGE    [{language}]", split, sampler)

    def avg(r, key, sub=None):
        vals = r[key][sub] if sub else r[key]
        return np.mean(vals), np.std(vals)

    design = (f"equal speakers: {equal_speakers} x {chunks_per_speaker} chunks" if equal_speakers
              else "sampled without replacement")
    row = {'language': language, 'n_per_class': min_size, 'tfidf_fit': tfidf_fit, 'split': split,
           'design': design}
    for prefix, r in [('tfidf', tfidf_r), ('bge', bge_r)]:
        for gender in ['MALE', 'FEMALE']:
            for metric, sub in [('precision', 'precision'), ('recall', 'recall'), ('f1', 'f1')]:
                row[f'{prefix}_{gender}_{metric}'], row[f'{prefix}_{gender}_{metric}_std'] = avg(r, gender, sub)
        row[f'{prefix}_accuracy'], row[f'{prefix}_accuracy_std'] = avg(r, 'accuracy')

    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bge_dataset_id",  type=str, required=True,
                        help="HuggingFace dataset ID for BGE embeddings")
    parser.add_argument("--n_iterations", type=int, default=50)
    parser.add_argument("--min_size",     type=int, default=1500)
    parser.add_argument("--output",       type=str, default="results/evaluation_results.txt")
    parser.add_argument("--tfidf_fit",    choices=["train", "all"], default="train",
                        help="Fit the TF-IDF vectorizer on the training split only (default) or, "
                             "as the original script does, on the whole balanced sample.")
    parser.add_argument("--split",        choices=["passage", "play"], default="passage",
                        help="Hold out random passages (chunks), as the original script does, or "
                             "whole plays.")
    parser.add_argument("--equal_speakers", type=int, default=None,
                        help="Draw this many distinct speakers per gender in every language "
                             "(robustness check, see the module docstring). Default: off.")
    parser.add_argument("--chunks_per_speaker", type=int, default=1,
                        help="With --equal_speakers: chunks per speaker. Default: 1")
    args = parser.parse_args()

    print("Loading BGE dataset...")
    bge_df  = load_dataset(args.bge_dataset_id,  split="train").to_pandas()

    if 'speaker_id' not in bge_df.columns:
        print("Warning: the BGE dataset has no speaker_id column; speakers are "
              "grouped by name, so same-named characters in one play count as one.")

    languages = sorted(bge_df['language'].unique())
    print(f"Languages found: {languages}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    all_results = []
    for language in languages:
        print(f"\n--- Language: {language} ---")
        result = evaluate_language(
            language    = language,
            df_bge      = bge_df[bge_df['language']   == language].copy(),
            n_iterations = args.n_iterations,
            min_size     = args.min_size,
            tfidf_fit    = args.tfidf_fit,
            split        = args.split,
            equal_speakers     = args.equal_speakers,
            chunks_per_speaker = args.chunks_per_speaker,
        )
        all_results.append(result)

    with open(args.output, "w") as f:
        for r in all_results:
            f.write(f"Language: {r['language']} (n per class: {r['n_per_class']}, {r['design']}, "
                    f"TF-IDF fitted on: {r['tfidf_fit']}, held out: {r['split']}s)\n")
            for label, prefix in [("TF-IDF", "tfidf"), ("BGE embeddings", "bge")]:
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
