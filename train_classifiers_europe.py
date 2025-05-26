import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import string

def clean_text(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

def evaluate_model_on_dataset(dataset_path, n_iterations=50):
    dataset = load_from_disk(dataset_path)
    df = dataset.to_pandas()
    df = df[df['gender'].isin(['MALE', 'FEMALE'])]

    male_df = df[df['gender'] == 'MALE']
    female_df = df[df['gender'] == 'FEMALE']
    min_size = min(len(male_df), len(female_df))

    tfidf_results = {
    'MALE': {'precision': [], 'recall': [], 'f1': []},
    'FEMALE': {'precision': [], 'recall': [], 'f1': []},
    'accuracy': []
    }
    embed_results = {
        'MALE': {'precision': [], 'recall': [], 'f1': []},
        'FEMALE': {'precision': [], 'recall': [], 'f1': []},
        'accuracy': []
    }

    for i in tqdm(range(n_iterations), desc=f"TF-IDF on {os.path.basename(dataset_path)}"):
        male_sample = resample(male_df, n_samples=min_size, random_state=i)
        female_sample = resample(female_df, n_samples=min_size, random_state=i)
        balanced_df = pd.concat([male_sample, female_sample]).sample(frac=1, random_state=i).reset_index(drop=True)

        X = balanced_df['speech_chunk']
        y = balanced_df['gender']

        vectorizer = TfidfVectorizer(ngram_range=(2, 3), min_df=3, max_df=0.9, preprocessor=clean_text)
        X_tfidf = vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=i, stratify=y)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        for gender in ['MALE', 'FEMALE']:
            tfidf_results[gender]['precision'].append(report[gender]['precision'])
            tfidf_results[gender]['recall'].append(report[gender]['recall'])
            tfidf_results[gender]['f1'].append(report[gender]['f1-score'])
            tfidf_results['accuracy'].append(report['accuracy'])


    for i in tqdm(range(n_iterations), desc=f"Embeddings on {os.path.basename(dataset_path)}"):
        male_sample = resample(male_df, n_samples=min_size, random_state=i)
        female_sample = resample(female_df, n_samples=min_size, random_state=i)
        balanced_df = pd.concat([male_sample, female_sample]).sample(frac=1, random_state=i).reset_index(drop=True)

        X = np.vstack(balanced_df['embedding'].values)
        y = balanced_df['gender'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i, stratify=y)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        for gender in ['MALE', 'FEMALE']:
            embed_results[gender]['precision'].append(report[gender]['precision'])
            embed_results[gender]['recall'].append(report[gender]['recall'])
            embed_results[gender]['f1'].append(report[gender]['f1-score'])
            embed_results['accuracy'].append(report['accuracy'])

    return {
    'dataset': os.path.basename(dataset_path),
    'tfidf_MALE_precision': np.mean(tfidf_results['MALE']['precision']),
    'tfidf_MALE_recall': np.mean(tfidf_results['MALE']['recall']),
    'tfidf_MALE_f1': np.mean(tfidf_results['MALE']['f1']),
    'tfidf_FEMALE_precision': np.mean(tfidf_results['FEMALE']['precision']),
    'tfidf_FEMALE_recall': np.mean(tfidf_results['FEMALE']['recall']),
    'tfidf_FEMALE_f1': np.mean(tfidf_results['FEMALE']['f1']),
    'tfidf_accuracy': np.mean(tfidf_results['accuracy']),
    'embed_MALE_precision': np.mean(embed_results['MALE']['precision']),
    'embed_MALE_recall': np.mean(embed_results['MALE']['recall']),
    'embed_MALE_f1': np.mean(embed_results['MALE']['f1']),
    'embed_FEMALE_precision': np.mean(embed_results['FEMALE']['precision']),
    'embed_FEMALE_recall': np.mean(embed_results['FEMALE']['recall']),
    'embed_FEMALE_f1': np.mean(embed_results['FEMALE']['f1']),
    'embed_accuracy': np.mean(embed_results['accuracy']),
    }

# Automatically gather all datasets in the "embeddings" folder
dataset_paths = [os.path.join("embeddings", d) for d in os.listdir("embeddings") if os.path.isdir(os.path.join("embeddings", d))]

# Evaluate all and collect results
all_results = []
for path in dataset_paths:
    result = evaluate_model_on_dataset(path)
    all_results.append(result)

# Save to TXT
with open("all_evaluation_results.txt", "w") as f:
    for r in all_results:
        f.write(f"📁 Dataset: {r['dataset']}\n")
        f.write("TF-IDF Results:\n")
        f.write(f"  MALE   → P: {r['tfidf_MALE_precision']:.4f}, R: {r['tfidf_MALE_recall']:.4f}, F1: {r['tfidf_MALE_f1']:.4f}\n")
        f.write(f"  FEMALE → P: {r['tfidf_FEMALE_precision']:.4f}, R: {r['tfidf_FEMALE_recall']:.4f}, F1: {r['tfidf_FEMALE_f1']:.4f}\n")
        f.write(f"  Accuracy: {r['tfidf_accuracy']:.4f}\n")

        f.write("Embedding Results:\n")
        f.write(f"  MALE   → P: {r['embed_MALE_precision']:.4f}, R: {r['embed_MALE_recall']:.4f}, F1: {r['embed_MALE_f1']:.4f}\n")
        f.write(f"  FEMALE → P: {r['embed_FEMALE_precision']:.4f}, R: {r['embed_FEMALE_recall']:.4f}, F1: {r['embed_FEMALE_f1']:.4f}\n")
        f.write(f"  Accuracy: {r['embed_accuracy']:.4f}\n\n")

print(f"✅ All results saved")