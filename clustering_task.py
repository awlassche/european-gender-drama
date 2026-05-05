import os
import re
import pandas as pd
import numpy as np
import ndjson
import torch
import einops
import hf_xet

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score

# ----------------------- CONFIG -----------------------

# Define language-specific model sets
MODEL_SETS = {
    'dutch': [
        "emanjavacas/GysBERT-v2",                    # historical Dutch
        "DTAI-KULeuven/robbert-2023-dutch-large",     # modern Dutch SOTA
        "intfloat/multilingual-e5-large",             # multilingual baseline
        "BAAI/bge-m3",                                # multilingual SOTA
        "jinaai/jina-embeddings-v3"                   # multilingual SOTA
    ],
    'eng': [
        "emanjavacas/MacBERTh",                       # historical English
        "BAAI/bge-large-en-v1.5",                     # modern English SOTA
        "intfloat/multilingual-e5-large",
        "BAAI/bge-m3",
        "jinaai/jina-embeddings-v3"
    ],
    'ger': [
        "dbmdz/bert-base-historic-german-cased",      # historical German
        "deutsche-telekom/gbert-large-paraphrase-cosine",  # modern German SOTA
        "intfloat/multilingual-e5-large",
        "BAAI/bge-m3",
        "jinaai/jina-embeddings-v3"
    ],
    'fre': [
        "dbmdz/bert-base-french-europeana-cased",     # historical French
        "dangvantuan/sentence-camembert-large",        # modern French SOTA
        "intfloat/multilingual-e5-large",
        "BAAI/bge-m3",
        "jinaai/jina-embeddings-v3"
    ],
    'ita': [
        "models/bertoldo-all/checkpoint",              # historical Italian (local)
        "nickprock/sentence-bert-base-italian-xxl-uncased",  # modern Italian SOTA
        "intfloat/multilingual-e5-large",
        "BAAI/bge-m3",
        "jinaai/jina-embeddings-v3"
    ],
    'cal': [
        "dccuchile/bert-base-spanish-wwm-cased",       # historical Spanish (BETO)
        "hiiamsid/sentence_similarity_spanish_es",    # modern Spanish SOTA
        "intfloat/multilingual-e5-large",
        "BAAI/bge-m3",
        "jinaai/jina-embeddings-v3"
    ]
}

CHUNK_SIZE = 400
TODAY = '260505'

# -------------------- FUNCTIONS -----------------------

def count_tokens(text):
    return len(text.split())

def chunk_speech(speech, chunk_size):
    words = speech.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def process_dataset(df, chunk_size):
    chunked_data = []
    for _, row in df.iterrows():
        chunks = chunk_speech(row['speech'], chunk_size)
        speaker_id_base = row['speaker'][:4]
        for i, chunk in enumerate(chunks):
            chunked_data.append({
                'speaker': row['speaker'],
                'play': row['play'],
                'gender': row['gender'],
                'speech_chunk': chunk,
                'unique_id': f"{speaker_id_base}_{i + 1}"
            })

    df_chunked = pd.DataFrame(chunked_data)
    return df_chunked[df_chunked['speech_chunk'].apply(lambda x: len(x.split()) >= chunk_size)].reset_index(drop=True)

def get_embeddings_flexible(model_name, text_chunks):
    try:
        model = SentenceTransformer(model_name, trust_remote_code=True)
        print(f"✅ Using SentenceTransformer for {model_name}")
        return model.encode(
            text_chunks,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
    except Exception as e:
        print(f"⚠️ Falling back to AutoModel for {model_name} due to: {e}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        embeddings = []

        for chunk in text_chunks:
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(emb)

        return np.array(embeddings)

def get_embeddings_advanced(model_name, text_chunks, pooling="cls"):
    try:
        model = SentenceTransformer(model_name)
        return np.array([model.encode(chunk) for chunk in text_chunks])
    except Exception as e:
        print(f"⚠️ SentenceTransformer failed for {model_name}: {e}. Falling back to AutoModel.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    embeddings = []
    for chunk in text_chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        if pooling == "cls":
            emb = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        elif pooling == "mean":
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        else:
            raise ValueError("Pooling strategy must be 'cls' or 'mean'.")
        embeddings.append(emb)

    return np.array(embeddings)

def evaluate_clustering(embeddings, true_labels, n_clusters):
    # create sample of size 650
    # alternative: vary sample size

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    predicted_labels = kmeans.fit_predict(embeddings)
    return v_measure_score(true_labels, predicted_labels)

def evaluate_clustering_stratified(embeddings, true_labels, sample_size=650, n_iterations=10):
    """
    Perform stratified sampling by speaker and run clustering multiple times.

    :param embeddings: NumPy array of embeddings
    :param true_labels: List or array of speaker labels
    :param sample_size: Number of samples to draw per iteration
    :param n_iterations: Number of repetitions
    :return: mean V-measure, std, list of all scores
    """
    v_scores = []
    embeddings = np.array(embeddings)
    true_labels = np.array(true_labels)

    df = pd.DataFrame({
        'index': range(len(true_labels)),
        'label': true_labels
    })

    for i in range(n_iterations):
        # Sample stratified subset of indices
        stratified_sample = (
            df.groupby('label', group_keys=False)
              .apply(lambda x: x.sample( 
                  max(1, int(np.floor(sample_size * len(x) / len(df)))),  # proportional allocation
                  random_state=i,
                  replace=False
              ), include_groups=False)
        )

        indices = stratified_sample['index'].values
        sampled_embeddings = embeddings[indices]
        sampled_labels = true_labels[indices]
        sampled_n_clusters = len(set(sampled_labels))

        kmeans = KMeans(n_clusters=sampled_n_clusters, random_state=i, n_init=10)
        predicted = kmeans.fit_predict(sampled_embeddings)
        v = v_measure_score(sampled_labels, predicted)
        v_scores.append(v)

    return np.mean(v_scores), np.std(v_scores), v_scores

def get_language_from_filename(filename):
    match = re.match(r"speech_gender_(\w+)\.ndjson", filename)
    return match.group(1) if match else None

def process_file(filepath, language, model_names, output_file):
    print(f"\n🚀 Processing file: {filepath} ({language})")

    with open(filepath) as fin:
        data = ndjson.load(fin)
    df = pd.DataFrame(data)

    required_cols = {'speaker', 'play', 'gender', 'speech'}
    assert required_cols.issubset(df.columns), f"Missing required columns in {filepath}"

    df['speech_length'] = df['speech'].apply(count_tokens)
    df_sorted = df.sort_values(by='speech_length', ascending=False).reset_index(drop=True)

    with open(output_file, "a", encoding="utf-8") as f:
        for n_speakers, sample_size in zip([10, 20, 30, 40], [150, 300, 450, 600]):
            print(f"\n🎭 Top {n_speakers} speakers, sample size = {sample_size}")
            df_top = df_sorted.iloc[:n_speakers]

            df_filtered = process_dataset(df_top, CHUNK_SIZE)
            text_chunks = df_filtered['speech_chunk'].tolist()
            true_labels = df_filtered['speaker'].tolist()
            num_chunks = len(text_chunks)

            for model_name in model_names:
                print(f"🧠 Evaluating: {model_name}")
                embeddings = get_embeddings_flexible(model_name, text_chunks)

                actual_sample_size = min(sample_size, len(df_filtered))

                mean_v, std_v, scores = evaluate_clustering_stratified(
                    embeddings,
                    true_labels,
                    sample_size=actual_sample_size,
                    n_iterations=20
                )

                f.write(f"{language}\t{n_speakers}\t{actual_sample_size}\t{model_name}\t{mean_v:.4f} ± {std_v:.4f}\t{num_chunks}\n")

    print(f"✅ Results written to: {output_file}")

# ------------------------ MAIN ------------------------

def main():
    os.makedirs("results", exist_ok=True)
    output_file = f"results/v_measure_results_{TODAY}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Language\tSpeakers\tSample Size\tModel Name\tV-Measure Score\tNum Chunks\n")

    for filename in sorted(os.listdir('data')):
        if filename.startswith("speech_gender_") and filename.endswith(".ndjson"):
            language = get_language_from_filename(filename)
            if not language:
                continue

            if language not in MODEL_SETS:
                print(f"⚠️ Skipping {filename}: No models defined for language '{language}'")
                continue

            filepath = os.path.join('data', filename)
            model_names = MODEL_SETS[language]
            process_file(filepath, language, model_names, output_file)

if __name__ == "__main__":
    main()