import os
import re
import pandas as pd
import numpy as np
import ndjson
import torch
import einops

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score

# ----------------------- CONFIG -----------------------

# Define language-specific model sets
MODEL_SETS = {
    'dutch': [
        "emanjavacas/GysBERT-v2",
        "DTAI-KULeuven/robbert-2023-dutch-large",
        "GroNLP/bert-base-dutch-cased",
        "xlm-roberta-large",
        "intfloat/multilingual-e5-large",
        "google-t5/t5-large",
        #"neulab/Pangea-7B",
        "jinaai/jina-embeddings-v3"
    ],
    'ger': [
        "xlm-roberta-large",
        "intfloat/multilingual-e5-large",
        "google-t5/t5-large",
        #"neulab/Pangea-7B",
        "jinaai/jina-embeddings-v3"
    ],
    'eng': [
        "xlm-roberta-large",
        "intfloat/multilingual-e5-large",
        "google-t5/t5-large",
        #"neulab/Pangea-7B",
        "jinaai/jina-embeddings-v3"
    ],
    'fre': [
        "xlm-roberta-large",
        "intfloat/multilingual-e5-large",
        "google-t5/t5-large",
        #"neulab/Pangea-7B",
        "jinaai/jina-embeddings-v3"
    ],
    'ita': [
        "xlm-roberta-large",
        "intfloat/multilingual-e5-large",
        "google-t5/t5-large",
       # "neulab/Pangea-7B",
        "jinaai/jina-embeddings-v3"
    ],
    'cal': [
        "xlm-roberta-large",
        "intfloat/multilingual-e5-large",
        "google-t5/t5-large",
        #"neulab/Pangea-7B",
        "jinaai/jina-embeddings-v3"
    ]
}

CHUNK_SIZES = [200, 300, 400]
TODAY = '250520'

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
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    predicted_labels = kmeans.fit_predict(embeddings)
    return v_measure_score(true_labels, predicted_labels)

def get_language_from_filename(filename):
    match = re.match(r"speech_gender_(\w+)\.ndjson", filename)
    return match.group(1) if match else None

def process_file(filepath, language, model_names):
    print(f"\n🚀 Processing file: {filepath} ({language})")

    with open(filepath) as fin:
        data = ndjson.load(fin)
    df = pd.DataFrame(data)

    df_grouped = df.groupby(['speaker', 'play']).agg({
        'gender': 'first',
        'speech': ' '.join
    }).reset_index()

    df_grouped['speech_length'] = df_grouped['speech'].apply(count_tokens)
    df_sorted = df_grouped.sort_values(by='speech_length', ascending=False).reset_index(drop=True)
    df_top = df_sorted.iloc[:40]

    os.makedirs("results", exist_ok=True)
    output_file = f"results/v_measure_results_{language}_{TODAY}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Chunk Size\tModel Name\tV-Measure Score\tNumber of Rows\n")

        for chunk_size in CHUNK_SIZES:
            print(f"\n🔍 Chunk size: {chunk_size}")
            df_filtered = process_dataset(df_top, chunk_size)
            text_chunks = df_filtered['speech_chunk'].tolist()
            true_labels = df_filtered['speaker'].tolist()
            n_clusters = len(set(true_labels))
            num_rows = df_filtered.shape[0]

            for model_name in model_names:
                print(f"🧠 Evaluating: {model_name}")
                embeddings = get_embeddings_flexible(model_name, text_chunks)
                v_score = evaluate_clustering(embeddings, true_labels, n_clusters)
                f.write(f"{chunk_size}\t{model_name}\t{v_score:.4f}\t{num_rows}\n")

    print(f"✅ Results saved: {output_file}")

# ------------------------ MAIN ------------------------

def main():
    for filename in os.listdir('data'):
        if filename.startswith("speech_gender_") and filename.endswith(".ndjson"):
            language = get_language_from_filename(filename)
            if not language:
                continue

            if language not in MODEL_SETS:
                print(f"⚠️ Skipping {filename}: No models defined for language '{language}'")
                continue

            filepath = os.path.join('data', filename)
            model_names = MODEL_SETS[language]
            process_file(filepath, language, model_names)

if __name__ == "__main__":
    main()
