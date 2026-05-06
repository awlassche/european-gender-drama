"""
hf auth login

python get_embeddings_europe.py \
  --input_dir data/ \
  --hub_dataset_id awlassche/european-gender-drama-bge-embeddings
"""

import os
import glob
import pandas as pd
from datasets import Dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
import ndjson
import torch


def chunk_speech(speech, chunk_size):
    words = speech.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def process_dataset(df, chunk_size, language):
    chunked_data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Chunking [{language}]"):
        chunks = chunk_speech(row['speech'], chunk_size)
        speaker_id_base = row['speaker'][:4]

        for i, chunk in enumerate(chunks):
            if len(chunk.split()) > 5:
                chunked_data.append({
                    'language': language,
                    'speaker': row['speaker'],
                    'play': row['play'],
                    'gender': row['gender'],
                    'speech_chunk': chunk,
                    'unique_id': f"{speaker_id_base}_{i + 1}"
                })

    return pd.DataFrame(chunked_data)


def embed_and_collect(input_path, language, model, chunk_size, batch_size):
    with open(input_path) as fin:
        data = ndjson.load(fin)
    df = pd.DataFrame(data)

    df_chunked = process_dataset(df, chunk_size, language)

    print(f"  Generating embeddings for {len(df_chunked)} chunks...")
    embeddings = model.encode(
        df_chunked['speech_chunk'].tolist(),
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,  # bge-m3 recommends normalized embeddings
    )

    df_chunked['embedding'] = embeddings.tolist()
    return Dataset.from_pandas(df_chunked, preserve_index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with .ndjson files")
    parser.add_argument("--hub_dataset_id", type=str, required=True, help="HuggingFace dataset ID, e.g. username/dataset-name")
    parser.add_argument("--chunk_size", type=int, default=400, help="Max number of words per chunk")
    parser.add_argument("--model", type=str, default="BAAI/bge-m3", help="SentenceTransformer model")
    parser.add_argument("--batch_size", type=int, default=64, help="Encoding batch size")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model, trust_remote_code=True, device=device)

    input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.ndjson")))
    if not input_files:
        raise FileNotFoundError(f"No .ndjson files found in {args.input_dir}")

    all_datasets = []
    for input_path in input_files:
        language = os.path.splitext(os.path.basename(input_path))[0].split("_")[-1]
        print(f"\nProcessing: {input_path} (language={language})")
        all_datasets.append(embed_and_collect(input_path, language, model, args.chunk_size, args.batch_size))

    print("\nConcatenating all language datasets...")
    combined = concatenate_datasets(all_datasets)

    print(f"Pushing {len(combined)} rows to HuggingFace Hub: {args.hub_dataset_id}")
    combined.push_to_hub(args.hub_dataset_id)
    print(f"Done. Dataset available at: https://huggingface.co/datasets/{args.hub_dataset_id}")


if __name__ == "__main__":
    main()
