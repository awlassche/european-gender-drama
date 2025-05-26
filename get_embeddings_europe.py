import os
import glob
import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
import ndjson
import einops

def chunk_speech(speech, chunk_size):
    """
    Splits a speech into chunks of a specified word length.

    :param speech: The full speech text.
    :param chunk_size: The maximum number of words per chunk.
    :return: A list of speech chunks.
    """
    words = speech.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def process_dataset(df, chunk_size):
    """
    Chunks speeches into smaller parts, filters out short chunks, and returns a DataFrame.

    :param df: The original DataFrame containing speeches.
    :param chunk_size: The chunk size to apply.
    :return: A filtered DataFrame with chunked speech.
    """

    chunked_data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking grouped speeches"):
        chunks = chunk_speech(row['speech'], chunk_size)
        speaker_id_base = row['speaker'][:4]

        for i, chunk in enumerate(chunks):
            if len(chunk.split()) > 5:  # Optional: filter out very short chunks
                chunked_data.append({
                    'speaker': row['speaker'],
                    'play': row['play'],
                    'gender': row['gender'],
                    'speech_chunk': chunk,
                    'unique_id': f"{speaker_id_base}_{i + 1}"
                })

    return pd.DataFrame(chunked_data)

def embed_and_save(input_path, output_path, chunk_size=400, model_name="jinaai/jina-embeddings-v3"):
    # 1. Load raw data
    with open(input_path) as fin:
        data = ndjson.load(fin)
    df = pd.DataFrame(data)

    # 2. Chunk speeches
    df_chunked = process_dataset(df, chunk_size)

    # 3. Load embedding model
    model = SentenceTransformer(model_name, trust_remote_code=True)

    # 4. Encode with batching
    print("Generating embeddings...")
    embeddings = model.encode(
        df_chunked['speech_chunk'].tolist(),
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )

    # 5. Attach and save
    df_chunked['embedding'] = embeddings.tolist()
    dataset = Dataset.from_pandas(df_chunked)
    dataset.save_to_disk(output_path)
    print(f"✅ Saved dataset with embeddings to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with .ndjson files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for saved datasets")
    parser.add_argument("--chunk_size", type=int, default=400, help="Max number of words per chunk")
    parser.add_argument("--model", type=str, default="jinaai/jina-embeddings-v3", help="SentenceTransformer model")
    args = parser.parse_args()

    input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.ndjson")))
    os.makedirs(args.output_dir, exist_ok=True)

    for input_path in input_files:
        language = os.path.splitext(os.path.basename(input_path))[0].split("_")[-1]
        output_path = os.path.join(args.output_dir, f"speech_{language}_embeddings")

        print(f"\n🚀 Processing: {input_path} → {output_path}")
        embed_and_save(input_path, output_path, args.chunk_size, args.model)
