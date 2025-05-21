import os
import ndjson
import pandas as pd

# --- Your existing functions ---

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

# --- New logic for chunk count inspection ---

def inspect_chunk_counts(data_dir='data', chunk_sizes=[200, 300, 400]):
    for filename in os.listdir(data_dir):
        if filename.startswith("speech_gender_") and filename.endswith(".ndjson"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath) as f:
                data = ndjson.load(f)
            df = pd.DataFrame(data)

            # Calculate speech length
            df['speech_length'] = df['speech'].apply(count_tokens)

            # Sort and take top 40 speakers
            df_sorted = df.sort_values(by='speech_length', ascending=False).reset_index(drop=True)
            df_top = df_sorted.iloc[:40]

            print(f"\n📄 File: {filename}")
            for chunk_size in chunk_sizes:
                df_chunked = process_dataset(df_top, chunk_size)
                print(f"  Chunk size {chunk_size}: {len(df_chunked)} chunks")

# --- Run it ---

if __name__ == "__main__":
    inspect_chunk_counts()
