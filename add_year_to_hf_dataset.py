"""
Add a 'year' column to an existing Hugging Face dataset by joining on the
'play' slug from the local NDJSON files (which must already have 'year'
added via add_year_to_ndjson.py).

Usage:
    python add_year_to_hf_dataset.py --repo_id awlassche/european-gender-drama-bge-embeddings AND awlassche/european-gender-drama-jina-embeddings
"""

import glob
import argparse
import ndjson
from datasets import load_dataset


def build_year_lookup(data_dir: str) -> dict:
    lookup = {}
    for path in glob.glob(f"{data_dir}/speech_gender_*.ndjson"):
        with open(path, encoding="utf-8") as f:
            for record in ndjson.load(f):
                play = record.get("play")
                year = record.get("year")
                if play and year is not None:
                    lookup[play] = year
    return lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True,
                        help="HuggingFace dataset repo ID, e.g. 'username/dataset-name'")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory with .ndjson files (default: data/)")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    print("Building year lookup from NDJSON files...")
    year_lookup = build_year_lookup(args.data_dir)
    print(f"  {len(year_lookup)} plays found with year info")

    print(f"Loading dataset from Hub: {args.repo_id}")
    ds = load_dataset(args.repo_id, split="train")
    print(f"  {len(ds):,} rows, columns: {ds.column_names}")

    ds = ds.map(lambda row: {"year": year_lookup.get(row["play"])})
    print(f"  Added 'year' column. Sample: {ds[0]['play']} → {ds[0]['year']}")

    print(f"Pushing back to Hub: {args.repo_id}")
    ds.push_to_hub(args.repo_id, private=args.private)
    print("Done!")


if __name__ == "__main__":
    main()
