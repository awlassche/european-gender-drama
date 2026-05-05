"""
Merge per-language embedding datasets into one combined dataset and push to Hugging Face Hub.

Usage:
    python push_to_hub.py \
        --embeddings_dir embeddings \
        --repo_id your-username/european-gender-drama-embeddings \
        [--private]
"""

import os
import glob
import argparse
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets


def get_language_tag(folder_name: str) -> str:
    """Derive a language tag from the folder name, e.g. 'speech_dutch_embeddings' → 'dutch'."""
    parts = folder_name.split("_")
    # folder pattern: speech_<lang>_embeddings
    if len(parts) >= 3:
        return parts[1]
    return folder_name


def load_and_tag(folder_path: str, language: str) -> Dataset:
    """Load a dataset from disk and add a 'language' column."""
    ds = load_from_disk(folder_path)
    ds = ds.add_column("language", [language] * len(ds))
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="embeddings",
        help="Directory containing per-language embedding folders (default: embeddings/)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face Hub repo ID, e.g. 'username/dataset-name'",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Push as a private dataset (default: public)",
    )
    args = parser.parse_args()

    # Find all per-language dataset folders (contain dataset_info.json)
    pattern = os.path.join(args.embeddings_dir, "*", "dataset_info.json")
    info_files = sorted(glob.glob(pattern))

    if not info_files:
        raise FileNotFoundError(
            f"No dataset folders found under '{args.embeddings_dir}'. "
            "Make sure get_embeddings_europe.py has been run first."
        )

    datasets = []
    for info_file in info_files:
        folder_path = os.path.dirname(info_file)
        folder_name = os.path.basename(folder_path)
        language = get_language_tag(folder_name)
        print(f"Loading '{language}' from {folder_path} ...")
        ds = load_and_tag(folder_path, language)
        print(f"  {len(ds):,} rows")
        datasets.append(ds)

    print(f"\nConcatenating {len(datasets)} language datasets ...")
    combined = concatenate_datasets(datasets)
    print(f"Combined dataset: {len(combined):,} rows, columns: {combined.column_names}")

    print(f"\nPushing to Hub as '{args.repo_id}' (private={args.private}) ...")
    combined.push_to_hub(args.repo_id, private=args.private)
    print(f"Done! Dataset available at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
