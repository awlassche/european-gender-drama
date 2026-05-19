import ndjson
import pandas as pd
from urllib.request import urlopen

CORPORA = ["fre", "ger", "eng", "dutch", "ita", "cal"]
DATA_DIR = "data"

for corpus in CORPORA:
    url = f"https://dracor.org/api/v1/corpora/{corpus}/metadata/csv"
    print(f"Fetching metadata for '{corpus}'...")
    with urlopen(url) as f:
        metadata = pd.read_csv(f)

    year_lookup = {
        name: int(year) if pd.notna(year) else None
        for name, year in zip(metadata["name"], metadata["yearNormalized"])
    }

    ndjson_path = f"{DATA_DIR}/speech_gender_{corpus}.ndjson"
    with open(ndjson_path, encoding="utf-8") as f:
        data = ndjson.load(f)

    for record in data:
        record["year"] = year_lookup.get(record["play"])

    with open(ndjson_path, "w", encoding="utf-8") as f:
        ndjson.dump(data, f, ensure_ascii=False)

    print(f"  Done: {len(data)} records updated in {ndjson_path}")
