"""
Extract the spoken text of every character from the DraCor corpora: one
record per character per play, with the character's gender, written to one
NDJSON file per corpus (data/speech_gender_<corpus>.ndjson).

Replaces Part 1 of notebooks/Read_DraCor_corpus.ipynb, which keyed its
records on the character's xml:id alone. Outside English, DraCor ids are
plain names ("lisette") reused in every play, so all characters with the
same id in a corpus were merged into one record, filed under the first play
in which the id occurred. Here a record is keyed on (play, character id).

Other differences from the notebook:
  - Text is collected from everything inside <sp> except <speaker>, <stage>
    and <note>, including text that follows inline elements. The notebook
    took only the .text of <l>, <lb>, <s> and <p>, which drops the text after
    each <lb/> line break and doubles text where <s> sits inside <p> or <l>.
  - An <sp> without a who attribute, or with several speakers (who="#a #b"),
    is skipped and counted. The notebook gave a speech without who to the
    previous speaker, and dropped multi-speaker turns.
  - Each record also gets the character id and the play's normalised year
    (what add_year_to_ndjson.py added afterwards).

Selection is unchanged: plays with yearNormalized in [1500, 1800).

The TEI files are cached in --cache_dir, so a re-run (or a run after an
interruption) only downloads what is missing.

python extract_speech_texts.py                          # all six corpora
python extract_speech_texts.py --corpora fre cal
python extract_speech_texts.py --output_dir data/v2     # keep the old files
"""

import argparse
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from io import StringIO

import pandas as pd
import requests
from tqdm import tqdm

API = "https://dracor.org/api/v1/corpora"
CORPORA = ["fre", "ger", "eng", "dutch", "ita", "cal"]
YEAR_MIN, YEAR_MAX = 1500, 1800  # yearNormalized in [YEAR_MIN, YEAR_MAX)

TEI = "{http://www.tei-c.org/ns/1.0}"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
NOT_SPOKEN = {f"{TEI}speaker", f"{TEI}stage", f"{TEI}note"}


def fetch(session, url, retries=4):
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=120)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(5 * attempt)


def select_plays(session, corpus):
    """(name, year) of the corpus's plays with a normalised year in range."""
    metadata = pd.read_csv(StringIO(fetch(session, f"{API}/{corpus}/metadata/csv")))
    in_range = metadata["yearNormalized"].between(YEAR_MIN, YEAR_MAX, inclusive="left")
    selected = metadata[in_range].sort_values("name")
    return list(zip(selected["name"], selected["yearNormalized"].astype(int)))


def load_tei(session, corpus, play, cache_dir):
    path = os.path.join(cache_dir, corpus, f"{play}.xml")
    if not os.path.exists(path):
        text = fetch(session, f"{API}/{corpus}/plays/{play}/tei")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return ET.parse(path).getroot()


def characters(root):
    """{xml:id: (name, gender)} from the play's list of persons."""
    cast = {}
    for person in root.iter():
        if person.tag not in (f"{TEI}person", f"{TEI}personGrp"):
            continue
        name_el = person.find(f"{TEI}persName")
        if name_el is None:
            name_el = person.find(f"{TEI}name")
        name = " ".join("".join(name_el.itertext()).split()) if name_el is not None else None
        cast[person.get(XML_ID)] = (name, person.get("sex"))
    return cast


def spoken_text(element):
    """All text inside element, without speaker labels, stage directions and
    notes, but keeping the text that follows them."""
    parts = [element.text or ""]
    for child in element:
        if child.tag not in NOT_SPOKEN:
            parts.append(spoken_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def extract_play(root, play, year):
    """One record per character with speech, in order of first speech, plus
    the counts of skipped speech turns."""
    cast = characters(root)
    records, skipped = {}, {"no_who": 0, "several_speakers": 0, "unknown_id": 0}
    for sp in root.iter(f"{TEI}sp"):
        who = (sp.get("who") or "").split()
        if not who:
            skipped["no_who"] += 1
            continue
        if len(who) > 1:
            skipped["several_speakers"] += 1
            continue
        char_id = who[0].lstrip("#")
        if char_id not in cast:
            skipped["unknown_id"] += 1
            continue
        text = " ".join(spoken_text(sp).split())
        if not text:
            continue
        if char_id not in records:
            name, gender = cast[char_id]
            records[char_id] = {"speaker": name, "speaker_id": char_id, "gender": gender,
                                "speech": [], "play": play, "year": year}
        records[char_id]["speech"].append(text)
    for record in records.values():
        record["speech"] = " ".join(record["speech"])
    return list(records.values()), skipped


def extract_corpus(corpus, output_dir, cache_dir, workers):
    session = requests.Session()
    plays = select_plays(session, corpus)
    print(f"\n[{corpus}] {len(plays)} plays with a normalised year in [{YEAR_MIN}, {YEAR_MAX})")

    def one(play_year):
        play, year = play_year
        return extract_play(load_tei(session, corpus, play, cache_dir), play, year)

    with ThreadPoolExecutor(workers) as pool:
        results = list(tqdm(pool.map(one, plays), total=len(plays), desc=f"[{corpus}]"))

    rows, skipped = [], {}
    for play_rows, play_skipped in results:
        rows.extend(play_rows)
        for key, n in play_skipped.items():
            skipped[key] = skipped.get(key, 0) + n

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"speech_gender_{corpus}.ndjson")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    df = pd.DataFrame(rows)
    words = df["speech"].str.split().str.len()
    by_gender = words.groupby(df["gender"].fillna("None")).sum()
    print(f"[{corpus}] {len(df)} characters with speech in {df['play'].nunique()} plays -> {out_path}")
    print(f"[{corpus}] words by gender: {by_gender.to_dict()}")
    print(f"[{corpus}] speech turns skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser(description="Extract per-character speech from DraCor.")
    parser.add_argument("--corpora", nargs="+", default=CORPORA, choices=CORPORA)
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--cache_dir", default="data/tei",
                        help="Where downloaded TEI files are kept. Default: %(default)s")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel downloads. Keep this low; DraCor is a shared service.")
    args = parser.parse_args()

    for corpus in args.corpora:
        extract_corpus(corpus, args.output_dir, args.cache_dir, args.workers)


if __name__ == "__main__":
    main()
