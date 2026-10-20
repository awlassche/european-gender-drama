"""
Per-play and per-character statistics (counts, distribution over acts,
co-presence network) for the DraCor corpora, written in the same format as
the corpus_stats files: plays_df_<corpus>.csv and character_df_<corpus>.csv,
separated by ";".

Replaces Part 1 of notebooks/Character_analysis_drama_corpus_TEI.ipynb.
Same plays as extract_speech_texts.py: every play in that script's output
(--speech_texts_dir), i.e. yearNormalized in [1500, 1800) and at least one
character with spoken text. That leaves out the three French plays without
speech (a plot summary and two pantomimes) and the eleven English plays in
which no speech names its speaker.

What is fixed compared with the notebook:
  - Scenes. The notebook found no speech at all in plays divided into acts
    without scenes (it iterated over the act's children instead of treating
    the act as one scene), in plays whose only "act" was a prologue, and in
    plays with neither acts nor scenes (it then took only the last division,
    often the epilogue). Those plays got no character rows and shares of
    lines of 0: 239 English, 21 German and 8 Italian plays. Here a scene
    (segment) is the nearest division enclosing a speech, so a play counts
    the same whether it has acts, scenes, both or neither.
  - Speeches are collected from the whole <text> element, like
    extract_speech_texts.py; some plays have several <body> elements, the
    first of them empty.
  - Female and male edges count all of a character's neighbours. The
    notebook stored each edge in one direction only, so it counted roughly
    half of them.
  - Year: DraCor's yearNormalized, as in the speech texts and the
    experiments (the notebook took the print year when there was one).
  - Lines: per speech, the number of verse lines (<l>) plus prose
    paragraphs (<p>), counted only inside speeches. The notebook also
    counted <s> and <lb> (so a sentence inside a paragraph counted twice)
    and took the play total over the whole document. Shares of tokens are
    added as an alternative that does not depend on how lines are encoded.
  - Tokens and types are counted (they were 0 throughout in the notebook's
    output): word tokens of the spoken text, without speaker labels, stage
    directions and notes, lower-cased.
  - Speeches by several speakers at once (who="#a #b") are not credited to
    one invented character "a b"; they count for the co-presence network
    (every speaker listed was on stage) but not for the speech counts of
    any one character. Speeches without a speaker are left out.
  - Characters who never speak have no character row, as before, and are
    not network nodes (degree and density are over speaking characters),
    but they do count in the cast figures of plays_df (characters, male,
    female and their shares).

Columns of plays_df: play_id, corpus, author, title, genre (DraCor's
normalizedGenre, "unknown" if empty), year, characters, male, female,
%_male_chars, %_female_chars, %_female_lines, acts, scenes, speech_turns,
lines, mean_sp_length, density, %_male_lines, plus tokens,
%_female_tokens, %_male_tokens.

Columns of character_df: as in corpus_stats. acts and scenes are the
numbers in which the character speaks; act 1-act 9 are shares of the
character's lines per act (acts after the 9th are added to act 9),
prologue is the number of lines in a prologue; the (rel) columns are
relative to the play; degree (rel) is the number of characters the
character shares a scene with, divided by the number of other speaking
characters; female/male edges count those characters by gender;
betweenness is networkx's normalised betweenness centrality.

python extract_character_stats.py
python extract_character_stats.py --corpora eng ger --output_dir data/corpus_stats_test
"""

import argparse
import json
import os
import re
from collections import defaultdict
from io import StringIO

import networkx as nx
import pandas as pd
import requests
from tqdm import tqdm

from extract_speech_texts import API, CORPORA, NOT_SPOKEN, TEI, characters, fetch, load_tei, spoken_text

TOKEN_RE = re.compile(r"\w+")


def play_names(speech_texts_dir, corpus):
    """The plays of the speech-text extraction, in file order."""
    path = os.path.join(speech_texts_dir, f"speech_gender_{corpus}.ndjson")
    names = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            play = json.loads(line)["play"]
            if not names or names[-1] != play:
                names.append(play)
    return list(dict.fromkeys(names))


def author_of(root):
    """Authors as the notebook wrote them: 'Forename Surname ' joined by ' & '."""
    names = []
    title_stmt = root.find(f".//{TEI}titleStmt")  # not the source description, which repeats them
    for person in (title_stmt if title_stmt is not None else root).iter(f"{TEI}author"):
        fore = person.find(f".//{TEI}forename")
        sur = person.find(f".//{TEI}surname")
        name = ((fore.text + " ") if fore is not None and fore.text else "") + \
               ((sur.text + " ") if sur is not None and sur.text else "")
        names.append(name or " ".join("".join(person.itertext()).split()) + " ")
    return " & ".join(names)


def speech_lines(sp):
    """Verse lines plus prose paragraphs, outside stage directions and notes."""
    count = 0
    stack = list(sp)
    while stack:
        el = stack.pop()
        if el.tag in NOT_SPOKEN:
            continue
        if el.tag in (f"{TEI}l", f"{TEI}p"):
            count += 1
        stack.extend(el)
    return count


def analyse_play(root, corpus, meta):
    """(plays_df row, list of character_df rows) for one play."""
    # the whole <text>, not just the first <body>: some plays have several
    # bodies (the first one empty), and prologues can sit in <front>
    body = root.find(f"{TEI}text")
    if body is None:
        body = root
    parent = {child: el for el in body.iter() for child in el}
    cast = characters(root)

    # acts: act and prologue divisions in the body, in document order
    acts = [d for d in body.iter(f"{TEI}div") if d.get("type") in ("act", "prologue")]
    act_label = {}
    act_number = 0
    for d in acts:
        if d.get("type") == "prologue":
            act_label[d] = "prologue"
        else:
            act_number += 1
            act_label[d] = f"act {min(act_number, 9)}"

    def enclosing(el, test):
        while el in parent:
            el = parent[el]
            if test(el):
                return el
        return None

    stats = defaultdict(lambda: {"speech turns": 0, "lines": 0, "tokens": [], "acts": set(),
                                 "scenes": set(), "act_lines": defaultdict(int)})
    scene_speakers = defaultdict(set)   # segment -> speakers present (for the network)
    segments = []
    speech_turns = lines_total = 0

    for sp in body.iter(f"{TEI}sp"):
        who = [w.lstrip("#") for w in (sp.get("who") or "").split()]
        who = [w for w in who if w in cast]
        if not who:
            continue
        segment = enclosing(sp, lambda e: e.tag == f"{TEI}div") or body
        if segment not in scene_speakers:
            segments.append(segment)
        scene_speakers[segment].update(who)
        if len(who) > 1:
            continue  # several speakers at once: co-presence only

        char = who[0]
        n_lines = speech_lines(sp)
        act = enclosing(sp, lambda e: e in act_label)
        s = stats[char]
        s["speech turns"] += 1
        s["lines"] += n_lines
        s["tokens"].extend(t.lower() for t in TOKEN_RE.findall(spoken_text(sp)))
        s["scenes"].add(segment)
        if act is not None:
            s["acts"].add(act)
            s["act_lines"][act_label[act]] += n_lines
        speech_turns += 1
        lines_total += n_lines

    speakers = [c for c in cast if c in stats]
    graph = nx.Graph()
    graph.add_nodes_from(speakers)
    for present in scene_speakers.values():
        present = [p for p in present if p in stats]
        graph.add_edges_from((a, b) for i, a in enumerate(present) for b in present[i + 1:])
    betweenness = nx.betweenness_centrality(graph) if len(graph) > 2 else {c: 0.0 for c in graph}

    gender = {c: cast[c][1] for c in cast}
    tokens_total = sum(len(stats[c]["tokens"]) for c in speakers)

    def share(g, key):
        total = lines_total if key == "lines" else tokens_total
        part = sum((stats[c]["lines"] if key == "lines" else len(stats[c]["tokens"]))
                   for c in speakers if gender[c] == g)
        return round(part / total, 2) if total else 0

    n_cast = len(cast)
    n_male = sum(g == "MALE" for g in gender.values())
    n_female = sum(g == "FEMALE" for g in gender.values())
    play_id = root.get("{http://www.w3.org/XML/1998/namespace}id")
    play_row = {
        "play_id": play_id, "corpus": corpus, "author": author_of(root), "title": meta["title"],
        "genre": meta["genre"], "year": meta["year"], "characters": n_cast, "male": n_male,
        "female": n_female,
        "%_male_chars": round(n_male / n_cast, 2) if n_cast else 0,
        "%_female_chars": round(n_female / n_cast, 2) if n_cast else 0,
        "%_female_lines": share("FEMALE", "lines"),
        "acts": sum(1 for d in acts if d.get("type") == "act") + sum(1 for d in acts if d.get("type") == "prologue"),
        "scenes": len(segments), "speech_turns": speech_turns, "lines": lines_total,
        "mean_sp_length": round(lines_total / speech_turns, 2) if speech_turns else 0,
        "density": round(nx.density(graph), 2) if len(graph) > 1 else 0,
        "%_male_lines": share("MALE", "lines"), "tokens": tokens_total,
        "%_female_tokens": share("FEMALE", "tokens"), "%_male_tokens": share("MALE", "tokens"),
    }

    char_rows = []
    others = max(len(speakers) - 1, 1)
    for c in speakers:
        s = stats[c]
        toks = s["tokens"]
        neighbours = list(graph.neighbors(c))
        line_base = s["lines"] or 1
        row = {
            "play_id": play_id, "corpus": corpus, "character_id": c, "genre": meta["genre"],
            "name": cast[c][0], "gender": gender[c], "acts": len(s["acts"]), "scenes": len(s["scenes"]),
            "speech turns": s["speech turns"], "lines": s["lines"], "tokens": len(toks),
            "types": len(set(toks)), "tok-types": round(len(set(toks)) / len(toks), 2) if toks else 0,
            "prologue": s["act_lines"]["prologue"],
        }
        for n in range(1, 10):
            row[f"act {n}"] = round(s["act_lines"][f"act {n}"] / line_base, 2)
        row.update({
            "scenes (rel)": round(len(s["scenes"]) / len(segments), 2) if segments else 0,
            "speech turns (rel)": round(s["speech turns"] / speech_turns, 2) if speech_turns else 0,
            "lines (rel)": round(s["lines"] / lines_total, 2) if lines_total else 0,
            "degree (rel)": round(len(neighbours) / others, 2),
            "female edges": sum(gender[n] == "FEMALE" for n in neighbours),
            "male edges": sum(gender[n] == "MALE" for n in neighbours),
            "betweenness": round(betweenness.get(c, 0.0), 2),
        })
        char_rows.append(row)
    return play_row, char_rows


def extract_corpus(corpus, args):
    session = requests.Session()
    metadata = pd.read_csv(StringIO(fetch(session, f"{API}/{corpus}/metadata/csv"))).set_index("name")
    plays = play_names(args.speech_texts_dir, corpus)
    play_rows, char_rows = [], []
    for name in tqdm(plays, desc=f"[{corpus}]"):
        m = metadata.loc[name]
        genre = m.get("normalizedGenre") if "normalizedGenre" in m else None
        meta = {"title": m["title"], "year": int(m["yearNormalized"]),
                "genre": genre if isinstance(genre, str) and genre else "unknown"}
        play_row, rows = analyse_play(load_tei(session, corpus, name, args.cache_dir), corpus, meta)
        play_rows.append(play_row)
        char_rows.extend(rows)

    os.makedirs(args.output_dir, exist_ok=True)
    plays_df = pd.DataFrame(play_rows).set_index("play_id", drop=False)
    plays_df.index.name = None
    plays_df.to_csv(os.path.join(args.output_dir, f"plays_df_{corpus}.csv"), sep=";")
    pd.DataFrame(char_rows).to_csv(os.path.join(args.output_dir, f"character_df_{corpus}.csv"), sep=";")
    without_rows = (~plays_df["play_id"].isin({r["play_id"] for r in char_rows})).sum()
    print(f"[{corpus}] {len(plays_df)} plays, {len(char_rows)} speaking characters, "
          f"{without_rows} plays without character rows -> {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Per-play and per-character statistics from DraCor TEI.")
    parser.add_argument("--corpora", nargs="+", default=CORPORA, choices=CORPORA)
    parser.add_argument("--speech_texts_dir", default="data/speech_texts_2026-10-15",
                        help="Output of extract_speech_texts.py; defines the plays. Default: %(default)s")
    parser.add_argument("--cache_dir", default="data/tei", help="Cached TEI files. Default: %(default)s")
    parser.add_argument("--output_dir", default="data/corpus_stats_2026-10-15")
    args = parser.parse_args()
    for corpus in args.corpora:
        extract_corpus(corpus, args)


if __name__ == "__main__":
    main()
