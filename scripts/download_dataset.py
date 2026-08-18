"""
Download & sample MSMARCO-XI, then build a deduplicated multilingual passage
corpus from it.

The real dataset (ai4bharat/MSMARCO-XI) is a QA-translation dataset, not a
flat document corpus: each row has one query plus ~10 candidate passages in
English (`English_passages`) and ~10 in the target language
(`Translated_passages`), both packed as Python-list-repr strings. There is
no separate corpus/queries/qrels split as older docs assumed.

This script streams a bounded number of rows per language/split (to avoid
downloading full ~450MB+ parquet files we'd mostly discard), extracts all
passages, tags them by language, and deduplicates by exact text to build
data/msmarco-xi/chunks.jsonl - one chunk per unique passage.
"""

import ast
import hashlib
import json
import os
import sys
import time

import pandas as pd
import requests

BASE_URL = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main"

TRAIN_LANGS = ["asm", "ben", "guj", "hin", "kan", "mal", "mar", "nep", "ori", "pan", "san", "tam", "urd"]
VAL_LANGS = ["asm", "ben", "guj", "hin", "kan", "mal", "mar", "nep", "ori", "pan", "san", "tam", "tel", "urd"]

# Language code map: dataset uses 3-letter codes, our pipeline uses 2-letter (ISO 639-1-ish)
LANG_MAP = {
    "asm": "as", "ben": "bn", "guj": "gu", "hin": "hi", "kan": "kn",
    "mal": "ml", "mar": "mr", "nep": "ne", "ori": "or", "pan": "pa",
    "san": "sa", "tam": "ta", "tel": "te", "urd": "ur",
}

ROWS_PER_FILE = 5000  # cap per language file (validation-only, 14 files) to bound corpus size
OUT_DIR = "data/msmarco-xi"


def parse_passage_list(raw):
    """Field comes back from pandas/pyarrow as a numpy array/list of strings already;
    only fall back to string parsing if it was actually serialized as text."""
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            return list(ast.literal_eval(raw))
        except (ValueError, SyntaxError):
            cleaned = raw.strip().strip("[]")
            return [p.strip().strip("'\"") for p in cleaned.split("'") if p.strip()]
    try:
        return [str(x) for x in raw]
    except TypeError:
        return []


TMP_PARQUET = "data/msmarco-xi/_tmp_download.parquet"


def download_and_sample_rows(url, limit):
    """Download a parquet file fully to a temp path, sample rows, then delete it."""
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(TMP_PARQUET, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    df = pd.read_parquet(TMP_PARQUET, columns=["query_id", "passages"])
    sample = df.head(limit)

    os.remove(TMP_PARQUET)

    return sample.to_dict("records")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    seen_hashes = set()
    chunks = []
    chunk_id = 0

    # Train files run 3GB+ each (vs ~450-500MB for validation) with no way to partially
    # read them (single row-group parquet, HF datasets-server can't serve this dataset
    # either - nested struct columns aren't supported there). Validation-only keeps the
    # download bounded (~6-7GB total) while still covering all 14 languages.
    files = []
    for lang in VAL_LANGS:
        files.append((f"{BASE_URL}/validation/{lang}val.parquet", "validation", lang))

    print(f"Processing {len(files)} files, up to {ROWS_PER_FILE} rows each...", flush=True)

    for file_idx, (url, split_name, lang3) in enumerate(files):
        lang2 = LANG_MAP[lang3]
        t0 = time.time()
        row_count = 0
        new_chunks = 0

        print(f"[{file_idx + 1}/{len(files)}] Downloading {lang3}/{split_name}...", flush=True)

        try:
            rows = download_and_sample_rows(url, ROWS_PER_FILE)
            for row in rows:
                row_count += 1
                query_id = row.get("query_id")
                passages = row.get("passages") or {}

                en_passages = parse_passage_list(passages.get("English_passages"))
                translated_passages = parse_passage_list(passages.get("Translated_passages"))

                for text in en_passages:
                    text = text.strip()
                    if len(text) < 20:
                        continue
                    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    chunks.append({
                        "chunk_id": f"chunk_{chunk_id}",
                        "content": text,
                        "language": "en",
                        "source": "MSMARCO-XI",
                        "query_id": query_id,
                    })
                    chunk_id += 1
                    new_chunks += 1

                for text in translated_passages:
                    text = text.strip()
                    if len(text) < 20:
                        continue
                    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    chunks.append({
                        "chunk_id": f"chunk_{chunk_id}",
                        "content": text,
                        "language": lang2,
                        "source": "MSMARCO-XI",
                        "query_id": query_id,
                    })
                    chunk_id += 1
                    new_chunks += 1

        except Exception as e:
            print(f"  ! {lang3}/{split_name}: failed after {row_count} rows ({e})", flush=True)
            if os.path.exists(TMP_PARQUET):
                os.remove(TMP_PARQUET)
            continue

        elapsed = time.time() - t0
        print(f"  {lang3}/{split_name}: {row_count} rows -> +{new_chunks} unique passages "
              f"({elapsed:.1f}s) | total unique so far: {len(chunks)}", flush=True)

        # Checkpoint after every file so progress survives an interruption
        save_chunks(chunks, files)

    print(f"\nDone. Total unique passages: {len(chunks)}", flush=True)


def save_chunks(chunks, files):
    chunks_path = os.path.join(OUT_DIR, "chunks.jsonl")
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    lang_counts = {}
    for c in chunks:
        lang_counts[c["language"]] = lang_counts.get(c["language"], 0) + 1

    summary = {
        "total_chunks": len(chunks),
        "rows_per_file_cap": ROWS_PER_FILE,
        "files_total": len(files),
        "language_distribution": lang_counts,
    }
    with open(os.path.join(OUT_DIR, "chunk_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
