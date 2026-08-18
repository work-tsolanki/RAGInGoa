"""
Incrementally expand data/msmarco-xi/chunks.jsonl using MORE of the same
already-relied-on dataset (ai4bharat/MSMARCO-XI validation split) - not a
new external source. download_dataset.py originally downloaded each full
validation parquet file but only kept the first OLD_ROWS_PER_FILE rows;
each file has ~98k rows, so almost all of what was already downloaded was
being discarded. This re-downloads the same files and takes the NEXT slice
of rows (OLD_ROWS_PER_FILE:NEW_ROWS_PER_FILE), appending only newly-seen
(deduplicated) passages to the existing chunks.jsonl with continuing
chunk_ids - so scripts/chunk_and_index.py's existing checkpoint (embedded_upto)
can resume and only embed the new tail, not redo the ~744k already-indexed
chunks.
"""

import ast
import hashlib
import json
import os
import time

import pandas as pd
import requests

BASE_URL = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main"

VAL_LANGS = ["asm", "ben", "guj", "hin", "kan", "mal", "mar", "nep", "ori", "pan", "san", "tam", "tel", "urd"]

LANG_MAP = {
    "asm": "as", "ben": "bn", "guj": "gu", "hin": "hi", "kan": "kn",
    "mal": "ml", "mar": "mr", "nep": "ne", "ori": "or", "pan": "pa",
    "san": "sa", "tam": "ta", "tel": "te", "urd": "ur",
}

OLD_ROWS_PER_FILE = 5000   # already consumed by the original download_dataset.py run
NEW_ROWS_PER_FILE = 7000   # new target - tune and re-run if size/coverage isn't right
OUT_DIR = "data/msmarco-xi"
CHUNKS_PATH = os.path.join(OUT_DIR, "chunks.jsonl")
SUMMARY_PATH = os.path.join(OUT_DIR, "chunk_summary.json")
TMP_PARQUET = os.path.join(OUT_DIR, "_tmp_download.parquet")


def parse_passage_list(raw):
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


def download_and_slice_rows(url, start, end):
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(TMP_PARQUET, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    df = pd.read_parquet(TMP_PARQUET, columns=["query_id", "passages"])
    sample = df.iloc[start:end]

    os.remove(TMP_PARQUET)

    return sample.to_dict("records")


def load_existing_state():
    seen_hashes = set()
    max_id = -1
    if not os.path.exists(CHUNKS_PATH):
        raise SystemExit(f"{CHUNKS_PATH} not found - expected the original corpus to already exist")
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            seen_hashes.add(hashlib.sha1(c["content"].encode("utf-8")).hexdigest())
            num = int(c["chunk_id"].split("_")[1])
            if num > max_id:
                max_id = num
    return seen_hashes, max_id


def main():
    print("Loading existing chunk hashes for dedup...", flush=True)
    seen_hashes, max_id = load_existing_state()
    next_id = max_id + 1
    print(f"Existing corpus: {len(seen_hashes)} unique passages, next chunk_id starts at chunk_{next_id}", flush=True)

    total_new = 0
    out_f = open(CHUNKS_PATH, "a", encoding="utf-8")

    try:
        for file_idx, lang3 in enumerate(VAL_LANGS):
            lang2 = LANG_MAP[lang3]
            url = f"{BASE_URL}/validation/{lang3}val.parquet"
            t0 = time.time()
            new_chunks = 0

            print(f"[{file_idx + 1}/{len(VAL_LANGS)}] Downloading {lang3}/validation, "
                  f"slicing rows[{OLD_ROWS_PER_FILE}:{NEW_ROWS_PER_FILE}]...", flush=True)

            try:
                rows = download_and_slice_rows(url, OLD_ROWS_PER_FILE, NEW_ROWS_PER_FILE)
            except Exception as e:
                print(f"  ! {lang3}/validation: failed ({e})", flush=True)
                if os.path.exists(TMP_PARQUET):
                    os.remove(TMP_PARQUET)
                continue

            for row in rows:
                query_id = row.get("query_id")
                passages = row.get("passages") or {}

                en_passages = parse_passage_list(passages.get("English_passages"))
                translated_passages = parse_passage_list(passages.get("Translated_passages"))

                for text, lang in [(t, "en") for t in en_passages] + [(t, lang2) for t in translated_passages]:
                    text = text.strip()
                    if len(text) < 20:
                        continue
                    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    rec = {
                        "chunk_id": f"chunk_{next_id}",
                        "content": text,
                        "language": lang,
                        "source": "MSMARCO-XI",
                        "query_id": query_id,
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    next_id += 1
                    new_chunks += 1
                    total_new += 1

            out_f.flush()
            elapsed = time.time() - t0
            print(f"  {lang3}/validation: +{new_chunks} new unique passages ({elapsed:.1f}s) "
                  f"| total new so far: {total_new}", flush=True)
    finally:
        out_f.close()

    final_total = len(seen_hashes)
    print(f"\nDone. Added {total_new} new chunks. Total unique passages now: {final_total}", flush=True)

    lang_counts = {}
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            lang_counts[c["language"]] = lang_counts.get(c["language"], 0) + 1

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "total_chunks": final_total,
            "old_rows_per_file": OLD_ROWS_PER_FILE,
            "new_rows_per_file": NEW_ROWS_PER_FILE,
            "files_total": len(VAL_LANGS),
            "language_distribution": lang_counts,
        }, f, ensure_ascii=False, indent=2)

    print("Next: run scripts/chunk_and_index.py (resumes from checkpoint, embeds only the new tail), "
          "then scripts/build_bm25s_index.py (full rebuild - bm25s has no incremental add).", flush=True)


if __name__ == "__main__":
    main()
