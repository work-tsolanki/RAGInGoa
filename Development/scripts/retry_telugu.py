"""Retry the Telugu validation file that failed with a network error, appending to
the existing chunks.jsonl instead of rerunning the whole 14-file job."""

import json

from download_dataset import (
    BASE_URL, OUT_DIR, ROWS_PER_FILE, LANG_MAP,
    download_and_sample_rows, parse_passage_list,
)
import hashlib
import os

CHUNKS_PATH = os.path.join(OUT_DIR, "chunks.jsonl")


def main():
    seen_hashes = set()
    max_id = -1
    existing = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            existing.append(c)
            seen_hashes.add(hashlib.sha1(c["content"].encode("utf-8")).hexdigest())
            n = int(c["chunk_id"].split("_")[1])
            max_id = max(max_id, n)

    print(f"Loaded {len(existing)} existing chunks, max_id={max_id}")

    url = f"{BASE_URL}/validation/telval.parquet"
    lang2 = LANG_MAP["tel"]
    chunk_id = max_id + 1
    new_chunks = []

    rows = download_and_sample_rows(url, ROWS_PER_FILE)
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
            new_chunks.append({
                "chunk_id": f"chunk_{chunk_id}",
                "content": text,
                "language": lang,
                "source": "MSMARCO-XI",
                "query_id": query_id,
            })
            chunk_id += 1

    print(f"Telugu: +{len(new_chunks)} unique passages")

    all_chunks = existing + new_chunks
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    lang_counts = {}
    for c in all_chunks:
        lang_counts[c["language"]] = lang_counts.get(c["language"], 0) + 1

    summary = {
        "total_chunks": len(all_chunks),
        "rows_per_file_cap": ROWS_PER_FILE,
        "files_total": 14,
        "language_distribution": lang_counts,
    }
    with open(os.path.join(OUT_DIR, "chunk_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Total chunks now: {len(all_chunks)}")


if __name__ == "__main__":
    main()
