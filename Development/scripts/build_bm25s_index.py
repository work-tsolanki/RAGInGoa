"""
Build the full-scale bm25s index from data/msmarco-xi/chunks.jsonl.

Reuses the already-chunked corpus (same source Whoosh's whoosh_index_full
was built from) - no re-embedding needed, this only rebuilds the BM25 side.
"""

import json
import sys
import time

sys.path.insert(0, '.')

from src.bm25s_service import Bm25sService

CHUNKS_PATH = "data/msmarco-xi/chunks.jsonl"
BM25S_INDEX_DIR = "bm25s_index_full"


def main():
    print("Loading chunks...", flush=True)
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunks.append({
                "doc_id": c["chunk_id"],
                "content": c["content"],
                "language": c["language"],
                "metadata": {"section": c.get("source", "")},
            })
    print(f"Loaded {len(chunks)} chunks", flush=True)

    t0 = time.time()
    Bm25sService(chunks=chunks, index_dir=BM25S_INDEX_DIR)
    print(f"bm25s index complete ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
