"""
Embed all chunks in data/msmarco-xi/chunks.jsonl and index them into Chroma
(dense) at full scale.

Runs in batches with periodic progress logging and checkpointing so it can
be interrupted and resumed without redoing completed work.

The sparse (BM25) side is built separately by build_bm25s_index.py, which
reads directly from CHUNKS_PATH and doesn't need the embeddings this script
produces.
"""

import json
import os
import sys
import time

sys.path.insert(0, '.')

from src.embedding_service import EmbeddingService
from src.chroma_service import ChromaService

CHUNKS_PATH = "data/msmarco-xi/chunks.jsonl"
PROGRESS_PATH = "data/msmarco-xi/index_progress.json"
BATCH_SIZE = 64


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"embedded_upto": 0}


def save_progress(progress):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f)


def main():
    print("Loading chunks...", flush=True)
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks", flush=True)

    progress = load_progress()
    start_idx = progress["embedded_upto"]
    print(f"Resuming from index {start_idx}", flush=True)

    print("Loading embedding model...", flush=True)
    embedding_service = EmbeddingService()

    print("Initializing Chroma (persistent)...", flush=True)
    chroma_service = ChromaService(collection_name="hhgoa_rag_full")

    t0 = time.time()
    total = len(chunks)

    for batch_start in range(start_idx, total, BATCH_SIZE):
        batch = chunks[batch_start:batch_start + BATCH_SIZE]
        texts = [c["content"] for c in batch]

        embeddings = embedding_service.embed_documents(texts, batch_size=BATCH_SIZE)

        items = []
        for chunk, emb in zip(batch, embeddings):
            items.append({
                "id": chunk["chunk_id"],
                "embedding": emb,
                "metadata": {
                    "content": chunk["content"][:1000],
                    "language": chunk["language"],
                    "source": chunk.get("source", "MSMARCO-XI"),
                }
            })

        chroma_service.upsert(items)

        progress["embedded_upto"] = batch_start + len(batch)
        save_progress(progress)

        if (batch_start // BATCH_SIZE) % 20 == 0:
            elapsed = time.time() - t0
            done = batch_start + len(batch) - start_idx
            rate = done / elapsed if elapsed > 0 else 0
            remaining = total - (batch_start + len(batch))
            eta_min = (remaining / rate / 60) if rate > 0 else float("inf")
            print(f"  {batch_start + len(batch)}/{total} embedded "
                  f"({rate:.1f}/s, ETA {eta_min:.0f}min)", flush=True)

    print(f"Dense indexing complete: {total} chunks in Chroma "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    print("Run scripts/build_bm25s_index.py to (re)build the sparse index.", flush=True)


if __name__ == "__main__":
    main()
