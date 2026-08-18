"""
Additive indexing pass: does NOT delete or modify existing passage-level
entries. Reads the existing corpus (data/msmarco-xi/chunks.jsonl), finds
passages longer than LENGTH_THRESHOLD_TOKENS, generates fixed-overlap and
semantic-boundary sub-chunks for those (see src/chunking/), embeds them, and:

  - Chroma (dense): upserts the new sub-chunks into the SAME collection
    (hhgoa_rag_full) as new ids - safe and idempotent, never touches the
    743,739 existing entries.
  - bm25s (sparse): bm25s has no incremental-add API (confirmed against
    src/bm25s_service.py - it's build-once, load-from-disk after that), so
    this writes a FULL rebuilt index (original passages + new sub-chunks)
    to a NEW directory, deliberately not overwriting the live
    bm25s_index_full. Swap it in manually (rename directories) once you've
    verified it - see the printed instructions at the end of a run.

Run this once, as a batch job, before the deadline - not on every server
start. Use --sample to test against a small slice first (see the
implementation doc's testing checklist) before committing to a full run.
"""

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, '.')

from src.embedding_service import EmbeddingService
from src.chroma_service import ChromaService
from src.bm25s_service import Bm25sService
from src.chunking.fixed_overlap import fixed_overlap_chunks
from src.chunking.semantic import semantic_chunks

CHUNKS_PATH = "data/msmarco-xi/chunks.jsonl"
LENGTH_THRESHOLD_TOKENS = 100  # only sub-chunk passages longer than this
EMBED_BATCH_SIZE = 64
CHROMA_COLLECTION = "hhgoa_rag_full"
NEW_BM25S_INDEX_DIR = "bm25s_index_full_with_chunks"  # new dir - see module docstring


def load_passages(sample: int = None) -> list:
    passages = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            passages.append(json.loads(line))
            if sample and len(passages) >= sample:
                break
    return passages


def generate_subchunks(passages: list, embedding_service: EmbeddingService) -> list:
    new_chunks = []
    long_passages = [p for p in passages if len(p["content"].split()) > LENGTH_THRESHOLD_TOKENS]
    print(f"{len(long_passages)}/{len(passages)} passages exceed "
          f"{LENGTH_THRESHOLD_TOKENS} tokens - only these get sub-chunked.", flush=True)

    for i, p in enumerate(long_passages):
        new_chunks.extend(fixed_overlap_chunks(p["content"], p["chunk_id"]))
        new_chunks.extend(semantic_chunks(
            p["content"], p["chunk_id"],
            embed_fn=embedding_service.embed_query,
        ))
        if (i + 1) % 200 == 0:
            print(f"  sub-chunked {i + 1}/{len(long_passages)} long passages "
                  f"({len(new_chunks)} sub-chunks so far)", flush=True)

    print(f"Generated {len(new_chunks)} additional sub-chunks from "
          f"{len(long_passages)} long passages.", flush=True)
    return new_chunks


def index_into_chroma(new_chunks: list, passages_by_id: dict,
                       embedding_service: EmbeddingService, chroma_service: ChromaService):
    if not new_chunks:
        return
    print(f"Embedding + upserting {len(new_chunks)} sub-chunks into Chroma "
          f"collection '{CHROMA_COLLECTION}'...", flush=True)
    t0 = time.time()
    for batch_start in range(0, len(new_chunks), EMBED_BATCH_SIZE):
        batch = new_chunks[batch_start:batch_start + EMBED_BATCH_SIZE]
        texts = [c["content"] for c in batch]
        embeddings = embedding_service.embed_documents(texts, batch_size=EMBED_BATCH_SIZE)

        items = []
        for chunk, emb in zip(batch, embeddings):
            parent_language = passages_by_id.get(chunk["parent_id"], {}).get("language", "en")
            items.append({
                "id": chunk["chunk_id"],
                "embedding": emb,
                "metadata": {
                    "content": chunk["content"][:1000],
                    "language": parent_language,  # inherited from parent passage
                    "source": "MSMARCO-XI",
                    "parent_id": chunk["parent_id"],
                    "chunking_strategy": chunk["chunking_strategy"],
                }
            })
        chroma_service.upsert(items)

        if (batch_start // EMBED_BATCH_SIZE) % 20 == 0:
            done = batch_start + len(batch)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  {done}/{len(new_chunks)} sub-chunks indexed ({rate:.1f}/s)", flush=True)

    print(f"Chroma indexing of sub-chunks complete ({(time.time() - t0) / 60:.1f} min).", flush=True)


def build_combined_bm25s_index(passages: list, new_chunks: list):
    # Bm25sService silently LOADS an existing index_dir instead of rebuilding
    # if one is already there (see src/bm25s_service.py's constructor) - bit
    # us once already (a stale dir from an earlier --sample run got silently
    # reused instead of rebuilt, "0.0 min" and no error). Always start clean.
    if os.path.exists(NEW_BM25S_INDEX_DIR):
        print(f"Removing existing '{NEW_BM25S_INDEX_DIR}' before rebuilding "
              f"(Bm25sService would otherwise silently reuse it as-is).", flush=True)
        shutil.rmtree(NEW_BM25S_INDEX_DIR)

    print(f"Building combined bm25s index ({len(passages)} original + "
          f"{len(new_chunks)} sub-chunks) at '{NEW_BM25S_INDEX_DIR}'...", flush=True)

    combined = [
        {
            "doc_id": p["chunk_id"],
            "content": p["content"],
            "language": p["language"],
            "metadata": {"section": p.get("source", "")},
        }
        for p in passages
    ]
    combined += [
        {
            "doc_id": c["chunk_id"],
            "content": c["content"],
            "language": None,  # backfilled below from parent
            "metadata": {"parent_id": c["parent_id"], "chunking_strategy": c["chunking_strategy"]},
        }
        for c in new_chunks
    ]
    passages_by_id = {p["chunk_id"]: p for p in passages}
    for c in combined:
        if c["language"] is None:
            parent_id = c["metadata"].get("parent_id")
            c["language"] = passages_by_id.get(parent_id, {}).get("language", "en")

    t0 = time.time()
    Bm25sService(chunks=combined, index_dir=NEW_BM25S_INDEX_DIR)
    print(f"bm25s combined index complete ({(time.time() - t0) / 60:.1f} min).", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                         help="Only process the first N passages (small-sample test run)")
    parser.add_argument("--skip-bm25s", action="store_true",
                         help="Skip the (slower) combined bm25s rebuild - Chroma-only dry run")
    args = parser.parse_args()

    passages = load_passages(sample=args.sample)
    print(f"Loaded {len(passages)} passages"
          f"{f' (sample of {args.sample})' if args.sample else ''}.", flush=True)
    passages_by_id = {p["chunk_id"]: p for p in passages}

    print("Loading embedding model...", flush=True)
    embedding_service = EmbeddingService()

    new_chunks = generate_subchunks(passages, embedding_service)

    print("Opening Chroma collection...", flush=True)
    chroma_service = ChromaService(collection_name=CHROMA_COLLECTION)
    index_into_chroma(new_chunks, passages_by_id, embedding_service, chroma_service)

    if not args.skip_bm25s:
        build_combined_bm25s_index(passages, new_chunks)
        print(f"\nDone. To go live: back up bm25s_index_full, then replace it "
              f"with the contents of {NEW_BM25S_INDEX_DIR} "
              f"(main_app.py loads bm25s_index_full by name).", flush=True)
    else:
        print("\n--skip-bm25s set: Chroma sub-chunks are live immediately "
              "(upsert is additive), but bm25s won't retrieve them until "
              "you re-run without --skip-bm25s and swap the index directory in.", flush=True)


if __name__ == "__main__":
    main()
