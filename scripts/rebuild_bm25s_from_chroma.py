"""
One-off fix: scripts/add_chunking_strategies.py's full run correctly
upserted 115,029 sub-chunks into Chroma, but its bm25s rebuild step silently
reused a stale index directory left over from an earlier --sample test run
instead of rebuilding (Bm25sService loads-if-exists rather than erroring -
see that script's now-fixed build_combined_bm25s_index). Chroma is already
correct and is the cheaper source of truth here - re-embedding 115K
sentences for semantic-boundary chunking again would be wasteful. This
pulls the already-embedded sub-chunks straight back out of Chroma by
metadata filter instead.
"""

import json
import os
import shutil
import sys

sys.path.insert(0, '.')

from src.chroma_service import ChromaService
from src.bm25s_service import Bm25sService

CHUNKS_PATH = "data/msmarco-xi/chunks.jsonl"
CHROMA_COLLECTION = "hhgoa_rag_full"
BM25S_INDEX_DIR = "bm25s_index_full_with_chunks"
FETCH_BATCH_SIZE = 5000


def fetch_all_subchunks(chroma_service: ChromaService) -> list:
    collection = chroma_service.collection
    subchunk_filter = {"chunking_strategy": {"$in": ["fixed_overlap", "semantic_boundary"]}}

    total = collection.count()
    print(f"Chroma collection has {total} total entries. Fetching sub-chunks...", flush=True)

    all_chunks = []
    offset = 0
    while True:
        result = collection.get(
            where=subchunk_filter,
            limit=FETCH_BATCH_SIZE,
            offset=offset,
            include=["metadatas"],
        )
        ids = result["ids"]
        if not ids:
            break
        for doc_id, metadata in zip(ids, result["metadatas"]):
            all_chunks.append({
                "doc_id": doc_id,
                "content": metadata.get("content", ""),
                "language": metadata.get("language", "en"),
                "metadata": {
                    "parent_id": metadata.get("parent_id", doc_id),
                    "chunking_strategy": metadata.get("chunking_strategy"),
                },
            })
        offset += len(ids)
        print(f"  fetched {offset} sub-chunks so far...", flush=True)
        if len(ids) < FETCH_BATCH_SIZE:
            break

    return all_chunks


def load_base_passages() -> list:
    passages = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            passages.append({
                "doc_id": c["chunk_id"],
                "content": c["content"],
                "language": c["language"],
                "metadata": {"section": c.get("source", "")},
            })
    return passages


def main():
    chroma_service = ChromaService(collection_name=CHROMA_COLLECTION)
    subchunks = fetch_all_subchunks(chroma_service)
    print(f"Fetched {len(subchunks)} sub-chunks from Chroma.", flush=True)

    base_passages = load_base_passages()
    print(f"Loaded {len(base_passages)} base passages from {CHUNKS_PATH}.", flush=True)

    combined = base_passages + subchunks
    print(f"Combined corpus: {len(combined)} documents "
          f"({len(base_passages)} base + {len(subchunks)} sub-chunks).", flush=True)

    if os.path.exists(BM25S_INDEX_DIR):
        print(f"Removing stale '{BM25S_INDEX_DIR}'...", flush=True)
        shutil.rmtree(BM25S_INDEX_DIR)

    print(f"Building bm25s index at '{BM25S_INDEX_DIR}'...", flush=True)
    Bm25sService(chunks=combined, index_dir=BM25S_INDEX_DIR)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
