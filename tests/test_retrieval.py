import json
import sys
import time

import pytest

sys.path.insert(0, '.')

from src.embedding_service import EmbeddingService
from src.bm25s_service import Bm25sService
from src.chroma_service import ChromaService
from src.retrieval import merge_and_rank, dedupe_by_parent


@pytest.fixture(scope="module")
def setup():
    """Setup services."""
    with open("data/test_chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)

    embedding_service = EmbeddingService()
    bm25_service = Bm25sService(chunks=chunks, index_dir="bm25s_index_pytest")
    chroma_service = ChromaService(collection_name="hhgoa_rag_pytest")

    embeddings_to_upsert = []
    for chunk in chunks:
        emb = embedding_service.embed_query(chunk["content"])
        metadata = dict(chunk.get("metadata", {}))
        metadata["content"] = chunk["content"]
        embeddings_to_upsert.append({
            "id": chunk["doc_id"],
            "embedding": emb.tolist(),
            "metadata": metadata
        })

    chroma_service.upsert(embeddings_to_upsert)

    return {
        "chunks": chunks,
        "embedding": embedding_service,
        "bm25s": bm25_service,
        "chroma": chroma_service
    }


def test_embedding(setup):
    """Test embedding service."""
    embedding_service = setup["embedding"]
    embedding = embedding_service.embed_query("Test query")
    assert len(embedding) == 384


def test_bm25s_retrieval(setup):
    """Test bm25s BM25 retrieval."""
    bm25_service = setup["bm25s"]
    results = bm25_service.query("Aadhaar", top_k=5)
    assert len(results) > 0
    assert "doc_id" in results[0]
    assert "score" in results[0]


def test_dense_retrieval(setup):
    """Test dense retrieval."""
    embedding_service = setup["embedding"]
    chroma_service = setup["chroma"]

    query_emb = embedding_service.embed_query("What is Aadhaar?")
    results = chroma_service.query(query_emb.tolist(), top_k=5)

    assert len(results) > 0
    assert "doc_id" in results[0]
    assert "score" in results[0]


def test_merge_results(setup):
    """Test result merging."""
    dense_results = [
        {"doc_id": "doc_001", "score": 0.95, "content": "Test", "metadata": {}},
    ]
    bm25_results = [
        {"doc_id": "doc_002", "score": 8.5, "content": "Test", "metadata": {}},
    ]

    merged = merge_and_rank(dense_results, bm25_results, top_k=2)
    assert len(merged) <= 2
    assert "final_score" in merged[0]


def test_dedupe_by_parent_collapses_whole_passage_and_its_own_subchunk():
    """A whole passage and one of its own sub-chunks (see src/chunking/)
    both appearing in the same candidate set must collapse to whichever
    scored higher - not occupy two of the top-k slots with near-duplicate
    content."""
    dense_results = [
        {"doc_id": "chunk_5", "score": 0.9, "content": "Whole passage.", "metadata": {}},
        {
            "doc_id": "chunk_5_fixed_0", "score": 0.95,
            "content": "Sub-chunk of chunk_5.",
            "metadata": {"parent_id": "chunk_5", "chunking_strategy": "fixed_overlap"},
        },
    ]
    merged = merge_and_rank(dense_results, [], top_k=5)

    assert len(merged) == 1
    assert merged[0]["doc_id"] == "chunk_5_fixed_0"  # the higher-scoring one survives


def test_dedupe_by_parent_no_op_when_no_shared_parents():
    """Regression guard: with no sub-chunks indexed (today's reality),
    every doc's parent_id defaults to its own doc_id, so nothing should
    ever collide - merge_and_rank must return all distinct docs unchanged."""
    dense_results = [
        {"doc_id": "chunk_1", "score": 0.9, "content": "A", "metadata": {}},
        {"doc_id": "chunk_2", "score": 0.8, "content": "B", "metadata": {}},
        {"doc_id": "chunk_3", "score": 0.7, "content": "C", "metadata": {}},
    ]
    merged = merge_and_rank(dense_results, [], top_k=5)
    assert {d["doc_id"] for d in merged} == {"chunk_1", "chunk_2", "chunk_3"}


def test_dedupe_by_parent_keeps_first_seen_order_for_ties():
    candidates = [
        {"doc_id": "a", "parent_id": "p1", "final_score": 0.5},
        {"doc_id": "b", "parent_id": "p2", "final_score": 0.9},
        {"doc_id": "c", "parent_id": "p1", "final_score": 0.5},  # tie with "a", same parent
    ]
    result = dedupe_by_parent(candidates)
    assert [d["doc_id"] for d in result] == ["a", "b"]


def test_merge_and_rank_exposes_chunking_strategy_from_bm25_only_match():
    """A doc that only matched on the sparse (bm25s) side must still carry
    its chunking_strategy through - not just dense/Chroma matches."""
    bm25_results = [
        {
            "doc_id": "chunk_9_semantic_0", "score": 5.0, "content": "Sub-chunk.",
            "language": "en", "parent_id": "chunk_9", "chunking_strategy": "semantic_boundary",
        },
    ]
    merged = merge_and_rank([], bm25_results, top_k=5)
    assert merged[0]["chunking_strategy"] == "semantic_boundary"
    assert merged[0]["parent_id"] == "chunk_9"


def test_full_pipeline(setup):
    """Test full retrieval pipeline."""
    embedding_service = setup["embedding"]
    bm25_service = setup["bm25s"]
    chroma_service = setup["chroma"]

    query = "How to apply for Aadhaar?"

    start = time.time()

    query_emb = embedding_service.embed_query(query)

    dense_results = chroma_service.query(query_emb.tolist(), top_k=10)
    bm25_results = bm25_service.query(query, top_k=10)

    merged = merge_and_rank(dense_results, bm25_results, top_k=5)

    elapsed = (time.time() - start) * 1000

    assert len(merged) > 0
    assert "final_score" in merged[0]

    print(f"\nFull pipeline latency: {elapsed:.1f}ms")
    print(f"Retrieved {len(merged)} documents")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
