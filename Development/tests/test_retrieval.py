import json
import sys
import time

import pytest

sys.path.insert(0, '.')

from src.embedding_service import EmbeddingService
from src.whoosh_service import WhooshService
from src.chroma_service import ChromaService
from src.retrieval import merge_and_rank


@pytest.fixture(scope="module")
def setup():
    """Setup services."""
    with open("data/test_chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)

    embedding_service = EmbeddingService()
    whoosh_service = WhooshService(chunks=chunks)
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
        "whoosh": whoosh_service,
        "chroma": chroma_service
    }


def test_embedding(setup):
    """Test embedding service."""
    embedding_service = setup["embedding"]
    embedding = embedding_service.embed_query("Test query")
    assert len(embedding) == 384


def test_whoosh_retrieval(setup):
    """Test Whoosh BM25 retrieval."""
    whoosh_service = setup["whoosh"]
    results = whoosh_service.query("Aadhaar", top_k=5)
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


def test_full_pipeline(setup):
    """Test full retrieval pipeline."""
    embedding_service = setup["embedding"]
    whoosh_service = setup["whoosh"]
    chroma_service = setup["chroma"]

    query = "How to apply for Aadhaar?"

    start = time.time()

    query_emb = embedding_service.embed_query(query)

    dense_results = chroma_service.query(query_emb.tolist(), top_k=10)
    bm25_results = whoosh_service.query(query, top_k=10)

    merged = merge_and_rank(dense_results, bm25_results, top_k=5)

    elapsed = (time.time() - start) * 1000

    assert len(merged) > 0
    assert "final_score" in merged[0]

    print(f"\nFull pipeline latency: {elapsed:.1f}ms")
    print(f"Retrieved {len(merged)} documents")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
