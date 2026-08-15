import os
import json
import bm25s
from config import DEBUG
from src.latency_tracker import track_latency
from src.query_utils import strip_query_stopwords


class Bm25sService:
    """BM25 search using bm25s (memory-mapped sparse-matrix scoring).

    Drop-in replacement for WhooshService: same query()/constructor shape,
    but scores as a sparse matmul instead of reopening disk segment files,
    which is what made Whoosh slow even after searcher caching.
    """

    def __init__(self, chunks: list = None, index_dir: str = "bm25s_index"):
        self.index_dir = index_dir

        if chunks and (not os.path.exists(index_dir) or len(os.listdir(index_dir)) == 0):
            self._create_index(chunks)
        elif os.path.exists(index_dir):
            self.retriever = bm25s.BM25.load(index_dir, load_corpus=True, mmap=True)
            if DEBUG:
                print(f"[Bm25sService] Opened existing index at {index_dir}")
        else:
            raise ValueError(f"Index directory {index_dir} not found and no chunks provided")

    def _create_index(self, chunks: list):
        if DEBUG:
            print(f"[Bm25sService] Creating index from {len(chunks)} chunks...")

        corpus = [
            {
                "doc_id": chunk["doc_id"],
                "content": chunk["content"],
                "language": chunk.get("language", "en"),
                "section": chunk.get("metadata", {}).get("section", ""),
            }
            for chunk in chunks
        ]

        tokens = bm25s.tokenize(
            [c["content"] for c in corpus],
            stopwords="english",
            show_progress=DEBUG,
        )

        self.retriever = bm25s.BM25(corpus=corpus)
        self.retriever.index(tokens, show_progress=DEBUG)

        os.makedirs(self.index_dir, exist_ok=True)
        self.retriever.save(self.index_dir, corpus=corpus)

        if DEBUG:
            print(f"Index created with {len(chunks)} documents")

    @track_latency("bm25s_search")
    def query(self, query_text: str, top_k: int = 10) -> list:
        """Search index with BM25."""
        if not query_text or len(query_text.strip()) == 0:
            return []

        results = []

        try:
            stripped = strip_query_stopwords(query_text)
            query_tokens = bm25s.tokenize(
                stripped, stopwords="english", show_progress=False, leave=False
            )

            k = min(top_k, len(self.retriever.corpus))
            if k == 0:
                return []

            docs, scores = self.retriever.retrieve(
                query_tokens, k=k, show_progress=False
            )

            for doc, score in zip(docs[0], scores[0]):
                results.append({
                    "doc_id": doc["doc_id"],
                    "content": doc["content"],
                    "score": float(score),
                    "language": doc.get("language", "en"),
                    "section": doc.get("section", ""),
                })

        except Exception as e:
            print(f"bm25s search failed: {e}")
            return []

        if DEBUG:
            print(f"[bm25s_query] Found {len(results)} results for: {query_text[:50]}")

        return results


if __name__ == "__main__":
    with open("data/test_chunks.json", encoding="utf-8") as f:
        test_chunks = json.load(f)

    service = Bm25sService(chunks=test_chunks, index_dir="bm25s_index_test")

    query = "What is Aadhaar?"
    results = service.query(query, top_k=5)

    print(f"\nQuery: {query}")
    print(f"Results: {len(results)}")
    for r in results:
        print(f"  - {r['doc_id']}: {r['content'][:60]}... (score: {r['score']:.2f})")
