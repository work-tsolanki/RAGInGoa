import numpy as np
from typing import List, Dict
from config import DEBUG, PINECONE_API_KEY
from src.latency_tracker import track_latency

class PineconeService:
    """Vector database service (Pinecone or mock)."""

    def __init__(self, use_mock: bool = False):
        """Initialize Pinecone or use mock."""
        self.use_mock = use_mock or PINECONE_API_KEY == "mock"

        if self.use_mock:
            self.mock_index = {}
            if DEBUG:
                print("[PineconeService] Using MOCK mode (in-memory)")
        else:
            self._init_pinecone()

    def _init_pinecone(self):
        """Initialize real Pinecone connection."""
        try:
            import pinecone
            pinecone.init(
                api_key=PINECONE_API_KEY,
                environment="us-west-2-aws"
            )
            self.ix = pinecone.Index("hhgoa-rag")
            if DEBUG:
                print("[PineconeService] Connected to Pinecone")
        except Exception as e:
            print(f"Pinecone init failed: {e}. Falling back to mock.")
            self.use_mock = True
            self.mock_index = {}

    @track_latency("pinecone_upsert")
    def upsert(self, embeddings: List[Dict]):
        """Index embeddings."""
        if self.use_mock:
            for item in embeddings:
                self.mock_index[item["id"]] = item
            if DEBUG:
                print(f"[pinecone_upsert] Stored {len(embeddings)} vectors (mock)")
        else:
            self.ix.upsert(vectors=[
                (item["id"], item["embedding"], item.get("metadata", {}))
                for item in embeddings
            ])
            if DEBUG:
                print(f"[pinecone_upsert] Upserted {len(embeddings)} vectors")

    @track_latency("pinecone_query")
    def query(self, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
        """Query vector database."""
        if self.use_mock:
            results = []
            query_vec = np.array(query_embedding)

            for doc_id, item in self.mock_index.items():
                doc_vec = np.array(item["embedding"])
                similarity = np.dot(query_vec, doc_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(doc_vec) + 1e-10
                )
                results.append({
                    "doc_id": doc_id,
                    "content": item.get("metadata", {}).get("content", ""),
                    "score": float(similarity),
                    "metadata": item.get("metadata", {})
                })

            results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

            if DEBUG:
                print(f"[pinecone_query] Retrieved {len(results)} results (mock)")

            return results
        else:
            results = self.ix.query(query_embedding, top_k=top_k, include_metadata=True)
            return [
                {
                    "doc_id": match["id"],
                    "score": match["score"],
                    "metadata": match.get("metadata", {})
                }
                for match in results["matches"]
            ]


if __name__ == "__main__":
    import json
    from src.embedding_service import EmbeddingService

    with open("data/test_chunks.json", encoding="utf-8") as f:
        test_chunks = json.load(f)

    embedding_service = EmbeddingService()
    pinecone_service = PineconeService(use_mock=True)

    embeddings_to_upsert = []
    for chunk in test_chunks:
        emb = embedding_service.embed_query(chunk["content"])
        embeddings_to_upsert.append({
            "id": chunk["doc_id"],
            "embedding": emb.tolist(),
            "metadata": chunk.get("metadata", {})
        })

    pinecone_service.upsert(embeddings_to_upsert)

    query = "What is Aadhaar?"
    query_emb = embedding_service.embed_query(query)
    results = pinecone_service.query(query_emb.tolist(), top_k=3)

    print(f"\nQuery: {query}")
    print(f"Results:")
    for r in results:
        print(f"  - {r['doc_id']}: {r['score']:.3f}")
