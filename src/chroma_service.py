import chromadb
from config import DEBUG
from src.latency_tracker import track_latency


class ChromaService:
    """Vector DB service using Chroma (local, persistent, no API key)."""

    def __init__(self, collection_name: str = "hhgoa_rag", persist_dir: str = "data/chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        # TODO (logged, not urgent): no explicit "hnsw:search_ef" override
        # here, so this runs on Chroma's low default ef_search. After the
        # Aug 2026 chunking pass grew hhgoa_rag_full from 743,739 to
        # 858,768 entries, a previously-reliable query ("What is Goa famous
        # for?") started returning irrelevant results from dense search -
        # none of the bad results were new sub-chunks, which points at
        # ef_search coverage becoming proportionally thinner as the corpus
        # grew, not a chunking-content problem. The fix to reach for when
        # this gets revisited is raising "hnsw:search_ef" on this
        # collection and benchmarking the recall improvement against the
        # added query latency (ef_search trades one for the other) - not
        # re-chunking or re-embedding anything.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        if DEBUG:
            print(f"[ChromaService] Initialized collection '{collection_name}' at {persist_dir}")

    @track_latency("chroma_upsert")
    def upsert(self, embeddings: list):
        """Index embeddings. Uses upsert semantics (safe to call again with same ids)."""
        ids = [item["id"] for item in embeddings]
        vectors = [item["embedding"] for item in embeddings]
        metadatas = [item.get("metadata", {}) for item in embeddings]

        self.collection.upsert(
            ids=ids,
            embeddings=vectors,
            metadatas=metadatas
        )

        if DEBUG:
            print(f"[chroma_upsert] Indexed {len(embeddings)} vectors")

    @track_latency("chroma_query")
    def query(self, query_embedding: list, top_k: int = 10) -> list:
        """Query vectors, returning results shaped like the old PineconeService.

        Skips the collection.count() pre-check that used to guard n_results:
        that call alone cost ~50ms per query (a full collection stat round
        trip), dwarfing the ~2ms HNSW search it was guarding. Chroma already
        handles n_results > actual size and empty collections by just
        returning fewer/no hits, so the check was pure overhead.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        output = []
        ids = results.get("ids") or [[]]
        if ids and ids[0]:
            distances = results["distances"][0]
            metadatas = results["metadatas"][0]
            for i, doc_id in enumerate(ids[0]):
                metadata = metadatas[i] or {}
                output.append({
                    "doc_id": doc_id,
                    "content": metadata.get("content", ""),
                    "score": 1.0 - distances[i],
                    "metadata": metadata
                })

        if DEBUG:
            print(f"[chroma_query] Retrieved {len(output)} results")

        return output


if __name__ == "__main__":
    import json
    from src.embedding_service import EmbeddingService

    with open("data/test_chunks.json", encoding="utf-8") as f:
        test_chunks = json.load(f)

    embedding_service = EmbeddingService()
    chroma_service = ChromaService(collection_name="hhgoa_rag_test")

    embeddings_to_upsert = []
    for chunk in test_chunks:
        emb = embedding_service.embed_query(chunk["content"])
        metadata = dict(chunk.get("metadata", {}))
        metadata["content"] = chunk["content"]
        embeddings_to_upsert.append({
            "id": chunk["doc_id"],
            "embedding": emb.tolist(),
            "metadata": metadata
        })

    chroma_service.upsert(embeddings_to_upsert)

    query = "What is Aadhaar?"
    query_emb = embedding_service.embed_query(query)
    results = chroma_service.query(query_emb.tolist(), top_k=3)

    print(f"\nQuery: {query}")
    print("Results:")
    for r in results:
        print(f"  - {r['doc_id']}: {r['score']:.3f}")
