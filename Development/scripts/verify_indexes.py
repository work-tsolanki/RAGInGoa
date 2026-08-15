import sys
sys.path.insert(0, '.')

from src.embedding_service import EmbeddingService
from src.chroma_service import ChromaService
from src.bm25s_service import Bm25sService

embedding_service = EmbeddingService()
chroma_service = ChromaService(collection_name="hhgoa_rag_full")
bm25_service = Bm25sService(index_dir="bm25s_index_full")

test_queries = [
    "What is a corporation?",
    "आधार क्या है?",
    "How to apply for a passport",
    "ஆதார் என்றால் என்ன?",
]

for query in test_queries:
    print(f"\nQuery: {query}")

    emb = embedding_service.embed_query(query)
    dense_results = chroma_service.query(emb.tolist(), top_k=3)
    print(f"  Dense: {len(dense_results)} results")
    for r in dense_results[:2]:
        print(f"    - {r['doc_id']} [{r['metadata'].get('language')}] score={r['score']:.3f}: {r['content'][:80]}")

    bm25_results = bm25_service.query(query, top_k=3)
    print(f"  BM25: {len(bm25_results)} results")
    for r in bm25_results[:2]:
        print(f"    - {r['doc_id']} [{r.get('language')}] score={r['score']:.1f}: {r['content'][:80]}")

print("\nVerification complete.")
