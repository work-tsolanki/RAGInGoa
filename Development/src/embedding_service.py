import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, DEBUG
from src.latency_tracker import track_latency

class EmbeddingService:
    """AI4Bharat IndicBERT embedding service."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Load embedding model."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if DEBUG:
            print(f"[EmbeddingService] Loading {model_name} on {device}...")

        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.dimension = self.model.get_sentence_embedding_dimension()

            if DEBUG:
                print(f"Model loaded. Dimension: {self.dimension}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

    @track_latency("embedding")
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        if not query or len(query.strip()) == 0:
            raise ValueError("Query cannot be empty")

        embedding = self.model.encode(query, convert_to_tensor=False)

        if DEBUG:
            print(f"[embed_query] Query: {query[:50]}... -> {len(embedding)}d vector")

        return embedding

    @track_latency("embedding_batch")
    def embed_documents(self, documents: list, batch_size: int = 32) -> list:
        """Embed multiple documents (batch)."""
        if not documents:
            raise ValueError("Documents list cannot be empty")

        embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            convert_to_tensor=False,
            show_progress_bar=DEBUG
        )

        if DEBUG:
            print(f"[embed_documents] Embedded {len(documents)} docs -> {embeddings.shape}")

        return embeddings.tolist()

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension


if __name__ == "__main__":
    service = EmbeddingService()

    query = "What is Aadhaar?"
    embedding = service.embed_query(query)
    print(f"Query embedding shape: {len(embedding)}")

    docs = [
        "Aadhaar is an ID",
        "Apply for Aadhaar here",
        "Aadhaar registration process"
    ]
    embeddings = service.embed_documents(docs)
    print(f"Batch embeddings: {len(embeddings)} docs x {len(embeddings[0])} dims")
