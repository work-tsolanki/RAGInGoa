import os
import json
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.qparser import QueryParser, OrGroup
from config import DEBUG
from src.latency_tracker import track_latency

# English question/auxiliary words carry no topical signal but aren't in
# Whoosh's small default stoplist (e.g. "what" isn't). With OrGroup these
# words match huge numbers of unrelated passages (anything else phrased as
# "What is X?") and can outrank the actually relevant document. Stripped
# from the query only - the index itself is untouched, so no reindex needed.
_QUERY_STOPWORDS = {
    "what", "who", "how", "where", "when", "which", "why", "whom", "whose",
    "is", "are", "was", "were", "do", "does", "did", "the", "a", "an",
    "of", "in", "on", "at", "to", "for",
}
# Includes Whoosh/Lucene query-syntax special characters (?, *, ~, ^, :,
# etc.) - left in place, a trailing "?" on a question turns that word into
# a single-character wildcard query instead of a plain term match.
_PUNCT = ".,?!;:\"'()[]{}*~^+-&|\\/"


def _strip_query_stopwords(text: str) -> str:
    words = [w.strip(_PUNCT) for w in text.split()]
    words = [w for w in words if w and w.lower() not in _QUERY_STOPWORDS]
    stripped = " ".join(words)
    return stripped or text


class WhooshService:
    """BM25 search using Whoosh."""

    def __init__(self, chunks: list = None, index_dir: str = "whoosh_index"):
        """Initialize Whoosh index."""
        self.index_dir = index_dir
        self.chunks_by_id = {}

        if chunks and (not os.path.exists(index_dir) or len(os.listdir(index_dir)) == 0):
            self._create_index(chunks)
        elif os.path.exists(index_dir):
            self.ix = open_dir(index_dir)
            if DEBUG:
                print(f"[WhooshService] Opened existing index at {index_dir}")
        else:
            raise ValueError(f"Index directory {index_dir} not found and no chunks provided")

    def _create_index(self, chunks: list):
        """Create Whoosh index from chunks."""
        if DEBUG:
            print(f"[WhooshService] Creating index from {len(chunks)} chunks...")

        os.makedirs(self.index_dir, exist_ok=True)

        schema = Schema(
            doc_id=ID(stored=True),
            content=TEXT(stored=True),
            language=STORED,
            section=STORED
        )

        self.ix = create_in(self.index_dir, schema)
        writer = self.ix.writer()

        for chunk in chunks:
            writer.add_document(
                doc_id=chunk["doc_id"],
                content=chunk["content"],
                language=chunk.get("language", "en"),
                section=chunk.get("metadata", {}).get("section", "")
            )
            self.chunks_by_id[chunk["doc_id"]] = chunk

        writer.commit()

        if DEBUG:
            print(f"Index created with {len(chunks)} documents")

    @track_latency("whoosh_search")
    def query(self, query_text: str, top_k: int = 10) -> list:
        """Search index with BM25."""
        if not query_text or len(query_text.strip()) == 0:
            return []

        results = []

        try:
            with self.ix.searcher() as searcher:
                # OrGroup: match any term and rank by BM25 relevance, not the
                # default AND (every term must co-occur in one passage,
                # which almost never happens for natural-language questions
                # and was silently zeroing out BM25's entire contribution).
                query_obj = QueryParser("content", self.ix.schema, group=OrGroup).parse(
                    _strip_query_stopwords(query_text)
                )
                hits = searcher.search(query_obj, limit=top_k)

                for hit in hits:
                    results.append({
                        "doc_id": hit["doc_id"],
                        "content": hit["content"],
                        "score": hit.score,
                        "language": hit.get("language", "en"),
                        "section": hit.get("section", "")
                    })

        except Exception as e:
            print(f"Whoosh search failed: {e}")
            return []

        if DEBUG:
            print(f"[whoosh_query] Found {len(results)} results for: {query_text[:50]}")

        return results


if __name__ == "__main__":
    with open("data/test_chunks.json", encoding="utf-8") as f:
        test_chunks = json.load(f)

    service = WhooshService(chunks=test_chunks)

    query = "What is Aadhaar?"
    results = service.query(query, top_k=5)

    print(f"\nQuery: {query}")
    print(f"Results: {len(results)}")
    for r in results:
        print(f"  - {r['doc_id']}: {r['content'][:60]}... (score: {r['score']:.2f})")
