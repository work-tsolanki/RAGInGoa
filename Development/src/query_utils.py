"""Query-text normalization shared by all BM25 backends."""

# English question/auxiliary words carry no topical signal but aren't in
# most default stoplists (e.g. "what" isn't in Whoosh's). With OR-style
# matching these words match huge numbers of unrelated passages (anything
# else phrased as "What is X?") and can outrank the actually relevant
# document. Stripped from the query only - the index itself is untouched.
QUERY_STOPWORDS = {
    "what", "who", "how", "where", "when", "which", "why", "whom", "whose",
    "is", "are", "was", "were", "do", "does", "did", "the", "a", "an",
    "of", "in", "on", "at", "to", "for",
}
# Includes Lucene/BM25 query-syntax special characters (?, *, ~, ^, :, etc.)
# - left in place, a trailing "?" on a question turns that word into a
# single-character wildcard/token instead of a plain term match.
_PUNCT = ".,?!;:\"'()[]{}*~^+-&|\\/"


def strip_query_stopwords(text: str) -> str:
    words = [w.strip(_PUNCT) for w in text.split()]
    words = [w for w in words if w and w.lower() not in QUERY_STOPWORDS]
    stripped = " ".join(words)
    return stripped or text
