"""Semantic-boundary sub-chunking for long passages: splits where
consecutive sentences diverge in meaning, rather than at a fixed token
count - the "semantic vs. fixed-size" contrast the spec asks to see.

Applied only to passages exceeding a length threshold (see
scripts/add_chunking_strategies.py). Additive: produces new chunk_ids
alongside the existing whole-passage index, never modifies or replaces the
parent passage.
"""

import re

import numpy as np

# Handles Latin-script sentence punctuation (. ! ?) plus Devanagari sentence-
# ending marks (। ॥, used across several of this corpus's Indic languages).
# Not a full NLP tokenizer - a lightweight, dependency-free splitter is
# sufficient for boundary-detection purposes here.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।॥])\s+")


def split_into_sentences(text: str) -> list:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def semantic_chunks(text: str, parent_id: str, embed_fn, similarity_drop_threshold: float = 0.5) -> list:
    """embed_fn: callable taking one sentence string, returning its
    embedding vector - pass EmbeddingService.embed_query for real use, or a
    fake deterministic function in tests to avoid loading the real model.

    Returns [] if the passage doesn't have enough sentences to meaningfully
    split (parent alone is fine).
    """
    sentences = split_into_sentences(text)
    if len(sentences) <= 2:
        return []

    embeddings = [embed_fn(s) for s in sentences]
    boundaries = [0]
    for i in range(1, len(sentences)):
        sim = _cosine_sim(embeddings[i - 1], embeddings[i])
        if sim < similarity_drop_threshold:
            boundaries.append(i)
    boundaries.append(len(sentences))

    chunks = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        content = " ".join(sentences[start:end])
        chunks.append({
            "chunk_id": f"{parent_id}_semantic_{i}",
            "parent_id": parent_id,
            "content": content,
            "chunking_strategy": "semantic_boundary",
        })
    return chunks
