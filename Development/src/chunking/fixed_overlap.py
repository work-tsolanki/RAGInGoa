"""Fixed-size overlapping-window sub-chunking for long passages.

Applied only to passages exceeding a length threshold (see
scripts/add_chunking_strategies.py) - short passages are left as-is, since
splitting them further would produce fragments smaller than a useful
retrieval unit. Additive: produces new chunk_ids alongside the existing
whole-passage index, never modifies or replaces the parent passage.
"""


def fixed_overlap_chunks(text: str, parent_id: str, window_tokens: int = 100,
                          overlap_tokens: int = 20, min_fragment_tokens: int = 20) -> list:
    """Splits `text` into overlapping word-count windows.

    Returns [] if the passage is already short enough that no sub-chunking
    is needed (parent alone is fine).
    """
    if overlap_tokens >= window_tokens:
        raise ValueError("overlap_tokens must be smaller than window_tokens")

    tokens = text.split()
    if len(tokens) <= window_tokens:
        return []

    chunks = []
    step = window_tokens - overlap_tokens
    for window_index, start in enumerate(range(0, len(tokens), step)):
        window = tokens[start:start + window_tokens]
        if len(window) < min_fragment_tokens:
            # trailing fragment too small to be a useful standalone unit -
            # its content is already covered by the previous window's tail
            break
        chunks.append({
            "chunk_id": f"{parent_id}_fixed_{window_index}",
            "parent_id": parent_id,
            "content": " ".join(window),
            "chunking_strategy": "fixed_overlap",
        })
    return chunks
