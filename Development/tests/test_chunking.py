import sys

import numpy as np
import pytest

sys.path.insert(0, '.')

from src.chunking.fixed_overlap import fixed_overlap_chunks
from src.chunking.semantic import semantic_chunks, split_into_sentences


def test_fixed_overlap_short_passage_returns_no_subchunks():
    text = "This is a short passage well under the window size."
    assert fixed_overlap_chunks(text, "parent1", window_tokens=100) == []


def test_fixed_overlap_splits_long_passage_with_overlap():
    tokens = [f"word{i}" for i in range(250)]
    text = " ".join(tokens)
    chunks = fixed_overlap_chunks(text, "parent1", window_tokens=100, overlap_tokens=20)

    assert len(chunks) >= 2
    for c in chunks:
        assert c["parent_id"] == "parent1"
        assert c["chunking_strategy"] == "fixed_overlap"
        assert c["chunk_id"].startswith("parent1_fixed_")

    # overlap: the tail of chunk 0 should reappear at the head of chunk 1
    chunk0_words = chunks[0]["content"].split()
    chunk1_words = chunks[1]["content"].split()
    assert chunk0_words[-20:] == chunk1_words[:20]


def test_fixed_overlap_rejects_overlap_ge_window():
    with pytest.raises(ValueError):
        fixed_overlap_chunks("some text", "parent1", window_tokens=50, overlap_tokens=50)


def test_fixed_overlap_drops_trailing_tiny_fragment():
    # 105 tokens, window=100 overlap=20 -> step=80: windows at [0:100], then
    # [80:105] is only 25 tokens... construct a case where the final window
    # is smaller than min_fragment_tokens to confirm it's dropped, not kept.
    tokens = [f"w{i}" for i in range(105)]
    text = " ".join(tokens)
    chunks = fixed_overlap_chunks(text, "p", window_tokens=100, overlap_tokens=20, min_fragment_tokens=30)
    # second window would be tokens[80:105] = 25 tokens < min_fragment_tokens=30
    assert len(chunks) == 1


def test_split_into_sentences_latin_punctuation():
    text = "Goa is famous for feni. It is made from cashew fruits! Do you like it?"
    sentences = split_into_sentences(text)
    assert sentences == [
        "Goa is famous for feni.",
        "It is made from cashew fruits!",
        "Do you like it?",
    ]


def test_split_into_sentences_devanagari_punctuation():
    text = "गोवा प्रसिद्ध है। यह काजू के लिए जाना जाता है।"
    sentences = split_into_sentences(text)
    assert len(sentences) == 2


def test_semantic_chunks_short_passage_returns_no_subchunks():
    text = "One sentence. Another sentence."
    def fake_embed(s):
        return np.array([1.0, 0.0, 0.0])
    assert semantic_chunks(text, "parent1", fake_embed) == []


def test_semantic_chunks_splits_at_similarity_drop():
    """Three sentences: first two embed identically (topic A), third embeds
    orthogonally (topic B) - must split before the third, not the second."""
    text = "Sentence A one. Sentence A two. Sentence B different topic."
    sentence_vectors = {
        "Sentence A one.": np.array([1.0, 0.0, 0.0]),
        "Sentence A two.": np.array([1.0, 0.0, 0.0]),
        "Sentence B different topic.": np.array([0.0, 1.0, 0.0]),
    }

    def fake_embed(s):
        return sentence_vectors[s]

    chunks = semantic_chunks(text, "parent1", fake_embed, similarity_drop_threshold=0.5)

    assert len(chunks) == 2
    assert chunks[0]["content"] == "Sentence A one. Sentence A two."
    assert chunks[1]["content"] == "Sentence B different topic."
    for c in chunks:
        assert c["parent_id"] == "parent1"
        assert c["chunking_strategy"] == "semantic_boundary"


def test_semantic_chunks_no_drop_stays_one_chunk():
    text = "Sentence A. Sentence A again. Sentence A once more."
    def fake_embed(s):
        return np.array([1.0, 0.0, 0.0])  # all identical -> never drops below threshold

    chunks = semantic_chunks(text, "parent1", fake_embed, similarity_drop_threshold=0.5)
    assert len(chunks) == 1
    assert chunks[0]["content"] == text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
