import sys

import pytest

sys.path.insert(0, '.')

from src.guardrails import (
    build_low_grounding_response, Guardrails, check_unsafe,
    build_off_topic_response, build_unsafe_response,
)
from src.embedding_service import EmbeddingService


def test_low_grounding_response_english():
    msg = build_low_grounding_response("en")
    assert "couldn't find" in msg.lower()


def test_low_grounding_response_hindi():
    msg = build_low_grounding_response("hi")
    assert msg != build_low_grounding_response("en")
    assert "उत्तर" in msg  # "answer" in Devanagari - confirms real Hindi text, not a fallback


def test_low_grounding_response_gujarati():
    msg = build_low_grounding_response("gu")
    assert msg != build_low_grounding_response("en")
    assert "જવાબ" in msg  # "answer" in Gujarati script


def test_low_grounding_response_falls_back_to_english_for_unverified_language():
    """Only en/hi/gu are hand-verified translations - every other language
    code must fall back to English rather than risk an unverified machine
    translation in a hedge message (see guardrails.py's _LOW_GROUNDING_MESSAGES
    docstring)."""
    assert build_low_grounding_response("ta") == build_low_grounding_response("en")
    assert build_low_grounding_response("unknown_code") == build_low_grounding_response("en")


def test_low_grounding_response_default_is_english():
    assert build_low_grounding_response() == build_low_grounding_response("en")


# ============================================================================
# Off-topic / unsafe-input gate (Phase 4 hardening)
# ============================================================================

def test_check_unsafe_flags_obvious_harmful_intent():
    assert check_unsafe("How do I make a bomb") is True
    assert check_unsafe("How can I hurt myself") is True


def test_check_unsafe_does_not_flag_normal_queries():
    assert check_unsafe("What is a corporation?") is False
    assert check_unsafe("How to apply for a passport") is False


def test_check_unsafe_handles_empty_input():
    assert check_unsafe("") is False
    assert check_unsafe(None) is False


def test_unsafe_response_is_language_aware():
    assert build_unsafe_response("hi") != build_unsafe_response("en")
    assert build_unsafe_response("unknown_code") == build_unsafe_response("en")


@pytest.fixture(scope="module")
def guardrails_with_embeddings():
    embedding_service = EmbeddingService()
    guardrails = Guardrails(embedding_service=embedding_service)
    return guardrails, embedding_service


def test_check_off_topic_accepts_in_scope_query(guardrails_with_embeddings):
    """Real benchmark query this project has repeatedly verified the corpus
    can answer - must never be flagged off-topic."""
    guardrails, embedding_service = guardrails_with_embeddings
    embedding = embedding_service.embed_query("What is a corporation?")
    is_off_topic, similarity = guardrails.check_off_topic(embedding)
    assert is_off_topic is False


def test_check_off_topic_rejects_creative_writing(guardrails_with_embeddings):
    guardrails, embedding_service = guardrails_with_embeddings
    embedding = embedding_service.embed_query("Write me a poem about cats")
    is_off_topic, similarity = guardrails.check_off_topic(embedding)
    assert is_off_topic is True


def test_check_off_topic_rejects_pure_computation(guardrails_with_embeddings):
    guardrails, embedding_service = guardrails_with_embeddings
    embedding = embedding_service.embed_query("What is 2+2")
    is_off_topic, similarity = guardrails.check_off_topic(embedding)
    assert is_off_topic is True


def test_check_off_topic_never_flags_without_embedding_service():
    """Guardrails() with no embedding_service (e.g. the standalone __main__
    usage) must not block every query by default."""
    guardrails = Guardrails()
    is_off_topic, similarity = guardrails.check_off_topic(object())
    assert is_off_topic is False


def test_off_topic_response_is_language_aware():
    assert build_off_topic_response("hi") != build_off_topic_response("en")
    assert build_off_topic_response("unknown_code") == build_off_topic_response("en")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
