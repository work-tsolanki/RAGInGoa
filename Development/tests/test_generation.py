import sys

import pytest

sys.path.insert(0, '.')

from src.generation_service import GenerationService, clean_answer_boilerplate


@pytest.fixture
def service():
    """No local model, no real Groq/Claude clients - isolates
    _generate_with_fallback's control flow from any real backend."""
    return GenerationService(use_local=False)


def _gen(*items):
    """Build a generator that yields `items` in order; a str item raised as
    an exception mid-sequence simulates a mid-stream failure."""
    def _run(prompt, max_tokens, temperature):
        for item in items:
            if isinstance(item, Exception):
                raise item
            yield item
    return _run


def test_groq_stream_raises_when_unconfigured(service):
    service.groq_client = None  # explicit, independent of whether GROQ_API_KEY is set in the environment
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        list(service._generate_groq_stream("prompt"))


class _FakeGroqChunk:
    def __init__(self, content):
        delta = type("Delta", (), {"content": content})()
        choice = type("Choice", (), {"delta": delta})()
        self.choices = [choice]


def test_groq_stream_yields_deltas_from_mocked_client(service):
    """Mocks the Groq SDK client itself (not our wrapper) to confirm
    _generate_groq_stream parses the real response shape correctly and
    yields plain text deltas - same shape as _generate_local_stream."""
    class _FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["stream"] is True
            return iter([_FakeGroqChunk("Hello"), _FakeGroqChunk(None), _FakeGroqChunk(" world")])

    service.groq_client = type("FakeGroqClient", (), {
        "chat": type("Chat", (), {"completions": _FakeCompletions()})(),
    })()

    deltas = list(service._generate_groq_stream("prompt"))
    assert deltas == ["Hello", " world"]  # None content chunks are skipped, same as local


def test_cerebras_stream_raises_when_unconfigured(service):
    service.cerebras_client = None
    with pytest.raises(RuntimeError, match="CEREBRAS_API_KEY"):
        list(service._generate_cerebras_stream("prompt"))


def test_cerebras_stream_yields_deltas_from_mocked_client(service):
    """Same OpenAI-compatible chunk shape as Groq. Model is gpt-oss-120b,
    not Llama-3.1-8B as originally planned - that model isn't available on
    this account (see benchmark/cerebras_status.md)."""
    class _FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["stream"] is True
            assert kwargs["model"] == "gpt-oss-120b"
            return iter([_FakeGroqChunk("Hello"), _FakeGroqChunk(None), _FakeGroqChunk(" world")])

    service.cerebras_client = type("FakeCerebrasClient", (), {
        "chat": type("Chat", (), {"completions": _FakeCompletions()})(),
    })()

    deltas = list(service._generate_cerebras_stream("prompt"))
    assert deltas == ["Hello", " world"]


def test_local_stream_raises_when_unconfigured(service):
    with pytest.raises(RuntimeError, match="Local llama.cpp"):
        list(service._generate_local_stream("prompt"))


def test_claude_stream_raises_when_unconfigured(service):
    with pytest.raises(RuntimeError, match="Claude client"):
        list(service._generate_claude_stream("prompt"))


def test_cerebras_registered_in_fallback_chain(service, monkeypatch):
    import src.generation_service as gen_module
    monkeypatch.setattr(gen_module, "GENERATION_BACKEND_ORDER", ["cerebras", "groq"])
    monkeypatch.setattr(service, "_generate_cerebras_stream", _gen("Cerebras", " answer"))
    monkeypatch.setattr(service, "_generate_groq_stream", _gen("should not run"))

    events = list(service._generate_with_fallback("prompt"))
    assert [e["delta"] for e in events] == ["Cerebras", " answer"]
    assert events[0]["backend"] == "cerebras"


def test_fallback_retries_before_first_token(service, monkeypatch):
    """Backend fails before yielding anything -> falls through cleanly."""
    monkeypatch.setattr(service, "_generate_groq_stream", _gen(ConnectionError("groq down")))
    monkeypatch.setattr(service, "_generate_local_stream", _gen("Hello", " world"))
    monkeypatch.setattr(service, "_generate_claude_stream", _gen("should not run"))

    events = list(service._generate_with_fallback("prompt"))
    assert [e["delta"] for e in events] == ["Hello", " world"]
    assert all(e["backend"] == "local" for e in events)


def test_fallback_commits_after_first_token(service, monkeypatch):
    """Backend fails mid-stream, after already yielding - must NOT retry
    with the next backend; partial output stands as the answer."""
    monkeypatch.setattr(service, "_generate_groq_stream", _gen("Partial", RuntimeError("groq dropped connection")))
    monkeypatch.setattr(service, "_generate_local_stream", _gen("should not run"))
    monkeypatch.setattr(service, "_generate_claude_stream", _gen("should not run"))

    events = list(service._generate_with_fallback("prompt"))
    assert [e["delta"] for e in events] == ["Partial"]
    assert events[0]["backend"] == "groq"


def test_fallback_all_backends_fail_raises(service, monkeypatch):
    monkeypatch.setattr(service, "_generate_groq_stream", _gen(ConnectionError("groq down")))
    monkeypatch.setattr(service, "_generate_local_stream", _gen(RuntimeError("no local model")))
    monkeypatch.setattr(service, "_generate_claude_stream", _gen(RuntimeError("no claude key")))

    with pytest.raises(RuntimeError, match="All generation backends failed"):
        list(service._generate_with_fallback("prompt"))


def test_generate_falls_back_to_extractive_when_all_backends_fail(service, monkeypatch):
    monkeypatch.setattr(service, "_generate_groq_stream", _gen(ConnectionError("groq down")))
    monkeypatch.setattr(service, "_generate_local_stream", _gen(RuntimeError("no local model")))
    monkeypatch.setattr(service, "_generate_claude_stream", _gen(RuntimeError("no claude key")))

    answer = service.generate(query="What is X?", context=["X is a thing."])
    assert answer == "X is a thing."
    assert service.last_backend == "extractive"


def test_generate_reports_which_backend_served_the_request(service, monkeypatch):
    monkeypatch.setattr(service, "_generate_groq_stream", _gen("The answer is Y."))

    answer = service.generate(query="What is Y?", context=["Y is a thing."])
    assert answer == "The answer is Y."
    assert service.last_backend == "groq"


def test_build_prompt_embeds_language_instruction_not_system_message(service):
    prompt = service._build_prompt("What is X?", ["X is a thing."], language="hi")
    assert "Hindi" in prompt
    assert "X is a thing." in prompt
    assert "What is X?" in prompt


def test_build_prompt_includes_natural_tone_instructions_and_few_shot(service):
    """Regression check for the natural-output rewrite (see
    Markdown/natural_output_and_accuracy_prompt.md) - confirms the specific
    anti-boilerplate instruction and the few-shot demonstration block are
    actually present, not just that the prompt builds without error."""
    prompt = service._build_prompt("What is X?", ["X is a thing."])
    assert "Based on the provided context" in prompt  # named as an anti-pattern to avoid
    assert "Good answer:" in prompt  # few-shot block marker
    assert "kaju katli" in prompt  # real corpus-grounded example, not an invented fact


@pytest.mark.parametrize("boilerplate_opener", [
    "Based on the provided context, ",
    "According to the documents, ",
    "As per the reference material: ",
    "Per the given passages, ",
])
def test_clean_answer_boilerplate_strips_leading_opener(boilerplate_opener):
    answer = clean_answer_boilerplate(boilerplate_opener + "Goa is famous for feni.")
    assert answer == "Goa is famous for feni."


def test_clean_answer_boilerplate_is_noop_without_opener():
    answer = clean_answer_boilerplate("Goa is famous for feni.")
    assert answer == "Goa is famous for feni."


def test_clean_answer_boilerplate_only_strips_leading_occurrence():
    """Must not mangle a legitimate later mention of "the documents" etc -
    only a LEADING opener clause is a boilerplate tell, per
    Markdown/natural_output_and_accuracy_prompt.md's failure-pattern list."""
    answer = clean_answer_boilerplate(
        "Feni is a liquor from Goa. You can read more about the documents "
        "required to visit a distillery at the tourism office."
    )
    assert answer.startswith("Feni is a liquor from Goa.")
    assert "the documents required to visit a distillery" in answer


def test_clean_answer_boilerplate_empty_string():
    assert clean_answer_boilerplate("") == ""


@pytest.mark.asyncio
async def test_stream_generate_yields_tokens_from_worker_thread(service, monkeypatch):
    monkeypatch.setattr(service, "_generate_groq_stream", _gen("Hello", " world"))

    events = [event async for event in service.stream_generate("What is X?", ["X is a thing."])]
    assert [e["delta"] for e in events] == ["Hello", " world"]
    assert all(e["backend"] == "groq" for e in events)
    assert service.last_backend == "groq"


@pytest.mark.asyncio
async def test_stream_generate_falls_back_to_extractive(service, monkeypatch):
    monkeypatch.setattr(service, "_generate_groq_stream", _gen(ConnectionError("groq down")))
    monkeypatch.setattr(service, "_generate_local_stream", _gen(RuntimeError("no local model")))
    monkeypatch.setattr(service, "_generate_claude_stream", _gen(RuntimeError("no claude key")))

    events = [event async for event in service.stream_generate("What is X?", ["X is a thing."])]
    assert len(events) == 1
    assert events[0]["delta"] == "X is a thing."
    assert events[0]["backend"] == "extractive"
    assert service.last_backend == "extractive"


@pytest.mark.asyncio
async def test_stream_generate_empty_context_short_circuits(service):
    events = [event async for event in service.stream_generate("What is X?", [])]
    assert len(events) == 1
    assert "could not find" in events[0]["delta"]
    assert service.last_backend is None


@pytest.mark.asyncio
async def test_stream_single_backend_yields_only_from_named_backend(service, monkeypatch):
    """Must not fall through to another backend on failure - that's the
    whole point vs. stream_generate's fallback chain."""
    monkeypatch.setattr(service, "_generate_groq_stream", _gen("Groq", " answer"))
    monkeypatch.setattr(service, "_generate_cerebras_stream", _gen("should not run"))

    events = [e async for e in service.stream_single_backend("groq", "What is X?", ["X is a thing."])]
    assert [e["delta"] for e in events] == ["Groq", " answer"]
    assert all(e["error"] is None for e in events)


@pytest.mark.asyncio
async def test_stream_single_backend_surfaces_error_without_fallthrough(service, monkeypatch):
    monkeypatch.setattr(service, "_generate_cerebras_stream", _gen(RuntimeError("cerebras down")))

    events = [e async for e in service.stream_single_backend("cerebras", "What is X?", ["X is a thing."])]
    assert len(events) == 1
    assert events[0]["delta"] is None
    assert "cerebras down" in events[0]["error"]


@pytest.mark.asyncio
async def test_stream_single_backend_empty_context_short_circuits(service):
    events = [e async for e in service.stream_single_backend("groq", "What is X?", [])]
    assert len(events) == 1
    assert "could not find" in events[0]["delta"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
