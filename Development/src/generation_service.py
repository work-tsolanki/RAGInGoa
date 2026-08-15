import asyncio
import os
import queue
import re
import sys
import threading

from config import (
    DEBUG, ANTHROPIC_API_KEY, GROQ_API_KEY, GROQ_MODEL, GROQ_MAX_TOKENS,
    GROQ_TEMPERATURE, CEREBRAS_API_KEY, CEREBRAS_MODEL,
    CEREBRAS_MAX_TOKENS, CEREBRAS_TEMPERATURE, GENERATION_BACKEND_ORDER,
    LOCAL_LLM_MODEL_PATH, LOCAL_LLM_N_CTX, LOCAL_LLM_MAX_TOKENS, LOCAL_LLM_N_GPU_LAYERS,
)
from src.latency_tracker import track_latency

# On Windows, llama_cpp's CUDA build (ggml-cuda.dll) dynamically links against
# cublas64_*.dll/cudart64_*.dll, which live under CUDA_PATH\bin\x64 (CUDA 13+
# moved them out of \bin). llama_cpp loads its DLLs via ctypes.CDLL(winmode=
# ctypes.RTLD_GLOBAL), and passing an explicit winmode makes Windows skip
# os.add_dll_directory()-registered paths entirely and fall back to the
# classic DLL search order - which does honor PATH. So the fix is to prepend
# to PATH, not add_dll_directory().
if sys.platform == "win32":
    import glob

    cuda_paths = [os.environ["CUDA_PATH"]] if os.environ.get("CUDA_PATH") else []
    cuda_paths += sorted(
        glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*"),
        reverse=True,  # prefer the newest version if multiple are installed
    )
    dll_dirs = []
    for cuda_path in cuda_paths:
        for sub in ("bin\\x64", "bin"):
            dll_dir = os.path.join(cuda_path, sub)
            if os.path.isdir(dll_dir):
                dll_dirs.append(dll_dir)
    if dll_dirs:
        os.environ["PATH"] = os.pathsep.join(dll_dirs) + os.pathsep + os.environ["PATH"]


# Named explicitly in the system prompt instead of leaving the model to
# infer language from the question text alone - "answer in the same
# language as the question" was unreliable at temperature=0.3 even with
# clean single-language context, since it makes the model do inference work
# a direct instruction removes entirely.
_LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "gu": "Gujarati", "mr": "Marathi",
    "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
    "bn": "Bengali", "pa": "Punjabi", "ur": "Urdu", "ne": "Nepali",
    "or": "Odia", "as": "Assamese",
}

# Safety-net backstop for the prompt+few-shot fix in _build_prompt, per
# Markdown/natural_output_and_accuracy_prompt.md: pure string/regex only,
# deliberately NOT a second LLM call - a cleanup pass that costs another
# generation round-trip would reintroduce exactly the latency variance the
# prompt rewrite is meant to avoid, and adds a second place accuracy could
# drift without its own grounding check. Only strips a *leading* boilerplate
# opener clause; never touches text elsewhere in the answer, to avoid
# mangling a sentence that legitimately mentions "the documents" etc.
# past the first clause.
_BOILERPLATE_OPENER_RE = re.compile(
    r"^\s*(?:based on|according to|as per|per)\s+the\s+(?:provided\s+|given\s+|available\s+)?"
    r"(?:context|reference material|documents?|passages?|information)\s*[,:]?\s*",
    re.IGNORECASE,
)


def clean_answer_boilerplate(answer: str) -> str:
    """Strips a leading "Based on the provided context, ..." / "According
    to the documents, ..." style opener if the model produced one anyway,
    and re-capitalizes the new first letter. No-op if the answer doesn't
    start with one of these patterns."""
    if not answer:
        return answer
    cleaned = _BOILERPLATE_OPENER_RE.sub("", answer, count=1)
    if cleaned != answer and cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


# Measured directly against the production prompt (see conversation, not
# guessed): for the SAME short factual answer, Groq's Llama-3.1 tokenizer
# needs ~45 tokens in English but ~200-325 in Gujarati/Tamil/Bengali - a
# flat 200-token cap silently truncates these three mid-word instead of
# just running a little longer. Tightening the prompt's brevity instruction
# was tried first and confirmed (via direct token-count comparison) NOT to
# help - the content was already near-minimal, so the gap is tokenizer
# efficiency, not verbosity. This extended cap is a deliberate, bounded
# latency-for-correctness tradeoff for exactly these three languages, not a
# blanket increase - see 400-token dry runs: gu settled at 325 tokens,
# ta at 177, both well under this ceiling with margin.
_TOKEN_HUNGRY_LANGUAGES = {"gu", "ta", "bn"}
_EXTENDED_MAX_TOKENS = 350


def _max_tokens_for_language(language: str, base_max_tokens: int) -> int:
    return _EXTENDED_MAX_TOKENS if language in _TOKEN_HUNGRY_LANGUAGES else base_max_tokens


class GenerationService:
    """LLM answer generation: Groq (primary) -> local llama.cpp (fallback) ->
    Claude API (fallback) -> extractive passage return (last resort).

    All three LLM backends receive the exact same composed prompt (see
    _build_prompt) as a single user-role message, so answer quality and
    grounding behavior stay comparable regardless of which backend actually
    served a given request - the only difference between them is speed and
    availability.
    """

    def __init__(self, use_local: bool = True):
        self.llama_model = None
        self.claude_client = None
        self.groq_client = None
        self.cerebras_client = None
        self.last_backend = None  # set after generate() - which backend actually served the last request

        if GROQ_API_KEY:
            from groq import Groq
            self.groq_client = Groq(api_key=GROQ_API_KEY)
            if DEBUG:
                print("[GenerationService] Groq primary backend enabled")
        elif DEBUG:
            print("[GenerationService] GROQ_API_KEY not set, skipping Groq backend")

        if CEREBRAS_API_KEY:
            from cerebras.cloud.sdk import Cerebras
            self.cerebras_client = Cerebras(api_key=CEREBRAS_API_KEY)
            if DEBUG:
                print("[GenerationService] Cerebras backend enabled (benchmarked-but-not-promoted)")
        elif DEBUG:
            print("[GenerationService] CEREBRAS_API_KEY not set, skipping Cerebras backend")

        if use_local and os.path.exists(LOCAL_LLM_MODEL_PATH):
            from llama_cpp import Llama
            if DEBUG:
                print(f"[GenerationService] Loading local model: {LOCAL_LLM_MODEL_PATH}")
            self.llama_model = Llama(
                model_path=LOCAL_LLM_MODEL_PATH,
                n_ctx=LOCAL_LLM_N_CTX,
                n_threads=os.cpu_count(),
                n_gpu_layers=LOCAL_LLM_N_GPU_LAYERS,
                verbose=DEBUG,
                # Confirmed via benchmark: loads cleanly on this build
                # ("flash_attn = enabled", no fallback warning even with
                # quantized KV cache tested alongside it). No measurable win
                # at our ~400-1000 token prompt lengths, but free and starts
                # to matter if context length grows later.
                flash_attn=True,
            )
            if DEBUG:
                print("[GenerationService] Local llama.cpp model loaded")
        elif use_local and DEBUG:
            print(f"[GenerationService] Local model not found at {LOCAL_LLM_MODEL_PATH}, skipping")

        if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "mock":
            import anthropic
            self.claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            if DEBUG:
                print("[GenerationService] Claude fallback enabled")

    @track_latency("generation")
    def generate(self, query: str, context: list, use_fast_path: bool = True, language: str = "en") -> str:
        """Generate an answer grounded in the retrieved context."""
        if not context:
            self.last_backend = None
            return "I could not find any relevant information to answer your question."

        prompt = self._build_prompt(query, context, language)
        max_tokens = _max_tokens_for_language(language, GROQ_MAX_TOKENS)

        try:
            answer_parts = []
            backend_used = None
            for event in self._generate_with_fallback(prompt, max_tokens=max_tokens):
                answer_parts.append(event["delta"])
                backend_used = event["backend"]
            self.last_backend = backend_used
            return clean_answer_boilerplate("".join(answer_parts).strip())
        except Exception as e:
            if DEBUG:
                print(f"[GenerationService] All generation backends failed: {e}. Falling back to extractive.")
            self.last_backend = "extractive"
            return self._generate_extractive(query, context)

    async def _bridge_sync_generator(self, sync_gen):
        """Runs a synchronous generator on a worker thread, yielding its
        items asynchronously without blocking the event loop between them.

        The backend SDKs (groq, llama-cpp-python, anthropic, openai) all
        expose synchronous streaming iterators, not async ones - iterating
        them directly here would block the event loop for the duration of
        each network read. Re-raises the generator's exception (if any)
        after the thread finishes, same as iterating it directly would -
        callers that want graceful degradation catch it themselves.
        """
        q = queue.Queue()
        sentinel = object()
        error_holder = {}

        def producer():
            try:
                for item in sync_gen:
                    q.put(item)
            except Exception as e:
                error_holder["error"] = e
            finally:
                q.put(sentinel)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        try:
            while True:
                item = await asyncio.to_thread(q.get)
                if item is sentinel:
                    break
                yield item
        finally:
            thread.join()
        if "error" in error_holder:
            raise error_holder["error"]

    def _backend_stream_fn(self, name: str):
        """Maps a backend name to its raw (unbridged) sync generator
        method - shared by the fallback chain and by single-backend modes
        that need one specific backend's real behavior without the
        fallback chain's retry logic."""
        return {
            "groq": self._generate_groq_stream,
            "cerebras": self._generate_cerebras_stream,
            "local": self._generate_local_stream,
            "claude": self._generate_claude_stream,
        }[name]

    async def stream_generate(self, query: str, context: list, language: str = "en"):
        """Async token-by-token generator for WebSocket clients, following
        the ordered fallback chain. Falls back to a single extractive
        "delta" if every backend fails, same as generate() - callers don't
        need to special-case that.
        """
        if not context:
            self.last_backend = None
            yield {"delta": "I could not find any relevant information to answer your question.", "backend": None}
            return

        prompt = self._build_prompt(query, context, language)
        max_tokens = _max_tokens_for_language(language, GROQ_MAX_TOKENS)
        backend_used = None
        got_any = False
        try:
            async for item in self._bridge_sync_generator(self._generate_with_fallback(prompt, max_tokens=max_tokens)):
                got_any = True
                backend_used = item["backend"]
                yield item
        except Exception:
            pass  # _generate_with_fallback only raises when got_any is still False - handled below

        if got_any:
            self.last_backend = backend_used
        else:
            if DEBUG:
                print("[GenerationService] All generation backends failed. Falling back to extractive.")
            self.last_backend = "extractive"
            yield {"delta": self._generate_extractive(query, context), "backend": "extractive"}

    async def stream_single_backend(self, backend_name: str, query: str, context: list, language: str = "en"):
        """Streams from exactly one named backend, bypassing the fallback
        chain entirely - for side-by-side race/comparison mode, where the
        caller wants each backend's real, unmodified behavior (including
        its own failures) rather than an automatic fallthrough to the next
        backend in line."""
        if not context:
            yield {"delta": "I could not find any relevant information to answer your question.", "error": None}
            return

        prompt = self._build_prompt(query, context, language)
        max_tokens = _max_tokens_for_language(language, GROQ_MAX_TOKENS)
        stream_fn = self._backend_stream_fn(backend_name)
        try:
            async for delta in self._bridge_sync_generator(stream_fn(prompt, max_tokens, None)):
                yield {"delta": delta, "error": None}
        except Exception as e:
            yield {"delta": None, "error": str(e)}

    def _build_prompt(self, query: str, context: list, language: str = "en") -> str:
        """Single composed prompt shared byte-for-byte across all backends -
        the language instruction lives here, not in a separate system
        message, so it can't drift between backends that do/don't support
        a system role the same way.

        See Markdown/natural_output_and_accuracy_prompt.md for the
        rationale - this instruction block + few-shot pair targets specific
        observed failure patterns (boilerplate openers, over-hedging,
        question-restating, length mismatch) rather than a vague "sound
        more natural" request, which doesn't reliably change model output.
        Bump config.PROMPT_VERSION when editing this meaningfully - see
        that constant's docstring for why."""
        language_name = _LANGUAGE_NAMES.get(language, language)
        context_text = "\n\n".join(context)
        return (
            "You are a knowledgeable, direct assistant answering general-knowledge and "
            "travel/civic questions, using only the reference material provided below.\n\n"
            "How to answer:\n"
            "- Answer the question directly, in the first sentence. Do not restate the "
            "question, and do not open with phrases like \"Based on the provided context\" "
            "or \"According to the documents.\"\n"
            "- Write the way a knowledgeable person would explain it to someone asking in "
            "person - clear, direct, conversational - not like a report or a list of "
            "extracted facts, unless the question specifically asks for a list or steps.\n"
            "- Match your answer's length to the question. A simple factual question gets a "
            "short, direct answer. A \"how do I...\" process question gets the actual steps, "
            "only as many as are needed.\n"
            "- Keep it brief: at most 1-2 sentences for a factual question, unless the "
            "question explicitly asks for a list, steps, or more detail. Do not add extra "
            "background, examples, or restatement beyond what directly answers the question.\n"
            "- If the reference material fully answers the question, answer with confidence - "
            "do not hedge or add unnecessary qualifiers like \"it appears\" or \"it seems\" "
            "when the source is clear.\n"
            "- If the reference material only partially answers the question, say plainly "
            "what is and isn't covered, without disclaiming the entire answer.\n"
            f"- Answer in {language_name} only, regardless of what language the reference "
            "material below is written in.\n"
            "- Do not mention \"the context,\" \"the documents,\" \"the provided passages,\" "
            "or similar meta-references to your own retrieval process. Just answer, the way "
            "you would if you simply knew the information.\n\n"
            "Example of the tone and directness to match:\n\n"
            "Question: What is a corporation?\n"
            "Good answer: A corporation is a legal entity created under law that exists "
            "independently of its owners - it has its own rights, powers, and liabilities "
            "separate from the people who founded or run it.\n\n"
            "Question: How do I apply for a passport?\n"
            "Good answer: Where you apply depends on how soon you need it - routine "
            "applications go through the standard passport agency process, while urgent "
            "requests have an expedited path with its own requirements.\n\n"
            "Question: What is Goa famous for?\n"
            "Good answer: Goa is famous for its cashew fruits, which are used to make a "
            "special liquor called feni, and for cashew-based sweets like kaju katli.\n\n"
            f"Reference material:\n{context_text}\n\nQuestion: {query}"
        )

    def _generate_with_fallback(self, prompt: str, max_tokens: int = None, temperature: float = None):
        """Try each backend in GENERATION_BACKEND_ORDER, yielding
        {"delta", "backend"} events as text streams in.

        Retry semantics: only fall through to the next backend if the
        current one fails before yielding any tokens (connection/timeout/
        rate-limit type failures). Once a backend has started streaming,
        commit to it for the rest of the request - a mid-stream failure
        after partial output has already been sent means we stop and let
        the partial text go to validate_answer as-is, rather than risk
        sending the client two different partial answers back to back.
        """
        last_error = None
        for name in GENERATION_BACKEND_ORDER:
            try:
                backend_fn = self._backend_stream_fn(name)
            except KeyError:
                continue
            yielded_any = False
            try:
                for delta in backend_fn(prompt, max_tokens, temperature):
                    yielded_any = True
                    yield {"delta": delta, "backend": name}
                return  # backend completed successfully
            except Exception as e:
                last_error = e
                if DEBUG:
                    print(f"[GenerationService] Backend '{name}' failed "
                          f"({'mid-stream' if yielded_any else 'before first token'}): {e}")
                if yielded_any:
                    # Already committed to this backend and sent partial
                    # output - do not retry with a different one.
                    return
                continue  # failed before any tokens - safe to try the next backend

        raise RuntimeError(f"All generation backends failed. Last error: {last_error}")

    def _generate_groq_stream(self, prompt: str, max_tokens: int = None, temperature: float = None):
        if self.groq_client is None:
            raise RuntimeError("GROQ_API_KEY not configured")

        stream = self.groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens or GROQ_MAX_TOKENS,
            temperature=temperature if temperature is not None else GROQ_TEMPERATURE,
            stream=True,
        )
        for chunk in stream:
            # Some OpenAI-compatible providers send a final chunk with an
            # empty choices list (e.g. a trailing usage-stats-only chunk) -
            # indexing [0] unconditionally crashes on that.
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _generate_cerebras_stream(self, prompt: str, max_tokens: int = None, temperature: float = None):
        """Originally planned as a same-weights-different-silicon speed test
        vs Groq (mercury2_and_alt_architectures.md Part 3.1); Llama-3.1-8B
        isn't available on this Cerebras account (confirmed via
        models.list()), so this runs gpt-oss-120b instead - a full
        accuracy+speed comparison against a different, larger model, not
        the narrow silicon-only test originally intended."""
        if self.cerebras_client is None:
            raise RuntimeError("CEREBRAS_API_KEY not configured")

        stream = self.cerebras_client.chat.completions.create(
            model=CEREBRAS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens or CEREBRAS_MAX_TOKENS,
            temperature=temperature if temperature is not None else CEREBRAS_TEMPERATURE,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _generate_local_stream(self, prompt: str, max_tokens: int = None, temperature: float = None):
        if self.llama_model is None:
            raise RuntimeError("Local llama.cpp model not loaded")

        stream = self.llama_model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens or LOCAL_LLM_MAX_TOKENS,
            temperature=temperature if temperature is not None else 0.3,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                yield delta

    def _generate_claude_stream(self, prompt: str, max_tokens: int = None, temperature: float = None):
        if self.claude_client is None:
            raise RuntimeError("Claude client not configured")

        with self.claude_client.messages.stream(
            model="claude-3-haiku-20240307",
            max_tokens=max_tokens or 200,
            temperature=temperature if temperature is not None else 0.3,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                if text:
                    yield text

    def _generate_extractive(self, query: str, context: list) -> str:
        """No LLM available: return the most relevant context passage directly."""
        return context[0]


if __name__ == "__main__":
    service = GenerationService(use_local=True)
    answer = service.generate(
        query="What is a corporation?",
        context=["A corporation is a company or group of people authorized to act as a "
                 "single entity and recognized as such in law."]
    )
    print(f"Answer: {answer}")
