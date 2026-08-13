import os

from config import (
    DEBUG, ANTHROPIC_API_KEY, LOCAL_LLM_MODEL_PATH,
    LOCAL_LLM_N_CTX, LOCAL_LLM_MAX_TOKENS,
)
from src.latency_tracker import track_latency


class GenerationService:
    """LLM answer generation: local llama.cpp (primary) -> Claude API (fallback) ->
    extractive passage return (last resort, no LLM available)."""

    def __init__(self, use_local: bool = True):
        self.llama_model = None
        self.claude_client = None

        if use_local and os.path.exists(LOCAL_LLM_MODEL_PATH):
            from llama_cpp import Llama
            if DEBUG:
                print(f"[GenerationService] Loading local model: {LOCAL_LLM_MODEL_PATH}")
            self.llama_model = Llama(
                model_path=LOCAL_LLM_MODEL_PATH,
                n_ctx=LOCAL_LLM_N_CTX,
                n_threads=os.cpu_count(),
                verbose=DEBUG,
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
    def generate(self, query: str, context: list, use_fast_path: bool = True) -> str:
        """Generate an answer grounded in the retrieved context."""
        if not context:
            return "I could not find any relevant information to answer your question."

        if self.llama_model is not None:
            try:
                return self._generate_with_llama(query, context)
            except Exception as e:
                if DEBUG:
                    print(f"[GenerationService] Local generation failed: {e}. Falling back.")

        if self.claude_client is not None:
            return self._generate_with_claude(query, context)

        return self._generate_extractive(query, context)

    def _build_prompt(self, query: str, context: list) -> str:
        context_text = "\n\n".join(context)
        return (
            f"Answer the question using only the context below. "
            f"If the answer isn't in the context, say so.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
        )

    def _generate_with_llama(self, query: str, context: list) -> str:
        prompt = self._build_prompt(query, context)
        messages = [
            {"role": "system", "content": "You are a helpful, concise multilingual assistant. "
                                           "Answer in the same language as the question."},
            {"role": "user", "content": prompt},
        ]
        response = self.llama_model.create_chat_completion(
            messages=messages,
            max_tokens=LOCAL_LLM_MAX_TOKENS,
            temperature=0.3,
        )
        return response["choices"][0]["message"]["content"].strip()

    def _generate_with_claude(self, query: str, context: list) -> str:
        prompt = self._build_prompt(query, context)
        try:
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            if DEBUG:
                print(f"[GenerationService] Claude call failed: {e}. Falling back to extractive.")
            return self._generate_extractive(query, context)

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
