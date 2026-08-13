from config import DEBUG, ANTHROPIC_API_KEY
from src.latency_tracker import track_latency


class GenerationService:
    """LLM answer generation, with Claude API fallback and an extractive mock path."""

    def __init__(self, use_local: bool = True):
        self.use_local = use_local
        self.claude_client = None

        if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "mock":
            import anthropic
            self.claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            if DEBUG:
                print("[GenerationService] Claude fallback enabled")
        elif DEBUG:
            print("[GenerationService] No real API key set - using extractive mock generation")

    @track_latency("generation")
    def generate(self, query: str, context: list, use_fast_path: bool = True) -> str:
        """Generate an answer grounded in the retrieved context."""
        if not context:
            return "I could not find any relevant information to answer your question."

        if self.claude_client is not None:
            return self._generate_with_claude(query, context)

        return self._generate_extractive(query, context)

    def _generate_with_claude(self, query: str, context: list) -> str:
        context_text = "\n\n".join(context)
        prompt = (
            f"Answer the question using only the context below.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
        )
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
        best = context[0]
        return best


if __name__ == "__main__":
    service = GenerationService(use_local=True)
    answer = service.generate(
        query="What is Aadhaar?",
        context=["Aadhaar is a 12-digit unique identity number issued to Indian residents."]
    )
    print(f"Answer: {answer}")
