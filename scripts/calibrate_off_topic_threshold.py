"""
Calibrates OFF_TOPIC_SIMILARITY_THRESHOLD in src/guardrails.py.

Two distributions:
  - IN_SCOPE: this project's own real benchmark queries (must NOT be flagged
    off-topic - these are exactly what the corpus is verified to answer).
  - OFF_TOPIC: deliberately off-topic categories the spec names explicitly -
    creative writing, pure computation, casual chat, meta questions about
    the assistant itself.

Prints max-similarity-to-reference-set for every query in both groups, then
the threshold that separates them (if one exists cleanly) or where the
overlap is, so the constant in guardrails.py is chosen from evidence, not
by eye.
"""

import json
import sys

sys.path.insert(0, '.')

from src.embedding_service import EmbeddingService
from src.guardrails import _OFF_TOPIC_REFERENCE_QUERIES
import numpy as np

IN_SCOPE_QUERIES_PATH = "benchmark/percentile_batch_queries.json"

OFF_TOPIC_QUERIES = [
    # Creative writing
    "Write me a poem about cats",
    "Write a short story about a dragon",
    "Compose a haiku about the ocean",
    # Pure computation
    "What is 2+2",
    "What is the square root of 144",
    "Calculate 15% of 200",
    # Casual chat / greetings
    "Hi how are you",
    "Tell me a joke",
    "What's up",
    # Meta questions about the assistant
    "What model are you",
    "Who created you",
    "Are you ChatGPT",
    # Code generation
    "Write python code to sort a list",
    "Give me a SQL query to join two tables",
]


def main():
    with open(IN_SCOPE_QUERIES_PATH, encoding="utf-8") as f:
        in_scope = [q["query"] for q in json.load(f)]

    embedding_service = EmbeddingService()

    refs = np.stack([embedding_service.embed_query(q) for q in _OFF_TOPIC_REFERENCE_QUERIES])
    refs = refs / (np.linalg.norm(refs, axis=1, keepdims=True) + 1e-9)

    def max_sim(query):
        q = embedding_service.embed_query(query)
        q = q / (np.linalg.norm(q) + 1e-9)
        return float(np.max(refs @ q))

    print("=== IN_SCOPE (must stay ABOVE threshold) ===")
    in_scope_sims = []
    for q in in_scope:
        sim = max_sim(q)
        in_scope_sims.append(sim)
        print(f"  {sim:.4f}  {q}")

    print("\n=== OFF_TOPIC (must stay BELOW threshold) ===")
    off_topic_sims = []
    for q in OFF_TOPIC_QUERIES:
        sim = max_sim(q)
        off_topic_sims.append(sim)
        print(f"  {sim:.4f}  {q}")

    min_in_scope = min(in_scope_sims)
    max_off_topic = max(off_topic_sims)

    print(f"\nmin(in_scope) = {min_in_scope:.4f}")
    print(f"max(off_topic) = {max_off_topic:.4f}")

    if min_in_scope > max_off_topic:
        threshold = (min_in_scope + max_off_topic) / 2
        print(f"\nCLEAN SEPARATION. Suggested threshold: {threshold:.4f}")
    else:
        print(f"\nOVERLAP: {max_off_topic:.4f} (off-topic) > {min_in_scope:.4f} (in-scope)")
        print("No threshold cleanly separates both sets - pick a threshold that")
        print("minimizes total misclassifications and report the tradeoff.")


if __name__ == "__main__":
    main()
