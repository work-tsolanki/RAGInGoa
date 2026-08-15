"""
Generates raw (query, answer, grounding_score, context) data for manual
labeling as part of grounding-threshold calibration (Priority 2 of
master_implementation_prompt-mercury2.md).

Produces benchmark/grounding_calibration_raw.json - each entry gets a
"label" field filled in by hand afterward (true = grounded/correct,
false = not), then scripts/analyze_grounding_calibration.py reads that.
"""

import json
import sys

sys.path.insert(0, '.')

from src.embedding_service import EmbeddingService
from src.chroma_service import ChromaService
from src.bm25s_service import Bm25sService
from src.retrieval import merge_and_rank
from src.generation_service import GenerationService
from src.guardrails import Guardrails

QUERIES = [
    "How to apply for a passport", "What is a corporation?", "income tax filing deadline",
    "What are voter ID requirements", "How does GST registration work",
    "What is Aadhaar?", "How does compound interest work", "What causes inflation",
    "How to renew a driving license", "What is a mutual fund",
    "How to file a police complaint", "What is the minimum wage",
    "How to apply for a ration card", "What is a notary public",
    "How does small claims court work", "What is a power of attorney",
    "How to check credit score", "What is a fixed deposit",
    "How to register a birth certificate", "What is capital gains tax",
    "What is an S-corporation", "How to start a corporation",
    "income tax refund status", "income tax slab rates",
    "how to update voter id address", "gst filing deadline",
    "gst registration fees", "how to update aadhaar address",
    "aadhaar card lost how to get duplicate", "how does simple interest work",
    "how to apply for a driving license", "what is a fixed deposit vs mutual fund",
    "how to check police complaint status", "what is overtime pay",
    "ration card eligibility criteria", "what is the boiling point of water",
    "how many players on a cricket team", "what is the capital of France",
]

def main():
    emb = EmbeddingService()
    chroma = ChromaService(collection_name='hhgoa_rag_full')
    bm25 = Bm25sService(index_dir='bm25s_index_full')
    gen = GenerationService(use_local=False)
    guardrails = Guardrails()

    entries = []
    for q in QUERIES:
        q_emb = emb.embed_query(q).tolist()
        dense = chroma.query(q_emb, top_k=10)
        sparse = bm25.query(q, top_k=10)
        merged = merge_and_rank(dense, sparse, top_k=5, target_language='en')
        context_docs = [d['content'] for d in merged]

        answer = gen.generate(query=q, context=context_docs, language='en')
        grounding_score = guardrails.check_grounding(answer, context_docs)

        entries.append({
            "query": q,
            "answer": answer,
            "grounding_score": round(grounding_score, 4),
            "top_context": context_docs[0][:200] if context_docs else "",
            "label": None,  # fill in by hand: true = grounded/correct, false = not
        })
        print(f"{grounding_score:.4f}  {q!r}", flush=True)

    with open("benchmark/grounding_calibration_raw.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(entries)} entries to benchmark/grounding_calibration_raw.json")

if __name__ == "__main__":
    main()
