# Improving answer naturalness and accuracy — prompt design + approach comparison

## Reality check before anything else

"Fine-tune the model" and "prompt-engineer the model" are very different scopes of work, and given ~6 days to the Aug 22 deadline, only one of them is realistic. Read this section before picking an approach — it determines which of the options below you should actually attempt.

- **Full or LoRA fine-tuning** requires: a labeled dataset of (query, context, ideal-answer) triples large enough to move the model's style (typically hundreds to low-thousands of examples for LoRA to meaningfully shift tone), a training pipeline, GPU time, and — critically — an evaluation loop to confirm the fine-tune didn't degrade grounding or introduce new hallucination patterns. This is realistically a multi-day-to-week effort even when it goes smoothly, and you're also relying on hosted APIs (Groq, Mercury) for generation now, which you don't control the weights of at all — you cannot fine-tune "the model" as currently deployed, only a self-hosted copy, which reopens the local-latency problem you just moved away from.
- **Prompt engineering + few-shot conditioning** achieves most of what "sounds natural, not templated" actually means, in hours not days, works identically across every backend in your fallback chain (Groq, Mercury, local, Claude) without retraining anything, and doesn't touch your already-calibrated grounding/retrieval pipeline.

**Recommendation given your timeline: prompt engineering is the correct approach. Skip fine-tuning entirely for this deadline.** The rest of this document is built around that call. If you have a specific business reason to fine-tune (product roadmap beyond the deadline, not this submission), say so and we can scope that separately — it shouldn't be mixed into this week's work.

## What "sounds trained, not natural" usually actually means

Before rewriting anything, it helps to name the specific failure patterns, since "make it more natural" is otherwise unfalsifiable. Common RAG-specific tells:

- Boilerplate scaffolding: *"Based on the provided context, the answer to your question is..."*
- Over-hedging: *"According to the documents, it appears that..."* when the source is actually clear and direct
- Listing instead of answering: dumping retrieved facts in sequence rather than synthesizing them into a direct response
- Refusal-flavored padding even when grounded: *"I don't have complete information, but based on what's available..."* when the retrieved passage actually does answer the question
- Repeating the question back before answering it
- Answer length mismatched to question complexity — a one-line factual question ("income tax filing deadline") getting a multi-paragraph answer, or vice versa

Go through your `benchmark/results.json` answers and mark which of these actually show up — that turns "make it sound natural" into a specific, testable fix list instead of a vibe.

## The prompt itself

Replace or extend your current `_build_prompt` instruction block with this. It's designed to be backend-agnostic — same text works across Groq, Mercury, local llama.cpp, and Claude, since it's instructing behavior, not exploiting a model-specific quirk.

```
You are a knowledgeable, direct assistant answering questions about Indian civic and
government processes (passports, taxes, voter ID, GST, and similar topics), using only
the reference material provided below.

How to answer:
- Answer the question directly, in the first sentence. Do not restate the question,
  and do not open with phrases like "Based on the provided context" or "According to
  the documents."
- Write the way a knowledgeable person would explain it to someone asking in person —
  clear, direct, conversational — not like a report or a list of extracted facts,
  unless the question specifically asks for a list or steps.
- Match your answer's length to the question. A simple factual question gets a short,
  direct answer. A "how do I..." process question gets the actual steps, only as many
  as are needed.
- If the reference material fully answers the question, answer with confidence — do
  not hedge or add unnecessary qualifiers like "it appears" or "it seems" when the
  source is clear.
- If the reference material only partially answers the question, say plainly what is
  and isn't covered, without disclaiming the entire answer.
- Answer in the same language as the question, regardless of what language the
  reference material below is in.
- Do not mention "the context," "the documents," "the provided passages," or similar
  meta-references to your own retrieval process. Just answer, the way you would if you
  simply knew the information.

Reference material:
{retrieved_passages}

Question: {query}
```

Notes on specific lines:
- The "do not mention the context/documents" instruction directly targets the most common tell of a RAG system sounding like a RAG system rather than a knowledgeable answerer.
- The language-match line is your existing fix, kept in place — don't drop it, this prompt is additive to that, not a replacement.
- The length-matching instruction is new and directly targets your earlier finding that concise answers (top_k=5) both perform better and are less prone to the noisy/garbled failure mode you saw at k=10 — this reinforces that at the generation-style level too.

## Few-shot examples — the highest-leverage single addition

Instructions alone (the prompt above) shift style somewhat. 2-3 concrete examples of the tone you want, embedded directly in the prompt, shift it much more reliably — models follow demonstrated patterns better than described ones. Add this block after the instructions and before "Reference material":

```
Example of the tone and directness to match:

Question: What is the deadline for filing income tax returns?
Good answer: For most individual taxpayers, the deadline is July 31st of the
assessment year. If your accounts need to be audited, it extends to October 31st.

Question: How do I apply for a passport?
Good answer: You'll apply online through the Passport Seva portal, fill out the
application, pay the fee, and book an appointment at your nearest Passport Seva
Kendra. Bring your original documents — proof of address, identity, and date of
birth — to the appointment; they'll take your photo and biometrics there.
```

Build these from your *actual* retrieved-passage content style (pull real examples from your dataset, don't invent facts for the few-shot block) so the demonstrated tone stays grounded in what your corpus actually supports.

## Approaches ranked by effort vs. payoff, for this deadline

| Approach | Effort | Payoff | Recommendation |
|---|---|---|---|
| Instruction rewrite (above) | ~30 min | Medium-high | Do this now |
| Few-shot examples in prompt | ~1 hr (curating real examples) | High | Do this now |
| Post-generation cleanup pass (regex/string strip common boilerplate openers as a safety net) | ~30 min | Low-medium, catches stragglers | Do if time allows, as a backstop not a primary fix |
| LoRA fine-tune on self-hosted model | Days | Uncertain, and reopens the latency problem you just solved | Do not attempt before Aug 22 |
| Full fine-tune / RLHF-style tuning | Weeks+ | N/A for this timeline | Out of scope entirely |

## How to validate this actually worked

Don't ship on vibes — reuse the benchmark harness you already built:

1. Re-run `benchmark/run_backend_comparison.py` with the new prompt against the same fixed query set.
2. For each answer, check off the specific failure patterns listed above (boilerplate opener, over-hedging, question-restating, length mismatch) — before and after, so you have a real before/after comparison, not just an impression.
3. Confirm `grounding_score` and `is_valid` rate haven't dropped — a more "natural-sounding" answer that's less grounded is a regression, not an improvement. Directness should not come at the cost of accuracy; if you see that tradeoff appearing, tighten the "if the reference material only partially answers" instruction rather than accepting the drop.
4. Spot-check the Hindi/Gujarati rows specifically — a stylistic prompt rewrite in English-authored instructions can sometimes interact unpredictably with non-English output; confirm the language-match behavior still holds after this change.

## What NOT to do here

- Don't try to fix this by asking the model to "sound more human" as a bare instruction — vague style requests without concrete examples or specific anti-patterns to avoid tend to produce inconsistent results. The specific failure-pattern list + few-shot examples above will work more reliably than a generic tone request.
- Don't chase fine-tuning this week — it's the wrong tool for the time available and the actual problem (which is a prompt-conditioning problem, not a weights problem).
- Don't skip the re-validation step — "sounds more natural" and "still accurate and grounded" are two different properties, and it's easy to improve one at the expense of the other if you don't check both.
