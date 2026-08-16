# Implementation: Grounding-gate fix + multi-strategy chunking

## Priority and scope note

Two independent fixes. Do Part 1 first — it's small, low-risk, and directly closes the most important guardrail gap. Part 2 is real work; it's scoped deliberately to be **additive** (new chunking runs alongside your existing 743,739-passage index, doesn't replace or require re-embedding it) so you can't accidentally break working retrieval this close to the deadline.

---

## Part 1: Grounding score gates the returned answer, not just the cache

### Problem
`main_app.py` currently only checks `grounding_score >= ANSWER_CACHE_MIN_GROUNDING` to decide whether to *cache* an answer. It never checks this before *returning* the answer to the user. Confirmed live: a Gujarati query retrieved irrelevant Nepali/Sanskrit passages (grounding 0.0868) and the system still returned a confident, unhedged answer.

### Fix

In `main_app.py`, find the block that currently looks like:

```python
grounding_score = check_grounding(answer, fused)
validated = validate_answer(answer, grounding_score)

if validated["is_valid"] and grounding_score >= config.ANSWER_CACHE_MIN_GROUNDING:
    answer_cache.set(literal_key, result)
    semantic_cache.set(literal_key, query_embedding, result)
```

Change to gate the answer itself first, then decide caching from the (possibly now-hedged) result:

```python
grounding_score = check_grounding(answer, fused)
validated = validate_answer(answer, grounding_score)

if validated["is_valid"] and grounding_score < config.ANSWER_CACHE_MIN_GROUNDING:
    # Answer passed basic validation (not empty/refusal-phrase) but isn't
    # actually grounded in the retrieved context — this is the confident-
    # hallucination case, not the "model said I don't know" case.
    final_answer = build_low_grounding_response(query_text)
    is_hedged = True
else:
    final_answer = validated["answer"]
    is_hedged = False

result = {
    "answer": final_answer,
    "grounding_score": grounding_score,
    "sources": doc_ids,
    "hedged": is_hedged,
}

# Only cache genuinely grounded, non-hedged answers — do not cache a hedge
# message as if it were the answer, and do not cache low-grounding answers
# under any circumstances (this rule already existed, keep it).
if validated["is_valid"] and grounding_score >= config.ANSWER_CACHE_MIN_GROUNDING:
    answer_cache.set(literal_key, result)
    semantic_cache.set(literal_key, query_embedding, result)
```

Add the hedge-message builder to `guardrails.py`, matching the tone/style of your existing `validate_answer` fallback message and respecting the query's language (reuse whatever language-detection you already have from the prompt template's language-match fix):

```python
def build_low_grounding_response(query_text: str) -> str:
    lang = detect_query_language(query_text)  # reuse existing detection if available
    messages = {
        "en": "I couldn't find information in the available sources that confidently answers this question. Could you rephrase it or ask about a related topic?",
        "hi": "मुझे उपलब्ध जानकारी में इस प्रश्न का विश्वसनीय उत्तर नहीं मिला। कृपया प्रश्न को दोबारा पूछें।",
        "gu": "મને ઉપલબ્ધ માહિતીમાં આ પ્રશ્નનો વિશ્વસનીય જવાબ મળ્યો નથી. કૃપા કરીને પ્રશ્નને ફરીથી પૂછો.",
    }
    return messages.get(lang, messages["en"])
```

If you don't already have reusable language detection outside the prompt template, a simple fallback is acceptable for now: default to English, or use the same script-detection heuristic (Unicode range check) your language-mismatch fix likely already uses somewhere.

### Apply to both paths
This needs to land in **both** the REST `/query` endpoint and the WebSocket handler in `main_app.py` — check both call sites, don't fix one and assume the other inherits it if they're not sharing a single function.

### Testing
- [ ] Re-run the exact Gujarati query that triggered the 0.0868 finding — confirm it now returns the hedge message, not a confident wrong answer.
- [ ] Run a known-good, well-grounded query — confirm it's unaffected (no hedge, normal answer, still caches).
- [ ] Confirm a hedge response is never written to either cache (check cache stats/logs after triggering a hedge).
- [ ] Re-run your benchmark harness across the full query set — confirm no previously-good answers start getting hedged (that would mean the threshold itself needs revisiting, not this gating logic).

---

## Part 2: Multi-strategy chunking (additive, doesn't touch the existing index)

### Scope decision, stated plainly
Your corpus is MSMARCO-XI's pre-segmented passages, already used as retrieval units. Re-chunking and re-embedding all 743,739 passages from scratch this close to the deadline is real risk for real reward — skip that. Instead: **add a second and third chunking strategy that run on top of the existing corpus, indexed as additional entries in the same Chroma collection, tagged with metadata identifying which strategy produced them.** This directly satisfies "should be vast... more than one strategy" and "metadata-aware chunking" without touching what's already working.

### Step 2.1: Fixed-size overlapping sub-chunking, for long passages

Not every MSMARCO passage is short. Identify passages over a length threshold and split them into overlapping windows — this is real chunking work applied to real data, not a no-op.

```python
# src/chunking/fixed_overlap.py
def fixed_overlap_chunks(text: str, parent_id: str, window_tokens=100, overlap_tokens=20):
    tokens = text.split()  # or use your embedding model's tokenizer for accuracy
    if len(tokens) <= window_tokens:
        return []  # short passage, no sub-chunking needed — parent alone is fine

    chunks = []
    step = window_tokens - overlap_tokens
    for i, start in enumerate(range(0, len(tokens), step)):
        window = tokens[start:start + window_tokens]
        if len(window) < 20:  # skip trailing tiny fragments
            break
        chunks.append({
            "chunk_id": f"{parent_id}_fixed_{i}",
            "parent_id": parent_id,
            "content": " ".join(window),
            "chunking_strategy": "fixed_overlap",
            "window_index": i,
        })
    return chunks
```

### Step 2.2: Semantic (similarity-boundary) chunking, for long passages

Split at points where consecutive sentences diverge semantically, rather than at a fixed token count — this is the "semantic vs. fixed-size" contrast the spec explicitly asks to see.

```python
# src/chunking/semantic.py
import numpy as np

def semantic_chunks(text: str, parent_id: str, embed_fn, similarity_drop_threshold=0.5):
    sentences = split_into_sentences(text)  # simple regex or existing sentence splitter
    if len(sentences) <= 2:
        return []  # too short to meaningfully split

    embeddings = [embed_fn(s) for s in sentences]
    boundaries = [0]
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i-1], embeddings[i])
        if sim < similarity_drop_threshold:
            boundaries.append(i)
    boundaries.append(len(sentences))

    chunks = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        if end - start < 1:
            continue
        content = " ".join(sentences[start:end])
        chunks.append({
            "chunk_id": f"{parent_id}_semantic_{i}",
            "parent_id": parent_id,
            "content": content,
            "chunking_strategy": "semantic_boundary",
            "segment_index": i,
        })
    return chunks

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

Reuse your existing embedding service (`embed_fn`) — don't stand up a second model just for this.

### Step 2.3: Metadata-aware retrieval — make the unused fields actually do something

The audit found `source` and `query_id` are stored but never read. Fix this alongside the new chunking work so "metadata-aware" is genuinely demonstrable, not just a stored-but-inert field:

- Use `parent_id` (new field from steps 2.1/2.2) to **deduplicate**: if both a parent passage and one of its sub-chunks appear in the same top-k result set, keep only the higher-scoring one — this is a direct, checkable use of chunk lineage metadata in ranking.
- Use `chunking_strategy` to **log which strategy contributed to each answer** — add this to your response payload's `sources` field (e.g., `{"doc_id": ..., "strategy": "semantic_boundary"}`). This gives you a concrete number for the submission writeup: "X% of retrieved chunks came from semantic chunking vs fixed-overlap vs whole-passage" — real evidence the multi-strategy approach is doing something, not just present in the code.

```python
# in retrieval.py::merge_and_rank, after fusion, before truncation to top_k
def dedupe_by_parent(fused_results: list) -> list:
    seen_parents = {}
    deduped = []
    for doc_id, score, metadata in fused_results:
        parent = metadata.get("parent_id", doc_id)  # whole passages are their own "parent"
        if parent not in seen_parents or score > seen_parents[parent]:
            seen_parents[parent] = score
            deduped.append((doc_id, score, metadata))
    return deduped
```

### Step 2.4: Build script — run once, additive to existing index

```python
# scripts/add_chunking_strategies.py
"""
Additive indexing pass: does NOT delete or modify existing passage-level entries.
Reads the existing corpus, identifies long passages, generates fixed-overlap and
semantic sub-chunks for those, embeds and adds them to the SAME Chroma collection
with chunking_strategy metadata.
"""
LENGTH_THRESHOLD_TOKENS = 100  # only sub-chunk passages longer than this

def main():
    passages = load_existing_passages()  # from chunks.jsonl
    new_chunks = []
    for p in passages:
        if len(p["content"].split()) > LENGTH_THRESHOLD_TOKENS:
            new_chunks.extend(fixed_overlap_chunks(p["content"], p["chunk_id"]))
            new_chunks.extend(semantic_chunks(p["content"], p["chunk_id"], embed_fn))

    print(f"Generated {len(new_chunks)} additional sub-chunks from "
          f"{sum(1 for p in passages if len(p['content'].split()) > LENGTH_THRESHOLD_TOKENS)} long passages")

    for chunk in new_chunks:
        embedding = embed_fn(chunk["content"])
        chroma_collection.add(
            ids=[chunk["chunk_id"]],
            embeddings=[embedding],
            documents=[chunk["content"]],
            metadatas=[{
                "parent_id": chunk["parent_id"],
                "chunking_strategy": chunk["chunking_strategy"],
                "language": get_parent_language(chunk["parent_id"]),  # inherit
            }],
        )
        # also add to bm25s sparse index if it supports incremental add;
        # otherwise rebuild the sparse index once at the end with the full
        # combined set (original + new chunks)
```

**Run this once, as a batch job, before the deadline — not on every server start.** Log how many new chunks were generated and confirm the count is nonzero and reasonable (sanity-check: if `LENGTH_THRESHOLD_TOKENS=100` and most MSMARCO passages are short, you may generate far fewer sub-chunks than you'd expect — that's fine and honest, just report the real number in your writeup rather than assuming a large number.)

### Testing
- [ ] Run the build script against a small sample first (e.g., 1,000 passages) before the full 743K corpus — confirm chunk generation, embedding, and Chroma insertion all work correctly and check timing to estimate full-corpus runtime.
- [ ] Confirm existing passage-level retrieval is completely unaffected — re-run your benchmark query set, confirm results and grounding scores are the same or better, never worse, versus pre-change baseline.
- [ ] Confirm `dedupe_by_parent` actually removes duplicates in a real query where both a parent and a sub-chunk would otherwise appear.
- [ ] Confirm the `chunking_strategy` field is visible in the response payload's sources, so it's demonstrable in the demo video / writeup.
- [ ] Update `Markdown/ARCHITECTURE.md` to match what's actually implemented now (the audit flagged this doc as previously describing chunking that didn't exist — close that gap, don't leave a second stale-doc problem).

### What to say in the submission writeup regardless of how much of Part 2 lands
Be precise about what's real: "MSMARCO-XI passages are used as base retrieval units; additionally, passages exceeding N tokens are further split using both fixed-size overlapping windows and semantic-boundary detection, indexed alongside the originals with strategy metadata used for deduplication in ranking." That's an honest, specific, defensible claim — don't round it up to "we chunk everything three ways" if the length threshold means most passages weren't actually short enough to need sub-chunking.

---

## Order of work

1. Part 1 (grounding gate) — same day, both code paths, test against the known 0.0868 case.
2. Part 2.4 small-sample test run — confirms the pipeline works before committing to the full corpus.
3. Part 2.1-2.3 full implementation and full-corpus run.
4. Update `ARCHITECTURE.md` to match reality.
5. Re-run the full benchmark harness one final time post-changes, confirm no regressions in latency or grounding versus the audit's baseline numbers.
