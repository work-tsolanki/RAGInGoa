"""
Analyzes benchmark/grounding_calibration_raw.json (38 hand-labeled
query/answer/grounding_score entries, labeled blind to score - see
master_implementation_prompt-mercury2.md Priority 2) to pick
config.ANSWER_CACHE_MIN_GROUNDING.

Label = "is this a genuinely correct, cacheable answer grounded in the
retrieved context" - NOT "did the model produce fluent text". A confident,
well-phrased answer that turns out to be wrong (usually from a homonym or
topic-adjacent retrieval mismatch, e.g. "gst registration fees" answered
from a vehicle-registration passage) is labeled False even when it scores
near 1.0.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, '.')

DATA_PATH = Path(__file__).parent.parent / "benchmark" / "grounding_calibration_raw.json"


def main():
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    true_entries = [d for d in data if d["label"] is True]
    false_entries = [d for d in data if d["label"] is False]

    true_scores = sorted(d["grounding_score"] for d in true_entries)
    false_scores = sorted(d["grounding_score"] for d in false_entries)

    print(f"TRUE (grounded/correct):  n={len(true_entries)}  "
          f"min={min(true_scores):.4f} max={max(true_scores):.4f} "
          f"mean={sum(true_scores)/len(true_scores):.4f}")
    print(f"FALSE (ungrounded/wrong): n={len(false_entries)}  "
          f"min={min(false_scores):.4f} max={max(false_scores):.4f} "
          f"mean={sum(false_scores)/len(false_scores):.4f}")

    print("\nFALSE entries scoring above 0.5 (dangerous: confidently wrong, would be cached if threshold is too low):")
    for d in sorted(false_entries, key=lambda d: -d["grounding_score"]):
        if d["grounding_score"] > 0.5:
            print(f"  {d['grounding_score']:.4f}  {d['query']!r} -> {d['answer'][:80]!r}")

    print("\nTRUE entries scoring below 0.5 (cost of a conservative threshold: real answers that won't get cached):")
    for d in sorted(true_entries, key=lambda d: d["grounding_score"]):
        if d["grounding_score"] < 0.5:
            print(f"  {d['grounding_score']:.4f}  {d['query']!r} -> {d['answer'][:80]!r}")

    print("\nThreshold sweep (recall = % of TRUE retained as cache-eligible, "
          "precision-guard = % of FALSE correctly excluded):")
    for threshold in [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.999]:
        true_retained = sum(1 for s in true_scores if s >= threshold)
        false_excluded = sum(1 for s in false_scores if s < threshold)
        false_leaked = len(false_scores) - false_excluded
        print(f"  threshold={threshold:<6} true_retained={true_retained}/{len(true_scores)} "
              f"({true_retained/len(true_scores):.0%})   "
              f"false_excluded={false_excluded}/{len(false_scores)} "
              f"({false_excluded/len(false_scores):.0%})   false_leaked={false_leaked}")


if __name__ == "__main__":
    main()
