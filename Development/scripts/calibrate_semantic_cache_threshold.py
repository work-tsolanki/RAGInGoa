"""
Calibrates SEMANTIC_CACHE_SIMILARITY_THRESHOLD (config.py) against the
production embedding model.

Re-run this once real query traffic accumulates - the pair list below is a
seed set (39 pairs across true-duplicate / related-but-distinct / unrelated
categories), not a substitute for tuning against actual usage patterns.

Result as of the initial calibration (see semantic_cache_implementation.md
for the full task spec this was built against):
  - No clean separation between true duplicates and related-but-distinct
    pairs (min true-dup = 0.7302, max related-distinct = 0.8963 - the gap
    is negative). This is expected for this domain: "gst registration
    process" vs "gst registration fees" is lexically close but needs a
    different answer.
  - Threshold set to 0.92: above the highest observed related-but-distinct
    score with ~0.024 margin, below the next true-duplicate cluster
    (0.9274+). Zero false positives on this calibration set; catches ~40%
    (6/15) of true duplicates - the closest-phrased ones.
"""

import sys

sys.path.insert(0, '.')

import numpy as np

from src.embedding_service import EmbeddingService

TRUE_DUPLICATES = [
    ('how to apply for a passport', 'passport application process'),
    ('how to apply for a passport', 'steps to get a passport'),
    ('what is a corporation', 'define corporation'),
    ('what is a corporation', 'corporation meaning'),
    ('income tax filing deadline', 'when is the tax filing due date'),
    ('income tax filing deadline', 'last date to file income tax'),
    ('voter id card requirements', 'documents needed for voter id'),
    ('gst registration process', 'how to register for gst'),
    ('what is aadhaar', 'aadhaar card meaning'),
    ('how does compound interest work', 'explain compound interest'),
    ('how to renew a driving license', 'driving license renewal process'),
    ('what is a mutual fund', 'define mutual fund'),
    ('how to file a police complaint', 'steps to file a police report'),
    ('what is the minimum wage', 'minimum wage definition'),
    ('how to apply for a ration card', 'ration card application steps'),
]

RELATED_BUT_DISTINCT = [
    ('how to apply for a passport', 'how to renew a passport'),
    ('how to apply for a passport', 'passport application fees'),
    ('how to apply for a passport', 'how to apply for a visa'),
    ('what is a corporation', 'what is an s-corporation'),
    ('what is a corporation', 'how to start a corporation'),
    ('income tax filing deadline', 'income tax refund status'),
    ('income tax filing deadline', 'income tax slab rates'),
    ('voter id card requirements', 'how to update voter id address'),
    ('gst registration process', 'gst filing deadline'),
    ('gst registration process', 'gst registration fees'),
    ('what is aadhaar', 'how to update aadhaar address'),
    ('what is aadhaar', 'aadhaar card lost how to get duplicate'),
    ('how does compound interest work', 'how does simple interest work'),
    ('how to renew a driving license', 'how to apply for a driving license'),
    ('what is a mutual fund', 'what is a fixed deposit'),
    ('how to file a police complaint', 'how to check police complaint status'),
    ('what is the minimum wage', 'what is overtime pay'),
    ('how to apply for a ration card', 'ration card eligibility criteria'),
]

UNRELATED = [
    ('how to apply for a passport', 'income tax filing deadline'),
    ('what is a corporation', 'voter id card requirements'),
    ('how does compound interest work', 'how to file a police complaint'),
    ('what is aadhaar', 'what is the minimum wage'),
    ('gst registration process', 'how to renew a driving license'),
    ('how to apply for a ration card', 'what is a mutual fund'),
]


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_category(emb, name, pairs):
    print(f'=== {name} ===')
    sims = []
    for a, b in pairs:
        sim = cos_sim(emb.embed_query(a), emb.embed_query(b))
        sims.append(sim)
        print(f'{sim:.4f}  {a!r} <-> {b!r}')
    print(f'min={min(sims):.4f} max={max(sims):.4f} mean={sum(sims)/len(sims):.4f}\n')
    return sims


def main():
    emb = EmbeddingService()

    dup_sims = run_category(emb, 'TRUE DUPLICATES (should match)', TRUE_DUPLICATES)
    rel_sims = run_category(emb, 'RELATED BUT DISTINCT (should NOT match)', RELATED_BUT_DISTINCT)
    unrel_sims = run_category(emb, 'UNRELATED (should NOT match)', UNRELATED)

    print('=== SEPARATION ANALYSIS ===')
    print(f'Lowest true-duplicate score: {min(dup_sims):.4f}')
    print(f'Highest related-distinct score: {max(rel_sims):.4f}')
    print(f'Highest unrelated score: {max(unrel_sims):.4f}')
    print(f'Gap (lowest dup - highest related-distinct): {min(dup_sims) - max(rel_sims):.4f}')

    for threshold in [0.90, 0.92, 0.94, 0.96]:
        false_positives = sum(1 for s in rel_sims + unrel_sims if s >= threshold)
        true_positive_rate = sum(1 for s in dup_sims if s >= threshold) / len(dup_sims)
        print(f'threshold={threshold}: false_positives={false_positives}, '
              f'true_duplicate_hit_rate={true_positive_rate:.0%}')


if __name__ == "__main__":
    main()
