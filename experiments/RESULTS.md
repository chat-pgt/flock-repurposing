# Results

One line per checked result. **Slides take numbers only from this file.**

## Analyses of the published paper

These use the paper's tables, the public Hetionet graph or a reading of the code. They are not FLOCK
runs.

| ID | Result | Evidence |
|---|---|---|
| A1 | Zero-shot entity MRR vs TRIX over 54 KGs: +0.001, 95% CI [−0.010, +0.012], Wilcoxon p = 0.36. No detectable difference | `results/a1_paper_significance.json` |
| A1 | Zero-shot entity Hits@10 vs TRIX: +0.012, 39 wins / 15 losses, Wilcoxon p = 0.002, family p = 0.037 (uncorrected) | same |
| A1 | Relation MRR beats TRIX and ULTRA, zero-shot and fine-tuned: 53–54 wins of 54, family-level Holm p = 0.004 | same |
| A2 | FLOCK's sampler on Hetionet (n = 512, ℓ = 128, one pass): sees 88% of nodes and 46% of connected node pairs. The true disease is reached in all 40 *treats* queries, even at ℓ = 16 | `results/a2_walk_coverage.txt` |
| A2 | At a fixed budget n·ℓ, coverage barely changes with ℓ | same |
| A3 | Estimated edge-type lookup on Hetionet: about 3,700 entries scanned per step, against about 17 comparisons with binary search. Not profiled | `paper_analysis/edge_lookup_cost.py` |

## Experiments

### E0. Reproduction and environment

*Pending.*

### E1. FLOCK vs ULTRA vs popularity

*Pending.*

### E2. Near-duplicate leakage

*Pending.*

### E3. Walk coverage

*Pending.*

### E4. Walk explanations

*Pending.*

### E5. Type-constrained negatives

*Pending.*

### E6. No-learning baselines for relation prediction

| Result | Evidence | Slide |
|---|---|---|
| Baseline C (no entity-type labels, only the training graph), all 112,510 test triples: MRR 0.912, Hits@1 0.853 | `results/e6_relation_baselines.json` | 2 |
| Baseline B (entity types and training frequency): MRR 0.885, Hits@1 0.796 | same | 2 |
| Baseline A (uniform among type-compatible relations): MRR 0.779, Hits@1 0.620 | same | |
| Published zero-shot scores for comparison: FLOCK 0.940 / 0.890, TRIX 0.809 / 0.707, ULTRA 0.634 / 0.524 (MRR / Hits@1) | Table 11 of the paper | 2 |
| 48,419 test triples (43%) have only one relation compatible with their entity types. On the other 64,091, MRR is 0.798 for B and 0.851 for C. FLOCK's published 0.940 implies at least 0.895 on those triples | same | 2 |
| Compound–Disease, *treats* vs *palliates* (56 triples): Hits@1 is 50% at random, 64.3% for B and 60.7% for C (34 of 56, 95% Wilson CI [47.6%, 72.4%]). Too few triples to separate the baselines | same | |

FLOCK's scores here are the published ones. They have not been reproduced in this study yet.

### E7. Relation similarity

*Pending.*
