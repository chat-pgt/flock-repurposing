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

| Result | Evidence | Slide |
|---|---|---|
| On 200 drug → disease queries (259 edges), fewer walks lower MRR from 0.457 (E1's n = 512, ℓ = 128, seed 0) to 0.387 at n = 256 (−0.071 [−0.139, −0.012]), 0.272 at n = 128 (−0.185 [−0.264, −0.109]) and 0.212 at n = 64 (−0.245 [−0.336, −0.153]) | `results/e3_coverage.json` | |
| At the same budget n·ℓ, shorter walks do not hurt. n = 1024, ℓ = 64 gives 0.494 (+0.036 [−0.019, +0.093]), and n = 2048, ℓ = 32 gives 0.530 (+0.073 [+0.009, +0.143]; Hits@10 +8.5 points [1.7, 17.2]) | same | |
| Removing all 526,407 Anatomy–*expresses*–Gene edges (23% of the graph) leaves MRR unchanged at 0.485 (+0.028 [−0.026, +0.089]) and halves the time per query, from 7.7 s to 3.9 s | same, `results/e3_run_log.txt` | |
| Noise floor: on the same edges, E1's seeds 1 and 2 differ from seed 0 by −0.004 and −0.006 MRR | `results/e3_coverage.json` | |

Shorter walks reuse the checkpoint trained at ℓ = 128, so they measure the released model's robustness, not
what a model trained at each length would do. Each setting has one walk seed.

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
