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

| Result | Evidence | Slide |
|---|---|---|
| Zero-shot entity prediction reproduces Table 9 on three inductive graphs. MRR is 0.501 vs 0.500 (FB15k-237 v1), 0.698 vs 0.698 (WN18RR v1) and 0.657 vs 0.658 (NELL v1) | `results/e0_zero_shot.json` | |
| The released walk recording differs from Appendix A.2. Each hop is labeled at random as (r, forward) or (r⁻¹, backward), consecutive hops over the same relation share an anonymous ID 50% of the time, and the query-relation flag fires on 50% of hops over that relation. The effect on performance was not measured | `results/e0_walker_check.json`, `results/e0_walker_check_colab.json` | |
| `--seed` alone does not make walks repeatable, because the released code does not seed NumPy. With NumPy seeded, walks are identical across runs, CPU counts and machines | same | |
| On a Colab T4 at n = 512, P = 2 and batch size 1, FLOCK takes about 15 s per Hetionet query and 3.4 GB of GPU memory. This is a timing check on 20 triples, so its MRR is not a result | `results/e0_hetionet_timing.json` | |
| ULTRA's public checkpoint scores a mean MRR of 0.416 here over the 18 graphs in its README, against the official 0.420 (Hits@10 0.569 vs 0.562). Runs in this study are labeled "ULTRA (public checkpoint)" and are not comparable with the ULTRA columns of the papers | `results/ultra_checkpoint_check.json` | |
| On a Kaggle T4 with 4 CPU cores, a Hetionet entity query takes 7.7–7.9 s and 3.3 GB of GPU memory for FLOCK (n = 512, P = 2), about 23 times the 0.34 s and 0.6 GB of ULTRA | `results/gpu_e1_summary.txt` | |

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
| Compound–Disease (56 triples), choosing between *treats* and *palliates*: 50% at random, 64.3% for B and 80.4% for C [59.3%, 95.1%]. Over all 24 relations, C's top choice is right for only 60.7% (34 of 56), because without entity types it can rank a more frequent, unrelated relation first. Too few triples to separate the baselines | same, `results/e6_compound_disease.json` | |
| FLOCK (relation checkpoint, 3 walk seeds) on the same 56 triples: *treats* vs *palliates* Hits@1 94.0% [84.3%, 100%], 13.7 points [1.5, 32.6] above C restricted to the pair. Its 24-way MRR is 0.970 [0.921, 1.000], against 0.657 for C | `results/e6_compound_disease.json` | 2 |
| ULTRA (public checkpoint) on the same 56 triples: *treats* vs *palliates* Hits@1 53.6% [31.2%, 75.7%], no better than chance. Its 24-way MRR is 0.756 [0.641, 0.869] | same | 2 |

The overall FLOCK, TRIX and ULTRA scores are the published ones. This study reran FLOCK and ULTRA only on the
56 Compound–Disease triples, with two-way bootstrap CIs over compounds and diseases.

### E7. Relation similarity

*Pending.*
