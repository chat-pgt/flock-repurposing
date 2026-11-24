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

| Result | Evidence | Slide |
|---|---|---|
| Popularity baseline, drug → disease (755 *treats* edges, 136 candidate diseases): MRR 0.208, 95% CI [0.083, 0.362]; Hits@10 47.0% [27.9%, 62.5%]. Random: MRR 0.041, Hits@10 7.6% | `results/e1_popularity.json` | 1 |
| Popularity baseline, disease → drug (1,538 candidate compounds): MRR 0.090 [0.039, 0.154]; Hits@10 20.8% [10.4%, 32.6%]. Random: MRR 0.005, Hits@10 0.7% | same | 1 |
| Drug → disease MRR is 0.153 [0.075, 0.242] when the drug keeps another indication in training (457 edges) and 0.293 [0.089, 0.529] when it does not (298 edges) | same | |
| Disease → drug never reaches the top 10 for the 298 edges whose drug has no other training indication | same | |
| Also filtering palliated diseases barely changes drug → disease: MRR 0.209, Hits@10 47.7% | same | |
| The score is concentrated: hypertension supplies 43% of the total reciprocal rank and the top five diseases 77% | `results/e1_popularity_edges.tsv` | |
| FLOCK, drug → disease (3 walk seeds): MRR 0.453 [0.336, 0.557]; Hits@10 74.8% [64.3%, 83.0%]. Above popularity by 0.245 MRR [0.154, 0.328] | `results/e1_models.json` | 1 |
| ULTRA (public checkpoint), drug → disease: MRR 0.368 [0.251, 0.486]; Hits@10 65.0% [52.5%, 75.0%]. Above popularity by 0.160 MRR [0.086, 0.235] | same | 1 |
| FLOCK minus ULTRA, drug → disease, paired per edge: +0.085 MRR [+0.027, +0.146]. The gap comes from drugs with no other training indication, +0.163 [+0.067, +0.273] on 298 edges, against +0.035 [−0.025, +0.093] on the other 457 | same | 1 |
| Disease → drug: FLOCK MRR 0.177 [0.118, 0.249] and ULTRA 0.219 [0.154, 0.300], both above popularity (+0.087 [+0.028, +0.146] and +0.129 [+0.057, +0.207]). ULTRA leads by 0.042 MRR [0.003, 0.082]. When drugs that palliate the disease are also filtered, the MRR gap is −0.015 [−0.052, +0.023], though ULTRA still leads on Hits@10 by 5.9 points [0.5, 11.1] | same | |
| As for popularity, drug → disease scores are higher for drugs with no other training indication: FLOCK 0.560 vs 0.383 MRR, ULTRA 0.398 vs 0.348 | same | |
| Also filtering palliated diseases raises drug → disease MRR to 0.506 for FLOCK and 0.425 for ULTRA. FLOCK's lead stays +0.081 [+0.015, +0.147] | same | |
| Ranking over all 45,158 connected entities (D4's secondary view) leaves MRR and Hits@10 unchanged to three decimals for both models in both directions, so non-disease entities almost never outrank the true disease | same | |
| Walk-seed spread is small: FLOCK's MRR is 0.452, 0.453 and 0.455 for seeds 0–2 drug → disease, and 0.178, 0.175 and 0.177 disease → drug | same | |

ULTRA is deterministic and runs once. FLOCK's per-edge metrics are averaged over three walk seeds. ULTRA's
public checkpoint differs from the ULTRA paper (see E0). Directions are not compared with each other,
because their candidate sets differ.

### E2. Near-duplicate leakage

*Pending.*

### E3. Walk coverage

*Pending.*

### E4. Walk explanations

*Pending.*

### E5. Type-constrained negatives

| Result | Evidence | Slide |
|---|---|---|
| FLOCK pilot, fold 0 (151 edges), one epoch of fine-tuning on *treats*: drug → disease MRR 0.422 zero-shot, 0.471 with uniform negatives and 0.517 with disease-only negatives. Disease-only minus zero-shot: +0.095 [+0.027, +0.170] | `results/e5_negatives_flock.json` | |
| On first-indication edges (53, the primary stratum): 0.542 zero-shot, 0.517 uniform and 0.636 disease-only. Disease-only minus uniform: +0.119 [+0.017, +0.244] | same | |
| Drugs with other training indications (98 edges) gain with both arms, +0.089 [−0.003, +0.179] uniform and +0.096 [+0.011, +0.193] disease-only, so the expected positive-unlabeled cost did not appear | same | |
| Cost: 302 training steps per arm at about 10.7 s each, about 52 minutes per arm, peaking at 4.8 GB of GPU memory on a Kaggle T4 | same | |

This is a pilot: one fold, one fine-tuning seed and one epoch. ULTRA's five-fold run is pending.

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

| Result | Evidence | Slide |
|---|---|---|
| ULTRA gives numerically identical representations to *upregulates* and *downregulates* for Compound, Disease and Anatomy–Gene, and to *treats* and *palliates*, on every query relation outside the pair (relative distance at most 1.3e-7) | `results/e7_relation_similarity.json` | 2 |
| FLOCK's relation states on *treats* queries show no stable up/down separation. A nearest-centroid probe on held-out drugs scores 49.6%, 47.9% and 56.2% for Compound, Disease and Anatomy–Gene (chance is 50%), against a median of 100% on pairs connecting different entity types | same | 2 |
| With a neutral query relation (*Compound causes Side Effect*), FLOCK's states still show no up/down separation: the probe scores 50.0%, 50.4% and 54.6% for Compound, Disease and Anatomy–Gene. *Treats* vs *palliates* separates at 75.8%, against a median of 99% on pairs connecting different entity types | same | |
| As a predictor, FLOCK tells *upregulates* from *downregulates* on 200 balanced Compound–Gene test triples (chance 50%): 88.2% [80.2%, 94.4%] with its relation checkpoint and 3 walk seeds. ULTRA (public checkpoint) scores 52.0% [39.4%, 64.1%] and Baseline C 71.5% [60.7%, 81.5%]. FLOCK leads ULTRA by 36.2 points [23.2, 49.3] and C by 16.7 [4.6, 28.9] | `results/e7_relation_pairs.json` | 2 |
| ULTRA's choices lean to *upregulates*: it is right on 70% of *upregulates* triples and 34% of *downregulates* ones. FLOCK is right on 92% and 84% | same | |

The state probes use FLOCK's entity checkpoint and the prediction test uses its relation checkpoint. FLOCK
separates up from down when predicting, although the probe finds no difference in the entity checkpoint's
relation states. With *treats* as the query relation, FLOCK's *treats* vs *palliates* probe is not
reported, because the query marker separates the two. ULTRA's chance-level result matches E6, where it
also cannot tell *treats* from *palliates*.
