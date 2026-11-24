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

| Result | Evidence | Slide |
|---|---|---|
| For 49.0% of held-out *treats* edges (95% CI [38.3%, 58.6%]), a compound the drug *resembles* treats the same disease in that fold's training graph | `results/e2_leakage.json` | 1 |
| The same rate over diseases the drug is not known to treat is 1.5% uniformly and 4.7% weighted by disease popularity. The excess over the weighted control is 44.3 points [34.3, 53.0], about 10 times the control rate | same | 1 |
| The rate is similar whether the drug keeps another indication in training (47.7%, 457 edges) or not (51.0%, 298 edges) | same | |
| Removing all 6,486 *resembles* edges lowers ULTRA's drug → disease MRR from 0.368 to 0.300 (Hits@10 65.0% → 57.6%). The drop is 0.120 [0.056, 0.202] on flagged edges and 0.017 [−0.005, +0.039] on unflagged ones, an extra 0.103 [0.039, 0.182] on flagged edges | `results/e2_ablation.json` | 1 |
| FLOCK (walk seeds 0, 1 and 2) goes from 0.453 to 0.429 (Hits@10 74.8% → 73.7%). The drop is 0.047 [0.010, 0.088] on flagged edges and 0.002 [−0.023, +0.027] on unflagged ones. The extra drop on flagged edges, +0.045 [−0.000, +0.093], is at the edge of detection, with a point estimate about half ULTRA's | same | 1 |

The fraction shows that the shortcut exists in the data, and the ablation shows that ULTRA uses it. FLOCK's
drop is also concentrated on flagged edges but smaller, and its excess over unflagged edges is at the edge
of detection. The two models were not compared directly.

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

| Result | Evidence | Slide |
|---|---|---|
| Survey of 40 random held-out edges (one walk seed): 65% of the disease's attention weight comes from walks that never visit the drug, and 27% from walks where the drug is more than 4 hops away. Short drug → disease stretches carry 8%: 5.4% end with another drug's treatment of the disease, 1.4% pass through a gene | `results/e4_walk_explanations.json` | |
| Deleting the facts on the 5 top stretches (8.6 on average) changes the disease's reciprocal rank by −0.154 [−0.303, −0.041]; deleting as many random facts touching the drug or disease changes it by −0.002 [−0.091, +0.093]. Paired difference −0.152 [−0.324, −0.015]: larger on 15 edges, smaller on 7, equal on 18 | same | |
| Acarbose → type 2 diabetes (rank 1 in all seeds): 54% of the weight sits on stretches ending with another drug's treatment of the disease. The top three run through miglitol, in the same pharmacologic class, via the class and shared side effects. Deleting the top stretches lowers the reciprocal rank by 0.83, deleting the disease's other treatments by 0.98 | same | |
| Dasatinib → hematologic cancer, the case chosen to favour a mechanism (reciprocal rank 0.78): the top stretch runs through a shared side effect to tioguanine, which treats the disease. Gene stretches carry 4% of the weight, and deleting all of dasatinib's gene facts does not lower the rank (+0.06). Deleting the disease's other treatments lowers it by 0.73 | same | |
| Valsartan → hypertension stays first whatever is deleted on the drug's side; deleting hypertension's other treatments lowers its reciprocal rank by 0.90 | same | |
| Metformin → type 2 diabetes (reciprocal rank 0.36): the top stretch is metformin treats metabolic syndrome X, which resembles type 2 diabetes. Deletion effects are mixed; removing metformin's gene facts raises the rank by 0.31 | same | |
| The recorded weights rebuild the model's consensus output to within 8.4e-6 on every run | `results/e4_walk_explanations_raw.json` | |

Attention weights alone are not attributions; the deletion tests support that the top stretches matter for
these cases. The survey has one walk seed, the showcase pairs three.

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
