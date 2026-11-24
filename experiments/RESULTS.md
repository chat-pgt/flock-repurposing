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

*Pending.*

### E6. No-learning baselines for relation prediction

*Pending.*

### E7. Relation similarity

*Pending.*
