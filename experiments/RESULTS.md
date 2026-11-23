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

FLOCK and ULTRA have not been run on these folds yet.

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
