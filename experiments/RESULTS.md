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

| Result | Evidence | Slide |
|---|---|---|
| For 49.0% of held-out *treats* edges (95% CI [38.3%, 58.6%]), a compound the drug *resembles* treats the same disease in that fold's training graph | `results/e2_leakage.json` | 1 |
| The same rate over diseases the drug is not known to treat is 1.5% uniformly and 4.7% weighted by disease popularity. The excess over the weighted control is 44.3 points [34.3, 53.0], about 10 times the control rate | same | 1 |
| The rate is similar whether the drug keeps another indication in training (47.7%, 457 edges) or not (51.0%, 298 edges) | same | |
| Removing all 6,486 *resembles* edges lowers ULTRA's drug → disease MRR from 0.368 to 0.300 (Hits@10 65.0% → 57.6%). The drop is 0.120 [0.056, 0.202] on flagged edges and 0.017 [−0.005, +0.039] on unflagged ones, an extra 0.103 [0.039, 0.182] on flagged edges | `results/e2_ablation.json` | 1 |
| FLOCK (walk seeds 0 and 1) goes from 0.452 to 0.432 (Hits@10 74.4% → 73.2%). The drop is 0.046 [0.002, 0.095] on flagged edges and −0.004 [−0.034, +0.025] on unflagged ones. The extra drop on flagged edges, +0.050 [−0.005, +0.107], is not distinguishable from zero | same | 1 |

The fraction shows that the shortcut exists in the data, and the ablation shows that ULTRA uses it. FLOCK's
drop is smaller and not clearly concentrated on flagged edges, but the two models were not compared
directly. FLOCK has two walk seeds here, one fewer than D7 requires.

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
