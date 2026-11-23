# Experiment protocol

**Status: draft.** Settle section 1 before running anything. Each experiment card is then reviewed
and marked `approved` before its full run.

**Question.** Does FLOCK's benchmark advantage carry over to drug repurposing?

| Slide | Message | Evidence |
|---|---|---|
| 1 | How well does it perform, and how much is leakage? | E1, E2 |
| 2 | Why? | E6, E7, E3 |
| Backup | Harder negatives, example explanations | E5, E4 |

---

## 1. Shared decisions

Changing one of these after results exist means rerunning everything that depends on it.

| ID | Decision | Proposed default | Status |
|---|---|---|---|
| D1 | Data | Official Hetionet v1.0 (`common/get_hetionet.sh`) | proposed |
| D2 | Split | 5 folds over the Compound–*treats*–Disease edges, fixed seed. A fold's edges are removed from the graph, including their inverses. Reproduction (E0) alone uses the authors' split | proposed |
| D3 | Queries | Tail-only: (compound, *treats*, ?) | proposed |
| D4 | Candidates | Primary: diseases only. Secondary: all entities (the paper's protocol) | proposed |
| D5 | Filtering | Remove other known *treats* **and** *palliates* pairs from the ranking | proposed; judgment call |
| D6 | Metrics | MRR and Hits@10, with the random baseline shown for scale (Hits@10 ≈ 7% among about 137 diseases) | proposed |
| D7 | Uncertainty | At least 3 walk seeds for stochastic models. Bootstrap CIs **grouped by disease** | proposed; judgment call |
| D8 | Models | Released 3-graph checkpoints, zero-shot. Training-seed variance can't be measured, and the slides must say so | proposed |
| D9 | FLOCK settings | Paper's Hetionet setting (n = 512, P = 2). Any change forced by memory is reported | proposed |

---

## 2. Experiments

Each card: question, design, how to read the outcome, compute.

### E0. Reproduction and environment (must)

- **Question.** Does the released code reproduce the paper here?
- **Design.**
  1. Build the walker and run it on the Figure 1 graph to check how r and r⁻¹ are recorded.
  2. Zero-shot entity prediction on 2–3 small inductive KGs (`FB15k237Inductive:v1`,
     `WN18RRInductive:v1`, `NELLInductive:v1`), compared with Table 9.
  3. Time 20 Hetionet queries and record peak memory before any long run.
- **Read-out.** Matching Table 9 within about 0.01 validates the setup. If the walker doesn't build by
  Tuesday night, drop the GPU work and keep the CPU experiments.
- **Compute.** Local for the walker check, Kaggle T4 for the rest.
- **Status.** Not started.

### E1. FLOCK vs ULTRA vs popularity (must)

- **Question.** Do they beat a baseline with no learning on *treats* prediction?
- **Design.** On the shared folds: FLOCK, ULTRA, and a popularity baseline ranking diseases by how many
  training drugs treat them.
- **Read-out.** If popularity is close, the benchmark mostly rewards hub diseases, and that's the
  finding.
- **Compute.** Popularity is local. FLOCK on Kaggle T4, ULTRA on Colab or Kaggle.
- **Status.** Not started.

### E2. Near-duplicate leakage (must)

- **Question.** How much of "repurposing" is "a similar drug already treats this"?
- **Design.** For each test (drug, treats, disease), check whether a drug it *resembles* treats that
  disease in training, and report the fraction. With GPU time, drop *resembles* edges and rerun E1.
- **Read-out.** A large fraction, or a large drop without *resembles*, means performance depends on
  near-duplicates.
- **Compute.** First part local; ablation on GPU.
- **Status.** Not started.

### E3. Coverage (could)

- **Question.** Does walk coverage explain FLOCK's Hetionet result?
- **Design.** Sweep (n, ℓ) keeping n·ℓ fixed, and vary n alone. Separately, remove high-volume relations
  off the drug–disease paths (e.g. Anatomy–expresses–Gene). With the released checkpoint, ℓ > 128 needs
  retraining and ℓ < 128 needs a small patch.
- **Read-out.** If performance tracks n·ℓ, coverage matters. If it doesn't move, the explanation is
  elsewhere. Removing relations tests whether KG design matters.
- **Compute.** Kaggle T4.
- **Status.** Not started.

### E4. Walk explanations (stretch)

- **Question.** Do high-confidence walks look like plausible mechanisms?
- **Design.** For one well-known pair, dump the highest-confidence walks and map them back to names.
- **Read-out.** Illustrative only: confidence weights aren't faithful attributions.
- **Compute.** Kaggle T4.
- **Status.** Not started.

### E5. Type-constrained negatives in fine-tuning (could)

- **Question.** Does fine-tuning improve with disease-only negatives?
- **Design.** For *treats* positives, draw tail negatives from diseases only, and compare with uniform
  negatives on the same folds. Run ULTRA first, then FLOCK if it fits. Training negatives are filtered
  against the training graph only, so held-out edges can be drawn as negatives.
- **Read-out.** A gain supports better negatives. A loss may reflect pushing down true unknown
  indications (positive-unlabeled learning).
- **Compute.** GPU; the most expensive experiment and the first to cut.
- **Status.** Not started.

### E6. No-learning baseline for relation prediction (must)

- **Question.** How much of FLOCK's 0.940 relation MRR on Hetionet (TRIX 0.809, ULTRA 0.634) can entity
  types and frequencies explain?
- **Design.** Match the released evaluation: **unfiltered** ranking over forward relations only.
  - Baseline A: uniform among relations compatible with (type(h), type(t)). With k candidates, expected
    reciprocal rank is H_k / k and Hits@1 is 1/k.
  - Baseline B: rank compatible relations by training frequency.
  - Report overall and per type pair.
- **Read-out.** If B approaches 0.94, relation prediction here is mostly types plus frequencies. If not,
  FLOCK really separates same-type relations.
- **Compute.** Local.
- **Status.** Not started.

### E7. Does ULTRA merge up- and downregulates? (could)

- **Question.** Does the like/dislike problem actually occur on a pharma KG?
- **Design.** Extract relation representations from ULTRA and FLOCK for a fixed query relation. Compute
  cosine similarity for same-type pairs (Compound, Disease and Anatomy up/down–Gene; *treats* vs
  *palliates*) and control pairs. Add a PCA plot like the paper's Fig. 6.
- **Read-out.** ULTRA cosine ≈ 1 demonstrates the paper's motivation on pharma data. Clearly below 1
  means the problem doesn't bind on Hetionet.
- **Compute.** Colab or Kaggle T4, about 2 hours.
- **Status.** Not started.

---

## 3. Loop for each experiment

1. Review the card and mark it `approved`, including what each outcome would mean.
2. Smoke-test on about 20 queries, and record time and peak memory.
3. Run in full, saving after each fold.
4. Save JSON to `results/` with commit, seed and config, and figures to `figures/`.
5. Check the result, then add one line to `RESULTS.md`.
