# FLOCK on drug repurposing: experiment protocol

[FLOCK](https://arxiv.org/abs/2510.01510) (Kim et al., ICLR 2026) is a knowledge graph foundation
model. It is pretrained on a few graphs and then applied to graphs it has never seen, including their
entities and relation types, by learning on random walks. This study tests how well the released model
carries over to drug repurposing on [Hetionet](https://het.io), a public biomedical knowledge graph.

## Background and question

Repurposing can be posed as link prediction: given a drug, rank the diseases it might treat. Hetionet
v1.0 has 47,031 nodes (45,158 with at least one edge), 24 relation types and about 2.25 million edges,
of which 755 are curated Compound–*treats*–Disease edges.

The paper's zero-shot results on Hetionet (MRR, higher is better) point in two directions:

| Task | ULTRA | TRIX | FLOCK |
|---|---|---|---|
| Entity prediction | 0.257 | 0.279 | 0.246 |
| Relation prediction | 0.634 | 0.809 | 0.940 |

FLOCK is the weakest of the three at finding a missing entity and the strongest at identifying a
missing relation. Both figures average over all 24 relations, so neither shows how it performs on
*treats*. The study therefore asks:

> **How well does FLOCK predict which diseases a drug treats, and what explains the result?**

| Sub-question | Experiments |
|---|---|
| How well does it perform, and how much comes from easy signal? | E1, E2 |
| What explains the result? | E6, E7, E3 |
| Extensions: harder negatives, example explanations | E5, E4 |

## Where the experiments come from

The experiments began as questions raised while reading the paper, its appendices, its public reviews
and the released code. Each card records its starting question under *Origin*. They check specific
claims and design choices, and are not intended as a new benchmark.

---

## 1. Shared decisions

These apply to every experiment unless its card says otherwise. Each run saves per-query disease scores
and the target's rank counts among all entities, so D5, D6 and D7 can be revised without rerunning a
model. Changing D1–D4, D8 or D9 means rerunning everything that depends on them.

| ID | Decision | Rationale | Status |
|---|---|---|---|
| D1 | **Data.** Official Hetionet v1.0, verified by checksum (`common/get_hetionet.sh`). | A fixed public release. | agreed |
| D2 | **Split.** Five folds over the 755 *treats* edges, stratified by disease, with a fixed seed. A fold's edges and their inverses are removed from the graph. Results are reported pooled, and separately for drugs with and without another known indication in that fold's training graph. E0 and E6 use the authors' split instead. | 258 edges are their drug's only indication. Holding one out asks for a first indication, which is a different task from finding a new use. E0 and E6 must stay comparable with the paper. | agreed |
| D3 | **Queries.** Primary: tail queries, (compound, *treats*, ?). Secondary: head queries, (?, *treats*, disease), ranked over the 1,538 connected compounds and run after the core experiments. | Tail queries ask "which diseases could this drug treat?" and head queries ask "which drugs could treat this disease?". Each is read against its own random baseline because the candidate sets differ. The paper averages head and tail queries over all relations, so its Hetionet figures are not directly comparable. | agreed |
| D4 | **Candidates.** Primary (the headline result): the 136 diseases in the graph. Secondary (a supporting view from the same run): all 45,158 connected entities. | A drug's indication is always a disease. The secondary view shows the effect of the paper's larger candidate set. | agreed |
| D5 | **Filtering.** Remove other known *treats* pairs from the ranking. Diseases the drug *palliates* stay as competitors. A version that also removes them is reported as a sensitivity check. | The two relations never overlap in Hetionet, so palliated diseases are curated examples of "related, but not disease-modifying". Removing them would hide that error. | agreed |
| D6 | **Metrics.** MRR and Hits@10 with the exact random baseline. Ties are broken uniformly at random, in expectation, for every model. | The released ranking counts every tie against the target. That would penalise a popularity baseline, where at least 59 diseases share a count of zero. | agreed |
| D7 | **Uncertainty.** At least three walk seeds for stochastic models, averaged per edge. 95% CIs from a two-way bootstrap that resamples drugs and diseases independently (Owen, 2007). Models are compared through paired per-edge differences. | Edges are dependent along both axes: 10 of the 77 treated diseases hold 43% of edges, and 66% of edges come from drugs with several indications. | agreed |
| D8 | **Models.** Released checkpoints pretrained on FB15k-237, WN18RR and CoDEx-Medium, used zero-shot. ULTRA's public checkpoint on the same three graphs, at a pinned commit. | Same pretraining data for both models. Training-seed variance cannot be measured from a single checkpoint, and results must state this. The public ULTRA checkpoint comes from a PyG reimplementation, so its per-graph scores differ from the ULTRA numbers in the papers. | agreed |
| D9 | **FLOCK settings.** The paper's zero-shot Hetionet setting (n = 512, P = 2), on which Table 18 and the released config agree. A change forced by memory applies to all folds and seeds, and is reported. | Keeps results comparable with the paper and across folds. Fine-tuning is less clear: Table 18 states P = 1 and n = 1024, while the released config runs P = 2 at n = 1024. E5 therefore evaluates at the zero-shot setting. | agreed |

---

## 2. Experiments

Each card gives the origin, question, design, how to read the outcome, and the compute needed.
Priorities are *core*, *extension* and *optional*.

### E0. Reproduction and environment (core)

- **Origin.** Every later result relies on the released code behaving as the paper describes. Reading
  the walker next to Appendix A.2 also raised a doubt about how a relation and its inverse are recorded.
- **Question.** Does the released code reproduce the paper in this environment?
- **Design.**
  1. Build the walker and run it on the Figure 1 graph to check how r and r⁻¹ are recorded.
  2. Check that two runs with the same seed produce identical walks.
  3. Run zero-shot entity prediction on `FB15k237Inductive:v1`, `WN18RRInductive:v1` and
     `NELLInductive:v1`, and compare with Table 9.
  4. Time 20 Hetionet queries and record peak memory.
- **Read-out.** Agreement with Table 9 within about 0.01 validates the setup. If the GPU environment
  cannot be built, the CPU experiments (E2, E6 and E1's popularity baseline) still stand.
- **Compute.** CPU for the walker checks, GPU for the rest.
- **Status.** Done. Table 9 reproduced within tolerance on all three graphs.

### E1. FLOCK vs ULTRA vs popularity (core)

- **Origin.** Is learning on random walks the right approach for repurposing, and do benchmark gains
  mean anything against simple or purpose-built alternatives?
- **Question.** Do the foundation models beat a baseline with no learning on *treats* prediction?
- **Design.** On the shared folds, compare FLOCK, ULTRA and a popularity baseline that ranks candidates
  by their number of *treats* edges in the training graph. Tail queries run first. Head queries follow
  once the core experiments are done, under the same rule: other drugs known to treat the disease are
  filtered, and drugs that palliate it stay as competitors.
- **Read-out.** If popularity comes close, the task mostly rewards hub diseases (tail) or drugs with many
  indications (head). Raw scores are not compared across directions.
- **Compute.** CPU for popularity, GPU for the models.
- **Status.** Popularity baseline approved and run. FLOCK (three walk seeds) and ULTRA run in both directions on a Kaggle T4. Results reviewed.

### E2. Near-duplicate leakage (core)

- **Origin.** A predicted indication can look like a discovery when a close analogue already has it, as
  with ketamine and esketamine. How often does that happen in a random split?
- **Question.** How much of the apparent repurposing signal is "a similar drug already treats this"?
- **Design.** For each held-out (drug, *treats*, disease), check whether a drug it *resembles* treats
  that disease in the training graph, and report the fraction. If GPU time allows, remove the
  *resembles* edges and rerun E1.
- **Read-out.** A large fraction, or a large drop without *resembles*, means performance depends on
  near-duplicates.
- **Compute.** CPU for the fraction, GPU for the ablation.
- **Status.** Not started.

### E3. Walk coverage (optional)

- **Origin.** The paper says shorter walks degrade results "as expected from coverage". Why is that
  expected, and what justifies the choice of walk length ℓ and walk count n?
- **Question.** Does walk coverage explain FLOCK's Hetionet result?
- **Design.** Sweep (n, ℓ) with n·ℓ fixed, and vary n alone. Separately, remove high-volume relations
  that lie off the drug–disease paths (for example Anatomy–*expresses*–Gene). With the released
  checkpoint, ℓ > 128 needs retraining and ℓ < 128 needs a small patch.
- **Read-out.** If performance tracks n·ℓ, coverage matters. If it does not move, the explanation lies
  elsewhere. The relation-removal arm shows whether the design of the graph matters.
- **Compute.** GPU.
- **Status.** Not started.

### E4. Walk explanations (extension)

- **Origin.** If this approach suits repurposing, its walks should read like candidate mechanisms
  (drug → gene → disease).
- **Question.** Do high-confidence walks look like plausible mechanisms?
- **Design.** For one well-known drug–disease pair, extract the highest-confidence walks and map them
  back to entity names.
- **Read-out.** Illustrative only, because confidence weights are not faithful attributions.
- **Compute.** GPU.
- **Status.** Not started. To be built now that the E1 and E2 results are in.

### E5. Type-constrained negatives in fine-tuning (extension)

- **Origin.** Working through how fine-tuning operates: its 256 negatives are drawn uniformly over about
  45,000 nodes, so on average fewer than one is a disease.
- **Question.** Does fine-tuning on *treats* improve with disease-only negatives?
- **Design.** Each fold starts from the released checkpoint and trains on that fold's training *treats*
  facts (about 604), corrupting tails only: 3 epochs for ULTRA, scored after each, and 1 for FLOCK. The two
  arms differ only in where the 256 negatives come from: every entity, as in the released sampler, or
  diseases only. Both draw with replacement, so the disease arm draws each of the roughly 135 eligible
  diseases about twice per fact. Both arms exclude the drug's training indications and see the same
  batches and walks. Training negatives are filtered against the
  training graph only, so held-out edges can be drawn as negatives, as would happen in practice. Other
  settings are the published ones: AdamW, learning rate 5e-4, self-adversarial temperature 1, batch 8
  for ULTRA and 2 for FLOCK. The epochs are fixed in advance, every scored epoch is reported, and none is
  selected or tuned on held-out folds.
  Scoring is E1's, at the D9 setting, with walks reseeded so that FLOCK's scoring walks match E1's seed
  0. Each arm is compared, paired per edge, with E1's zero-shot run on the same edges.
  - ULTRA (public checkpoint), all five folds. Its kernel replacement gets a backward pass that
    recomputes messages in chunks, so training fits in memory.
  - FLOCK, a pilot on fold 0. Measured on CPU, one training query keeps about 0.023·n GB of activations,
    so the published n = 1024 with batch 2 would need about 48 GB. FLOCK therefore trains at n = 512 with
    one query per micro-batch, accumulated to batch 2, and its sequence network is recomputed in the
    backward pass (about 2.5 GB). Checkpointing leaves gradients unchanged. n = 512 rather than 1024
    and a single epoch keep each run near an hour.

  Unlike the published fine-tuning, which uses all relations, this fine-tunes on *treats* only, so it
  tests task-specific fine-tuning.
- **Read-out.** A gain from disease-only negatives over uniform ones supports better negatives. The
  uniform arm separates the effect of fine-tuning itself from that of the negatives. The primary read-out
  is on edges whose drug has no other training indication: those drugs never appear in fine-tuning, so any
  change is generalization. For the other edges, the disease arm draws the held-out disease as a negative
  for the drug in about 85% of the drug's training facts, against 0.6% with uniform negatives. A loss there
  measures the cost of pushing down true but unrecorded indications (positive-unlabeled learning). With one
  fine-tuning seed and one fold for FLOCK, this is a proof of concept for FLOCK and a narrow test for ULTRA.
- **Compute.** GPU. ULTRA: 10 runs of about 228 steps, roughly 3 hours. FLOCK: 2 runs of about 302 steps,
  roughly 2 hours.
- **Status.** Approved. FLOCK pilot run on a Kaggle T4, results recorded, awaiting review. ULTRA running.

### E6. No-learning baseline for relation prediction (core)

- **Origin.** The paper attributes FLOCK's large relation-prediction gain to its joint encoding of
  entities and relations. Is there evidence for that, or could simpler signal explain it?
- **Question.** How much of FLOCK's 0.940 relation MRR on Hetionet (TRIX 0.809, ULTRA 0.634) can entity
  types and relation frequencies explain?
- **Design.** Match the released evaluation on the authors' test triples: **unfiltered** ranking over
  forward relations only.
  - Baseline A: uniform among relations compatible with (type(h), type(t)). With k candidates, the
    expected reciprocal rank is H_k / k and Hits@1 is 1/k.
  - Baseline B: rank compatible relations by training frequency.
  - Baseline C: use no entity-type labels, which the models do not receive. Rank first the relations
    that h has used as a head and t as a tail in training, ordering each group by training frequency.
  - Report overall and per type pair, including Compound–Disease (*treats* vs *palliates*).
- **Read-out.** If B or C approaches 0.94, relation prediction here is mostly graph signature and
  frequency. C is the fairer comparison, because it uses only what the models also see. If FLOCK stays
  clearly above it, FLOCK is separating relations beyond those counts.
- **Compute.** CPU.
- **Status.** Approved and run. FLOCK and ULTRA also run on the 56 Compound–Disease triples, results reviewed.

### E7. Does ULTRA merge up- and downregulates? (optional)

- **Origin.** Earlier foundation models assume that structurally isomorphic relations mean the same
  thing. Does this hold on drug graphs, where opposite relations such as *upregulates* and
  *downregulates* connect the same entity types?
- **Question.** Does this failure, the paper's like/dislike example, actually occur on a pharmacological
  graph?
- **Design.** Extract relation representations on the full graph for the query relation *treats*. ULTRA
  is deterministic. FLOCK uses 20 sampled drugs, three walk seeds and both ensemble passes. Compute cosine
  similarity for same-type pairs (Compound, Disease and Anatomy up/down–Gene; *treats* vs *palliates*)
  and for control pairs that connect different entity types, both raw and after subtracting the mean
  relation. Add a PCA plot like the paper's Figure 6.
  A follow-up repeats FLOCK with a neutral query relation (*Compound causes Side Effect*), and scores 100
  *upregulates* and 100 *downregulates* Compound–Gene test triples as relation queries, where chance is 50%.
- **Read-out.** ULTRA cosine near 1, well above the controls, demonstrates the paper's motivation on
  pharmacological data. Values close to the controls mean the problem does not bind on Hetionet.
- **Compute.** GPU, about 60 FLOCK queries, plus about 2 hours for the follow-up.
- **Status.** First run and follow-up done on a Kaggle T4 and reviewed.

---

## 3. Reproducibility

- **CPU environment.** `uv sync` in `experiments/` installs the locked analysis environment.
- **Commit hooks.** `uv run pre-commit install`, once per clone. The hooks strip notebook outputs, run
  ruff and the tests, verify the lockfile, block large files and private keys, and check that commit
  messages are one-line and conventional. The same file checks run on every pull request.
- **GPU environment.** The base image fixes PyTorch and CUDA, so E0 records the exact packages with
  `pip freeze` in `requirements-gpu.txt`.
- **GPU runs.** `gpu/build_notebooks.py` builds self-contained notebooks in `notebooks/` from a commit.
  They use pinned FLOCK and ULTRA commits and save each fold as it finishes.
- **Seeds.** The released code does not seed NumPy, which drives walk sampling, so `--seed` alone does
  not make walks repeatable. Experiment runners seed it explicitly, and E0 verifies the result.
- **Artifacts.** Each result JSON records the commit, seed, config, environment and checkpoint hash.
- **Upstream code.** `src_*` and `graph-walker` change only when an experiment requires it, in separate
  commits.

## 4. Workflow

Each experiment is developed on its own branch and merged into `experiments` through a pull request.
The PR description records the question, steps, decisions, result and the commands to reproduce it.

1. Review the card and mark it `approved`, including what each outcome would mean.
2. Smoke-test on about 20 queries, recording time and peak memory.
3. Run in full, saving after each fold.
4. Save JSON to `results/` and figures to `figures/`.
5. Check the result, then add one line to `RESULTS.md`.
