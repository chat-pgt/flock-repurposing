# Experiments

The study of how well FLOCK predicts which diseases a drug treats on Hetionet, and what explains the
result. Two files hold it together:

- [`PROTOCOL.md`](PROTOCOL.md) gives the question, the shared decisions (data, split, candidates,
  metrics, uncertainty, model settings) and one card per experiment. Each card sets out the design and
  what each outcome would mean before the experiment runs.
- [`RESULTS.md`](RESULTS.md) records each checked result, negative ones included, on one line with the
  file it comes from.

## Experiments

| ID | Question | Code |
|---|---|---|
| A1–A4 | Do the paper's tables, the Hetionet graph and the released code support the paper's claims? | `paper_analysis/` |
| E0 | Does the released code reproduce the paper in this environment, and does its walker match Appendix A.2? | `e0_reproduction/` |
| E1 | Do FLOCK and ULTRA beat a popularity baseline at drug → disease prediction? | `e1_baselines/`, `gpu/` |
| E2 | How much of the signal is "a similar drug already treats this disease"? | `e2_leakage/`, `gpu/` |
| E3 | Does walk coverage explain FLOCK's result? | `e3_coverage/` |
| E4 | Do FLOCK's highest-weight walks read like mechanisms or like shortcuts? | `e4_explanations/` |
| E5 | Does fine-tuning on *treats* improve with disease-only negatives? | `e5_negatives/` |
| E6 | How much of FLOCK's relation-prediction score can simple counts reach? | `e6_relation_baseline/`, `gpu/` |
| E7 | Do ULTRA and FLOCK tell *upregulates* from *downregulates*? | `e7_relation_similarity/` |

## Layout

| Path | Contents |
|---|---|
| `common/` | Hetionet download and loading, the shared split, ranking with random tie-breaking, the two-way bootstrap and run provenance |
| `gpu/` | FLOCK and ULTRA loaders, the shared entity and relation runners, analysis of saved runs, and the E1, E2 and E6 notebook builder |
| `e0_*/` to `e7_*/` | One folder per experiment: its runner, its analysis and, for GPU experiments, its notebook builder |
| `paper_analysis/` | CPU analyses of the paper's tables and the released code. See its [README](paper_analysis/README.md) |
| `notebooks/` | Generated GPU notebooks. Rebuild them rather than editing them |
| `splits/` | The five-fold split of the 755 *treats* edges |
| `results/` | One file per result. Each records the commit, seed, config and data checksums |
| `tests/` | Unit tests for the shared code, the baselines and the NumPy-only helpers |

The analysis code needs only NumPy, pandas and SciPy, so everything except the model runs works on a
CPU.

## How a GPU run works

1. A builder script packs the code committed at `HEAD` into a notebook, so commit before building.
2. On a Kaggle or Colab T4, the notebook clones FLOCK and ULTRA at pinned commits, unpacks this
   study's code over them, installs the GPU environment and runs the experiment. Each fold is saved as
   it finishes, and no new fold starts after a time budget.
3. The downloaded outputs are analysed on a CPU with the shared bootstrap, and the checked numbers go
   to `results/` and `RESULTS.md`.

## Reproduce

From `experiments/`:

```bash
uv sync                              # locked CPU environment
uv run pre-commit install            # once per clone
bash common/get_hetionet.sh          # Hetionet v1.0, checked against its checksums
uv run pytest
```

CPU experiments:

```bash
uv run python -m e1_baselines.popularity
uv run python -m e2_leakage.near_duplicates
bash e6_relation_baseline/get_authors_split.sh
uv run python -m e6_relation_baseline.relation_baselines
```

GPU experiments are built into notebooks, run on a hosted T4, then analysed locally:

```bash
uv run python gpu/build_notebooks.py           # E1, E2 and E6
uv run python -m e3_coverage.build_notebook    # likewise e4_explanations, e5_negatives, e7_relation_similarity
uv run python -m gpu.analyze_e1 --help         # each analysis script takes the downloaded runs
```

The pull request that added each experiment lists its full commands. The split in `splits/` is
committed, and `uv run python -m common.make_split` regenerates it.
