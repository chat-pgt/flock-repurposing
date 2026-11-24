# FLOCK on drug repurposing

This repo is a copy of [jw9730/flock](https://github.com/jw9730/flock) (MIT) plus a small study of how
FLOCK performs on drug repurposing with Hetionet. The study lives in `experiments/`.

**Start with** `../experiments-handoff.md` if it exists, then `experiments/PROTOCOL.md` and
`experiments/RESULTS.md`.

## Git

- `main` mirrors upstream. All work happens on `experiments`.
- Remotes: `origin` is the private repo, `upstream` is the original (push disabled).
- Upstream code (`src_*`, `graph-walker`) changes only when an experiment needs it, in small separate
  commits, so `git diff upstream/main -- src_entity src_relation graph-walker` shows every patch.
- Keep personal, employer and interview details out of tracked files. This repo may become public.
- Commits are one-line, atomic and conventional, with no co-author trailer.
- Run `uv run --project experiments pre-commit install` once per clone. Never bypass the hooks.
- Each experiment gets its own branch and a PR into `experiments`. The PR body records the question,
  steps, decisions, result and reproduction commands.

## Rules for experiments

- Nothing runs in full until its card in `PROTOCOL.md` is marked `approved`.
- Don't change a shared decision (section 1 of the protocol) without flagging what must be rerun.
- **Never re-pretrain.** It takes about 3 days. Use `checkpoints/`.
- Smoke-test on about 20 queries, recording time and peak memory, before any long run.
- Save after each fold. Every result JSON records commit, seed and config.
- `RESULTS.md` holds only checked results, including negative ones. Slides take numbers only from it.

## Compute

- **Local machine:** no GPU, 7 GB RAM. CPU analysis only (`uv sync` in `experiments/`).
- **Kaggle T4:** main GPU machine. Not the P100.
- **Colab:** second machine, for ULTRA and debugging. Only 2 CPU cores, and walk sampling is on the CPU.
- GPU environment: see the root `README.md` (PyTorch 2.8, CUDA 12.6, `torch_geometric==2.4.0`,
  `torch-scatter`, `bash install_walker.sh`).

## Known code details

- Relation prediction ranks **unfiltered** over forward relations, although the paper says filtered.
- Ensemble passes run in parallel (`repeat_interleave`), so memory grows with P.
- Walk sampling draws its seed from NumPy, which the released code never seeds. Experiment runners must
  seed it.
- `compute_ranking` counts ties against the target. Experiments use `common/evaluation.py` instead.
- The released code gives r and r⁻¹ different anonymous IDs and labels each hop at random as
  (r, forward) or (r⁻¹, backward), unlike Appendix A.2. `experiments/e0_reproduction/check_walker.py`
  confirms this.
