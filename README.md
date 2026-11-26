# FLOCK on drug repurposing

This repository tests how well [FLOCK](https://arxiv.org/abs/2510.01510), a knowledge graph foundation
model, carries over to drug repurposing on [Hetionet](https://het.io), a public biomedical knowledge
graph. It is a copy of the authors' code, [jw9730/flock](https://github.com/jw9730/flock), with a study
added in [`experiments/`](experiments/).

## What it asks

Repurposing can be posed as link prediction: given a drug, rank the diseases it might treat. The paper
reports zero-shot Hetionet scores averaged over all 24 relations. There, FLOCK is the weakest of three
models at entity prediction and the strongest at relation prediction, and neither score shows how it
does on *treats*. The study asks:

> **How well does FLOCK predict which diseases a drug treats, and what explains the result?**

## How

- **Released checkpoints only.** FLOCK's checkpoints are used zero-shot, without re-pretraining. They
  are compared with ULTRA's public checkpoint, which was pretrained on the same three graphs, and with
  no-learning baselines built from popularity, entity types and relation counts.
- **Shared evaluation.** Drug → disease experiments score every model on the same five-fold split of
  the 755 *treats* edges, stratified by disease. Results are reported pooled, and separately for drugs
  with and without another known indication in training. Ties are broken at random, and 95% confidence
  intervals come from a two-way bootstrap over drugs and diseases.
- **Fixed before running.** Each experiment (E0–E7) has a card in
  [`PROTOCOL.md`](experiments/PROTOCOL.md) that sets its question, its design and what each outcome
  would mean before it runs. [`RESULTS.md`](experiments/RESULTS.md) records only checked results,
  including negative ones.
- **Free GPUs.** CPU analyses run locally. GPU runs use generated, self-contained notebooks for a
  Kaggle or Colab T4. Each notebook clones FLOCK and ULTRA at pinned commits, unpacks this study's code
  over them and saves each fold as it finishes. Every result file records the commit, seed, config and
  data checksums that produced it.

## Layout

| Path | Contents |
|---|---|
| [`experiments/`](experiments/) | The study: protocol, results, code and tests. See its [README](experiments/README.md) |
| `src_entity/`, `src_relation/`, `src_synthetic/`, `graph-walker/`, `scripts/`, `checkpoints/`, `generate_petals.py`, `install_walker.sh` | The authors' code and checkpoints |
| `.github/`, `.pre-commit-config.yaml` | Checks run on every commit and pull request: notebook output stripping, lint, tests and commit format |
| `CLAUDE.md` | Working notes for coding agents in this repository |

The authors' code is unchanged from
[`f35103d`](https://github.com/jw9730/flock/tree/f35103d25a78bdf4075de5c673a51de4979aa4d7). The
`experiments` branch holds the study, and `main` mirrors the original repository.

FLOCK is the work of Kim et al. (ICLR 2026) and is released under the MIT license, kept in
[`LICENSE`](LICENSE). Their README follows, unchanged apart from heading levels.

---

## Original README: Flock (PyTorch)

[![arXiv](https://img.shields.io/badge/arXiv-2510.01510-b31b1b.svg)](https://arxiv.org/abs/2510.01510) \
**Flock: A Knowledge Graph Foundation Model via Learning on Random Walks** \
[Jinwoo Kim*](https://jw9730.github.io), [Xingyue Huang*](https://www.cs.ox.ac.uk/people/xingyue.huang/), [Krzysztof Olejniczak](https://scholar.google.com/citations?user=0DqGU5oAAAAJ&hl=en), [Kyungbin Min](https://vllab.kaist.ac.kr/members.php), [Michael Bronstein](https://www.cs.ox.ac.uk/people/michael.bronstein/), [Seunghoon Hong](https://maga33.github.io), [İsmail İlkan Ceylan](https://www.cs.ox.ac.uk/people/ismaililkan.ceylan/) (* equal contribution) \
ICLR 2026

![image-flock](./docs/flock.png)

This codebase contains training and evaluation scripts for Flock, including pretraining scripts and checkpoints on FB15k-237, WN18RR, and CoDEx Medium, zero-shot and finetuning evaluation over 54 knowledge graphs, and our new synthetic dataset Petals.

The codebase has been tested with NVIDIA H100 GPUs.

### Setup

We recommend using the official PyTorch Docker image with CUDA support.

```bash
docker pull pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel
docker run -it --gpus all --ipc host --name flock -v /home:/home pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel bash
```

Assuming the codebase is located at `~/flock` inside Docker container, install the required packages:

```bash
cd ~/flock
pip3 install torch_geometric==2.4.0 easydict pybind11 pyyaml jinja2
pip3 install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
bash install_walker.sh
```

### Pretraining and Evaluation

Entity prediction pretraining

```bash
cd ~/flock
python3 src_entity/pretrain.py --config src_entity/config/pretrain_3g.yaml --gpus [0]
```

Entity prediction evaluation (using the checkpoint at `checkpoints/flock_entity.pth`)

```bash
cd ~/flock
mkdir -p ~/flock/output-entity

# zero-shot
CKPT=~/flock/checkpoints/flock_entity.pth bash scripts/entity_zeroshot.sh

# finetuning
CKPT=~/flock/checkpoints/flock_entity.pth bash scripts/entity_finetune.sh
```

Relation prediction pretraining

```bash
cd ~/flock
python3 src_relation/pretrain.py --config src_relation/config/pretrain_3g.yaml --gpus [0]
```

Relation prediction evaluation (using the checkpoint at `checkpoints/flock_relation.pth`)

```bash
cd ~/flock
mkdir -p ~/flock/output-relation

# zero-shot
CKPT=~/flock/checkpoints/flock_relation.pth bash scripts/relation_zeroshot.sh

# finetuning
CKPT=~/flock/checkpoints/flock_relation.pth bash scripts/relation_finetune.sh
```

### Synthetic Dataset Generation

Execute the following commands to generate the Petals synthetic dataset at `data/petals/`:

```bash
python3 generate_petals.py
```

### Synthetic Dataset Experiments

```bash
cd ~/flock

# flock
python3 src_synthetic/run_petals_flock.py --config src_synthetic/config/flock.yaml --gpus [0]

# ultra
python3 src_synthetic/run_petals_ultra.py --config src_synthetic/config/ultra.yaml --gpus [0]

# motif
python3 src_synthetic/run_petals_motif.py --config src_synthetic/config/motif.yaml --gpus [0]

# trix
python3 src_synthetic/run_petals_trix.py --config src_synthetic/config/trix.yaml --gpus [0]

# trix + noise injection
python3 src_synthetic/run_petals_trix.py --config src_synthetic/config/trix_noise_injection.yaml --gpus [0]
```

### References

Our implementation is based on the code from the following repositories:

- [RWNN](https://github.com/jw9730/random-walk) and [graph-walker](https://github.com/kerighan/graph-walker) for random walks and their anonymizations
- [MOTIF](https://github.com/HxyScotthuang/MOTIF) and [ULTRA](https://github.com/DeepGraphLearning/ULTRA) for entity prediction experiments
- [TRIX](https://github.com/yuchengz99/TRIX) for relation prediction experiments

### Citation

If you find our work useful, please consider citing it:

```bib
@article{kim2025flock,
  author    = {Jinwoo Kim and Xingyue Huang and Krzysztof Olejniczak and Kyungbin Min and Michael Bronstein and Seunghoon Hong and İsmail İlkan Ceylan},
  title     = {Flock: A Knowledge Graph Foundation Model via Learning on Random Walks},
  journal   = {arXiv},
  volume    = {abs/2510.01510},
  year      = {2025},
  url       = {https://arxiv.org/abs/2510.01510}
}
```
