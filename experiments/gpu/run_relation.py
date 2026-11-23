"""Score the Compound-Disease test triples with FLOCK or ULTRA as relation queries (E6).

Uses the authors' Hetionet split. Each query (compound, ?, disease) is scored against all 24 forward
relations, as in the released evaluation, and also against *treats* and *palliates* only. FLOCK uses its
relation checkpoint. ULTRA has no relation head, so each candidate relation is scored as a triple.

Usage, from experiments/:
    python -m gpu.run_relation --model flock --seeds 0,1,2 --output-dir runs
    python -m gpu.run_relation --model ultra --output-dir runs
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from common.evaluation import rank_metrics
from common.hetionet import EXPERIMENTS, PALLIATES, TREATS
from gpu.graphs import encode_triples
from gpu.models import build_data, load_flock, load_ultra, peak_gpu_gb, run_metadata, seed_everything, sha256
from gpu.run_entity import save_shard

AUTHOR_FILES = {
    "train.txt": (
        "https://www.dropbox.com/s/y47bt9oq57h6l5k/train.txt?dl=1",
        "cc8edc84e9e62a955affc88e21eeba7f272c04d552e53e780624c5b75ad929b2",
    ),
    "valid.txt": (
        "https://www.dropbox.com/s/a0pbrx9tz3dgsff/valid.txt?dl=1",
        "7c8301fb0e3635e347812111afd02580c2dd6c915dea5c4641d4aff2108f36ed",
    ),
    "test.txt": (
        "https://www.dropbox.com/s/4dhrvg3fyq5tnu4/test.txt?dl=1",
        "53afde5426cb915241887e9a7e83ba979a6bfb43a433cc25e8b552c7ec768728",
    ),
}
ULTRA_RELATIONS_PER_BATCH = 4


def ensure_authors_split(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, checksum) in AUTHOR_FILES.items():
        path = directory / name
        if not path.exists():
            subprocess.run(["curl", "-fsSL", "-o", str(path), url], check=True)
        if sha256(path) != checksum:
            raise ValueError(f"unexpected checksum for {path}")


def read_triples(path):
    with open(path) as handle:
        for line in handle:
            head, relation, tail = line.rstrip("\n").split("\t")
            yield head, relation, tail


@torch.no_grad()
def score_relations(model, model_name, data, head, tail, n_relations, device):
    relations = torch.arange(n_relations, device=device)
    heads, tails = torch.full_like(relations, head), torch.full_like(relations, tail)
    batch = torch.stack([heads, tails, relations], dim=-1)
    if model_name == "flock":
        return model(data, batch[None])[0].float().cpu().numpy()
    parts = [model(data, chunk[:, None])[:, 0] for chunk in batch.split(ULTRA_RELATIONS_PER_BATCH)]
    return torch.cat(parts).float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["flock", "ultra"], required=True)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deadline", type=float, help="Unix time after which no new seed starts")
    parser.add_argument("--ultra-repo", type=Path)
    parser.add_argument("--data-dir", type=Path, default=EXPERIMENTS / "data/hetionet_authors")
    parser.add_argument("--skip-download", action="store_true", help="Use local files without checksums, for tests")
    parser.add_argument("--max-triples", type=int, help="Score only this many triples, for smoke tests")
    parser.add_argument("--walk-num", type=int, help="Override n, for smoke tests only")
    parser.add_argument("--ensemble", type=int, help="Override P, for smoke tests only")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        ensure_authors_split(args.data_dir)

    train = list(read_triples(args.data_dir / "train.txt"))
    valid = list(read_triples(args.data_dir / "valid.txt"))
    test = list(read_triples(args.data_dir / "test.txt"))
    graph = encode_triples(train + valid + test)
    keep = np.arange(len(graph.heads)) < len(train)
    first_test = len(train) + len(valid)
    treats, palliates = graph.relation_names.index(TREATS), graph.relation_names.index(PALLIATES)
    is_target = (graph.relations[first_test:] == treats) | (graph.relations[first_test:] == palliates)
    targets = first_test + np.flatnonzero(is_target)
    if args.max_triples is not None:
        targets = targets[: args.max_triples]

    device = torch.device(args.device)
    if args.model == "flock":
        model, settings = load_flock("relation", device, args.walk_num, args.ensemble)
        seeds = [int(seed) for seed in args.seeds.split(",")]
    else:
        model, settings = load_ultra(args.ultra_repo, device)
        seeds = [0]  # ULTRA is deterministic
    metadata = run_metadata(device, model=args.model, task="compound_disease_relation", **settings)
    data = build_data(graph, keep, args.model, device).to(device)
    n_relations = len(graph.relation_names)
    two_way = np.zeros(n_relations, dtype=bool)
    two_way[[treats, palliates]] = True

    for seed in seeds:
        path = args.output_dir / f"relation_{args.model}_seed{seed}.npz"
        if path.exists():
            print(f"skip {path.name}", flush=True)
            continue
        if args.deadline is not None and time.time() > args.deadline:
            print(f"time budget reached before {path.name}", flush=True)
            return
        seed_everything(seed)
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        scores = np.stack(
            [
                score_relations(
                    model, args.model, data, int(graph.heads[row]), int(graph.tails[row]), n_relations, device
                )
                for row in targets
            ]
        )
        seconds = time.perf_counter() - start
        true_relations = graph.relations[targets]
        all_relations = np.ones(n_relations, dtype=bool)
        full = [rank_metrics(s, int(r), all_relations, k=1) for s, r in zip(scores, true_relations, strict=True)]
        pair = [rank_metrics(s, int(r), two_way, k=1) for s, r in zip(scores, true_relations, strict=True)]
        info = {**metadata, "seed": seed, "seconds": seconds, "peak_gpu_gb": peak_gpu_gb(device)}
        save_shard(
            path,
            metadata=np.array(json.dumps(info)),
            heads=np.array([graph.entities[node] for node in graph.heads[targets]]),
            tails=np.array([graph.entities[node] for node in graph.tails[targets]]),
            relation_names=np.array(graph.relation_names),
            true_relations=true_relations,
            scores=scores.astype(np.float32),
            all_relations_rr=np.array([value[0] for value in full]),
            all_relations_hits1=np.array([value[1] for value in full]),
            treats_vs_palliates_rr=np.array([value[0] for value in pair]),
            treats_vs_palliates_hits1=np.array([value[1] for value in pair]),
        )
        print(
            f"{path.name}: {len(targets)} triples, 24-way MRR {np.mean([v[0] for v in full]):.3f}, "
            f"treats vs palliates Hits@1 {np.mean([v[1] for v in pair]):.3f}, "
            f"{seconds / len(targets):.2f} s/triple, peak GPU {info['peak_gpu_gb']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
