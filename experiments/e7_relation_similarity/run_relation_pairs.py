"""E7 follow-up: can a model tell two same-type relations apart when predicting them?

Scores a balanced sample of the authors' Hetionet test triples for a pair of relations, by default
Compound *upregulates* and *downregulates* Gene. Each query (h, ?, t) is scored against all 24 forward
relations, so the pair can be read directly. With a balanced sample, chance is 50%.

Usage, from experiments/:
    python -m e7_relation_similarity.run_relation_pairs --model flock --seeds 0,1,2 --output-dir e7
    python -m e7_relation_similarity.run_relation_pairs --model ultra --ultra-repo ULTRA --output-dir e7
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common.evaluation import rank_metrics
from common.hetionet import EXPERIMENTS
from e7_relation_similarity.similarity import sample_balanced
from gpu.graphs import encode_triples
from gpu.models import build_data, load_flock, load_ultra, peak_gpu_gb, run_metadata, seed_everything
from gpu.run_entity import save_shard
from gpu.run_relation import ensure_authors_split, read_triples, score_relations


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["flock", "ultra"], required=True)
    parser.add_argument("--pair", default="CuG,CdG")
    parser.add_argument("--per-relation", type=int, default=100)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deadline", type=float, help="Unix time after which no new seed starts")
    parser.add_argument("--ultra-repo", type=Path)
    parser.add_argument("--data-dir", type=Path, default=EXPERIMENTS / "data/hetionet_authors")
    parser.add_argument("--skip-download", action="store_true", help="Use local files without checksums, for tests")
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
    pair_names = args.pair.split(",")
    pair = [graph.relation_names.index(name) for name in pair_names]
    targets = first_test + sample_balanced(graph.relations[first_test:], pair, args.per_relation)

    device = torch.device(args.device)
    if args.model == "flock":
        model, settings = load_flock("relation", device, args.walk_num, args.ensemble)
        seeds = [int(seed) for seed in args.seeds.split(",")]
    else:
        model, settings = load_ultra(args.ultra_repo, device)
        seeds = [0]  # ULTRA is deterministic
    metadata = run_metadata(device, model=args.model, task="relation_pair", pair=pair_names, **settings)
    data = build_data(graph, keep, args.model, device).to(device)
    n_relations = len(graph.relation_names)
    everyone = np.ones(n_relations, dtype=bool)
    in_pair = np.zeros(n_relations, dtype=bool)
    in_pair[pair] = True

    for seed in seeds:
        path = args.output_dir / f"pair_{pair_names[0]}_{pair_names[1]}_{args.model}_seed{seed}.npz"
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
        full = [rank_metrics(s, int(r), everyone, k=1) for s, r in zip(scores, true_relations, strict=True)]
        restricted = [rank_metrics(s, int(r), in_pair, k=1) for s, r in zip(scores, true_relations, strict=True)]
        info = {**metadata, "seed": seed, "seconds": seconds, "peak_gpu_gb": peak_gpu_gb(device)}
        save_shard(
            path,
            metadata=np.array(json.dumps(info)),
            heads=np.array([graph.entities[node] for node in graph.heads[targets]]),
            tails=np.array([graph.entities[node] for node in graph.tails[targets]]),
            relation_names=np.array(graph.relation_names),
            pair=np.array(pair),
            true_relations=true_relations,
            scores=scores.astype(np.float32),
            all_relations_rr=np.array([value[0] for value in full]),
            all_relations_hits1=np.array([value[1] for value in full]),
            pair_hits1=np.array([value[1] for value in restricted]),
        )
        print(
            f"{path.name}: {len(targets)} triples, pair accuracy {np.mean([v[1] for v in restricted]):.3f}, "
            f"24-way MRR {np.mean([v[0] for v in full]):.3f}, {seconds / len(targets):.2f} s/triple, "
            f"peak GPU {info['peak_gpu_gb']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
