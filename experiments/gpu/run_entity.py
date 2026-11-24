"""Score the held-out treats edges with FLOCK or ULTRA on the shared folds (E1, E2).

Each fold is saved as its own file, and existing files are skipped, so an interrupted run can resume.

Usage, from experiments/:
    python -m gpu.run_entity --model flock --direction tail --seeds 0,1,2 --output-dir runs
    python -m gpu.run_entity --model ultra --direction tail --condition no_resembles --output-dir runs
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from common.evaluation import candidate_mask, rank_metrics
from common.hetionet import EDGES_PATH, PALLIATES, RESEMBLES, SPLIT_PATH, TREATS, read_edges, read_split
from gpu.graphs import encode_triples, plan_queries, rank_counts, training_mask
from gpu.models import build_data, load_flock, load_ultra, peak_gpu_gb, run_metadata, seed_everything

DROPPED = {"full": (), "no_resembles": (RESEMBLES,)}
CANDIDATE_KIND = {"tail": "Disease::", "head": "Compound::"}


@torch.no_grad()
def score_query(model, data, query, relation, direction, n_entities, device):
    everyone = torch.arange(n_entities, device=device)
    fixed = torch.full_like(everyone, query)
    relations = torch.full_like(everyone, relation)
    columns = (fixed, everyone, relations) if direction == "tail" else (everyone, fixed, relations)
    return model(data, torch.stack(columns, dim=-1)[None])[0].float().cpu().numpy()


def save_shard(path, **arrays):
    temporary = path.with_suffix(".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def run_fold(model, data, graph, split_ids, fold, args, device, plan=None):
    query_ids, answer_ids, folds, known, candidates, treats = split_ids
    position = {int(entity): index for index, entity in enumerate(candidates)}
    if plan is None:
        plan = plan_queries(query_ids, folds, fold)
        if args.max_queries is not None:
            plan = dict(list(plan.items())[: args.max_queries])

    edge_rows, query_positions, reciprocal_ranks, hits, above, tied, query_scores = [], [], [], [], [], [], []
    for query_position, (query, rows) in enumerate(plan.items()):
        scores = score_query(model, data, query, treats, args.direction, len(graph.entities), device)
        candidate_scores = scores[candidates]
        query_scores.append(candidate_scores)
        for row in rows:
            target = int(answer_ids[row])
            others = known[query] - {target}
            mask = candidate_mask(
                len(candidates), range(len(candidates)), [position[node] for node in others], position[target]
            )
            reciprocal_rank, hit = rank_metrics(candidate_scores, position[target], mask)
            count_above, count_tied = rank_counts(scores, target, others)
            edge_rows.append(row)
            query_positions.append(query_position)
            reciprocal_ranks.append(reciprocal_rank)
            hits.append(hit)
            above.append(count_above)
            tied.append(count_tied)
    return {
        "query_entities": np.array([graph.entities[query] for query in plan]),
        "candidate_scores": np.stack(query_scores).astype(np.float32),
        "edge_rows": np.array(edge_rows),
        "edge_query_positions": np.array(query_positions),
        "reciprocal_rank": np.array(reciprocal_ranks),
        "hits10": np.array(hits),
        "all_entities_above": np.array(above),
        "all_entities_tied": np.array(tied),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["flock", "ultra"], required=True)
    parser.add_argument("--direction", choices=["tail", "head"], default="tail")
    parser.add_argument("--condition", choices=sorted(DROPPED), default="full")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deadline", type=float, help="Unix time after which no new fold starts")
    parser.add_argument("--ultra-repo", type=Path)
    parser.add_argument("--edges", type=Path, default=EDGES_PATH)
    parser.add_argument("--split", type=Path, default=SPLIT_PATH)
    parser.add_argument("--max-queries", type=int, help="Score only this many queries per fold, for smoke tests")
    parser.add_argument("--walk-num", type=int, help="Override n, for smoke tests only")
    parser.add_argument("--ensemble", type=int, help="Override P, for smoke tests only")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    graph = encode_triples(read_edges(args.edges))
    split = read_split(args.split)
    index = {name: position for position, name in enumerate(graph.entities)}
    drugs = np.array([index[name] for name in split.drugs.tolist()])
    diseases = np.array([index[name] for name in split.diseases.tolist()])
    query_ids, answer_ids = (drugs, diseases) if args.direction == "tail" else (diseases, drugs)
    candidates = np.array(
        [position for position, name in enumerate(graph.entities) if name.startswith(CANDIDATE_KIND[args.direction])]
    )
    known = defaultdict(set)
    for query, answer in zip(query_ids.tolist(), answer_ids.tolist(), strict=True):
        known[query].add(answer)
    treats = graph.relation_names.index(TREATS)
    dropped = [graph.relation_names.index(name) for name in DROPPED[args.condition]]
    if PALLIATES not in graph.relation_names:
        raise ValueError("the graph has no palliates relation")

    device = torch.device(args.device)
    if args.model == "flock":
        model, settings = load_flock("entity", device, args.walk_num, args.ensemble)
        seeds = [int(seed) for seed in args.seeds.split(",")]
    else:
        model, settings = load_ultra(args.ultra_repo, device)
        seeds = [0]  # ULTRA is deterministic
    metadata = run_metadata(device, model=args.model, direction=args.direction, condition=args.condition, **settings)

    split_ids = (query_ids, answer_ids, split.folds, known, candidates, treats)
    for seed in seeds:
        for fold in np.unique(split.folds).tolist():
            name = f"entity_{args.model}_{args.condition}_{args.direction}_seed{seed}_fold{fold}.npz"
            path = args.output_dir / name
            if path.exists():
                print(f"skip {name}", flush=True)
                continue
            if args.deadline is not None and time.time() > args.deadline:
                print(f"time budget reached before {name}", flush=True)
                return
            seed_everything(1000 * seed + fold)
            held = split.folds == fold
            keep = training_mask(graph, drugs[held], diseases[held], treats, dropped)
            data = build_data(graph, keep, args.model, device).to(device)
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            arrays = run_fold(model, data, graph, split_ids, fold, args, device)
            seconds = time.perf_counter() - start
            info = {**metadata, "seed": seed, "fold": fold, "seconds": seconds, "peak_gpu_gb": peak_gpu_gb(device)}
            save_shard(path, metadata=np.array(json.dumps(info)), **arrays)
            print(
                f"{name}: {len(arrays['edge_rows'])} edges, {len(arrays['query_entities'])} queries, "
                f"MRR {arrays['reciprocal_rank'].mean():.3f}, Hits@10 {arrays['hits10'].mean():.3f}, "
                f"{seconds / len(arrays['query_entities']):.2f} s/query, peak GPU {info['peak_gpu_gb']}",
                flush=True,
            )
            del data
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
