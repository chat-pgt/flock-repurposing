"""E3: measure how FLOCK's drug -> disease ranking changes with walk coverage and graph content.

Runs a fixed sample of queries on the released checkpoint, varying the number of walks n, the walk length l
at a fixed budget n * l, and removing the largest relation that lies off the drug-disease paths. Walks
shorter than the trained length of 128 reuse the trained embedding tables, without changing upstream code.

Usage, from experiments/: python -m e3_coverage.run_coverage --output-dir e3
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from common.hetionet import EDGES_PATH, SPLIT_PATH, TREATS, read_edges, read_split
from e3_coverage.plan import QUERIES_PER_FOLD, SETTINGS, sample_plan
from gpu.graphs import encode_triples, training_mask
from gpu.models import build_data, load_flock, peak_gpu_gb, run_metadata, seed_everything
from gpu.run_entity import run_fold, save_shard


def set_walk_length(model, length):
    """Run a trained model on shorter walks.

    The anonymous-relation table reserves its last row for "no relation", at the trained length. The
    released code writes that marker as `L + 1`, so it is moved back to the trained row after each walk.
    """
    trained = model.emb_anon_type[0].num_embeddings - 1
    if length > trained:
        raise ValueError(f"walks of length {length} need retraining: the checkpoint was trained on {trained}")
    original = model._walk

    def walk(*args, **kwargs):
        output = list(original(*args, **kwargs))
        output[5] = torch.where(output[5] == length + 1, trained + 1, output[5])
        return tuple(output)

    model.L = length
    model._walk = walk
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", default=",".join(SETTINGS))
    parser.add_argument("--queries-per-fold", type=int, default=QUERIES_PER_FOLD)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deadline", type=float, help="Unix time after which no new fold starts")
    parser.add_argument("--edges", type=Path, default=EDGES_PATH)
    parser.add_argument("--split", type=Path, default=SPLIT_PATH)
    parser.add_argument("--walk-num", type=int, help="Override n, for smoke tests only")
    parser.add_argument("--ensemble", type=int, help="Override P, for smoke tests only")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    graph = encode_triples(read_edges(args.edges))
    split = read_split(args.split)
    index = {name: position for position, name in enumerate(graph.entities)}
    drugs = np.array([index[name] for name in split.drugs.tolist()])
    diseases = np.array([index[name] for name in split.diseases.tolist()])
    candidates = np.array([position for position, name in enumerate(graph.entities) if name.startswith("Disease::")])
    known = defaultdict(set)
    for drug, disease in zip(drugs.tolist(), diseases.tolist(), strict=True):
        known[drug].add(disease)
    treats = graph.relation_names.index(TREATS)
    split_ids = (drugs, diseases, split.folds, known, candidates, treats)
    run_args = SimpleNamespace(direction="tail", max_queries=None)
    device = torch.device(args.device)

    for name in args.settings.split(","):
        setting = SETTINGS[name]
        dropped = [graph.relation_names.index(relation) for relation in setting.dropped_relations]
        model, settings = load_flock("entity", device, args.walk_num or setting.walk_num, args.ensemble)
        set_walk_length(model, setting.walk_len)
        settings = {**settings, "walk_len": setting.walk_len, "dropped_relations": list(setting.dropped_relations)}
        metadata = run_metadata(device, model="flock", experiment="e3", setting=name, seed=args.seed, **settings)
        for fold in np.unique(split.folds).tolist():
            path = args.output_dir / f"e3_{name}_fold{fold}.npz"
            if path.exists():
                print(f"skip {path.name}", flush=True)
                continue
            if args.deadline is not None and time.time() > args.deadline:
                print(f"time budget reached before {path.name}", flush=True)
                return
            seed_everything(1000 * args.seed + fold)
            held = split.folds == fold
            keep = training_mask(graph, drugs[held], diseases[held], treats, dropped)
            data = build_data(graph, keep, "flock", device).to(device)
            plan = sample_plan(drugs, split.folds, fold, args.queries_per_fold, args.seed)
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            arrays = run_fold(model, data, graph, split_ids, fold, run_args, device, plan=plan)
            seconds = time.perf_counter() - start
            info = {**metadata, "fold": fold, "seconds": seconds, "peak_gpu_gb": peak_gpu_gb(device)}
            save_shard(path, metadata=np.array(json.dumps(info)), **arrays)
            print(
                f"{path.name}: {len(arrays['edge_rows'])} edges, {len(plan)} queries, "
                f"MRR {arrays['reciprocal_rank'].mean():.3f}, {seconds / len(plan):.2f} s/query, "
                f"peak GPU {info['peak_gpu_gb']}",
                flush=True,
            )
            del data
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        del model


if __name__ == "__main__":
    main()
