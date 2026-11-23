"""Run the released entity checkpoint zero-shot and compare it with Table 9 of the paper.

Uses the upstream dataset, model and test code unchanged, with the released config for each graph.
`--max-triples` scores a random subset, which is useful for timing but not comparable with Table 9.

Usage:
    python reproduce_zero_shot.py --datasets WN18RRInductive:v1 --device cpu --batch-size 1 \
        --output ../results/e0_zero_shot.json
"""

import argparse
import hashlib
import json
import logging
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from check_walker import metadata
from torch_geometric.data import Data

REPO = Path(__file__).resolve().parents[2]
CHECKPOINT = REPO / "checkpoints/flock_entity.pth"
INDUCTIVE_CONFIG = "src_entity/config/zeroshot_inductive/n16_ensemble16.yaml"
TRANSDUCTIVE_CONFIG = "src_entity/config/zeroshot_transductive/n512_ensemble2.yaml"
TABLE_9_NAMES = {
    "FB15k237Inductive:v1": "FB-v1",
    "WN18RRInductive:v1": "WN-v1",
    "NELLInductive:v1": "NL-v1",
    "Hetionet": "Hetionet",
}
TOLERANCE = 0.01


def upstream():
    sys.path.insert(0, str(REPO / "src_entity"))
    from flock import util
    from flock.models import Flock
    from run import test

    return util, Flock, test


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(util, name, data_root, device, batch_size):
    dataset, _, version = name.partition(":")
    path = TRANSDUCTIVE_CONFIG if dataset == "Hetionet" else INDUCTIVE_CONFIG
    context = {
        "dataset": dataset,
        "gpus": "[0]" if device == "cuda" else "null",
        "epochs": 0,
        "bpe": "null",
        "ckpt": str(CHECKPOINT),
    }
    if version:
        context["version"] = version
    cfg = util.load_config(REPO / path, context=context)
    cfg.dataset.root = str(data_root)
    cfg.train.test_batch_size = batch_size
    return cfg, path


def filtering_graph(cfg, dataset, test_data):
    """Build the graph of known facts exactly as `run_many.py` does for these datasets."""
    if cfg.task["name"] == "InductiveInference":
        return Data(
            edge_index=torch.cat([test_data.edge_index, test_data.target_edge_index], dim=1),
            edge_type=torch.cat([test_data.edge_type, test_data.target_edge_type]),
            num_nodes=test_data.num_nodes,
        )
    return Data(
        edge_index=dataset._data.target_edge_index,
        edge_type=dataset._data.target_edge_type,
        num_nodes=dataset[0].num_nodes,
    )


def subsample(test_data, max_triples):
    n_triples = test_data.target_edge_index.shape[1]
    if max_triples is None or max_triples >= n_triples:
        return test_data
    index = torch.randperm(n_triples)[:max_triples]
    subset = test_data.clone()
    subset.target_edge_index = test_data.target_edge_index[:, index]
    subset.target_edge_type = test_data.target_edge_type[index]
    return subset


def paper_values(name):
    tables = json.loads((REPO / "experiments/paper_analysis/paper_tables.json").read_text())
    row = next(row for row in tables["9"] if row["ds"] == TABLE_9_NAMES[name])
    return {"mrr": row["F"], "hits10": row["Fh"]}


def evaluate(name, args, modules):
    util, flock_model, test = modules
    logger = logging.getLogger("e0")
    seed_everything(args.seed)
    cfg, config_path = load_config(util, name, args.data_root, args.device, args.batch_size)
    device = util.get_device(cfg)
    dataset = util.build_dataset(cfg, device=device)
    test_data = dataset[2].to(device)
    known_facts = filtering_graph(cfg, dataset, test_data).to(device)
    evaluated = subsample(test_data, args.max_triples)

    model = flock_model(model_cfg=cfg.model)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["model"])
    model = model.to(device)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    with torch.no_grad():
        metrics = test(cfg, model, evaluated, device, logger, filtered_data=known_facts, return_metrics=True)
    seconds = time.perf_counter() - start

    n_triples = evaluated.target_edge_index.shape[1]
    full_run = n_triples == test_data.target_edge_index.shape[1]
    measured = {"mrr": float(metrics["mrr"]), "hits10": float(metrics["hits@10"])}
    paper = paper_values(name)
    return {
        "dataset": name,
        "config": config_path,
        "walks_n": cfg.model.walk_num,
        "ensemble_p": cfg.model.test_samples,
        "batch_size": args.batch_size,
        "test_triples_in_dataset": test_data.target_edge_index.shape[1],
        "test_triples_evaluated": n_triples,
        "full_run": full_run,
        "paper_all_relations": paper,
        "measured": measured,
        "mrr_difference": measured["mrr"] - paper["mrr"],
        "within_tolerance": abs(measured["mrr"] - paper["mrr"]) <= TOLERANCE if full_run else None,
        "seconds": seconds,
        "seconds_per_triple": seconds / n_triples,
        "peak_process_memory_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3 if args.device == "cuda" else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="FB15k237Inductive:v1,WN18RRInductive:v1,NELLInductive:v1")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-triples", type=int)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--data-root", type=Path, default=REPO / "experiments/data/kg")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    modules = upstream()
    info = metadata(args.seed)
    info.pop("config")
    info["checkpoint_sha256"] = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    info["device"] = torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"
    results = {"metadata": info, "runs": []}
    for name in args.datasets.split(","):
        results["runs"].append(evaluate(name, args, modules))
        print(json.dumps(results["runs"][-1], indent=2), flush=True)
        if args.output:
            args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
