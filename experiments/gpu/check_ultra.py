"""Check the public ULTRA checkpoint against its official results on the 18 inductive entity graphs.

ULTRA's README reports a mean MRR of 0.420 and Hits@10 of 0.562 for this checkpoint. Those are lower than
the original paper's 0.430 and 0.566, because the public checkpoint comes from a PyG reimplementation.

Usage, from experiments/: python -m gpu.check_ultra --ultra-repo ULTRA --data-root data/ultra --output check.json
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from gpu.models import load_ultra, run_metadata

OFFICIAL_MRR = 0.420
OFFICIAL_HITS10 = 0.562
TOLERANCE = 0.01
GRAPHS = [
    *[("FB15k237Inductive", version) for version in ("v1", "v2", "v3", "v4")],
    *[("WN18RRInductive", version) for version in ("v1", "v2", "v3", "v4")],
    *[("NELLInductive", version) for version in ("v1", "v2", "v3", "v4")],
    ("ILPC2022", "small"),
    ("ILPC2022", "large"),
    *[("HM", version) for version in ("1k", "3k", "5k", "indigo")],
]


def known_facts(dataset_name, valid_data, test_data):
    """Build the filtering graph exactly as ULTRA's `run.py` does."""
    if "ILPC" in dataset_name:
        edges = [valid_data.edge_index, valid_data.target_edge_index, test_data.target_edge_index]
        types = [valid_data.edge_type, valid_data.target_edge_type, test_data.target_edge_type]
    else:
        edges = [test_data.edge_index, test_data.target_edge_index]
        types = [test_data.edge_type, test_data.target_edge_type]
    return Data(edge_index=torch.cat(edges, dim=1), edge_type=torch.cat(types), num_nodes=test_data.num_nodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ultra-repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--deadline", type=float, help="Unix time after which no new graph starts")
    args = parser.parse_args()

    # ULTRA's dataset classes predate PyTorch's safer default for loading their own processed files.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    device = torch.device(args.device)
    model, settings = load_ultra(args.ultra_repo, device)
    sys.path.insert(0, str(args.ultra_repo / "script"))
    from run import test
    from ultra import util

    results = {"metadata": run_metadata(device, **settings), "graphs": []}
    for dataset_name, version in GRAPHS:
        if args.deadline is not None and time.time() > args.deadline:
            print("time budget reached", flush=True)
            break
        context = {
            "dataset": dataset_name,
            "version": version,
            "gpus": "[0]" if str(device).startswith("cuda") else "null",
            "epochs": 0,
            "bpe": "null",
            "ckpt": "none",
        }
        cfg = util.load_config(str(args.ultra_repo / "config/inductive/inference.yaml"), context=context)
        cfg.dataset.root = str(args.data_root)
        cfg.train.batch_size = args.batch_size
        dataset = util.build_dataset(cfg)
        valid_data, test_data = dataset[1].to(device), dataset[2].to(device)
        known = known_facts(dataset_name, valid_data, test_data).to(device)
        with torch.no_grad():
            metrics = test(cfg, model, test_data, device, logging.getLogger("ultra"), known, return_metrics=True)
        row = {"graph": f"{dataset_name}:{version}", "mrr": float(metrics["mrr"]), "hits10": float(metrics["hits@10"])}
        results["graphs"].append(row)
        print(row, flush=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")

    if len(results["graphs"]) == len(GRAPHS):
        mrr = float(np.mean([row["mrr"] for row in results["graphs"]]))
        hits10 = float(np.mean([row["hits10"] for row in results["graphs"]]))
        results["summary"] = {
            "mean_mrr": mrr,
            "mean_hits10": hits10,
            "official_mrr": OFFICIAL_MRR,
            "official_hits10": OFFICIAL_HITS10,
            "matches_official": abs(mrr - OFFICIAL_MRR) <= TOLERANCE and abs(hits10 - OFFICIAL_HITS10) <= TOLERANCE,
        }
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(results["summary"], flush=True)


if __name__ == "__main__":
    main()
