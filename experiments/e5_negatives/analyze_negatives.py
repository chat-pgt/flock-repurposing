"""E5: compare fine-tuning with uniform and disease-only negatives against zero-shot E1 on the same edges.

Uses the folds that both arms finished. The zero-shot reference is E1's run with the same model and walk
seed, so FLOCK's scoring walks are identical across the three conditions and only the weights differ. All
comparisons are paired per edge, with two-way bootstrap CIs over drugs and diseases.

Usage, from experiments/:
    uv run python -m e5_negatives.analyze_negatives --runs path/to/e5 --reference path/to/e1/runs --model ultra
"""

import argparse
import json
from pathlib import Path

import numpy as np

from common.hetionet import EDGES_PATH, EXPERIMENTS, SPLIT_PATH, read_split
from common.provenance import run_metadata
from gpu.analysis import interval

CONDITIONS = {"zero_shot": "full", "uniform": "ft_uniform", "disease": "ft_disease"}
CONTRASTS = (("uniform", "zero_shot"), ("disease", "zero_shot"), ("disease", "uniform"))


def load_condition(directory, model, condition, seed, n_edges):
    """Return {fold: (rows, reciprocal ranks, Hits@10)} for the folds present, plus training details."""
    folds, training = {}, {}
    for path in sorted(Path(directory).glob(f"entity_{model}_{condition}_tail_seed{seed}_fold*.npz")):
        with np.load(path) as shard:
            info = json.loads(str(shard["metadata"]))
            rows = shard["edge_rows"]
            if rows.max() >= n_edges:
                raise ValueError(f"{path.name} does not follow the shared split")
            folds[info["fold"]] = (rows, shard["reciprocal_rank"], shard["hits10"])
            if "train_loss" in shard:
                loss = shard["train_loss"]
                training[info["fold"]] = {
                    "steps": int(len(loss)),
                    "loss_first_10": float(loss[:10].mean()),
                    "loss_last_10": float(loss[-10:].mean()),
                    "train_minutes": info["train_seconds"] / 60,
                    "train_peak_gpu_gb": info["train_peak_gpu_gb"],
                }
    return folds, training


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True, help="Directory with the E5 shards")
    parser.add_argument("--reference", type=Path, required=True, help="Directory with the E1 shards")
    parser.add_argument("--model", choices=["flock", "ultra"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=Path, default=SPLIT_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or EXPERIMENTS / f"results/e5_negatives_{args.model}.json"

    split = read_split(args.split)
    n_edges = len(split.drugs)
    loaded, training = {}, {}
    for name, condition in CONDITIONS.items():
        directory = args.reference if name == "zero_shot" else args.runs
        loaded[name], training[name] = load_condition(directory, args.model, condition, args.seed, n_edges)
    folds = sorted(set.intersection(*(set(folds) for folds in loaded.values())))
    if not folds:
        raise SystemExit("no fold is complete in all three conditions")

    metrics = {name: {"rr": np.full(n_edges, np.nan), "hits10": np.full(n_edges, np.nan)} for name in CONDITIONS}
    for name, by_fold in loaded.items():
        for fold in folds:
            rows, rr, hits = by_fold[fold]
            metrics[name]["rr"][rows], metrics[name]["hits10"][rows] = rr, hits
    in_folds = np.isin(split.folds, folds)
    keep = in_folds & ~np.any([np.isnan(metrics[name]["rr"]) for name in CONDITIONS], axis=0)
    if (in_folds & ~keep).any():
        print(f"warning: {int((in_folds & ~keep).sum())} edges of these folds are missing from some condition")

    drugs, diseases = split.drugs, split.diseases
    strata = {
        "all": keep,
        "drug_has_other_indication": keep & split.has_other_indication,
        "first_indication": keep & ~split.has_other_indication,
    }
    results = {
        "metadata": run_metadata(data_files=(EDGES_PATH, args.split)),
        "model": args.model,
        "seed": args.seed,
        "folds": folds,
        "n_edges": int(keep.sum()),
        "training": {name: training[name] for name in ("uniform", "disease")},
        "conditions": {
            name: {metric: interval(metrics[name][metric], drugs, diseases, keep) for metric in ("rr", "hits10")}
            for name in CONDITIONS
        },
        "contrasts": {},
    }
    for first, second in CONTRASTS:
        results["contrasts"][f"{first}_minus_{second}"] = {
            stratum: {
                metric: interval(metrics[first][metric] - metrics[second][metric], drugs, diseases, mask)
                for metric in ("rr", "hits10")
            }
            for stratum, mask in strata.items()
            if mask.any()
        }
    output.write_text(json.dumps(results, indent=2) + "\n")

    print(f"{args.model}, seed {args.seed}, folds {folds}, {keep.sum()} edges")
    for name in CONDITIONS:
        row = results["conditions"][name]["rr"]
        print(f"  {name}: MRR {row['mean']:.3f} [{row['ci95'][0]:.3f}, {row['ci95'][1]:.3f}]")
    for contrast, by_stratum in results["contrasts"].items():
        row = by_stratum["all"]["rr"]
        print(f"  {contrast}: MRR {row['mean']:+.3f} [{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
