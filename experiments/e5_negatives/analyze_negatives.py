"""E5: compare fine-tuning with uniform and disease-only negatives against zero-shot E1 on the same edges.

Uses the folds that every condition finished. The zero-shot reference is E1's run with the same model and
walk seed, so FLOCK's scoring walks are identical across conditions and only the weights differ. Every epoch
that was scored is reported, and none is selected. Comparisons are paired per edge, with two-way bootstrap
CIs over drugs and diseases.

The primary stratum is edges whose drug has no other indication in training. Those drugs never appear in
fine-tuning, so any change there is generalization. For drugs with other indications, the held-out disease
can be drawn as a training negative for the drug, which is the positive-unlabeled cost.

Usage, from experiments/:
    uv run python -m e5_negatives.analyze_negatives --runs path/to/e5 --reference path/to/e1/runs --model ultra
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common.hetionet import EDGES_PATH, EXPERIMENTS, SPLIT_PATH, read_split
from common.provenance import run_metadata
from gpu.analysis import interval

STRATA = ("first_indication", "drug_has_other_indication", "all")


def read_shard(path, n_edges):
    with np.load(path) as shard:
        info = json.loads(str(shard["metadata"]))
        rows = shard["edge_rows"]
        if rows.max() >= n_edges:
            raise ValueError(f"{path.name} does not follow the shared split")
        loss = shard["train_loss"] if "train_loss" in shard else None
        return info, (rows, shard["reciprocal_rank"], shard["hits10"]), loss


def load_runs(runs, reference, model, seed, n_edges):
    """Return {condition: {fold: (rows, rr, hits10)}} and training details per fine-tuned condition."""
    conditions, training = defaultdict(dict), defaultdict(dict)
    for path in sorted(Path(reference).glob(f"entity_{model}_full_tail_seed{seed}_fold*.npz")):
        info, arrays, _ = read_shard(path, n_edges)
        conditions["zero_shot"][info["fold"]] = arrays
    for path in sorted(Path(runs).glob(f"entity_{model}_ft_*_tail_seed{seed}_fold*.npz")):
        info, arrays, loss = read_shard(path, n_edges)
        epoch = info.get("epoch", info.get("epochs", 1))  # early one-epoch runs record only `epochs`
        name = f"{info['arm']}_ep{epoch}"
        conditions[name][info["fold"]] = arrays
        training[name][info["fold"]] = {
            "steps": int(len(loss)),
            "loss_first_10": float(loss[:10].mean()),
            "loss_last_10": float(loss[-10:].mean()),
            "train_minutes": info["train_seconds"] / 60,
            "train_peak_gpu_gb": info["train_peak_gpu_gb"],
        }
    return dict(conditions), dict(training)


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
    conditions, training = load_runs(args.runs, args.reference, args.model, args.seed, n_edges)
    epochs = sorted({int(name.rsplit("_ep", 1)[1]) for name in conditions if name != "zero_shot"})
    complete = [e for e in epochs if f"uniform_ep{e}" in conditions and f"disease_ep{e}" in conditions]
    if "zero_shot" not in conditions or not complete:
        raise SystemExit("need E1 zero-shot shards and both E5 arms for at least one epoch")
    conditions = {
        name: by_fold
        for name, by_fold in conditions.items()
        if name == "zero_shot" or int(name.rsplit("_ep", 1)[1]) in complete
    }
    folds = sorted(set.intersection(*(set(by_fold) for by_fold in conditions.values())))
    if not folds:
        raise SystemExit("no fold is complete in every condition")

    metrics = {}
    for name, by_fold in conditions.items():
        metrics[name] = {"rr": np.full(n_edges, np.nan), "hits10": np.full(n_edges, np.nan)}
        for fold in folds:
            rows, rr, hits = by_fold[fold]
            metrics[name]["rr"][rows], metrics[name]["hits10"][rows] = rr, hits
    in_folds = np.isin(split.folds, folds)
    keep = in_folds & ~np.any([np.isnan(values["rr"]) for values in metrics.values()], axis=0)
    if (in_folds & ~keep).any():
        print(f"warning: {int((in_folds & ~keep).sum())} edges of these folds are missing from some condition")

    drugs, diseases = split.drugs, split.diseases
    masks = {
        "first_indication": keep & ~split.has_other_indication,
        "drug_has_other_indication": keep & split.has_other_indication,
        "all": keep,
    }

    def summarise(values):
        return {
            stratum: {metric: interval(values[metric], drugs, diseases, mask) for metric in ("rr", "hits10")}
            for stratum, mask in masks.items()
            if mask.any()
        }

    def difference(first, second):
        return summarise({metric: metrics[first][metric] - metrics[second][metric] for metric in ("rr", "hits10")})

    results = {
        "metadata": run_metadata(data_files=(EDGES_PATH, args.split)),
        "model": args.model,
        "seed": args.seed,
        "folds": folds,
        "epochs": complete,
        "primary_stratum": "first_indication",
        "n_edges": {stratum: int(mask.sum()) for stratum, mask in masks.items()},
        "training": {name: details for name, details in training.items() if name in conditions},
        "conditions": {name: summarise(values) for name, values in metrics.items()},
        "contrasts": {
            f"ep{epoch}": {
                "uniform_minus_zero_shot": difference(f"uniform_ep{epoch}", "zero_shot"),
                "disease_minus_zero_shot": difference(f"disease_ep{epoch}", "zero_shot"),
                "disease_minus_uniform": difference(f"disease_ep{epoch}", f"uniform_ep{epoch}"),
            }
            for epoch in complete
        },
    }
    output.write_text(json.dumps(results, indent=2) + "\n")

    print(f"{args.model}, seed {args.seed}, folds {folds}, edges {results['n_edges']}")
    for stratum in STRATA:
        if stratum not in results["conditions"]["zero_shot"]:
            continue
        print(f"  {stratum}: zero-shot MRR {results['conditions']['zero_shot'][stratum]['rr']['mean']:.3f}")
        for epoch, contrasts in results["contrasts"].items():
            parts = []
            for name, by_stratum in contrasts.items():
                row = by_stratum[stratum]["rr"]
                parts.append(f"{name} {row['mean']:+.3f} [{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}]")
            print(f"    {epoch}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
