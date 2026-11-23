"""E2: measure what removing *resembles* edges does to flagged and unflagged edges.

An edge is flagged when a compound the drug resembles treats the same disease in training. If models use
that shortcut, scores should fall more on flagged edges. Each model is compared with itself on the seeds
it completed in both conditions.

Usage, from experiments/: uv run python -m gpu.analyze_e2 --runs path/to/runs
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common.hetionet import EDGES_PATH, EXPERIMENTS, SPLIT_PATH, load_repurposing_graph, read_split
from common.provenance import run_metadata
from gpu.analysis import average_over_seeds, bootstrap_contrast, complete_seeds, interval, seed_metrics

MODELS = ("flock", "ultra")


def read_flags(path, split):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if [(row["compound"], row["disease"]) for row in rows] != list(
        zip(split.drugs.tolist(), split.diseases.tolist(), strict=True)
    ):
        raise ValueError("the leakage file does not follow the shared split")
    return np.array([row["flagged"] == "1" for row in rows])


def condition_metrics(runs, model, condition, seeds, graph, split):
    paths = complete_seeds(runs, model, condition, "tail")
    per_seed = {
        seed: seed_metrics(paths[seed], split.drugs.tolist(), split.diseases.tolist(), graph.diseases, graph.palliates)
        for seed in seeds
    }
    return average_over_seeds(per_seed)["candidates"]


def summarise_metric(full, ablated, flagged, split):
    drop = full - ablated
    drugs, diseases = split.drugs, split.diseases
    return {
        "full": interval(full, drugs, diseases),
        "without_resembles": interval(ablated, drugs, diseases),
        "drop_all": interval(drop, drugs, diseases),
        "drop_flagged": interval(drop, drugs, diseases, flagged),
        "drop_unflagged": interval(drop, drugs, diseases, ~flagged),
        "extra_drop_on_flagged": bootstrap_contrast(drop, flagged, drugs, diseases),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--leakage", type=Path, default=EXPERIMENTS / "results/e2_leakage_edges.tsv")
    parser.add_argument("--edges", type=Path, default=EDGES_PATH)
    parser.add_argument("--split", type=Path, default=SPLIT_PATH)
    parser.add_argument("--output", type=Path, default=EXPERIMENTS / "results/e2_ablation.json")
    args = parser.parse_args()

    graph = load_repurposing_graph(args.edges)
    split = read_split(args.split)
    flagged = read_flags(args.leakage, split)
    results = {
        "metadata": run_metadata(data_files=(args.edges, args.split, args.leakage)),
        "n_flagged": int(flagged.sum()),
        "n_unflagged": int((~flagged).sum()),
        "models": {},
    }
    for model in MODELS:
        seeds = sorted(
            set(complete_seeds(args.runs, model, "full", "tail"))
            & set(complete_seeds(args.runs, model, "no_resembles", "tail"))
        )
        if not seeds:
            print(f"{model}: no seed is complete in both conditions")
            continue
        full = condition_metrics(args.runs, model, "full", seeds, graph, split)
        ablated = condition_metrics(args.runs, model, "no_resembles", seeds, graph, split)
        results["models"][model] = {
            "seeds": seeds,
            "mrr": summarise_metric(full["rr"], ablated["rr"], flagged, split),
            "hits10": summarise_metric(full["hits10"], ablated["hits10"], flagged, split),
        }
        row = results["models"][model]["mrr"]
        extra = row["extra_drop_on_flagged"]
        print(
            f"{model} (seeds {seeds}): MRR {row['full']['mean']:.3f} -> {row['without_resembles']['mean']:.3f} | "
            f"drop flagged {row['drop_flagged']['mean']:+.3f}, unflagged {row['drop_unflagged']['mean']:+.3f} | "
            f"extra on flagged {extra['mean']:+.3f} [{extra['ci95'][0]:+.3f}, {extra['ci95'][1]:+.3f}]"
        )
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
