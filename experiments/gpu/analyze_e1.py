"""E1: compare FLOCK and ULTRA with the popularity baseline on the shared folds.

Only seeds with all five folds are used, and each edge is averaged over them. Models are compared
through paired per-edge differences, with two-way bootstrap CIs over drugs and diseases.

Usage, from experiments/: uv run python -m gpu.analyze_e1 --runs path/to/runs
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common.hetionet import EDGES_PATH, EXPERIMENTS, SPLIT_PATH, load_repurposing_graph, read_split
from common.provenance import run_metadata
from gpu.analysis import VIEWS, average_over_seeds, complete_seeds, interval, seed_metrics

MODELS = ("flock", "ultra")
DIRECTIONS = ("tail", "head")


def read_popularity(path, split):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if [(row["compound"], row["disease"]) for row in rows] != list(
        zip(split.drugs.tolist(), split.diseases.tolist(), strict=True)
    ):
        raise ValueError("the popularity file does not follow the shared split")
    return {column: np.array([float(row[column]) for row in rows]) for column in rows[0] if "_" in column}


def load_model(runs, model, direction, graph, split):
    seeds = complete_seeds(runs, model, "full", direction)
    if not seeds:
        return None
    drugs, diseases = split.drugs.tolist(), split.diseases.tolist()
    if direction == "tail":
        queries, answers, candidates, palliated = drugs, diseases, graph.diseases, graph.palliates
    else:
        flipped = {(disease, drug) for drug, disease in graph.palliates}
        queries, answers, candidates, palliated = diseases, drugs, graph.compounds, flipped
    per_seed = {seed: seed_metrics(paths, queries, answers, candidates, palliated) for seed, paths in seeds.items()}
    return {
        "seeds": sorted(per_seed),
        "seed_mrr": {seed: float(run["candidates"]["rr"].mean()) for seed, run in per_seed.items()},
        "views": average_over_seeds(per_seed),
    }


def compare(view, popularity, column, split, keep):
    drugs, diseases = split.drugs, split.diseases
    return {
        "n_edges": int(keep.sum()),
        "mrr": interval(view["rr"], drugs, diseases, keep),
        "hits10": interval(view["hits10"], drugs, diseases, keep),
        "mrr_minus_popularity": interval(view["rr"] - popularity[f"{column}_rr"], drugs, diseases, keep),
        "hits10_minus_popularity": interval(view["hits10"] - popularity[f"{column}_hits10"], drugs, diseases, keep),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--popularity", type=Path, default=EXPERIMENTS / "results/e1_popularity_edges.tsv")
    parser.add_argument("--edges", type=Path, default=EDGES_PATH)
    parser.add_argument("--split", type=Path, default=SPLIT_PATH)
    parser.add_argument("--output", type=Path, default=EXPERIMENTS / "results/e1_models.json")
    args = parser.parse_args()

    graph = load_repurposing_graph(args.edges)
    split = read_split(args.split)
    popularity = read_popularity(args.popularity, split)
    strata = {
        "all": np.ones(len(split.drugs), dtype=bool),
        "drug_has_other_indication": split.has_other_indication,
        "first_indication": ~split.has_other_indication,
    }
    results = {
        "metadata": run_metadata(data_files=(args.edges, args.split, args.popularity)),
        "coverage": {},
        "views": {},
    }
    for direction in DIRECTIONS:
        loaded = {model: load_model(args.runs, model, direction, graph, split) for model in MODELS}
        for model, run in loaded.items():
            results["coverage"][f"{model}_{direction}"] = (
                {"seeds": run["seeds"], "seed_mrr": run["seed_mrr"]} if run else {"seeds": []}
            )
        for view_name in VIEWS:
            column = f"{direction}_{view_name}"
            section = {}
            for model, run in loaded.items():
                if run:
                    section[model] = {
                        stratum: compare(run["views"][view_name], popularity, column, split, keep)
                        for stratum, keep in strata.items()
                    }
            if all(loaded.values()):
                flock, ultra = loaded["flock"]["views"][view_name], loaded["ultra"]["views"][view_name]
                section["flock_minus_ultra"] = {
                    stratum: {
                        "mrr": interval(flock["rr"] - ultra["rr"], split.drugs, split.diseases, keep),
                        "hits10": interval(flock["hits10"] - ultra["hits10"], split.drugs, split.diseases, keep),
                    }
                    for stratum, keep in strata.items()
                }
            if section:
                results["views"][column] = section
    args.output.write_text(json.dumps(results, indent=2) + "\n")

    print("coverage:", {name: row["seeds"] for name, row in results["coverage"].items()})
    for column, section in results["views"].items():
        for model in MODELS:
            if model in section:
                row = section[model]["all"]
                difference = row["mrr_minus_popularity"]
                print(
                    f"{column} | {model} | MRR {row['mrr']['mean']:.3f} "
                    f"[{row['mrr']['ci95'][0]:.3f}, {row['mrr']['ci95'][1]:.3f}] | "
                    f"Hits@10 {row['hits10']['mean']:.3f} | minus popularity {difference['mean']:+.3f} "
                    f"[{difference['ci95'][0]:+.3f}, {difference['ci95'][1]:+.3f}]"
                )


if __name__ == "__main__":
    main()
