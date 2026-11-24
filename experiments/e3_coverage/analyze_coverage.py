"""E3: compare each coverage setting with FLOCK's reference run on the same held-out edges.

The reference is seed 0 of the E1 run (n = 512, l = 128, full graph). The difference between E1's own seeds
on the same edges gives the noise floor: a setting only matters if it moves the score by more than that.

Usage, from experiments/: uv run python -m e3_coverage.analyze_coverage --runs e3 --reference runs/e1
"""

import argparse
import json
from pathlib import Path

import numpy as np

from common.hetionet import EXPERIMENTS, SPLIT_PATH, read_split
from common.provenance import run_metadata
from e3_coverage.plan import SETTINGS
from gpu.analysis import N_FOLDS, complete_seeds, interval


def read_edges_metrics(paths):
    """Return edge rows, reciprocal ranks and Hits@10 from a set of shards, sorted by row."""
    rows, reciprocal_ranks, hits = [], [], []
    for path in paths:
        with np.load(path) as shard:
            rows.append(shard["edge_rows"])
            reciprocal_ranks.append(shard["reciprocal_rank"])
            hits.append(shard["hits10"])
    rows, reciprocal_ranks, hits = map(np.concatenate, (rows, reciprocal_ranks, hits))
    order = np.argsort(rows)
    return rows[order], reciprocal_ranks[order], hits[order]


def load_setting(runs, name):
    paths = [Path(runs) / f"e3_{name}_fold{fold}.npz" for fold in range(N_FOLDS)]
    return read_edges_metrics(paths) if all(path.exists() for path in paths) else None


def select(reference, rows):
    """Align a reference run with the edges of a setting."""
    reference_rows, reciprocal_ranks, hits = reference
    positions = np.searchsorted(reference_rows, rows)
    if (positions >= len(reference_rows)).any() or not np.array_equal(reference_rows[positions], rows):
        raise ValueError("the reference run does not cover these edges")
    return reciprocal_ranks[positions], hits[positions]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="Directory with the E1 FLOCK shards")
    parser.add_argument("--split", type=Path, default=SPLIT_PATH)
    parser.add_argument("--output", type=Path, default=EXPERIMENTS / "results/e3_coverage.json")
    args = parser.parse_args()

    split = read_split(args.split)
    seeds = complete_seeds(args.reference, "flock", "full", "tail")
    if 0 not in seeds:
        raise SystemExit("the reference run has no complete seed 0")
    references = {seed: read_edges_metrics(paths) for seed, paths in seeds.items()}
    results = {
        "metadata": run_metadata(data_files=(args.split,)),
        "reference": {"setting": "n512_l128", "budget": 512 * 128, "seeds": sorted(seeds)},
        "settings": {},
    }
    for name, setting in SETTINGS.items():
        loaded = load_setting(args.runs, name)
        if loaded is None:
            print(f"{name}: not complete")
            continue
        rows, reciprocal_ranks, hits = loaded
        drugs, diseases = split.drugs[rows], split.diseases[rows]
        reference_rr, reference_hits = select(references[0], rows)
        row = {
            "walk_num": setting.walk_num,
            "walk_len": setting.walk_len,
            "budget": setting.budget,
            "dropped_relations": list(setting.dropped_relations),
            "n_edges": len(rows),
            "mrr": interval(reciprocal_ranks, drugs, diseases),
            "hits10": interval(hits, drugs, diseases),
            "reference_mrr": float(reference_rr.mean()),
            "reference_hits10": float(reference_hits.mean()),
            "mrr_minus_reference": interval(reciprocal_ranks - reference_rr, drugs, diseases),
            "hits10_minus_reference": interval(hits - reference_hits, drugs, diseases),
        }
        noise = {}
        for seed in sorted(seeds):
            if seed != 0:
                other_rr, _ = select(references[seed], rows)
                noise[f"seed{seed}_minus_seed0"] = interval(other_rr - reference_rr, drugs, diseases)
        row["noise_floor_mrr"] = noise
        results["settings"][name] = row
        difference = row["mrr_minus_reference"]
        floors = ", ".join(f"{key} {value['mean']:+.3f}" for key, value in noise.items()) or "no second seed"
        print(
            f"{name}: budget {setting.budget}, MRR {row['mrr']['mean']:.3f} vs {row['reference_mrr']:.3f} | "
            f"difference {difference['mean']:+.3f} [{difference['ci95'][0]:+.3f}, {difference['ci95'][1]:+.3f}] | "
            f"noise {floors}"
        )
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
