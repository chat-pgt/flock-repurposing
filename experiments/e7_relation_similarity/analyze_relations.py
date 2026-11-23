"""E7: test whether ULTRA and FLOCK separate relations that connect the same entity types.

Cosines are reported raw and after subtracting the mean relation, and target pairs are read against
control pairs that connect different entity types. The FLOCK figures average over seeds, queries and passes.

Usage, from experiments/: uv run python -m e7_relation_similarity.analyze_relations --runs e7
"""

import argparse
import json
from pathlib import Path

import numpy as np

from common.hetionet import EXPERIMENTS, TREATS
from common.provenance import run_metadata
from e7_relation_similarity.similarity import TARGET_PAIRS, pca_2d, summarise


def load_ultra(runs):
    path = Path(runs) / "e7_ultra.npz"
    if not path.exists():
        return None
    with np.load(path) as shard:
        names = shard["relation_names"].tolist()
        return {
            "states": shard["states"][names.index(TREATS), : len(names)],
            "names": names,
            "head_kinds": shard["head_kinds"].tolist(),
            "tail_kinds": shard["tail_kinds"].tolist(),
        }


def load_flock(runs):
    paths = sorted(Path(runs).glob("e7_flock_seed*.npz"))
    if not paths:
        return None
    shards = [dict(np.load(path)) for path in paths]
    names = shards[0]["relation_names"].tolist()
    for shard in shards:
        if not np.array_equal(shard["query_entities"], shards[0]["query_entities"]):
            raise ValueError("FLOCK seeds cover different queries")
    states = np.stack([shard["states"] for shard in shards])[..., : len(names), :]
    visited = np.stack([shard["visited"] for shard in shards])[..., : len(names)]
    return {
        "states": states,
        "visited_share": float(visited.mean()),
        "n_seeds": len(shards),
        "names": names,
        "head_kinds": shards[0]["head_kinds"].tolist(),
        "tail_kinds": shards[0]["tail_kinds"].tolist(),
    }


def pca_output(vectors, names):
    coordinates, explained = pca_2d(vectors)
    return {
        "explained_variance": explained,
        "coordinates": {name: coordinates[index].tolist() for index, name in enumerate(names)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=EXPERIMENTS / "results/e7_relation_similarity.json")
    args = parser.parse_args()

    results = {"metadata": run_metadata(), "target_pairs": TARGET_PAIRS, "models": {}}
    ultra = load_ultra(args.runs)
    if ultra:
        results["models"]["ultra"] = {
            **summarise(ultra["states"], ultra["names"], ultra["head_kinds"], ultra["tail_kinds"]),
            "pca": pca_output(ultra["states"], ultra["names"]),
        }
    flock = load_flock(args.runs)
    if flock:
        per_pass = summarise(flock["states"], flock["names"], flock["head_kinds"], flock["tail_kinds"])
        averaged = flock["states"].mean(axis=(0, 1, 2))
        results["models"]["flock"] = {
            **per_pass,
            "n_seeds": flock["n_seeds"],
            "visited_share": flock["visited_share"],
            "averaged_state": summarise(averaged, flock["names"], flock["head_kinds"], flock["tail_kinds"]),
            "pca": pca_output(averaged, flock["names"]),
        }
    if not results["models"]:
        raise SystemExit("no E7 runs found")
    args.output.write_text(json.dumps(results, indent=2) + "\n")

    for model, row in results["models"].items():
        for label in ("raw", "centred"):
            targets = ", ".join(f"{name}: {value['mean']:.3f}" for name, value in row[label]["targets"].items())
            controls = row[label]["controls"]
            print(
                f"{model} {label} | {targets} | controls median {controls['median']:.3f} "
                f"[{controls['p05']:.3f}, {controls['p95']:.3f}]"
            )


if __name__ == "__main__":
    main()
