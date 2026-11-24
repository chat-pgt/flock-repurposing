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
from e7_relation_similarity.similarity import (
    TARGET_PAIRS,
    across_pass_cosines,
    centre,
    control_pairs,
    leave_one_query_out_accuracy,
    pca_2d,
    relative_distance,
    summarise,
)

N_CONTROL_PROBES = 40
CONTROL_SAMPLE_SEED = 0


def ultra_merge_check(all_states, names):
    """Measure each target pair on query relations outside the pair, where no query marker separates it."""
    output = {}
    for label, (first, second) in TARGET_PAIRS.items():
        i, j = names.index(first), names.index(second)
        queries = [query for query in range(len(names)) if query not in (i, j)]
        distances = relative_distance(all_states[queries, i], all_states[queries, j])
        output[label] = {
            "n_queries_outside_pair": len(queries),
            "max_relative_distance": float(distances.max()),
            "relative_distance_when_first_is_the_query": float(relative_distance(all_states[i, i], all_states[i, j])),
        }
    return output


def flock_noise_check(states, names, head_kinds, tail_kinds, query_relation=TREATS):
    """Test whether FLOCK's relation differences exceed walk noise and hold across held-out drugs."""
    centred = centre(states)
    pairs = {}
    for label, (first, second) in TARGET_PAIRS.items():
        i, j = names.index(first), names.index(second)
        same, different = across_pass_cosines(centred, i, j)
        pairs[label] = {
            "same_relation_across_passes": same,
            "different_relation_across_passes": different,
            "leave_one_drug_out_accuracy": leave_one_query_out_accuracy(states, i, j),
            "query_relation_in_pair": query_relation in (first, second),
        }
    controls = control_pairs(names, head_kinds, tail_kinds)
    sample = np.random.default_rng(CONTROL_SAMPLE_SEED).choice(
        len(controls), size=min(N_CONTROL_PROBES, len(controls)), replace=False
    )
    accuracy = [
        leave_one_query_out_accuracy(states, names.index(controls[k][0]), names.index(controls[k][1]))
        for k in sample.tolist()
    ]
    return {
        "pairs": pairs,
        "control_accuracy": {
            "n_pairs": len(accuracy),
            "median": float(np.median(accuracy)),
            "min": float(min(accuracy)),
        },
    }


def load_ultra(runs):
    path = Path(runs) / "e7_ultra.npz"
    if not path.exists():
        return None
    with np.load(path) as shard:
        names = shard["relation_names"].tolist()
        return {
            "states": shard["states"][names.index(TREATS), : len(names)],
            "all_states": shard["states"][:, : len(names)],
            "names": names,
            "head_kinds": shard["head_kinds"].tolist(),
            "tail_kinds": shard["tail_kinds"].tolist(),
        }


def load_flock(runs, pattern="e7_flock_seed*.npz"):
    paths = sorted(Path(runs).glob(pattern))
    if not paths:
        return None
    shards = [dict(np.load(path)) for path in paths]
    names = shards[0]["relation_names"].tolist()
    query_relations = {json.loads(str(shard["metadata"])).get("query_relation", TREATS) for shard in shards}
    if len(query_relations) != 1:
        raise ValueError(f"FLOCK seeds use different query relations: {sorted(query_relations)}")
    for shard in shards:
        if not np.array_equal(shard["query_entities"], shards[0]["query_entities"]):
            raise ValueError("FLOCK seeds cover different queries")
    states = np.stack([shard["states"] for shard in shards])[..., : len(names), :]
    visited = np.stack([shard["visited"] for shard in shards])[..., : len(names)]
    return {
        "query_relation": query_relations.pop(),
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
            "merge_check": ultra_merge_check(ultra["all_states"], ultra["names"]),
            "pca": pca_output(ultra["states"], ultra["names"]),
        }
    for model_name, pattern in (("flock", "e7_flock_seed*.npz"), ("flock_neutral_query", "e7_flock_query-*_seed*.npz")):
        flock = load_flock(args.runs, pattern)
        if not flock:
            continue
        labels = (flock["names"], flock["head_kinds"], flock["tail_kinds"])
        averaged = flock["states"].mean(axis=(0, 1, 2))
        results["models"][model_name] = {
            **summarise(flock["states"], *labels),
            "query_relation": flock["query_relation"],
            "n_seeds": flock["n_seeds"],
            "visited_share": flock["visited_share"],
            "averaged_state": summarise(averaged, *labels),
            "noise_check": flock_noise_check(flock["states"], *labels, query_relation=flock["query_relation"]),
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
    if "ultra" in results["models"]:
        for label, row in results["models"]["ultra"]["merge_check"].items():
            print(f"ultra merge | {label}: max relative distance {row['max_relative_distance']:.1e}")
    for model_name in ("flock", "flock_neutral_query"):
        if model_name not in results["models"]:
            continue
        check = results["models"][model_name]["noise_check"]
        for label, row in check["pairs"].items():
            marker = " (query relation in pair)" if row["query_relation_in_pair"] else ""
            print(
                f"{model_name} noise | {label}{marker}: same across passes {row['same_relation_across_passes']:.3f}, "
                f"different {row['different_relation_across_passes']:.3f}, "
                f"held-out accuracy {row['leave_one_drug_out_accuracy']:.3f}"
            )
        print(f"{model_name} noise | control pairs: {check['control_accuracy']}")


if __name__ == "__main__":
    main()
