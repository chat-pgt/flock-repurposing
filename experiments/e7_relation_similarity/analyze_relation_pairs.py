"""E7 follow-up: compare how well each model separates a pair of same-type relations.

The sample is balanced, so 50% accuracy is chance. Each triple is averaged over walk seeds, and intervals
come from a two-way bootstrap over heads and tails.

Usage, from experiments/: uv run python -m e7_relation_similarity.analyze_relation_pairs --runs e7
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common.evaluation import rank_metrics
from common.hetionet import EXPERIMENTS
from common.provenance import run_metadata
from e6_relation_baseline.relation_baselines import read_triples
from gpu.analysis import interval

MODELS = ("flock", "ultra")


def load_model(runs, model, pair_names):
    paths = sorted(Path(runs).glob(f"pair_{pair_names[0]}_{pair_names[1]}_{model}_seed*.npz"))
    if not paths:
        return None
    shards = [dict(np.load(path)) for path in paths]
    first = shards[0]
    for shard in shards[1:]:
        if not (np.array_equal(shard["heads"], first["heads"]) and np.array_equal(shard["tails"], first["tails"])):
            raise ValueError(f"{model}: seeds cover different triples")
    return {
        "n_seeds": len(shards),
        "heads": first["heads"],
        "tails": first["tails"],
        "relation_names": first["relation_names"].tolist(),
        "pair": first["pair"].tolist(),
        "true_relations": first["true_relations"],
        "pair_hits1": np.mean([shard["pair_hits1"] for shard in shards], axis=0),
        "all_rr": np.mean([shard["all_relations_rr"] for shard in shards], axis=0),
    }


def pair_baselines(train_triples, heads, tails, true_relations, names, pair):
    """Return per-triple accuracy of the frequency and label-free signature baselines on the pair."""
    frequency = Counter()
    as_head, as_tail = defaultdict(set), defaultdict(set)
    for head, relation, tail in train_triples:
        frequency[relation] += 1
        as_head[head].add(relation)
        as_tail[tail].add(relation)
    counts = np.array([frequency[name] for name in names], dtype=float)
    priority = counts.sum() + 1
    in_pair = np.zeros(len(names), dtype=bool)
    in_pair[list(pair)] = True
    frequency_hits, signature_hits = [], []
    for head, tail, target in zip(list(heads), list(tails), list(true_relations), strict=True):
        seen = as_head[head] & as_tail[tail]
        signature = counts + priority * np.array([name in seen for name in names])
        frequency_hits.append(rank_metrics(counts, int(target), in_pair, k=1)[1])
        signature_hits.append(rank_metrics(signature, int(target), in_pair, k=1)[1])
    return {"frequency": np.array(frequency_hits), "signature": np.array(signature_hits)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--pair", default="CuG,CdG")
    parser.add_argument("--train", type=Path, default=EXPERIMENTS / "data/hetionet_authors/train.txt")
    parser.add_argument("--output", type=Path, default=EXPERIMENTS / "results/e7_relation_pairs.json")
    args = parser.parse_args()

    pair_names = args.pair.split(",")
    loaded = {model: load_model(args.runs, model, pair_names) for model in MODELS}
    available = [run for run in loaded.values() if run]
    if not available:
        raise SystemExit("no relation-pair runs found")
    reference = available[0]
    heads, tails, true_relations = reference["heads"], reference["tails"], reference["true_relations"]
    baselines = pair_baselines(
        read_triples(args.train),
        heads.tolist(),
        tails.tolist(),
        true_relations.tolist(),
        reference["relation_names"],
        reference["pair"],
    )
    results = {
        "metadata": run_metadata(data_files=(args.train,)),
        "pair": pair_names,
        "n_triples": len(heads),
        "chance": 0.5,
        "baselines": {name: interval(values, heads, tails) for name, values in baselines.items()},
        "models": {},
    }
    for model, run in loaded.items():
        if not run:
            continue
        if not (np.array_equal(run["heads"], heads) and np.array_equal(run["tails"], tails)):
            raise ValueError(f"{model}: runs cover different triples")
        results["models"][model] = {
            "n_seeds": run["n_seeds"],
            "pair_accuracy": interval(run["pair_hits1"], heads, tails),
            "accuracy_by_true_relation": {
                run["relation_names"][relation]: float(run["pair_hits1"][true_relations == relation].mean())
                for relation in run["pair"]
            },
            "all_relations_mrr": interval(run["all_rr"], heads, tails),
            "minus_frequency": interval(run["pair_hits1"] - baselines["frequency"], heads, tails),
            "minus_signature": interval(run["pair_hits1"] - baselines["signature"], heads, tails),
        }
    if all(loaded.values()):
        results["flock_minus_ultra"] = interval(
            loaded["flock"]["pair_hits1"] - loaded["ultra"]["pair_hits1"], heads, tails
        )
    args.output.write_text(json.dumps(results, indent=2) + "\n")

    for name, row in results["baselines"].items():
        print(f"{name}: accuracy {row['mean']:.3f} [{row['ci95'][0]:.3f}, {row['ci95'][1]:.3f}]")
    for model, row in results["models"].items():
        accuracy = row["pair_accuracy"]
        print(
            f"{model} ({row['n_seeds']} seeds): accuracy {accuracy['mean']:.3f} "
            f"[{accuracy['ci95'][0]:.3f}, {accuracy['ci95'][1]:.3f}] | "
            f"by relation {row['accuracy_by_true_relation']} | "
            f"minus signature {row['minus_signature']['mean']:+.3f}"
        )


if __name__ == "__main__":
    main()
