"""E6: compare FLOCK and ULTRA with the count baselines on the Compound-Disease test triples.

Each query asks which relation links a known compound-disease pair. Results are reported over all 24
relations, as in the released evaluation, and over *treats* vs *palliates* only.

Usage, from experiments/: uv run python -m gpu.analyze_e6 --runs path/to/runs
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common.evaluation import rank_metrics
from common.hetionet import EXPERIMENTS, PALLIATES, TREATS
from common.provenance import run_metadata
from e6_relation_baseline.relation_baselines import read_triples
from gpu.analysis import interval

MODELS = ("flock", "ultra")


def load_model(runs, model):
    paths = sorted(Path(runs).glob(f"relation_{model}_seed*.npz"))
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
        "true_relations": first["true_relations"],
        "all_rr": np.mean([shard["all_relations_rr"] for shard in shards], axis=0),
        "all_hits1": np.mean([shard["all_relations_hits1"] for shard in shards], axis=0),
        "pair_hits1": np.mean([shard["treats_vs_palliates_hits1"] for shard in shards], axis=0),
    }


def count_baselines(train_triples, heads, tails, true_relations, relation_names):
    """Return per-triple metrics of the frequency and label-free signature baselines."""
    frequency = Counter()
    as_head, as_tail = defaultdict(set), defaultdict(set)
    for head, relation, tail in train_triples:
        frequency[relation] += 1
        as_head[head].add(relation)
        as_tail[tail].add(relation)
    counts = np.array([frequency[name] for name in relation_names], dtype=float)
    priority = counts.sum() + 1
    pair = np.zeros(len(relation_names), dtype=bool)
    pair[[relation_names.index(TREATS), relation_names.index(PALLIATES)]] = True
    everyone = np.ones(len(relation_names), dtype=bool)

    rows = defaultdict(list)
    for head, tail, target in zip(heads.tolist(), tails.tolist(), true_relations.tolist(), strict=True):
        seen = as_head[head] & as_tail[tail]
        signature = counts + priority * np.array([name in seen for name in relation_names])
        rr, hits = rank_metrics(signature, target, everyone, k=1)
        rows["signature_all_rr"].append(rr)
        rows["signature_all_hits1"].append(hits)
        rows["signature_pair_hits1"].append(rank_metrics(signature, target, pair, k=1)[1])
        rows["frequency_pair_hits1"].append(rank_metrics(counts, target, pair, k=1)[1])
    return {name: np.array(values) for name, values in rows.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=EXPERIMENTS / "data/hetionet_authors/train.txt")
    parser.add_argument("--output", type=Path, default=EXPERIMENTS / "results/e6_compound_disease.json")
    args = parser.parse_args()

    loaded = {model: load_model(args.runs, model) for model in MODELS}
    available = [run for run in loaded.values() if run]
    if not available:
        raise SystemExit("no relation runs found")
    reference = available[0]
    heads, tails = reference["heads"], reference["tails"]
    baselines = count_baselines(
        read_triples(args.train), heads, tails, reference["true_relations"], reference["relation_names"]
    )
    results = {
        "metadata": run_metadata(data_files=(args.train,)),
        "n_triples": len(heads),
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
            "all_relations_mrr": interval(run["all_rr"], heads, tails),
            "all_relations_hits1": interval(run["all_hits1"], heads, tails),
            "treats_vs_palliates_hits1": interval(run["pair_hits1"], heads, tails),
            "pair_hits1_minus_frequency": interval(run["pair_hits1"] - baselines["frequency_pair_hits1"], heads, tails),
            "pair_hits1_minus_signature": interval(run["pair_hits1"] - baselines["signature_pair_hits1"], heads, tails),
        }
        row = results["models"][model]
        print(
            f"{model} ({run['n_seeds']} seeds): 24-way MRR {row['all_relations_mrr']['mean']:.3f} | "
            f"treats vs palliates Hits@1 {row['treats_vs_palliates_hits1']['mean']:.3f} "
            f"[{row['treats_vs_palliates_hits1']['ci95'][0]:.3f}, {row['treats_vs_palliates_hits1']['ci95'][1]:.3f}]"
        )
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(
        f"baselines: frequency {results['baselines']['frequency_pair_hits1']['mean']:.3f}, "
        f"signature {results['baselines']['signature_pair_hits1']['mean']:.3f}"
    )


if __name__ == "__main__":
    main()
