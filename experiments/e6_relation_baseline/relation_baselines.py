"""E6: how much of relation prediction on Hetionet can entity types and frequencies explain?

Matches the released evaluation on the authors' test triples: one query (h, ?, t) per triple, ranked
without filtering over forward relations. Both baselines only consider relations seen in training
between the entity types of h and t.
- Baseline A picks uniformly among those relations.
- Baseline B ranks them by training frequency.

Usage, from experiments/: bash e6_relation_baseline/get_authors_split.sh
                          uv run python -m e6_relation_baseline.relation_baselines
"""

import json
from collections import Counter, defaultdict

import numpy as np

from common.evaluation import rank_metrics
from common.hetionet import EXPERIMENTS, node_kind
from common.provenance import run_metadata

DATA = EXPERIMENTS / "data/hetionet_authors"
TABLES = EXPERIMENTS / "paper_analysis/paper_tables.json"


def count_triples(path):
    counts = Counter()
    with open(path) as handle:
        for line in handle:
            head, relation, tail = line.rstrip("\n").split("\t")
            counts[(node_kind(head), relation, node_kind(tail))] += 1
    return counts


def baseline_rows(train_counts, test_counts):
    """Return one row per (head type, relation, tail type) in the test set."""
    compatible = defaultdict(dict)
    for (head_type, relation, tail_type), count in train_counts.items():
        compatible[(head_type, tail_type)][relation] = count

    rows = []
    for (head_type, relation, tail_type), n_test in sorted(test_counts.items()):
        group = compatible.get((head_type, tail_type), {})
        if relation not in group:
            raise ValueError(f"{relation} between {head_type} and {tail_type} never appears in training")
        relations = sorted(group)
        target = relations.index(relation)
        everyone = np.ones(len(relations), dtype=bool)
        frequencies = np.array([group[name] for name in relations], dtype=float)
        uniform_rr, uniform_hits1 = rank_metrics(np.zeros(len(relations)), target, everyone, k=1)
        frequency_rr, frequency_hits1 = rank_metrics(frequencies, target, everyone, k=1)
        rows.append(
            {
                "head_type": head_type,
                "relation": relation,
                "tail_type": tail_type,
                "n_test": n_test,
                "n_compatible": len(relations),
                "uniform_rr": uniform_rr,
                "uniform_hits1": uniform_hits1,
                "frequency_rr": frequency_rr,
                "frequency_hits1": frequency_hits1,
            }
        )
    return rows


def weighted_summary(rows):
    n_test = sum(row["n_test"] for row in rows)
    if n_test == 0:
        return {"n_test": 0}
    summary = {"n_test": n_test}
    for name, key in (
        ("uniform_mrr", "uniform_rr"),
        ("uniform_hits1", "uniform_hits1"),
        ("frequency_mrr", "frequency_rr"),
        ("frequency_hits1", "frequency_hits1"),
    ):
        summary[name] = sum(row[key] * row["n_test"] for row in rows) / n_test
    return summary


def type_pair_summaries(rows):
    pairs = defaultdict(list)
    for row in rows:
        pairs[f"{row['head_type']} -> {row['tail_type']}"].append(row)
    return {
        pair: {
            "relations": sorted(row["relation"] for row in group),
            "n_compatible": group[0]["n_compatible"],
            **weighted_summary(group),
        }
        for pair, group in sorted(pairs.items(), key=lambda item: -sum(row["n_test"] for row in item[1]))
    }


def main():
    train_path, test_path = DATA / "train.txt", DATA / "test.txt"
    rows = baseline_rows(count_triples(train_path), count_triples(test_path))
    paper = next(row for row in json.loads(TABLES.read_text())["11"] if row["ds"] == "Hetionet")
    results = {
        "metadata": run_metadata(data_files=(train_path, test_path)),
        "paper_zero_shot": {
            "ULTRA": {"mrr": paper["U"], "hits1": paper["Uh"]},
            "TRIX": {"mrr": paper["T"], "hits1": paper["Th"]},
            "FLOCK": {"mrr": paper["F"], "hits1": paper["Fh"]},
        },
        "overall": weighted_summary(rows),
        "single_candidate": weighted_summary([row for row in rows if row["n_compatible"] == 1]),
        "several_candidates": weighted_summary([row for row in rows if row["n_compatible"] > 1]),
        "per_type_pair": type_pair_summaries(rows),
        "per_relation": rows,
    }
    (EXPERIMENTS / "results/e6_relation_baselines.json").write_text(json.dumps(results, indent=2) + "\n")

    for name in ("overall", "single_candidate", "several_candidates"):
        row = results[name]
        print(
            f"{name}: n={row['n_test']} | uniform MRR {row['uniform_mrr']:.3f}, Hits@1 {row['uniform_hits1']:.3f} | "
            f"frequency MRR {row['frequency_mrr']:.3f}, Hits@1 {row['frequency_hits1']:.3f}"
        )
    print("paper:", results["paper_zero_shot"])
    print("type pair | n | k | relations | uniform MRR | frequency MRR | frequency Hits@1")
    for pair, row in results["per_type_pair"].items():
        print(
            f"{pair} | {row['n_test']} | {row['n_compatible']} | {','.join(row['relations'])} | "
            f"{row['uniform_mrr']:.3f} | {row['frequency_mrr']:.3f} | {row['frequency_hits1']:.3f}"
        )


if __name__ == "__main__":
    main()
