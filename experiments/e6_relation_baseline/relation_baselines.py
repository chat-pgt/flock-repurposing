"""E6: how much of relation prediction on Hetionet can simple counts explain?

Matches the released evaluation on the authors' test triples: one query (h, ?, t) per triple, ranked
without filtering over forward relations.
- Baseline A picks uniformly among relations seen in training between the entity types of h and t.
- Baseline B ranks those relations by training frequency.
- Baseline C uses no entity-type labels, only the graph the models also receive. Relations that h has
  used as a head and t has used as a tail in training come first, and each group is ordered by training
  frequency.

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
SUMMARY_METRICS = (
    ("uniform_mrr", "uniform_rr"),
    ("uniform_hits1", "uniform_hits1"),
    ("frequency_mrr", "frequency_rr"),
    ("frequency_hits1", "frequency_hits1"),
    ("signature_mrr", "signature_rr"),
    ("signature_hits1", "signature_hits1"),
)


def read_triples(path):
    with open(path) as handle:
        for line in handle:
            head, relation, tail = line.rstrip("\n").split("\t")
            yield head, relation, tail


def count_triples(path):
    counts = Counter()
    for head, relation, tail in read_triples(path):
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


def signature_metrics(train_triples, test_triples):
    """Return Baseline C's mean metrics per (head type, relation, tail type)."""
    relation_ids, frequency = {}, Counter()
    as_head, as_tail = defaultdict(int), defaultdict(int)
    for head, relation, tail in train_triples:
        bit = 1 << relation_ids.setdefault(relation, len(relation_ids))
        frequency[relation] += 1
        as_head[head] |= bit
        as_tail[tail] |= bit

    relations = sorted(relation_ids)
    counts = np.array([frequency[name] for name in relations], dtype=float)
    bits = np.array([1 << relation_ids[name] for name in relations], dtype=np.int64)
    priority = counts.sum() + 1
    everyone = np.ones(len(relations), dtype=bool)

    cache = {}
    totals = defaultdict(lambda: [0, 0.0, 0.0])
    for head, relation, tail in test_triples:
        if relation not in relation_ids:
            raise ValueError(f"{relation} never appears in training")
        signature = as_head.get(head, 0) & as_tail.get(tail, 0)
        if (signature, relation) not in cache:
            scores = counts + priority * ((bits & signature) != 0)
            cache[signature, relation] = rank_metrics(scores, relations.index(relation), everyone, k=1)
        reciprocal_rank, hit = cache[signature, relation]
        total = totals[(node_kind(head), relation, node_kind(tail))]
        total[0] += 1
        total[1] += reciprocal_rank
        total[2] += hit
    return {key: {"signature_rr": rr / n, "signature_hits1": hits / n} for key, (n, rr, hits) in totals.items()}


def weighted_summary(rows):
    n_test = sum(row["n_test"] for row in rows)
    if n_test == 0:
        return {"n_test": 0}
    summary = {"n_test": n_test}
    for name, key in SUMMARY_METRICS:
        if key in rows[0]:
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
    signature = signature_metrics(read_triples(train_path), read_triples(test_path))
    for row in rows:
        row.update(signature[(row["head_type"], row["relation"], row["tail_type"])])
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
            f"{name}: n={row['n_test']} | A MRR {row['uniform_mrr']:.3f}, Hits@1 {row['uniform_hits1']:.3f} | "
            f"B MRR {row['frequency_mrr']:.3f}, Hits@1 {row['frequency_hits1']:.3f} | "
            f"C MRR {row['signature_mrr']:.3f}, Hits@1 {row['signature_hits1']:.3f}"
        )
    print("paper:", results["paper_zero_shot"])
    print("type pair | n | k | A MRR | B MRR | B Hits@1 | C MRR | C Hits@1")
    for pair, row in results["per_type_pair"].items():
        print(
            f"{pair} | {row['n_test']} | {row['n_compatible']} | {row['uniform_mrr']:.3f} | "
            f"{row['frequency_mrr']:.3f} | {row['frequency_hits1']:.3f} | "
            f"{row['signature_mrr']:.3f} | {row['signature_hits1']:.3f}"
        )


if __name__ == "__main__":
    main()
