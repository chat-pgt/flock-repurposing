"""E2, CPU part: how often does a drug's close analogue already treat the held-out disease?

An edge is flagged when a compound that the drug *resembles* treats the same disease in that fold's
training graph. Hub diseases are flagged more easily, so each edge is compared with two controls over
the diseases the drug is not known to treat: a uniform average, and an average weighted by each
disease's number of training drugs.

Usage, from experiments/: uv run python -m e2_leakage.near_duplicates
"""

import csv
import json
from collections import defaultdict

import numpy as np

from common.evaluation import twoway_bootstrap_ci
from common.hetionet import EDGES_PATH, EXPERIMENTS, SPLIT_PATH, load_repurposing_graph, read_split
from common.provenance import run_metadata

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0
NOTHING = frozenset()


def leakage_by_edge(drugs, diseases, folds, resembles, candidates):
    """Return, per edge, the flag, both control rates and whether the drug has any analogue."""
    known = defaultdict(set)
    training = {int(fold): defaultdict(set) for fold in np.unique(folds)}
    edges = list(zip(drugs.tolist(), diseases.tolist(), folds.tolist(), strict=True))
    for drug, disease, edge_fold in edges:
        known[drug].add(disease)
        for fold, treaters in training.items():
            if fold != edge_fold:
                treaters[disease].add(drug)

    table = {name: np.zeros(len(edges)) for name in ("flagged", "uniform", "weighted", "has_analogue")}
    for row, (drug, disease, fold) in enumerate(edges):
        analogues = resembles.get(drug, NOTHING)
        treaters = training[fold]
        negatives = [node for node in candidates if node not in known[drug]]
        flags = np.array([bool(analogues & treaters.get(node, NOTHING)) for node in negatives], dtype=float)
        weights = np.array([len(treaters.get(node, NOTHING)) for node in negatives], dtype=float)
        table["flagged"][row] = bool(analogues & treaters.get(disease, NOTHING))
        table["uniform"][row] = flags.mean()
        table["weighted"][row] = flags @ weights / weights.sum() if weights.sum() else 0.0
        table["has_analogue"][row] = bool(analogues)
    return table


def summarise(table, drugs, diseases, keep):
    def interval(values):
        mean, low, high = twoway_bootstrap_ci(
            values[keep], drugs[keep], diseases[keep], draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
        )
        return {"mean": mean, "ci95": [low, high]}

    flagged = float(table["flagged"][keep].mean())
    weighted = float(table["weighted"][keep].mean())
    return {
        "n_edges": int(keep.sum()),
        "share_with_any_analogue": float(table["has_analogue"][keep].mean()),
        "flagged": interval(table["flagged"]),
        "control_uniform": float(table["uniform"][keep].mean()),
        "control_popularity_weighted": weighted,
        "flagged_minus_weighted_control": interval(table["flagged"] - table["weighted"]),
        "enrichment_over_weighted_control": flagged / weighted if weighted else None,
    }


def main():
    graph = load_repurposing_graph()
    split = read_split()
    table = leakage_by_edge(split.drugs, split.diseases, split.folds, graph.resembles, graph.diseases)
    strata = {
        "all": np.ones(len(split.drugs), dtype=bool),
        "drug_has_other_indication": split.has_other_indication,
        "first_indication": ~split.has_other_indication,
    }
    results = {
        "metadata": run_metadata(
            data_files=(EDGES_PATH, SPLIT_PATH), bootstrap_draws=BOOTSTRAP_DRAWS, bootstrap_seed=BOOTSTRAP_SEED
        ),
        "strata": {name: summarise(table, split.drugs, split.diseases, keep) for name, keep in strata.items()},
    }
    (EXPERIMENTS / "results/e2_leakage.json").write_text(json.dumps(results, indent=2) + "\n")
    with open(EXPERIMENTS / "results/e2_leakage_edges.tsv", "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["compound", "disease", "fold", "flagged", "control_uniform", "control_weighted"])
        for row in range(len(split.drugs)):
            writer.writerow(
                [
                    split.drugs[row],
                    split.diseases[row],
                    split.folds[row],
                    int(table["flagged"][row]),
                    f"{table['uniform'][row]:.6f}",
                    f"{table['weighted'][row]:.6f}",
                ]
            )

    print("stratum | n | has analogue | flagged [95% CI] | uniform | weighted | diff [95% CI] | enrichment")
    for name, row in results["strata"].items():
        difference = row["flagged_minus_weighted_control"]
        print(
            f"{name} | {row['n_edges']} | {row['share_with_any_analogue']:.3f} | {row['flagged']['mean']:.3f} "
            f"[{row['flagged']['ci95'][0]:.3f}, {row['flagged']['ci95'][1]:.3f}] | {row['control_uniform']:.3f} | "
            f"{row['control_popularity_weighted']:.3f} | {difference['mean']:.3f} "
            f"[{difference['ci95'][0]:.3f}, {difference['ci95'][1]:.3f}] | "
            f"{row['enrichment_over_weighted_control']:.2f}"
        )


if __name__ == "__main__":
    main()
