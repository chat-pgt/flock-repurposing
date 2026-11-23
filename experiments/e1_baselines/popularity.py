"""E1, CPU part: rank candidates by their number of treats edges in each fold's training graph.

Tail queries rank diseases for a drug, and head queries rank drugs for a disease. The score ignores the
query entirely, so anything a model gains over it comes from more than hub popularity.

Usage, from experiments/: uv run python -m e1_baselines.popularity
"""

import csv
import json
from collections import defaultdict

import numpy as np

from common.evaluation import candidate_mask, rank_metrics, twoway_bootstrap_ci
from common.hetionet import EDGES_PATH, EXPERIMENTS, SPLIT_PATH, load_repurposing_graph, read_split
from common.provenance import run_metadata

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0
METRICS = ("rr", "hits10", "random_rr", "random_hits10")


def score_direction(query_nodes, answer_nodes, folds, candidates, n_entities, palliated_pairs):
    """Return per-edge metrics for the candidate set, the palliates filter and all entities."""
    index = {node: position for position, node in enumerate(candidates)}
    answers = np.array([index[node] for node in answer_nodes])
    n_candidates = len(candidates)
    counts = {int(fold): np.bincount(answers[folds != fold], minlength=n_candidates) for fold in np.unique(folds)}

    known, palliated = defaultdict(set), defaultdict(set)
    for query, answer in zip(query_nodes.tolist(), answers.tolist(), strict=True):
        known[query].add(answer)
    for query, node in palliated_pairs:
        if node in index:
            palliated[query].add(index[node])

    views = {
        name: {metric: np.empty(len(answers)) for metric in METRICS}
        for name in ("candidates", "filter_palliates", "all_entities")
    }
    every_candidate = range(n_candidates)
    every_entity = range(n_entities)
    padding = np.zeros(n_entities - n_candidates)
    for row, (query, target, fold) in enumerate(
        zip(query_nodes.tolist(), answers.tolist(), folds.tolist(), strict=True)
    ):
        other_answers = known[query] - {target}
        scores = counts[fold].astype(float)
        cases = {
            "candidates": (scores, candidate_mask(n_candidates, every_candidate, other_answers, target)),
            "filter_palliates": (
                scores,
                candidate_mask(n_candidates, every_candidate, other_answers | palliated[query], target),
            ),
            "all_entities": (
                np.concatenate([scores, padding]),
                candidate_mask(n_entities, every_entity, other_answers, target),
            ),
        }
        for name, (case_scores, mask) in cases.items():
            views[name]["rr"][row], views[name]["hits10"][row] = rank_metrics(case_scores, target, mask)
            random_scores = np.zeros(len(case_scores))
            views[name]["random_rr"][row], views[name]["random_hits10"][row] = rank_metrics(random_scores, target, mask)
    return views


def summarise(view, drugs, diseases, keep):
    def interval(values):
        mean, low, high = twoway_bootstrap_ci(
            values[keep], drugs[keep], diseases[keep], draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
        )
        return {"mean": mean, "ci95": [low, high]}

    return {
        "n_edges": int(keep.sum()),
        "mrr": interval(view["rr"]),
        "hits10": interval(view["hits10"]),
        "random_mrr": float(view["random_rr"][keep].mean()),
        "random_hits10": float(view["random_hits10"][keep].mean()),
        "mrr_minus_random": interval(view["rr"] - view["random_rr"]),
        "hits10_minus_random": interval(view["hits10"] - view["random_hits10"]),
    }


def write_edges(path, split, views):
    columns = [f"{view}_{metric}" for view in views for metric in METRICS]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["compound", "disease", "fold", *columns])
        for row in range(len(split.drugs)):
            values = [f"{views[view][metric][row]:.6f}" for view in views for metric in METRICS]
            writer.writerow([split.drugs[row], split.diseases[row], split.folds[row], *values])


def main():
    graph = load_repurposing_graph()
    split = read_split()
    flipped_palliates = {(disease, drug) for drug, disease in graph.palliates}
    tail = score_direction(split.drugs, split.diseases, split.folds, graph.diseases, graph.n_entities, graph.palliates)
    head = score_direction(
        split.diseases, split.drugs, split.folds, graph.compounds, graph.n_entities, flipped_palliates
    )
    views = {f"tail_{name}": view for name, view in tail.items()} | {
        f"head_{name}": view for name, view in head.items()
    }

    strata = {
        "all": np.ones(len(split.drugs), dtype=bool),
        "drug_has_other_indication": split.has_other_indication,
        "first_indication": ~split.has_other_indication,
    }
    results = {
        "metadata": run_metadata(
            data_files=(EDGES_PATH, SPLIT_PATH),
            bootstrap_draws=BOOTSTRAP_DRAWS,
            bootstrap_seed=BOOTSTRAP_SEED,
            n_diseases=len(graph.diseases),
            n_compounds=len(graph.compounds),
            n_entities=graph.n_entities,
        ),
        "views": {
            name: {stratum: summarise(view, split.drugs, split.diseases, keep) for stratum, keep in strata.items()}
            for name, view in views.items()
        },
    }
    output = EXPERIMENTS / "results/e1_popularity.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    write_edges(EXPERIMENTS / "results/e1_popularity_edges.tsv", split, views)

    print("view | stratum | n | MRR [95% CI] | random | Hits@10 [95% CI] | random")
    for name, view in results["views"].items():
        for stratum, row in view.items():
            print(
                f"{name} | {stratum} | {row['n_edges']} | {row['mrr']['mean']:.3f} "
                f"[{row['mrr']['ci95'][0]:.3f}, {row['mrr']['ci95'][1]:.3f}] | {row['random_mrr']:.3f} | "
                f"{row['hits10']['mean']:.3f} [{row['hits10']['ci95'][0]:.3f}, {row['hits10']['ci95'][1]:.3f}] | "
                f"{row['random_hits10']:.3f}"
            )


if __name__ == "__main__":
    main()
