"""Turn saved GPU runs into per-edge metrics and paired comparisons. NumPy only."""

import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from common.evaluation import candidate_mask, rank_metrics, twoway_bootstrap_ci

N_FOLDS = 5
VIEWS = ("candidates", "filter_palliates", "all_entities")
METRICS = ("rr", "hits10")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0


def expected_metrics(above, tied, k=10):
    """Return the expected reciprocal rank and Hits@k under random tie-breaking."""
    ranks = np.arange(above + 1, above + tied + 2)
    return float((1 / ranks).mean()), float((ranks <= k).mean())


def complete_seeds(runs, model, condition, direction):
    """Return the shard paths of each seed that has all five folds."""
    pattern = re.compile(rf"entity_{model}_{condition}_{direction}_seed(\d+)_fold(\d+)\.npz")
    found = defaultdict(dict)
    for path in Path(runs).glob("entity_*.npz"):
        match = pattern.fullmatch(path.name)
        if match:
            found[int(match[1])][int(match[2])] = path
    return {
        seed: [folds[fold] for fold in range(N_FOLDS)]
        for seed, folds in sorted(found.items())
        if set(folds) == set(range(N_FOLDS))
    }


def seed_metrics(paths, queries, answers, candidates, palliated):
    """Recompute each edge's metrics for one seed, in the order of the shared split."""
    position = {name: index for index, name in enumerate(candidates)}
    known, palliated_by_query = defaultdict(set), defaultdict(set)
    for query, answer in zip(queries, answers, strict=True):
        known[query].add(position[answer])
    for query, node in palliated:
        if node in position:
            palliated_by_query[query].add(position[node])

    views = {view: {metric: np.full(len(queries), np.nan) for metric in METRICS} for view in VIEWS}
    every_candidate = range(len(candidates))
    for path in paths:
        with np.load(path) as shard:
            scores = shard["candidate_scores"]
            entities = shard["query_entities"].tolist()
            edges = zip(
                shard["edge_rows"].tolist(),
                shard["edge_query_positions"].tolist(),
                shard["reciprocal_rank"].tolist(),
                shard["all_entities_above"].tolist(),
                shard["all_entities_tied"].tolist(),
                strict=True,
            )
            for row, query_position, stored_rr, above, tied in edges:
                query = entities[query_position]
                if query != queries[row]:
                    raise ValueError(f"{path.name}: edge {row} does not belong to query {query}")
                target = position[answers[row]]
                others = known[query] - {target}
                query_scores = scores[query_position]
                mask = candidate_mask(len(candidates), every_candidate, others, target)
                rr, hits = rank_metrics(query_scores, target, mask)
                if not np.isclose(rr, stored_rr):
                    raise ValueError(f"{path.name}: edge {row} does not match its stored rank")
                views["candidates"]["rr"][row], views["candidates"]["hits10"][row] = rr, hits
                mask = candidate_mask(len(candidates), every_candidate, others | palliated_by_query[query], target)
                views["filter_palliates"]["rr"][row], views["filter_palliates"]["hits10"][row] = rank_metrics(
                    query_scores, target, mask
                )
                views["all_entities"]["rr"][row], views["all_entities"]["hits10"][row] = expected_metrics(above, tied)
    if np.isnan(views["candidates"]["rr"]).any():
        raise ValueError("some edges are missing from these shards")
    return views


def average_over_seeds(per_seed):
    """Average each edge's metrics over walk seeds (D7)."""
    runs = list(per_seed.values())
    return {
        view: {metric: np.mean([run[view][metric] for run in runs], axis=0) for metric in METRICS} for view in VIEWS
    }


def interval(values, drugs, diseases, keep=None):
    keep = np.ones(len(values), dtype=bool) if keep is None else keep
    mean, low, high = twoway_bootstrap_ci(
        values[keep], drugs[keep], diseases[keep], draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
    )
    return {"mean": mean, "ci95": [low, high]}


def bootstrap_contrast(values, first, drugs, diseases, draws=BOOTSTRAP_DRAWS, level=0.95, seed=BOOTSTRAP_SEED):
    """Return the mean over `first` edges minus the mean over the rest, with a two-way bootstrap CI."""
    values, first = np.asarray(values, dtype=float), np.asarray(first, dtype=bool)
    if not first.any() or first.all():
        raise ValueError("both groups must contain at least one edge")
    _, drug_index = np.unique(drugs, return_inverse=True)
    _, disease_index = np.unique(diseases, return_inverse=True)
    n_drugs, n_diseases = drug_index.max() + 1, disease_index.max() + 1
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for draw in range(draws):
        while True:
            weights = (
                rng.multinomial(n_drugs, np.full(n_drugs, 1 / n_drugs))[drug_index]
                * rng.multinomial(n_diseases, np.full(n_diseases, 1 / n_diseases))[disease_index]
            )
            first_weight, second_weight = weights[first].sum(), weights[~first].sum()
            if first_weight and second_weight:
                break
        estimates[draw] = (
            weights[first] @ values[first] / first_weight - weights[~first] @ values[~first] / second_weight
        )
    tail = (1 - level) / 2
    low, high = np.quantile(estimates, [tail, 1 - tail])
    return {"mean": float(values[first].mean() - values[~first].mean()), "ci95": [float(low), float(high)]}
