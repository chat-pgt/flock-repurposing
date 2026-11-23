"""Split, ranking and uncertainty code shared by the experiments (PROTOCOL.md, section 1)."""

import numpy as np


def make_folds(drugs, diseases, n_folds=5, seed=0):
    """Assign each treats edge to a fold, stratified by disease (D2).

    The assignment depends on the set of edges and the seed, not on their order in the input.
    """
    drugs, diseases = np.asarray(drugs), np.asarray(diseases)
    if len(drugs) != len(diseases):
        raise ValueError("drugs and diseases must have the same length")

    rng = np.random.default_rng(seed)
    order = np.lexsort((drugs, diseases))
    folds = np.empty(len(drugs), dtype=np.int64)
    next_fold = 0
    for disease in rng.permutation(np.unique(diseases)):
        members = rng.permutation(order[diseases[order] == disease])
        folds[members] = (next_fold + np.arange(len(members))) % n_folds
        next_fold = (next_fold + len(members)) % n_folds
    return folds


def training_triples(triples, held_out):
    """Return the graph for one fold: all triples except held-out edges and their inverses (D2)."""
    removed = {tuple(triple) for triple in held_out}
    removed |= {(tail, relation, head) for head, relation, tail in removed}
    return [triple for triple in triples if tuple(triple) not in removed]


def has_other_indication(drugs, folds):
    """Flag edges whose drug keeps another treats edge in that fold's training graph (D2)."""
    drugs, folds = np.asarray(drugs), np.asarray(folds)
    flags = np.zeros(len(drugs), dtype=bool)
    for drug in np.unique(drugs):
        members = np.flatnonzero(drugs == drug)
        flags[members] = [(folds[members] != folds[index]).any() for index in members]
    return flags


def candidate_mask(num_entities, allowed, known_answers, target):
    """Select the entities ranked for one query: allowed, minus other known answers (D4, D5)."""
    mask = np.zeros(num_entities, dtype=bool)
    mask[list(allowed)] = True
    mask[list(known_answers)] = False
    mask[target] = True
    return mask


def rank_metrics(scores, target, candidates, k=10):
    """Return the expected reciprocal rank and Hits@k under uniformly random tie-breaking (D6)."""
    scores = np.asarray(scores, dtype=float)
    competitors = np.asarray(candidates, dtype=bool).copy()
    competitors[target] = False
    if np.isnan(scores[target]) or np.isnan(scores[competitors]).any():
        raise ValueError("scores contain NaN")

    above = int((scores[competitors] > scores[target]).sum())
    tied = int((scores[competitors] == scores[target]).sum())
    possible_ranks = np.arange(above + 1, above + tied + 2)
    return float((1 / possible_ranks).mean()), float((possible_ranks <= k).mean())


def twoway_bootstrap_ci(values, drugs, diseases, draws=10_000, level=0.95, seed=0):
    """Return the mean and a two-way (pigeonhole) bootstrap CI over drug-disease edges (D7).

    Drugs and diseases are resampled independently (Owen, 2007). An edge's weight is the number of
    times its drug was drawn times the number of times its disease was drawn. To compare two models,
    pass their per-edge differences, each already averaged over walk seeds.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or len(values) != len(drugs) or len(values) != len(diseases):
        raise ValueError("values, drugs and diseases must be non-empty and have the same length")

    _, drug_index = np.unique(drugs, return_inverse=True)
    _, disease_index = np.unique(diseases, return_inverse=True)
    n_drugs, n_diseases = drug_index.max() + 1, disease_index.max() + 1
    rng = np.random.default_rng(seed)

    estimates = np.empty(draws)
    for draw in range(draws):
        weights = np.zeros(len(values))
        while weights.sum() == 0:
            drug_counts = rng.multinomial(n_drugs, np.full(n_drugs, 1 / n_drugs))
            disease_counts = rng.multinomial(n_diseases, np.full(n_diseases, 1 / n_diseases))
            weights = drug_counts[drug_index] * disease_counts[disease_index]
        estimates[draw] = weights @ values / weights.sum()

    tail = (1 - level) / 2
    low, high = np.quantile(estimates, [tail, 1 - tail])
    return float(values.mean()), float(low), float(high)
