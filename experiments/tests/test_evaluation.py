import numpy as np
import pytest

from common.evaluation import (
    candidate_mask,
    has_other_indication,
    make_folds,
    rank_metrics,
    training_triples,
    twoway_bootstrap_ci,
)


def synthetic_edges(seed=0, n_edges=400, n_drugs=120, n_diseases=25):
    rng = np.random.default_rng(seed)
    popularity = rng.dirichlet(np.full(n_diseases, 0.5))
    pairs = set()
    while len(pairs) < n_edges:
        pairs.add((int(rng.integers(n_drugs)), int(rng.choice(n_diseases, p=popularity))))
    drugs, diseases = zip(*sorted(pairs), strict=True)
    return np.array(drugs), np.array(diseases)


def test_folds_partition_edges_and_balance_each_disease():
    drugs, diseases = synthetic_edges()
    folds = make_folds(drugs, diseases, n_folds=5, seed=1)

    assert set(folds) == set(range(5))
    assert np.ptp(np.bincount(folds)) <= 1
    for disease in np.unique(diseases):
        assert np.ptp(np.bincount(folds[diseases == disease], minlength=5)) <= 1


def test_folds_do_not_depend_on_input_order():
    drugs, diseases = synthetic_edges()
    permutation = np.random.default_rng(2).permutation(len(drugs))
    folds = make_folds(drugs, diseases, seed=3)
    shuffled = make_folds(drugs[permutation], diseases[permutation], seed=3)

    assert np.array_equal(folds[permutation], shuffled)


def test_training_graph_excludes_held_out_edges_and_inverses_only():
    triples = [("c1", "CtD", "d1"), ("d1", "CtD", "c1"), ("c1", "CpD", "d1"), ("c2", "CtD", "d1")]

    remaining = training_triples(triples, [("c1", "CtD", "d1")])

    assert remaining == [("c1", "CpD", "d1"), ("c2", "CtD", "d1")]


def test_other_indication_is_evaluated_within_each_fold():
    drugs = ["a", "a", "b", "b", "c"]
    folds = [0, 1, 2, 2, 3]

    assert has_other_indication(drugs, folds).tolist() == [True, True, False, False, False]


def test_all_tied_candidates_match_the_closed_form():
    k = 7
    reciprocal_rank, hits_at_1 = rank_metrics(np.zeros(k), 3, np.ones(k, dtype=bool), k=1)

    assert reciprocal_rank == pytest.approx(sum(1 / i for i in range(1, k + 1)) / k)
    assert hits_at_1 == pytest.approx(1 / k)


def test_ties_are_not_counted_against_the_target():
    scores = np.array([5.0, 3.0, 3.0, 1.0])

    reciprocal_rank, hits_at_2 = rank_metrics(scores, 1, np.ones(4, dtype=bool), k=2)

    assert reciprocal_rank == pytest.approx((1 / 2 + 1 / 3) / 2)
    assert hits_at_2 == pytest.approx(0.5)


def test_filter_removes_other_treats_but_keeps_palliated_diseases():
    scores = np.array([9.0, 8.0, 7.0, 0.0])  # 0: other treated, 1: palliated, 2: target, 3: non-disease
    mask = candidate_mask(4, allowed=[0, 1, 2], known_answers=[0, 2], target=2)

    assert mask.tolist() == [False, True, True, False]
    assert rank_metrics(scores, 2, mask)[0] == pytest.approx(1 / 2)


def test_nan_scores_are_rejected():
    with pytest.raises(ValueError, match="NaN"):
        rank_metrics(np.array([1.0, np.nan]), 0, np.ones(2, dtype=bool))


def test_bootstrap_is_seeded_and_exact_for_constant_values():
    drugs, diseases = synthetic_edges()

    mean, low, high = twoway_bootstrap_ci(np.full(len(drugs), 0.25), drugs, diseases, draws=50)
    assert (mean, low, high) == pytest.approx((0.25, 0.25, 0.25))

    values = np.random.default_rng(4).random(len(drugs))
    first = twoway_bootstrap_ci(values, drugs, diseases, draws=100, seed=5)
    assert first == twoway_bootstrap_ci(values, drugs, diseases, draws=100, seed=5)


def test_bootstrap_widens_when_outcomes_cluster_by_disease():
    drugs, diseases = synthetic_edges()
    rng = np.random.default_rng(6)
    disease_effect = rng.normal(0, 1, diseases.max() + 1)
    values = disease_effect[diseases] + rng.normal(0, 0.2, len(diseases))

    _, low, high = twoway_bootstrap_ci(values, drugs, diseases, draws=1000, seed=7)
    resampled = values[rng.integers(0, len(values), (1000, len(values)))].mean(axis=1)
    naive_low, naive_high = np.quantile(resampled, [0.025, 0.975])

    assert high - low > 1.5 * (naive_high - naive_low)
