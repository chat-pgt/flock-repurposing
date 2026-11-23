from collections import Counter

import pytest

from e6_relation_baseline.relation_baselines import baseline_rows, weighted_summary


def test_baselines_match_hand_computed_values():
    train = Counter({("C", "up", "G"): 5, ("C", "down", "G"): 2, ("D", "assoc", "G"): 4})
    test = Counter({("C", "up", "G"): 1, ("C", "down", "G"): 1, ("D", "assoc", "G"): 2})

    summary = weighted_summary(baseline_rows(train, test))

    assert summary["n_test"] == 4
    assert summary["uniform_mrr"] == pytest.approx((0.75 + 0.75 + 1 + 1) / 4)
    assert summary["uniform_hits1"] == pytest.approx((0.5 + 0.5 + 1 + 1) / 4)
    assert summary["frequency_mrr"] == pytest.approx((1 + 0.5 + 1 + 1) / 4)
    assert summary["frequency_hits1"] == pytest.approx((1 + 0 + 1 + 1) / 4)


def test_relation_unseen_between_these_types_is_an_error():
    train = Counter({("C", "up", "G"): 5})
    test = Counter({("C", "down", "G"): 1})

    with pytest.raises(ValueError, match="never appears"):
        baseline_rows(train, test)
