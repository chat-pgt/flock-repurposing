from collections import Counter

import pytest

from e6_relation_baseline.relation_baselines import baseline_rows, signature_metrics, weighted_summary


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


def test_signature_baseline_prefers_relations_seen_on_both_endpoints():
    train = [
        ("C::a", "up", "G::x1"),
        ("C::b", "up", "G::x2"),
        ("C::c", "down", "G::y1"),
        ("C::c", "down", "G::y2"),
        ("C::d", "down", "G::y3"),
    ]
    test = [
        ("C::a", "up", "G::x2"),  # both endpoints have used "up", so it beats the more frequent "down"
        ("C::c", "up", "G::x1"),  # no shared relation, so the global frequency puts "down" first
        ("C::a", "down", "G::y1"),  # no shared relation, and "down" is the most frequent
    ]

    metrics = signature_metrics(train, test)

    assert metrics["C", "up", "G"]["signature_rr"] == pytest.approx((1 + 0.5) / 2)
    assert metrics["C", "up", "G"]["signature_hits1"] == pytest.approx(0.5)
    assert metrics["C", "down", "G"]["signature_rr"] == pytest.approx(1)


def test_signature_baseline_rejects_relations_unseen_in_training():
    with pytest.raises(ValueError, match="never appears"):
        signature_metrics([("C::a", "up", "G::x")], [("C::a", "down", "G::x")])
