import numpy as np
import pytest

from e2_leakage.near_duplicates import leakage_by_edge

RESEMBLES = {"a": frozenset({"b"}), "b": frozenset({"a"})}


def test_edge_is_flagged_when_an_analogue_treats_the_disease_in_training():
    drugs = np.array(["a", "b", "c"])
    diseases = np.array(["x", "x", "y"])
    folds = np.array([0, 1, 1])

    table = leakage_by_edge(drugs, diseases, folds, RESEMBLES, ("x", "y"))

    assert table["flagged"].tolist() == [1, 1, 0]
    assert table["has_analogue"].tolist() == [1, 1, 0]


def test_analogue_held_out_in_the_same_fold_does_not_count():
    drugs = np.array(["a", "b", "c"])
    diseases = np.array(["x", "x", "y"])
    folds = np.array([0, 0, 1])

    table = leakage_by_edge(drugs, diseases, folds, RESEMBLES, ("x", "y"))

    assert table["flagged"].tolist() == [0, 0, 0]


def test_controls_average_over_diseases_the_drug_is_not_known_to_treat():
    drugs = np.array(["a", "b", "c", "c"])
    diseases = np.array(["p", "q", "q", "r"])
    folds = np.array([0, 1, 1, 1])

    table = leakage_by_edge(drugs, diseases, folds, RESEMBLES, ("p", "q", "r", "w"))

    # For (a, p) in fold 0, the negatives are q, r and w. Only q is treated by the analogue b.
    # Training drugs per negative: q=2, r=1, w=0.
    assert table["flagged"][0] == 0
    assert table["uniform"][0] == pytest.approx(1 / 3)
    assert table["weighted"][0] == pytest.approx(2 / 3)
