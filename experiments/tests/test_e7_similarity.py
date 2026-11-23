import numpy as np
import pytest

from e7_relation_similarity.similarity import centre, control_pairs, cosine, pair_cosines, pca_2d, summarise


def test_cosine_works_along_the_last_axis():
    first = np.array([[1.0, 0.0], [1.0, 1.0]])
    second = np.array([[0.0, 2.0], [2.0, 2.0]])

    assert cosine(first, second).tolist() == pytest.approx([0.0, 1.0])


def test_centring_removes_a_component_shared_by_all_relations():
    shared = np.array([100.0, 100.0])
    states = np.stack([shared + [1.0, 0.0], shared - [1.0, 0.0]])

    assert cosine(states[0], states[1]) == pytest.approx(1.0, abs=1e-3)
    centred = centre(states)
    assert cosine(centred[0], centred[1]) == pytest.approx(-1.0)


def test_controls_only_pair_relations_between_different_types():
    names = ["CuG", "CdG", "CtD"]

    pairs = control_pairs(names, ["Compound", "Compound", "Compound"], ["Gene", "Gene", "Disease"])

    assert pairs == [("CuG", "CtD"), ("CdG", "CtD")]


def test_pair_cosines_keep_leading_axes():
    states = np.zeros((3, 2, 4, 5))
    states[..., 0, 0] = 1
    states[..., 1, 0] = 1
    states[..., 2, 1] = 1

    values = pair_cosines(states, ["a", "b", "c", "d"], [("a", "b"), ("a", "c")])

    assert values.shape == (3, 2, 2)
    assert values[..., 0] == pytest.approx(np.ones((3, 2)))
    assert values[..., 1] == pytest.approx(np.zeros((3, 2)))


def test_pca_recovers_the_main_axis():
    vectors = np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, -0.1, 0.0]])

    coordinates, explained = pca_2d(vectors)

    assert coordinates.shape == (4, 2)
    assert explained[0] > 0.99


def test_summary_separates_targets_from_controls():
    names = ["CuG", "CdG", "DuG", "DdG", "AuG", "AdG", "CtD", "CpD"]
    heads = ["Compound", "Compound", "Disease", "Disease", "Anatomy", "Anatomy", "Compound", "Compound"]
    tails = ["Gene", "Gene", "Gene", "Gene", "Gene", "Gene", "Disease", "Disease"]
    states = np.eye(8)
    states[1] = states[0]

    summary = summarise(states, names, heads, tails)

    assert summary["raw"]["targets"]["Compound up vs down Gene"]["mean"] == pytest.approx(1.0)
    assert summary["raw"]["targets"]["Disease up vs down Gene"]["mean"] == pytest.approx(0.0)
    assert summary["n_control_pairs"] > 0
