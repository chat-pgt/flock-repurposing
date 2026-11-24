import numpy as np
import pytest

from e1_baselines.popularity import score_direction


def test_popularity_uses_training_folds_and_filters_only_known_treats():
    drugs = np.array(["a", "a", "b", "c", "d"])
    diseases = np.array(["x", "y", "y", "y", "z"])
    folds = np.array([0, 1, 1, 0, 1])
    palliated = {("a", "z")}

    views = score_direction(drugs, diseases, folds, ("x", "y", "z", "w"), 6, palliated)

    # Query (a, ?) with target x in fold 0. Training counts are x=0, y=2, z=1, w=0.
    # y is another known answer, so it is filtered. z outranks x, which ties with w.
    assert views["candidates"]["rr"][0] == pytest.approx((1 / 2 + 1 / 3) / 2)
    assert views["candidates"]["random_rr"][0] == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)

    # Removing the palliated disease z leaves only the tie with w.
    assert views["filter_palliates"]["rr"][0] == pytest.approx((1 + 1 / 2) / 2)

    # Two extra entities score zero and join the tie: x can be ranked 2, 3, 4 or 5.
    assert views["all_entities"]["rr"][0] == pytest.approx((1 / 2 + 1 / 3 + 1 / 4 + 1 / 5) / 4)


def test_held_out_edges_do_not_count_towards_popularity():
    drugs = np.array(["a", "b"])
    diseases = np.array(["x", "y"])
    folds = np.array([0, 1])

    views = score_direction(drugs, diseases, folds, ("x", "y"), 2, set())

    # In fold 0, y has one training edge and x has none, so the target x ranks second.
    assert views["candidates"]["rr"][0] == pytest.approx(1 / 2)
    assert views["candidates"]["hits10"][0] == 1
