import numpy as np

from common.evaluation import has_other_indication, make_folds
from common.hetionet import read_split
from common.make_split import N_FOLDS, SEED


def test_committed_split_covers_each_treats_edge_once():
    split = read_split()
    edges = set(zip(split.drugs.tolist(), split.diseases.tolist(), strict=True))

    assert len(split.drugs) == 755
    assert len(edges) == 755
    assert set(split.folds.tolist()) == set(range(N_FOLDS))


def test_committed_split_is_balanced_overall_and_per_disease():
    split = read_split()

    assert np.ptp(np.bincount(split.folds)) <= 1
    for disease in np.unique(split.diseases):
        assert np.ptp(np.bincount(split.folds[split.diseases == disease], minlength=N_FOLDS)) <= 1


def test_committed_split_matches_its_generator():
    split = read_split()

    assert np.array_equal(split.folds, make_folds(split.drugs, split.diseases, n_folds=N_FOLDS, seed=SEED))
    assert np.array_equal(split.has_other_indication, has_other_indication(split.drugs, split.folds))
