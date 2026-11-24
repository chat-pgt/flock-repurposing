import numpy as np
import pytest

from e5_negatives.negatives import NegativeSampler, batches, training_indications

DRUGS = np.array([0, 0, 1, 2])
DISEASES = np.array([10, 11, 10, 12])
DISEASE_IDS = np.array([10, 11, 12, 13])


def test_training_indications_use_only_training_rows():
    known = training_indications(DRUGS, DISEASES, train_rows=[0, 2, 3])

    assert dict(known) == {0: {10}, 1: {10}, 2: {12}}


def test_disease_arm_draws_only_diseases_the_drug_does_not_treat_in_training():
    known = training_indications(DRUGS, DISEASES, train_rows=[0, 1])
    sampler = NegativeSampler("disease", 14, DISEASE_IDS, known, 50, np.random.default_rng(0))

    negatives = sampler.sample([0, 1])

    assert negatives.shape == (2, 50)
    assert set(negatives[0].tolist()) == {12, 13}
    assert set(negatives[1].tolist()) == {10, 11, 12, 13}


def test_uniform_arm_draws_from_every_entity_except_training_indications():
    known = training_indications(DRUGS, DISEASES, train_rows=[0, 1])
    sampler = NegativeSampler("uniform", 14, DISEASE_IDS, known, 2000, np.random.default_rng(0))

    drawn = set(sampler.sample([0])[0].tolist())

    assert drawn == set(range(14)) - {10, 11}


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError):
        NegativeSampler("hard", 14, DISEASE_IDS, {}, 5, np.random.default_rng(0))


def test_batches_cover_every_row_once_per_epoch_in_a_seeded_order():
    rows = np.arange(10)

    first = list(batches(rows, 4, 2, np.random.default_rng(3)))
    second = list(batches(rows, 4, 2, np.random.default_rng(3)))

    assert [len(batch) for batch in first] == [4, 4, 2, 4, 4, 2]
    assert sorted(np.concatenate(first[:3]).tolist()) == rows.tolist()
    assert sorted(np.concatenate(first[3:]).tolist()) == rows.tolist()
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
