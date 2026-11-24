import numpy as np
import pytest

from e3_coverage.analyze_coverage import read_edges_metrics, select
from e3_coverage.plan import SETTINGS, sample_plan


def test_fixed_budget_settings_match_the_reference_budget():
    assert SETTINGS["n1024_l64"].budget == 512 * 128
    assert SETTINGS["n2048_l32"].budget == 512 * 128
    assert SETTINGS["no_anatomy_expression"].dropped_relations == ("AeG",)


def test_query_sample_is_reproducible_and_keeps_all_edges_of_a_query():
    query_ids = np.array([5, 5, 6, 7, 8, 9, 5, 6])
    folds = np.array([0, 0, 0, 0, 0, 0, 1, 1])

    first = sample_plan(query_ids, folds, 0, n_queries=3, seed=1)

    assert first == sample_plan(query_ids, folds, 0, n_queries=3, seed=1)
    assert len(first) == 3
    if 5 in first:
        assert first[5] == [0, 1]
    assert sample_plan(query_ids, folds, 1, n_queries=10) == {5: [6], 6: [7]}


def write_shard(path, rows, reciprocal_ranks):
    np.savez(path, edge_rows=np.array(rows), reciprocal_rank=np.array(reciprocal_ranks), hits10=np.ones(len(rows)))


def test_edges_are_sorted_and_aligned_with_the_reference(tmp_path):
    paths = [tmp_path / "a.npz", tmp_path / "b.npz"]
    write_shard(paths[0], [4, 1], [0.25, 1.0])
    write_shard(paths[1], [2], [0.5])

    reference = read_edges_metrics(paths)

    assert reference[0].tolist() == [1, 2, 4]
    assert reference[1].tolist() == [1.0, 0.5, 0.25]
    assert select(reference, np.array([2, 4]))[0].tolist() == [0.5, 0.25]
    with pytest.raises(ValueError, match="does not cover"):
        select(reference, np.array([3]))
