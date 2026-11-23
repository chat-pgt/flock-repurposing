import numpy as np
import pytest

from gpu.analysis import average_over_seeds, bootstrap_contrast, complete_seeds, expected_metrics, seed_metrics

CANDIDATES = ("d1", "d2", "d3")
QUERIES = ["a", "a", "b"]
ANSWERS = ["d1", "d2", "d1"]


def write_shard(path, entities, scores, rows, positions, reciprocal_ranks, above, tied):
    np.savez(
        path,
        query_entities=np.array(entities, dtype="<U8"),
        candidate_scores=np.array(scores, dtype=np.float32).reshape(len(entities), len(CANDIDATES)),
        edge_rows=np.array(rows, dtype=int),
        edge_query_positions=np.array(positions, dtype=int),
        reciprocal_rank=np.array(reciprocal_ranks, dtype=float),
        hits10=np.ones(len(rows)),
        all_entities_above=np.array(above, dtype=int),
        all_entities_tied=np.array(tied, dtype=int),
    )


def write_seed(directory, seed, first_rr=0.5):
    paths = [directory / f"entity_flock_full_tail_seed{seed}_fold{fold}.npz" for fold in range(5)]
    write_shard(paths[0], ["a", "b"], [[1, 3, 2], [5, 1, 1]], [0, 2], [0, 1], [first_rr, 1.0], [3, 0], [1, 0])
    write_shard(paths[1], ["a"], [[3, 1, 2]], [1], [0], [0.5], [1], [0])
    for path in paths[2:]:
        write_shard(path, [], [], [], [], [], [], [])
    return paths


def test_expected_metrics_average_over_tied_ranks():
    assert expected_metrics(3, 1, k=4) == pytest.approx(((1 / 4 + 1 / 5) / 2, 0.5))


def test_only_seeds_with_all_folds_are_complete(tmp_path):
    write_seed(tmp_path, 0)
    paths = write_seed(tmp_path, 1)
    paths[4].unlink()

    assert list(complete_seeds(tmp_path, "flock", "full", "tail")) == [0]
    assert complete_seeds(tmp_path, "flock", "no_resembles", "tail") == {}


def test_seed_metrics_recompute_all_views(tmp_path):
    paths = write_seed(tmp_path, 0)

    views = seed_metrics(paths, QUERIES, ANSWERS, CANDIDATES, {("a", "d3")})

    assert views["candidates"]["rr"].tolist() == pytest.approx([0.5, 0.5, 1.0])
    assert views["filter_palliates"]["rr"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert views["all_entities"]["rr"][0] == pytest.approx((1 / 4 + 1 / 5) / 2)


def test_stored_rank_mismatch_is_rejected(tmp_path):
    paths = write_seed(tmp_path, 0, first_rr=0.25)

    with pytest.raises(ValueError, match="stored rank"):
        seed_metrics(paths, QUERIES, ANSWERS, CANDIDATES, set())


def test_missing_edges_are_rejected(tmp_path):
    paths = write_seed(tmp_path, 0)

    with pytest.raises(ValueError, match="missing"):
        seed_metrics(paths[:1], QUERIES, ANSWERS, CANDIDATES, set())


def test_seeds_are_averaged_per_edge():
    first = {view: {"rr": np.array([0.0, 1.0]), "hits10": np.array([0.0, 1.0])} for view in _views()}
    second = {view: {"rr": np.array([1.0, 1.0]), "hits10": np.array([1.0, 0.0])} for view in _views()}

    average = average_over_seeds({0: first, 1: second})

    assert average["candidates"]["rr"].tolist() == [0.5, 1.0]


def _views():
    return ("candidates", "filter_palliates", "all_entities")


def test_contrast_recovers_a_group_difference():
    drugs = np.array([f"c{i}" for i in range(40)])
    diseases = np.array([f"d{i % 8}" for i in range(40)])
    first = np.arange(40) % 2 == 0
    values = np.where(first, 0.75, 0.25)

    result = bootstrap_contrast(values, first, drugs, diseases, draws=100)

    assert result["mean"] == pytest.approx(0.5)
    assert result["ci95"] == pytest.approx([0.5, 0.5])
