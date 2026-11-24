"""Compare relation representations: target pairs, control pairs and a 2-D PCA. NumPy only."""

from itertools import combinations

import numpy as np

TARGET_PAIRS = {
    "Compound up vs down Gene": ("CuG", "CdG"),
    "Disease up vs down Gene": ("DuG", "DdG"),
    "Anatomy up vs down Gene": ("AuG", "AdG"),
    "treats vs palliates": ("CtD", "CpD"),
}


def cosine(first, second):
    """Return the cosine similarity along the last axis."""
    numerator = (first * second).sum(-1)
    denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    return numerator / np.maximum(denominator, 1e-12)


def centre(states):
    """Subtract the mean relation, so a component shared by all relations cannot inflate cosines."""
    return states - states.mean(axis=-2, keepdims=True)


def control_pairs(names, head_kinds, tail_kinds):
    """Return all pairs of relations that connect different entity types."""
    signature = {name: (head, tail) for name, head, tail in zip(names, head_kinds, tail_kinds, strict=True)}
    return [(first, second) for first, second in combinations(names, 2) if signature[first] != signature[second]]


def pair_cosines(states, names, pairs):
    """Return cosines for each pair, keeping any leading axes such as seed, query and pass."""
    index = {name: position for position, name in enumerate(names)}
    return np.stack([cosine(states[..., index[a], :], states[..., index[b], :]) for a, b in pairs], axis=-1)


def pca_2d(vectors):
    """Project row vectors on their first two principal components."""
    centred = vectors - vectors.mean(axis=0)
    left, singular, _ = np.linalg.svd(centred, full_matrices=False)
    coordinates = left[:, :2] * singular[:2]
    variance = singular**2
    return coordinates, (variance[:2] / max(variance.sum(), 1e-12)).tolist()


def relative_distance(first, second):
    """Return the distance between vectors, relative to their mean norm."""
    scale = 0.5 * (np.linalg.norm(first, axis=-1) + np.linalg.norm(second, axis=-1))
    return np.linalg.norm(first - second, axis=-1) / np.maximum(scale, 1e-12)


def across_pass_cosines(states, first, second):
    """Compare one relation across two passes with two different relations across those passes.

    `states` has axes (seed, query, pass, relation, dim). If the two values match, the difference
    between the relations is no larger than the walk noise.
    """
    pass_a, pass_b = states[:, :, 0], states[:, :, 1]
    same = [cosine(pass_a[..., relation, :], pass_b[..., relation, :]).ravel() for relation in (first, second)]
    different = [
        cosine(pass_a[..., first, :], pass_b[..., second, :]).ravel(),
        cosine(pass_a[..., second, :], pass_b[..., first, :]).ravel(),
    ]
    return float(np.concatenate(same).mean()), float(np.concatenate(different).mean())


def leave_one_query_out_accuracy(states, first, second):
    """Label held-out vectors as one of two relations with nearest centroids. Chance is 0.5."""
    dim = states.shape[-1]
    correct = total = 0
    for held in range(states.shape[1]):
        train = np.delete(states, held, axis=1)
        centroid_first = train[..., first, :].reshape(-1, dim).mean(axis=0)
        centroid_second = train[..., second, :].reshape(-1, dim).mean(axis=0)
        for relation, is_second in ((first, False), (second, True)):
            vectors = states[:, held, :, relation, :].reshape(-1, dim)
            nearer_second = np.linalg.norm(vectors - centroid_second, axis=1) < np.linalg.norm(
                vectors - centroid_first, axis=1
            )
            correct += int((nearer_second == is_second).sum())
            total += len(vectors)
    return correct / total


def sample_balanced(relations, pair, per_relation, seed=0):
    """Pick the same number of rows for each relation in a pair, reproducibly and in sorted order."""
    rng = np.random.default_rng(seed)
    chosen = []
    for relation in pair:
        rows = np.flatnonzero(np.asarray(relations) == relation)
        if len(rows) < per_relation:
            raise ValueError(f"relation {relation} has only {len(rows)} rows")
        chosen.append(rng.choice(rows, size=per_relation, replace=False))
    return np.sort(np.concatenate(chosen))


def summarise(states, names, head_kinds, tail_kinds):
    """Summarise raw and centred cosines for target pairs against control pairs."""
    controls = control_pairs(names, head_kinds, tail_kinds)
    output = {"n_control_pairs": len(controls)}
    for label, values in (("raw", states), ("centred", centre(states))):
        targets = pair_cosines(values, names, list(TARGET_PAIRS.values()))
        control_values = pair_cosines(values, names, controls)
        output[label] = {
            "targets": {
                name: {"mean": float(targets[..., i].mean()), "std": float(targets[..., i].std())}
                for i, name in enumerate(TARGET_PAIRS)
            },
            "controls": {
                "median": float(np.median(control_values.mean(axis=tuple(range(control_values.ndim - 1))))),
                "p05": float(np.percentile(control_values.mean(axis=tuple(range(control_values.ndim - 1))), 5)),
                "p95": float(np.percentile(control_values.mean(axis=tuple(range(control_values.ndim - 1))), 95)),
            },
        }
    return output
