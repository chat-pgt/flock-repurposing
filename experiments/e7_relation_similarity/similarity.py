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
