"""Negative sampling for E5's two fine-tuning arms. NumPy only, so it is testable without a GPU.

Both arms corrupt only the tail of (drug, treats, disease) training facts and draw the same number of
negatives with replacement. They differ only in the pool: every entity, or diseases only. As in the released
code's strict sampling, a drug's training indications are never drawn as its negatives. Held-out facts are
not known at training time, so they can be drawn.
"""

from collections import defaultdict

import numpy as np

ARMS = ("uniform", "disease")


def training_indications(drugs, diseases, train_rows):
    """Map each drug to the diseases it treats in the training graph."""
    known = defaultdict(set)
    for row in np.asarray(train_rows).tolist():
        known[int(drugs[row])].add(int(diseases[row]))
    return known


class NegativeSampler:
    def __init__(self, arm, n_entities, disease_ids, known, n_negative, rng):
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
        self.base = np.arange(n_entities) if arm == "uniform" else np.sort(np.asarray(disease_ids))
        self.known, self.n_negative, self.rng = known, n_negative, rng
        self.pools = {}

    def pool(self, drug):
        if drug not in self.pools:
            excluded = np.fromiter(self.known.get(drug, ()), dtype=self.base.dtype)
            self.pools[drug] = np.setdiff1d(self.base, excluded, assume_unique=True)
        return self.pools[drug]

    def sample(self, drugs):
        """Return an array [len(drugs), n_negative] of negative tails."""
        return np.stack([self.rng.choice(self.pool(int(drug)), size=self.n_negative) for drug in drugs])


def batches(train_rows, batch_size, epochs, rng):
    """Yield the training rows of each step: a fresh shuffle per epoch, keeping the last partial batch."""
    for _ in range(epochs):
        order = rng.permutation(np.asarray(train_rows))
        for start in range(0, len(order), batch_size):
            yield order[start : start + batch_size]
