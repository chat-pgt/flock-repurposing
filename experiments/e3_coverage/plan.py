"""Settings and query sample of the E3 coverage sweep. NumPy only."""

from dataclasses import dataclass

import numpy as np

from gpu.graphs import plan_queries

QUERIES_PER_FOLD = 40


@dataclass(frozen=True)
class Setting:
    walk_num: int
    walk_len: int
    dropped_relations: tuple[str, ...] = ()

    @property
    def budget(self):
        return self.walk_num * self.walk_len


# Cheapest first, so that a time limit cuts the most expensive settings. The reference setting
# (n = 512, l = 128 on the full graph) comes from the E1 runs on the same queries.
SETTINGS = {
    "n64_l128": Setting(64, 128),
    "n128_l128": Setting(128, 128),
    "n256_l128": Setting(256, 128),
    "n1024_l64": Setting(1024, 64),
    "n2048_l32": Setting(2048, 32),
    "no_anatomy_expression": Setting(512, 128, ("AeG",)),
}


def sample_plan(query_ids, folds, fold, n_queries=QUERIES_PER_FOLD, seed=0):
    """Pick a fixed sample of a fold's queries, keeping all held-out edges of each query."""
    plan = plan_queries(query_ids, folds, fold)
    queries = sorted(plan)
    chosen = np.random.default_rng([seed, int(fold)]).choice(queries, size=min(n_queries, len(queries)), replace=False)
    return {query: plan[query] for query in sorted(chosen.tolist())}
