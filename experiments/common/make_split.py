"""Write the shared five-fold split of the treats edges (PROTOCOL.md, D2).

Usage, from experiments/: uv run python -m common.make_split
"""

import csv

import numpy as np

from common.evaluation import has_other_indication, make_folds
from common.hetionet import SPLIT_PATH, load_repurposing_graph

N_FOLDS = 5
SEED = 0


def main():
    graph = load_repurposing_graph()
    drugs = np.array([drug for drug, _ in graph.treats])
    diseases = np.array([disease for _, disease in graph.treats])
    folds = make_folds(drugs, diseases, n_folds=N_FOLDS, seed=SEED)
    other = has_other_indication(drugs, folds)

    SPLIT_PATH.parent.mkdir(exist_ok=True)
    with open(SPLIT_PATH, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["compound", "disease", "fold", "has_other_indication"])
        for row in zip(drugs, diseases, folds, other.astype(int), strict=True):
            writer.writerow(row)
    print(f"wrote {len(drugs)} edges to {SPLIT_PATH}")
    print("edges per fold:", np.bincount(folds).tolist())
    print("edges whose drug keeps another indication in training:", int(other.sum()))


if __name__ == "__main__":
    main()
