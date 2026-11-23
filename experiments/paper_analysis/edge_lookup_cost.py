"""Estimate how many adjacency entries FLOCK's edge-type lookup scans per walk step on Hetionet.

Usage: python edge_lookup_cost.py ../data/hetionet/hetionet-v1.0-edges.sif.gz
Based on reading graph-walker/src/edgeTypes.cpp. An estimate under the stationary walk distribution, not a profile.
"""

import gzip
import sys

import numpy as np

ids = {}
A = []
B = []
with gzip.open(sys.argv[1], "rt") as f:
    next(f)
    for l in f:
        s, _, t = l.rstrip("\n").split("\t")
        A.append(ids.setdefault(s, len(ids)))
        B.append(ids.setdefault(t, len(ids)))
A, B = np.array(A), np.array(B)
rng = np.random.default_rng(0)
p = rng.permutation(len(A))[: int(0.9 * len(A))]
A, B = A[p], B[p]
N = len(ids)
# row length of the CSR built from edge_index containing fact + flipped fact = incident facts
m = np.bincount(A, minlength=N) + np.bincount(B, minlength=N)
k = A != B
key = np.unique(np.concatenate([A[k].astype(np.int64) * N + B[k], B[k].astype(np.int64) * N + A[k]]))
d = np.bincount((key // N).astype(int), minlength=N)
pi = d / d.sum()
scan = 2 * m  # forward + transposed rows are both scanned in full
print(
    "mean row (uniform node) %.0f; walk-weighted row %.0f; entries scanned per step (both rows) %.0f; binary-search comparisons ~ %.0f"
    % (m[m > 0].mean(), (pi * m).sum(), (pi * scan).sum(), (pi * 2 * np.log2(np.maximum(m, 2))).sum())
)
print("ratio %.0fx; max row %d" % ((pi * scan).sum() / (pi * 2 * np.log2(np.maximum(m, 2))).sum(), m.max()))
