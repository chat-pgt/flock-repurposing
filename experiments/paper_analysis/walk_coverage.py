"""Simulate FLOCK's walk sampler (App. A.1) on Hetionet and measure coverage per update and per pass.

Usage: python walk_coverage.py ../data/hetionet/hetionet-v1.0-edges.sif.gz  (7-15 minutes on CPU)
Uses a random 90% training split, which has the same node and edge counts as FLOCK's Hetionet.
Measures the sampler only, on 40 held-out treats queries. It does not measure model performance.
"""

import gzip
import sys
import time

import numpy as np

path = sys.argv[1]
rng = np.random.default_rng(0)
t0 = time.time()
ids = {}
S = []
R = []
T = []
rels = {}
with gzip.open(path, "rt") as f:
    next(f)
    for line in f:
        s, r, t = line.rstrip("\n").split("\t")
        S.append(ids.setdefault(s, len(ids)))
        T.append(ids.setdefault(t, len(ids)))
        R.append(rels.setdefault(r, len(rels)))
S, R, T = np.array(S), np.array(R), np.array(T)
names = np.array(list(ids))
kinds = np.array([x.split("::")[0] for x in names])
perm = rng.permutation(len(S))
ntr = int(0.9 * len(S))
tr = perm[:ntr]
te = perm[ntr:]
s, r, t = S[tr], R[tr], T[tr]
N = len(ids)
print("nodes with edges", N, "train edges", len(s), "relations", len(rels), flush=True)
# undirected unique neighbours; self loops ignored
keep = s != t
u = np.concatenate([s[keep], t[keep]])
v = np.concatenate([t[keep], s[keep]])
key = np.unique(u.astype(np.int64) * N + v)
U = (key // N).astype(np.int32)
V = (key % N).astype(np.int32)
ptr = np.zeros(N + 1, dtype=np.int64)
np.add.at(ptr, U + 1, 1)
ptr = ptr.cumsum()
deg = np.diff(ptr)
print("unique directed pairs", len(key), "mean deg", deg.mean(), "median", np.median(deg), "max", deg.max(), flush=True)
pairs = np.minimum(s, t).astype(np.int64) * N + np.maximum(s, t)
upairs = np.unique(pairs)
print("unique undirected pairs", len(upairs))
# type-start edges: per relation edge idx
byrel = [np.flatnonzero(r == i) for i in range(len(rels))]


def step(cur, prev):
    d = deg[cur]
    nxt = cur.copy()
    ok = d > 0
    has = ok & (prev >= 0)
    # choose uniformly among neighbours except prev
    k = np.zeros(len(cur), dtype=np.int64)
    q = has & (d > 1)
    k[q] = (rng.random(q.sum()) * (d[q] - 1)).astype(np.int64)
    pos = np.searchsorted(key, cur[q].astype(np.int64) * N + prev[q]) - ptr[cur[q]]
    kk = k[q]
    kk = kk + (kk >= pos)
    k[q] = kk
    fresh = ok & (prev < 0)
    k[fresh] = (rng.random(fresh.sum()) * d[fresh]).astype(np.int64)
    nxt[ok] = V[ptr[cur[ok]] + k[ok]]
    return nxt


def walks(starts, firsts, L):
    W = np.empty((len(starts), L), dtype=np.int32)
    W[:, 0] = starts
    prev = -np.ones(len(starts), dtype=np.int32)
    cur = starts.copy()
    i0 = 1
    has = firsts >= 0
    if has.any():
        W[:, 1] = np.where(has, firsts, 0)
        nxt = step(cur, prev)
        nxt = np.where(has, firsts, nxt)
        W[:, 1] = nxt
        prev = cur
        cur = nxt
        i0 = 2
    for i in range(i0, L):
        nxt = step(cur, prev)
        W[:, i] = nxt
        prev = cur
        cur = nxt
    return W


ctd = rels["CtD"]
test = te[R[te] == ctd]
q = rng.choice(test, size=min(40, len(test)), replace=False)
print("held-out CtD", len(test), "sampled", len(q), flush=True)
dis = np.flatnonzero(kinds == "Disease")
isdis = np.zeros(N, bool)
isdis[dis] = True
nonisolated = (deg > 0).sum()


def round_walks(h, n, L):
    a = np.full(n, h, dtype=np.int32)
    fa = -np.ones(n, dtype=np.int32)
    c = rng.integers(0, N, n).astype(np.int32)
    fc = -np.ones(n, dtype=np.int32)
    rr = rng.integers(0, len(rels), n)
    e = np.array([byrel[x][rng.integers(len(byrel[x]))] for x in rr])
    flip = rng.random(n) < 0.5
    b = np.where(flip, t[e], s[e]).astype(np.int32)
    fb = np.where(flip, s[e], t[e]).astype(np.int32)
    return walks(np.concatenate([a, b, c]), np.concatenate([fa, fb, fc]), L)


def run(n, L, rounds=6):
    out = []
    for e in q:
        h, tgt = S[e], T[e]
        ws = [round_walks(h, n, L) for _ in range(rounds)]
        w1 = ws[0]
        wa = np.concatenate(ws)
        nodes = np.unique(wa)
        nodes1 = np.unique(w1)
        A = np.minimum(wa[:, :-1], wa[:, 1:]).astype(np.int64).ravel() * N + np.maximum(wa[:, :-1], wa[:, 1:]).ravel()
        Au = np.unique(A)
        cov = np.isin(upairs, Au, assume_unique=True).mean()
        A1 = np.unique(
            np.minimum(w1[:, :-1], w1[:, 1:]).astype(np.int64).ravel() * N + np.maximum(w1[:, :-1], w1[:, 1:]).ravel()
        )
        cov1 = np.isin(upairs, A1, assume_unique=True).mean()
        vis = (wa == tgt).sum()
        headw = np.concatenate([x[:n] for x in ws])
        out.append(
            (
                (deg[nodes] > 0).sum() / nonisolated,
                cov,
                isdis[nodes].sum() / len(dis),
                int(vis > 0),
                vis,
                (headw == tgt).any(1).mean(),
                (deg[nodes1] > 0).sum() / nonisolated,
                cov1,
                isdis[nodes1].sum() / len(dis),
                int((w1 == tgt).any()),
            )
        )
    return np.array(out, dtype=float).mean(0), np.median(np.array(out, dtype=float)[:, 4])


print(
    "n L steps/round | ROUND nodes edges diseases target | PASS(6 rounds) nodes edges diseases target meanVisitsTarget medianVisits fracHeadWalksHitTarget",
    flush=True,
)
for n, L in [(512, 16), (512, 32), (512, 64), (512, 128), (512, 256), (1024, 64), (256, 256), (2048, 32), (1024, 128)]:
    t1 = time.time()
    m, med = run(n, L)
    print(
        n,
        L,
        3 * n * (L - 1),
        "| %.3f %.3f %.3f %.2f | %.3f %.3f %.3f %.2f %.1f %.1f %.4f"
        % (m[6], m[7], m[8], m[9], m[0], m[1], m[2], m[3], m[4], med, m[5]),
        "(%.0fs)" % (time.time() - t1),
        flush=True,
    )
