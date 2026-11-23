"""Load the official Hetionet v1.0 release and the shared treats split (PROTOCOL.md, D1 and D2)."""

import csv
import gzip
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[1]
EDGES_PATH = EXPERIMENTS / "data/hetionet/hetionet-v1.0-edges.sif.gz"
SPLIT_PATH = EXPERIMENTS / "splits/treats_5fold_seed0.tsv"
TREATS = "CtD"
PALLIATES = "CpD"
RESEMBLES = "CrC"


@dataclass(frozen=True)
class RepurposingGraph:
    """The parts of Hetionet that the repurposing experiments share."""

    treats: tuple[tuple[str, str], ...]
    palliates: frozenset[tuple[str, str]]
    resembles: dict[str, frozenset[str]]
    diseases: tuple[str, ...]
    compounds: tuple[str, ...]
    n_entities: int


@dataclass(frozen=True)
class TreatsSplit:
    """One row per treats edge, in a fixed order."""

    drugs: np.ndarray
    diseases: np.ndarray
    folds: np.ndarray
    has_other_indication: np.ndarray


def read_edges(path=EDGES_PATH):
    with gzip.open(path, "rt") as handle:
        next(handle)
        for line in handle:
            head, relation, tail = line.rstrip("\n").split("\t")
            yield head, relation, tail


def node_kind(node):
    return node.split("::", 1)[0]


def load_repurposing_graph(path=EDGES_PATH):
    treats, palliates, connected = set(), set(), set()
    resembles = defaultdict(set)
    for head, relation, tail in read_edges(path):
        connected.update((head, tail))
        if relation == TREATS:
            treats.add((head, tail))
        elif relation == PALLIATES:
            palliates.add((head, tail))
        elif relation == RESEMBLES:
            resembles[head].add(tail)
            resembles[tail].add(head)
    return RepurposingGraph(
        treats=tuple(sorted(treats)),
        palliates=frozenset(palliates),
        resembles={drug: frozenset(others) for drug, others in resembles.items()},
        diseases=tuple(sorted(node for node in connected if node_kind(node) == "Disease")),
        compounds=tuple(sorted(node for node in connected if node_kind(node) == "Compound")),
        n_entities=len(connected),
    )


def read_split(path=SPLIT_PATH):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return TreatsSplit(
        drugs=np.array([row["compound"] for row in rows]),
        diseases=np.array([row["disease"] for row in rows]),
        folds=np.array([int(row["fold"]) for row in rows]),
        has_other_indication=np.array([row["has_other_indication"] == "1" for row in rows]),
    )
