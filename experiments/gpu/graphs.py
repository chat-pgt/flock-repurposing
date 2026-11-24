"""Graph and query bookkeeping for the GPU runs. NumPy only, so it is testable without a GPU."""

from array import array
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EncodedGraph:
    heads: np.ndarray
    relations: np.ndarray
    tails: np.ndarray
    entities: tuple[str, ...]
    relation_names: tuple[str, ...]


def encode_triples(triples):
    """Encode (head, relation, tail) strings with sorted, reproducible vocabularies."""
    entity_ids, relation_ids = {}, {}
    heads, relations, tails = array("i"), array("i"), array("i")
    for head, relation, tail in triples:
        heads.append(entity_ids.setdefault(head, len(entity_ids)))
        relations.append(relation_ids.setdefault(relation, len(relation_ids)))
        tails.append(entity_ids.setdefault(tail, len(entity_ids)))

    entities, relation_names = tuple(sorted(entity_ids)), tuple(sorted(relation_ids))
    entity_map = np.empty(len(entities), dtype=np.int32)
    entity_map[[entity_ids[name] for name in entities]] = np.arange(len(entities))
    relation_map = np.empty(len(relation_names), dtype=np.int32)
    relation_map[[relation_ids[name] for name in relation_names]] = np.arange(len(relation_names))
    return EncodedGraph(
        heads=entity_map[np.frombuffer(heads, dtype=np.int32)],
        relations=relation_map[np.frombuffer(relations, dtype=np.int32)],
        tails=entity_map[np.frombuffer(tails, dtype=np.int32)],
        entities=entities,
        relation_names=relation_names,
    )


def training_mask(graph, held_heads, held_tails, relation, dropped_relations=()):
    """Select all triples except the held-out (head, relation, tail) facts and any dropped relation."""
    n_entities = len(graph.entities)
    keys = graph.heads.astype(np.int64) * n_entities + graph.tails
    held_keys = np.asarray(held_heads, dtype=np.int64) * n_entities + np.asarray(held_tails, dtype=np.int64)
    keep = ~((graph.relations == relation) & np.isin(keys, held_keys))
    for dropped in dropped_relations:
        keep &= graph.relations != dropped
    return keep


def plan_queries(query_ids, folds, fold):
    """Group a fold's held-out edges by query entity, so that each query is scored once."""
    plan = {}
    for row in np.flatnonzero(np.asarray(folds) == fold).tolist():
        plan.setdefault(int(query_ids[row]), []).append(row)
    return plan


def rank_counts(scores, target, excluded):
    """Count the entities scoring above and tied with the target, ignoring the excluded ones."""
    competitors = np.ones(len(scores), dtype=bool)
    competitors[list(excluded)] = False
    competitors[target] = False
    rivals = scores[competitors]
    return int((rivals > scores[target]).sum()), int((rivals == scores[target]).sum())
