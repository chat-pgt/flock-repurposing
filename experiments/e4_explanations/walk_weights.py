"""Credit FLOCK's walks with their share of a node's attention-weighted state. NumPy only.

FLOCK updates a node at each refinement with a softmax-weighted mean over every walk position that visits
it, one softmax per attention head, and adds the updates over refinements. A walk's share of the target's
state is therefore the sum of its positions' softmax weights, averaged over heads and over the
(refinement, pass) groups in which the target was visited.
"""

import numpy as np

# Hetionet v1.0 metaedges: abbreviation -> (head kind, verb, tail kind).
METAEDGES = {
    "AdG": ("Anatomy", "downregulates", "Gene"),
    "AeG": ("Anatomy", "expresses", "Gene"),
    "AuG": ("Anatomy", "upregulates", "Gene"),
    "CbG": ("Compound", "binds", "Gene"),
    "CcSE": ("Compound", "causes", "Side Effect"),
    "CdG": ("Compound", "downregulates", "Gene"),
    "CpD": ("Compound", "palliates", "Disease"),
    "CrC": ("Compound", "resembles", "Compound"),
    "CtD": ("Compound", "treats", "Disease"),
    "CuG": ("Compound", "upregulates", "Gene"),
    "DaG": ("Disease", "associates", "Gene"),
    "DdG": ("Disease", "downregulates", "Gene"),
    "DlA": ("Disease", "localizes", "Anatomy"),
    "DpS": ("Disease", "presents", "Symptom"),
    "DrD": ("Disease", "resembles", "Disease"),
    "DuG": ("Disease", "upregulates", "Gene"),
    "GcG": ("Gene", "covaries", "Gene"),
    "GiG": ("Gene", "interacts", "Gene"),
    "GpBP": ("Gene", "participates", "Biological Process"),
    "GpCC": ("Gene", "participates", "Cellular Component"),
    "GpMF": ("Gene", "participates", "Molecular Function"),
    "GpPW": ("Gene", "participates", "Pathway"),
    "Gr>G": ("Gene", "regulates", "Gene"),
    "PCiC": ("Pharmacologic Class", "includes", "Compound"),
}
DISTANCE_BINS = ((1, "1 hop"), (2, "2 hops"), (3, "3 hops"), (4, "4 hops"), (10, "5-10 hops"), (None, "over 10 hops"))


def occurrence_weights(logits):
    """Softmax each head over one group's occurrences, then average the heads. Returns weights summing to 1."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = np.exp(logits - logits.max(axis=0, keepdims=True))
    return (shifted / shifted.sum(axis=0, keepdims=True)).mean(axis=1)


def occurrence_shares(groups, logits):
    """Return each occurrence's share of the target's state, in input order."""
    groups = np.asarray(groups)
    shares = np.zeros(len(groups))
    unique_groups = np.unique(groups)
    for group in unique_groups:
        rows = np.flatnonzero(groups == group)
        shares[rows] = occurrence_weights(np.asarray(logits)[rows]) / len(unique_groups)
    return shares


def nearest_stretch(walk, drug, position):
    """Return the walk positions from the drug occurrence nearest to `position`, ending at `position`.

    Returns None when the walk never visits the drug.
    """
    visits = np.flatnonzero(np.asarray(walk) == drug)
    if visits.size == 0:
        return None
    start = int(visits[np.argmin(np.abs(visits - position))])
    step = 1 if start <= position else -1
    return list(range(start, position + step, step))


def distance_label(hops):
    if hops is None:
        return "drug not on walk"
    for limit, label in DISTANCE_BINS:
        if limit is None or hops <= limit:
            return label
    raise AssertionError("unreachable")


class EdgeLookup:
    """Orient a walk hop by finding the stored Hetionet fact behind it."""

    def __init__(self, heads, relations, tails, n_entities, n_relations):
        self.n_entities, self.n_relations = n_entities, n_relations
        self.keys = np.sort(self._key(heads, relations, tails))

    def _key(self, heads, relations, tails):
        heads, relations, tails = (np.asarray(x, dtype=np.int64) for x in (heads, relations, tails))
        return (heads * self.n_entities + tails) * self.n_relations + relations

    def has(self, head, relation, tail):
        key = self._key([head], [relation], [tail])[0]
        index = np.searchsorted(self.keys, key)
        return bool(index < len(self.keys) and self.keys[index] == key)

    def orient(self, source, relation, target):
        """Return +1 if the fact is source -> target, -1 if target -> source, 0 if neither exists."""
        if self.has(source, relation, target):
            return 1
        if self.has(target, relation, source):
            return -1
        return 0


def hop_relation(recorded_type, n_relations):
    """Map a recorded edge type, where r + |R| marks the flipped copy of r, back to r. None if no type."""
    if not 0 <= recorded_type < 2 * n_relations:
        return None
    return recorded_type % n_relations


def describe_stretch(nodes, recorded_types, lookup, relation_names, entity_label):
    """Render a stretch as text and as a metapath, and list the stored facts behind its hops.

    `recorded_types[i]` is the type of the hop that reaches `nodes[i + 1]` from `nodes[i]`. Facts are
    (head, relation, tail) as stored; a hop with no recorded type or no matching fact contributes none.
    """
    text, metapath, facts = [entity_label(nodes[0])], [entity_label(nodes[0], kind_only=True)], []
    for source, target, recorded in zip(nodes[:-1], nodes[1:], recorded_types, strict=True):
        relation = hop_relation(recorded, len(relation_names))
        if relation is None:
            arrow = "-?-"
        else:
            verb = METAEDGES[relation_names[relation]][1]
            orientation = lookup.orient(source, relation, target)
            arrow = {1: f"-{verb}->", -1: f"<-{verb}-", 0: f"?{verb}?"}[orientation]
            if orientation == 1:
                facts.append((int(source), int(relation), int(target)))
            elif orientation == -1:
                facts.append((int(target), int(relation), int(source)))
        text += [arrow, entity_label(target)]
        metapath += [arrow, entity_label(target, kind_only=True)]
    return " ".join(text), " ".join(metapath), facts


def stretch_category(kinds, ends_with_treatment):
    """Classify a short drug -> disease stretch by the evidence it carries."""
    if ends_with_treatment:
        return "another drug's treatment"
    if "Gene" in kinds[1:-1]:
        return "gene"
    return "other"


def drop_facts(graph, keep, facts):
    """Return `keep` with the given (head, relation, tail) facts also removed."""
    n_entities, n_relations = len(graph.entities), len(graph.relation_names)
    keys = (graph.heads.astype(np.int64) * n_entities + graph.tails) * n_relations + graph.relations
    dropped = np.array([(h * n_entities + t) * n_relations + r for h, r, t in facts], dtype=np.int64)
    return keep & ~np.isin(keys, dropped)


def incident_facts(graph, keep, nodes, relations=None):
    """Return the kept facts touching any of `nodes`, optionally limited to some relations, as tuples."""
    touching = keep & (np.isin(graph.heads, nodes) | np.isin(graph.tails, nodes))
    if relations is not None:
        touching &= np.isin(graph.relations, relations)
    rows = np.flatnonzero(touching)
    return [(int(graph.heads[i]), int(graph.relations[i]), int(graph.tails[i])) for i in rows]
