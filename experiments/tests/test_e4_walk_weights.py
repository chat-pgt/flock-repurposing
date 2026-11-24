import numpy as np
import pytest

from e4_explanations.walk_weights import (
    EdgeLookup,
    describe_stretch,
    distance_label,
    drop_facts,
    hop_relation,
    incident_facts,
    nearest_stretch,
    occurrence_shares,
    occurrence_weights,
    stretch_category,
)
from gpu.graphs import encode_triples


def test_occurrence_weights_softmax_each_head_then_average():
    logits = np.array([[0.0, 0.0], [np.log(3), 0.0]])

    weights = occurrence_weights(logits)

    # head 1 gives 1/4 and 3/4, head 2 gives 1/2 and 1/2
    assert weights == pytest.approx([0.375, 0.625])


def test_each_visited_group_contributes_equally():
    groups = [0, 0, 1]
    logits = np.zeros((3, 4))

    shares = occurrence_shares(groups, logits)

    assert shares.tolist() == pytest.approx([0.25, 0.25, 0.5])
    assert shares.sum() == pytest.approx(1.0)


def test_nearest_stretch_runs_from_the_closest_drug_visit_to_the_target():
    walk = [5, 1, 9, 2, 5, 7]

    assert nearest_stretch(walk, drug=5, position=3) == [4, 3]
    assert nearest_stretch(walk, drug=5, position=2) == [0, 1, 2]
    assert nearest_stretch(walk, drug=8, position=2) is None


def test_distance_labels():
    assert distance_label(None) == "drug not on walk"
    assert distance_label(2) == "2 hops"
    assert distance_label(7) == "5-10 hops"
    assert distance_label(40) == "over 10 hops"


def test_hops_are_oriented_by_the_stored_fact():
    # relation 0 is CtD: compound 0 treats disease 1. Relation 1 is CrC: compound 2 resembles compound 0.
    lookup = EdgeLookup(heads=[0, 2], relations=[0, 1], tails=[1, 0], n_entities=3, n_relations=2)

    assert lookup.orient(0, 0, 1) == 1
    assert lookup.orient(1, 0, 0) == -1
    assert lookup.orient(0, 1, 1) == 0
    assert hop_relation(3, n_relations=2) == 1
    assert hop_relation(4, n_relations=2) is None


def test_stretches_render_with_names_and_metapaths():
    lookup = EdgeLookup(heads=[0, 2], relations=[0, 1], tails=[1, 0], n_entities=3, n_relations=2)
    names = {0: ("aspirin", "Compound"), 1: ("pain", "Disease"), 2: ("salsalate", "Compound")}

    def label(node, kind_only=False):
        name, kind = names[node]
        return kind if kind_only else f"{name} ({kind})"

    # aspirin -> salsalate by the flipped copy of CrC (type 1 + 2), then salsalate -> pain with no type.
    text, metapath, facts = describe_stretch([0, 2], [3], lookup, ("CtD", "CrC"), label)
    assert text == "aspirin (Compound) <-resembles- salsalate (Compound)"
    assert metapath == "Compound <-resembles- Compound"
    assert facts == [(2, 1, 0)]

    text, _, facts = describe_stretch([2, 0, 1], [1, 2], lookup, ("CtD", "CrC"), label)
    assert text == "salsalate (Compound) -resembles-> aspirin (Compound) -treats-> pain (Disease)"
    assert facts == [(2, 1, 0), (0, 0, 1)]


def test_hops_without_a_recorded_type_or_fact_give_no_fact():
    lookup = EdgeLookup(heads=[0], relations=[0], tails=[1], n_entities=3, n_relations=2)

    _, _, facts = describe_stretch([0, 2, 1], [9, 1], lookup, ("CtD", "CrC"), lambda node, kind_only=False: "x")

    assert facts == []


def test_stretch_categories():
    assert stretch_category(["Compound", "Compound", "Disease"], ends_with_treatment=True) == "another drug's treatment"
    assert stretch_category(["Compound", "Gene", "Disease"], ends_with_treatment=False) == "gene"
    assert stretch_category(["Compound", "Side Effect", "Compound", "Disease"], False) == "other"


GRAPH = encode_triples(
    [
        ("Compound::a", "CtD", "Disease::x"),
        ("Compound::b", "CtD", "Disease::x"),
        ("Compound::a", "CrC", "Compound::b"),
        ("Compound::c", "CbG", "Gene::g"),
    ]
)


def test_drop_facts_removes_only_the_named_facts():
    index = GRAPH.entities.index
    fact = (index("Compound::a"), GRAPH.relation_names.index("CrC"), index("Compound::b"))
    keep = np.ones(len(GRAPH.heads), dtype=bool)

    kept = drop_facts(GRAPH, keep, [fact])

    removed = {(GRAPH.heads[i], GRAPH.relations[i], GRAPH.tails[i]) for i in np.flatnonzero(~kept)}
    assert removed == {fact}
    assert drop_facts(GRAPH, keep, []).all()


def test_incident_facts_respect_the_kept_graph_and_relations():
    index = GRAPH.entities.index
    keep = np.ones(len(GRAPH.heads), dtype=bool)
    treats = GRAPH.relation_names.index("CtD")

    touching = incident_facts(GRAPH, keep, [index("Disease::x")], [treats])
    assert sorted(touching) == sorted(
        [(index("Compound::a"), treats, index("Disease::x")), (index("Compound::b"), treats, index("Disease::x"))]
    )
    keep[0] = False
    assert len(incident_facts(GRAPH, keep, [index("Disease::x")], [treats])) == 1
