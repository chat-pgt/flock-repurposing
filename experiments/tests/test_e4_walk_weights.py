import numpy as np
import pytest

from e4_explanations.walk_weights import (
    EdgeLookup,
    describe_stretch,
    distance_label,
    hop_relation,
    nearest_stretch,
    occurrence_shares,
    occurrence_weights,
)


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
    text, metapath = describe_stretch([0, 2], [3], lookup, ("CtD", "CrC"), label)
    assert text == "aspirin (Compound) <-resembles- salsalate (Compound)"
    assert metapath == "Compound <-resembles- Compound"

    text, _ = describe_stretch([2, 0, 1], [1, 2], lookup, ("CtD", "CrC"), label)
    assert text == "salsalate (Compound) -resembles-> aspirin (Compound) -treats-> pain (Disease)"
