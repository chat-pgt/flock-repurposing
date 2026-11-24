import numpy as np

from gpu.graphs import encode_triples, plan_queries, rank_counts, training_mask

TRIPLES = [
    ("Compound::b", "CtD", "Disease::x"),
    ("Compound::a", "CtD", "Disease::x"),
    ("Compound::a", "CpD", "Disease::y"),
    ("Compound::a", "CrC", "Compound::b"),
]


def test_encoding_does_not_depend_on_triple_order():
    first = encode_triples(TRIPLES)
    second = encode_triples(reversed(TRIPLES))

    assert first.entities == second.entities
    assert first.relation_names == ("CpD", "CrC", "CtD")
    decoded = {
        (first.entities[h], first.relation_names[r], first.entities[t])
        for h, r, t in zip(first.heads.tolist(), first.relations.tolist(), first.tails.tolist(), strict=True)
    }
    assert decoded == set(TRIPLES)


def test_training_mask_removes_only_held_out_facts_and_dropped_relations():
    graph = encode_triples(TRIPLES + [("Compound::a", "CpD", "Disease::x")])
    index = graph.entities.index
    treats = graph.relation_names.index("CtD")
    resembles = graph.relation_names.index("CrC")

    keep = training_mask(graph, [index("Compound::a")], [index("Disease::x")], treats)
    assert keep.tolist() == [True, False, True, True, True]

    keep = training_mask(graph, [index("Compound::a")], [index("Disease::x")], treats, [resembles])
    assert keep.tolist() == [True, False, True, False, True]


def test_each_query_is_planned_once_per_fold():
    plan = plan_queries(np.array([7, 7, 8, 7]), np.array([0, 0, 0, 1]), 0)

    assert plan == {7: [0, 1], 8: [2]}


def test_rank_counts_ignore_the_target_and_excluded_entities():
    scores = np.array([5.0, 3.0, 3.0, 9.0, 1.0])

    assert rank_counts(scores, 1, {3}) == (1, 1)
