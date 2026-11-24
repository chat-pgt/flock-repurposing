"""Needs PyTorch and PyG, which the CPU analysis environment does not install, so it is skipped there."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from e5_negatives import trainable_rspmm  # noqa: E402
from e5_negatives.finetune import adversarial_bce  # noqa: E402
from gpu.models import tensor_rspmm  # noqa: E402


def random_graph(seed, n_nodes=7, n_edges=30, n_relations=4, columns=6):
    generator = torch.Generator().manual_seed(seed)
    edge_index = torch.randint(n_nodes, (2, n_edges), generator=generator)
    edge_type = torch.randint(n_relations, (n_edges,), generator=generator)
    edge_weight = torch.rand(n_edges, generator=generator, dtype=torch.float64)
    relation = torch.randn(n_relations, columns, generator=generator, dtype=torch.float64, requires_grad=True)
    node_input = torch.randn(n_nodes, columns, generator=generator, dtype=torch.float64, requires_grad=True)
    return edge_index, edge_type, edge_weight, relation, node_input


def test_forward_matches_the_kernel_replacement():
    edge_index, edge_type, edge_weight, relation, node_input = random_graph(0)

    expected = tensor_rspmm(edge_index, edge_type, edge_weight, relation, node_input)
    actual = trainable_rspmm.trainable_rspmm(edge_index, edge_type, edge_weight, relation, node_input)

    assert torch.equal(actual, expected)


def test_gradients_match_autograd_through_the_kernel_replacement(monkeypatch):
    monkeypatch.setattr(trainable_rspmm, "ELEMENTS_PER_CHUNK", 12)  # force several chunks
    edge_index, edge_type, edge_weight, relation, node_input = random_graph(1)
    upstream = torch.randn(7, 6, dtype=torch.float64)

    expected = torch.autograd.grad(
        (tensor_rspmm(edge_index, edge_type, edge_weight, relation, node_input) * upstream).sum(),
        (relation, node_input),
    )
    actual = torch.autograd.grad(
        (trainable_rspmm.trainable_rspmm(edge_index, edge_type, edge_weight, relation, node_input) * upstream).sum(),
        (relation, node_input),
    )

    for got, want in zip(actual, expected, strict=True):
        assert torch.allclose(got, want)


def test_gradcheck():
    edge_index, edge_type, edge_weight, relation, node_input = random_graph(2)

    def function(relation, node_input):
        return trainable_rspmm.trainable_rspmm(edge_index, edge_type, edge_weight, relation, node_input)

    assert torch.autograd.gradcheck(function, (relation, node_input))


def test_adversarial_loss_matches_the_released_training_loop():
    pred = torch.tensor([[2.0, 0.5, -1.0, 0.0], [-0.3, 1.5, 1.0, -2.0]])

    target = torch.zeros_like(pred)
    target[:, 0] = 1
    loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
    weight = torch.ones_like(pred)
    weight[:, 1:] = torch.softmax(pred[:, 1:], dim=-1)
    expected = ((loss * weight).sum(-1) / weight.sum(-1)).mean()

    assert torch.allclose(adversarial_bce(pred, 1.0), expected)
