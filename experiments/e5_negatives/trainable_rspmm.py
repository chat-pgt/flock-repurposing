"""A memory-light backward pass for ULTRA's relational message passing, for fine-tuning on hosted GPUs.

`gpu.models.tensor_rspmm` replaces ULTRA's compiled kernel, output[head] += weight * relation[type] *
input[tail]. Through autograd it would keep every edge message for the backward pass, about 2 GB per layer
on Hetionet at batch size 1. This function computes the forward pass with that same code and recomputes the
messages in chunks during the backward pass instead.
"""

import sys

import torch

from gpu.models import tensor_rspmm

ELEMENTS_PER_CHUNK = 32_000_000


def _chunks(n_edges, columns):
    step = max(1, ELEMENTS_PER_CHUNK // max(1, columns))
    for start in range(0, n_edges, step):
        yield slice(start, start + step)


class RelationalSpmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input)
        with torch.no_grad():
            return tensor_rspmm(edge_index, edge_type, edge_weight, relation, input)

    @staticmethod
    def backward(ctx, grad_output):
        edge_index, edge_type, edge_weight, relation, input = ctx.saved_tensors
        heads, tails = edge_index
        grad_relation = torch.zeros_like(relation) if ctx.needs_input_grad[3] else None
        grad_input = torch.zeros_like(input) if ctx.needs_input_grad[4] else None
        for part in _chunks(heads.numel(), input.shape[1]):
            upstream = grad_output[heads[part]] * edge_weight[part].unsqueeze(-1)
            if grad_input is not None:
                grad_input.index_add_(0, tails[part], upstream * relation[edge_type[part]])
            if grad_relation is not None:
                grad_relation.index_add_(0, edge_type[part], upstream * input[tails[part]])
        return None, None, None, grad_relation, grad_input


def trainable_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul="mul"):
    if (sum, mul) != ("add", "mul"):
        raise NotImplementedError(f"unsupported kernel: sum={sum}, mul={mul}")
    if edge_weight.requires_grad:
        raise NotImplementedError("edge weights are not trained")
    return RelationalSpmm.apply(edge_index, edge_type, edge_weight, relation, input)


def install():
    """Route ULTRA's kernel calls to the trainable version. Call after `gpu.models.load_ultra`."""
    sys.modules["ultra.rspmm"].generalized_rspmm = trainable_rspmm
