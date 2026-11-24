"""Load the released FLOCK and ULTRA checkpoints and build their input graphs."""

import hashlib
import os
import platform
import random
import subprocess
import sys
import types
from pathlib import Path

import jinja2
import numpy as np
import torch
import torch_geometric
import yaml
from torch_geometric.data import Data

REPO = Path(__file__).resolve().parents[2]
ULTRA_COMMIT = "427966ad8ed60420eef034063d44f3153addff90"
ULTRA_CHECKPOINT_SHA256 = "fdedc01b0045fc089d2ad5da08569466b7b221691fe978a18e91082bbb133c18"
EDGES_PER_CHUNK = 500_000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_everything(seed):
    """Seed Python, PyTorch and NumPy. The released walker draws its seed from NumPy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _render(path):
    template = jinja2.Template(Path(path).read_text())
    return yaml.safe_load(template.render(dataset="Hetionet", gpus="null", epochs=0, bpe="null", ckpt="none"))


def load_flock(task, device, walk_num=None, ensemble=None):
    """Load FLOCK for `entity` or `relation` prediction at the paper's zero-shot Hetionet setting."""
    sys.path.insert(0, str(REPO / f"src_{task}"))
    from flock.models import Flock

    config = _render(REPO / f"src_{task}/config/zeroshot_transductive/n512_ensemble2.yaml")["model"]
    if walk_num is not None:
        config["walk_num"] = walk_num
    if ensemble is not None:
        config["test_samples"] = ensemble
    model = Flock(model_cfg=config)
    checkpoint = REPO / f"checkpoints/flock_{task}.pth"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    return model.to(device).eval(), {"checkpoint_sha256": sha256(checkpoint), **_flock_settings(config)}


def _flock_settings(config):
    return {"walk_num": config["walk_num"], "ensemble": config["test_samples"], "walk_len": config["walk_len"]}


def tensor_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul="mul"):
    """Compute what ULTRA's compiled kernel does: output[head] += weight * relation[type] * input[tail].

    ULTRA's own `message` and `aggregate` functions send messages the opposite way, so they do not
    reproduce the kernel that its published checkpoints were evaluated with.
    """
    if (sum, mul) != ("add", "mul"):
        raise NotImplementedError(f"unsupported kernel: sum={sum}, mul={mul}")
    heads, tails = edge_index
    output = torch.zeros_like(input)
    for start in range(0, heads.numel(), EDGES_PER_CHUNK):
        stop = start + EDGES_PER_CHUNK
        weights = edge_weight[start:stop].unsqueeze(-1)
        output.index_add_(0, heads[start:stop], relation[edge_type[start:stop]] * input[tails[start:stop]] * weights)
    return output


def load_ultra(repo, device, use_compiled_kernel=False):
    """Load ULTRA's three-graph checkpoint.

    ULTRA compiles a CUDA kernel at import, which needs a compiler that hosted notebooks lack. By default
    the kernel is replaced with an equivalent PyTorch function.
    """
    repo = Path(repo)
    checkpoint = repo / "ckpts/ultra_3g.pth"
    if sha256(checkpoint) != ULTRA_CHECKPOINT_SHA256:
        raise ValueError("unexpected ULTRA checkpoint")
    sys.path.insert(0, str(repo))
    if not use_compiled_kernel:
        replacement = types.ModuleType("ultra.rspmm")
        replacement.generalized_rspmm = tensor_rspmm
        sys.modules["ultra.rspmm"] = replacement
    from ultra.models import Ultra

    config = _render(repo / "config/transductive/inference.yaml")["model"]
    model = Ultra(rel_model_cfg=config["relation_model"], entity_model_cfg=config["entity_model"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    settings = {"checkpoint_sha256": ULTRA_CHECKPOINT_SHA256, "ultra_commit": ULTRA_COMMIT}
    return model.to(device).eval(), settings


def build_data(graph, keep, model_name, device):
    """Build the model input: each kept fact plus a flipped copy with relation r + |R|."""
    n_relations = len(graph.relation_names)
    edges = torch.from_numpy(np.stack([graph.heads[keep], graph.tails[keep]])).long()
    types = torch.from_numpy(graph.relations[keep].copy()).long()
    edge_index = torch.cat([edges, edges.flip(0)], dim=1)
    edge_type = torch.cat([types, types + n_relations])
    if model_name == "flock":
        return Data(
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=len(graph.entities),
            num_relations=torch.tensor(2 * n_relations),
            device=device,
        )
    from ultra.tasks import build_relation_graph

    data = Data(
        edge_index=edge_index, edge_type=edge_type, num_nodes=len(graph.entities), num_relations=2 * n_relations
    )
    return build_relation_graph(data)


def run_metadata(device, **settings):
    commit = os.environ.get("EXPERIMENT_COMMIT")
    if commit is None:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    return {
        "commit": commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "numpy": np.__version__,
        "device": torch.cuda.get_device_name(0) if str(device).startswith("cuda") else "cpu",
        **settings,
    }


def peak_gpu_gb(device):
    return torch.cuda.max_memory_allocated() / 1024**3 if str(device).startswith("cuda") else None
