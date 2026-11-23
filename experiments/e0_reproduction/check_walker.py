"""Compare the released walk recording with Appendix A.2, and check that seeded walks repeat.

Runs the real `Flock.walks` on the graph of the paper's Figure 1, at the zero-shot Hetionet setting.
The dashed links in that figure are queries, so they are not part of the graph. A CPU build of the
environment in the root README is enough.

Usage: python check_walker.py [--seed 1024] [--output ../results/e0_walker_check.json]
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import jinja2
import numpy as np
import torch
import torch_geometric
import yaml
from torch_geometric.data import Data

REPO = Path(__file__).resolve().parents[2]
CONFIG = "src_entity/config/zeroshot_transductive/n512_ensemble2.yaml"
RELATIONS = ["like", "dislike", "friendWith"]
NODES = ["Luke", "HanSolo", "Emperor", "Yoda", "DarthVader", "Chewbacca", "Jabba", "Leia"]
EDGES = [
    ("Luke", "like", "HanSolo"),
    ("Luke", "dislike", "Emperor"),
    ("Luke", "friendWith", "Yoda"),
    ("Luke", "dislike", "DarthVader"),
    ("Luke", "like", "Chewbacca"),
    ("HanSolo", "dislike", "Jabba"),
    ("Emperor", "dislike", "Jabba"),
    ("DarthVader", "like", "Leia"),
    ("Chewbacca", "like", "Leia"),
]


def figure_1_graph():
    """Build the graph as the dataset classes do: each fact plus a flipped copy with type r + |R|."""
    heads = [NODES.index(head) for head, _, _ in EDGES]
    tails = [NODES.index(tail) for _, _, tail in EDGES]
    types = torch.tensor([RELATIONS.index(relation) for _, relation, _ in EDGES])
    edge_index = torch.tensor([heads, tails])
    return Data(
        edge_index=torch.cat([edge_index, edge_index.flip(0)], dim=1),
        edge_type=torch.cat([types, types + len(RELATIONS)]),
        num_nodes=len(NODES),
        num_relations=torch.tensor(2 * len(RELATIONS)),
    )


def load_model():
    sys.path.insert(0, str(REPO / "src_entity"))
    from flock.models import Flock

    template = jinja2.Template((REPO / CONFIG).read_text())
    config = yaml.safe_load(template.render(dataset="Figure1", gpus=[0], epochs=0, bpe=0, ckpt="none"))
    return Flock(config["model"]).eval()


def sample(model, data, head, seed=None, seed_numpy=False):
    if seed is not None:
        torch.manual_seed(seed)
    np.random.seed(seed if seed_numpy else None)
    with torch.no_grad():
        return model.walks(data, torch.tensor([head]))


def digest(record):
    hasher = hashlib.sha256()
    for tensor in record:
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def hop_statistics(record, query_relation="like"):
    """Measure how hops are recorded. Position 1 is skipped because edge-started walks fix it."""
    walks, _, _, _, types, named_types, directions = [tensor.flatten(0, 2).numpy() for tensor in record]
    n_relations = len(RELATIONS)
    relation = np.full((len(NODES), len(NODES)), -1)
    against = np.zeros((len(NODES), len(NODES)), dtype=bool)
    for head, name, tail in EDGES:
        head_id, tail_id = NODES.index(head), NODES.index(tail)
        relation[head_id, tail_id] = relation[tail_id, head_id] = RELATIONS.index(name)
        against[tail_id, head_id] = True

    previous, current = walks[:, 1:-1], walks[:, 2:]
    hop_relation = relation[previous, current]
    hop_against = against[previous, current]
    recorded = types[:, 2:]
    is_inverse = recorded >= n_relations
    is_backward = directions[:, 2:] == 1
    if (hop_relation < 0).any() or not (directions[:, 2:] < 2).all():
        raise RuntimeError("found a hop that is not a single edge of the Figure 1 graph")

    along = ~hop_against
    same_relation = hop_relation[:, :-1] == hop_relation[:, 1:]
    same_id = named_types[:, 2:-1] == named_types[:, 3:]
    query_hops = hop_relation == RELATIONS.index(query_relation)
    return {
        "hops": int(hop_relation.size),
        "recorded_relation_is_the_traversed_one": bool((recorded % n_relations == hop_relation).all()),
        "head_to_tail_hops": {
            "recorded_r_forward": float((~is_inverse[along] & ~is_backward[along]).mean()),
            "recorded_r_inverse_backward": float((is_inverse[along] & is_backward[along]).mean()),
        },
        "tail_to_head_hops": {
            "recorded_r_inverse_forward": float((is_inverse[~along] & ~is_backward[~along]).mean()),
            "recorded_r_backward": float((~is_inverse[~along] & is_backward[~along]).mean()),
        },
        "share_matching_appendix_a2": float(((is_inverse == hop_against) & (is_backward == hop_against)).mean()),
        "true_direction_recoverable_from_type_xor_direction": bool(((is_inverse ^ is_backward) == hop_against).all()),
        "consecutive_same_relation_hops": int(same_relation.sum()),
        "consecutive_same_relation_share_with_same_anonymous_id": float(same_id[same_relation].mean()),
        "query_relation_hops_flagged": float((recorded[query_hops] == RELATIONS.index(query_relation)).mean()),
    }


def seed_checks(model, data, head, seed):
    released = [digest(sample(model, data, head, seed=seed)) for _ in range(2)]
    seeded = [digest(sample(model, data, head, seed=seed, seed_numpy=True)) for _ in range(2)]
    return {
        "torch_seed_only_repeats": released[0] == released[1],
        "torch_and_numpy_seed_repeats": seeded[0] == seeded[1],
        "seeded_walks_sha256": seeded[0],
    }


def metadata(seed):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True)
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src_entity", "graph-walker", "experiments/e0_reproduction"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "commit": commit.stdout.strip(),
        "code_modified_since_commit": bool(dirty.stdout.strip()),
        "seed": seed,
        "config": CONFIG,
        "cpu_threads_available": len(os.sched_getaffinity(0)),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "numpy": np.__version__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--output")
    args = parser.parse_args()

    model = load_model()
    data = figure_1_graph()
    head = NODES.index("Luke")
    record = sample(model, data, head, seed=args.seed, seed_numpy=True)
    results = {
        "metadata": metadata(args.seed),
        "walk_tensor_shape": list(record[0].shape),
        "recording": hop_statistics(record),
        "seeding": seed_checks(model, data, head, args.seed),
    }
    print(json.dumps(results, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
