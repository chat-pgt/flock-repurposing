"""E7: extract the relation representations of ULTRA and FLOCK on Hetionet.

ULTRA computes one deterministic set of relation vectors per query relation, and all 24 are saved. FLOCK's
relation states depend on the query and on the walks, so they are saved per seed, query and pass for
(drug, *treats*, ?) queries on a fixed sample of drugs.

Usage, from experiments/:
    python -m e7_relation_similarity.extract_relations --model ultra --ultra-repo ULTRA --output-dir e7
    python -m e7_relation_similarity.extract_relations --model flock --seeds 0,1,2 --output-dir e7
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common.hetionet import EDGES_PATH, TREATS, node_kind, read_edges
from gpu.graphs import encode_triples
from gpu.models import build_data, load_flock, load_ultra, peak_gpu_gb, run_metadata, seed_everything
from gpu.run_entity import save_shard

QUERY_SAMPLE_SEED = 0


class RelationStateRecorder:
    """Accumulate FLOCK's relation updates, which `_forward` keeps in a local variable."""

    def __init__(self, module, n_type_ids, n_nodes):
        if n_type_ids == n_nodes:
            raise ValueError("cannot tell relation updates from node updates")
        self.module, self.n_type_ids = module, n_type_ids
        self.original = module.consensus_softmax_stable
        self.update, self.visited = None, None

    def __call__(self, x, logits, ids, num_ids):
        output = self.original(x, logits, ids, num_ids)
        if num_ids == self.n_type_ids:
            self.update = output if self.update is None else self.update + output
            visited = torch.zeros(ids.shape[0], num_ids, dtype=torch.bool, device=ids.device)
            visited.scatter_(1, ids.flatten(1), True)
            self.visited = visited if self.visited is None else self.visited | visited
        return output

    def __enter__(self):
        self.update, self.visited = None, None
        self.module.consensus_softmax_stable = self
        return self

    def __exit__(self, *_):
        self.module.consensus_softmax_stable = self.original


def relation_kinds(graph):
    head_kinds, tail_kinds = [], []
    for relation in range(len(graph.relation_names)):
        row = np.flatnonzero(graph.relations == relation)[0]
        head_kinds.append(node_kind(graph.entities[graph.heads[row]]))
        tail_kinds.append(node_kind(graph.entities[graph.tails[row]]))
    return head_kinds, tail_kinds


def sample_drugs(graph, n_queries):
    treats = graph.relation_names.index(TREATS)
    drugs = np.unique(graph.heads[graph.relations == treats])
    chosen = np.random.default_rng(QUERY_SAMPLE_SEED).choice(drugs, size=min(n_queries, len(drugs)), replace=False)
    return np.sort(chosen)


@torch.no_grad()
def ultra_states(model, data, n_relations, device):
    queries = torch.arange(n_relations, device=device)
    return np.stack(
        [model.relation_model(data.relation_graph, query=query[None])[0].float().cpu().numpy() for query in queries]
    )


@torch.no_grad()
def flock_states(model, data, drugs, treats, n_relations, device):
    module = sys.modules[type(model).__module__]
    if not (model.attention_scatter and model.additive_refinement):
        raise ValueError("expected the released attention and additive-update configuration")
    states, visited = [], []
    for drug in drugs.tolist():
        batch = torch.tensor([[[drug, 0, treats]]], device=device)
        with RelationStateRecorder(module, 2 * n_relations + 1, data.num_nodes) as recorder:
            model(data, batch)
        states.append((model.type_init[None, None] + recorder.update).float().cpu().numpy())
        visited.append(recorder.visited.cpu().numpy())
    return np.stack(states), np.stack(visited)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["flock", "ultra"], required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-queries", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deadline", type=float, help="Unix time after which no new seed starts")
    parser.add_argument("--ultra-repo", type=Path)
    parser.add_argument("--edges", type=Path, default=EDGES_PATH)
    parser.add_argument("--walk-num", type=int, help="Override n, for smoke tests only")
    parser.add_argument("--ensemble", type=int, help="Override P, for smoke tests only")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    graph = encode_triples(read_edges(args.edges))
    n_relations = len(graph.relation_names)
    head_kinds, tail_kinds = relation_kinds(graph)
    keep = np.ones(len(graph.heads), dtype=bool)
    device = torch.device(args.device)
    labels = {
        "relation_names": np.array(graph.relation_names),
        "head_kinds": np.array(head_kinds),
        "tail_kinds": np.array(tail_kinds),
    }

    if args.model == "ultra":
        model, settings = load_ultra(args.ultra_repo, device)
        data = build_data(graph, keep, "ultra", device).to(device)
        start = time.perf_counter()
        states = ultra_states(model, data, n_relations, device)
        info = {**run_metadata(device, model="ultra", **settings), "seconds": time.perf_counter() - start}
        save_shard(args.output_dir / "e7_ultra.npz", metadata=np.array(json.dumps(info)), states=states, **labels)
        print(f"e7_ultra: states {states.shape}, {info['seconds']:.1f}s", flush=True)
        return

    model, settings = load_flock("entity", device, args.walk_num, args.ensemble)
    data = build_data(graph, keep, "flock", device).to(device)
    drugs = sample_drugs(graph, args.n_queries)
    treats = graph.relation_names.index(TREATS)
    for seed in [int(value) for value in args.seeds.split(",")]:
        path = args.output_dir / f"e7_flock_seed{seed}.npz"
        if path.exists():
            print(f"skip {path.name}", flush=True)
            continue
        if args.deadline is not None and time.time() > args.deadline:
            print(f"time budget reached before {path.name}", flush=True)
            return
        seed_everything(seed)
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        states, visited = flock_states(model, data, drugs, treats, n_relations, device)
        info = {
            **run_metadata(device, model="flock", **settings),
            "seed": seed,
            "seconds": time.perf_counter() - start,
            "peak_gpu_gb": peak_gpu_gb(device),
        }
        save_shard(
            path,
            metadata=np.array(json.dumps(info)),
            states=states,
            visited=visited,
            query_entities=np.array([graph.entities[drug] for drug in drugs.tolist()]),
            **labels,
        )
        forward_visited = visited[..., :n_relations].mean()
        print(
            f"{path.name}: states {states.shape}, forward relations visited {forward_visited:.3f}, "
            f"{info['seconds'] / len(drugs):.2f} s/query",
            flush=True,
        )


if __name__ == "__main__":
    main()
