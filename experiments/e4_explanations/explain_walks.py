"""E4: which of FLOCK's walks carry its prediction for a few well-known drug -> disease pairs.

Each pair is scored zero-shot on the fold where it is held out, at the D9 setting. The runner records the
walks and the target disease's attention logits without changing upstream code, checks that the recorded
weights reproduce the model's consensus output, and credits each walk with its share of the target's state.

Usage, from experiments/:
    python -m e4_explanations.explain_walks --seeds 0,1,2 --output-dir e4
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from common.evaluation import candidate_mask, rank_metrics
from common.hetionet import EDGES_PATH, SPLIT_PATH, TREATS, read_edges, read_split
from e4_explanations.walk_weights import (
    EdgeLookup,
    describe_stretch,
    distance_label,
    nearest_stretch,
    occurrence_shares,
)
from gpu.graphs import encode_triples, training_mask
from gpu.models import build_data, load_flock, peak_gpu_gb, run_metadata, seed_everything
from gpu.run_entity import score_query

NODES_PATH = EDGES_PATH.parent / "hetionet-v1.0-nodes.tsv"
# Chosen before running, from E1 and E2 (PROTOCOL.md, E4).
PAIRS = (
    ("Compound::DB00177", "Disease::DOID:10763"),  # valsartan -> hypertension: flagged in E2
    ("Compound::DB00284", "Disease::DOID:9352"),  # acarbose -> type 2 diabetes: first indication, FLOCK >> ULTRA
    ("Compound::DB00331", "Disease::DOID:9352"),  # metformin -> type 2 diabetes: best known
)
START_KINDS = ("drug", "random node", "relation-balanced edge")
MAX_STRETCH_HOPS = 4
TOP_STRETCHES = 10


class WalkRecorder:
    """Keep the walks of one forward pass and the target's node-consensus inputs at each refinement."""

    def __init__(self, model, module, target, n_nodes):
        self.model, self.module, self.target, self.n_nodes = model, module, target, n_nodes
        self.original_consensus = module.consensus_softmax_stable
        self.walks, self.occurrences = None, []

    def record_walks(self, *args, **kwargs):
        walks = self.original_walks(*args, **kwargs)
        self.walks = (walks[0].cpu().numpy(), walks[4].cpu().numpy())  # node ids and edge types
        return walks

    def consensus(self, x, logits, ids, num_ids):
        output = self.original_consensus(x, logits, ids, num_ids)
        if num_ids == self.n_nodes:
            batch, samples, length = ids.shape
            mask = ids == self.target
            heads = logits.shape[-1]
            where = mask.nonzero().cpu().numpy()  # (pass, walk, position)
            self.occurrences.append(
                {
                    "where": where,
                    "logits": logits.view(batch, samples, length, heads)[mask].double().cpu().numpy(),
                    "values": x.view(batch, samples, length, -1)[mask].double().cpu().numpy(),
                    "output": output[:, self.target].double().cpu().numpy(),
                }
            )
        return output

    def __enter__(self):
        self.original_walks = self.model.walks
        self.model.walks = self.record_walks
        self.module.consensus_softmax_stable = self.consensus
        return self

    def __exit__(self, *_):
        del self.model.walks
        self.module.consensus_softmax_stable = self.original_consensus


def consensus_error(occurrences, heads):
    """Largest gap between the model's consensus output and the one rebuilt from the recorded weights."""
    worst = 0.0
    for step in occurrences:
        for batch in range(step["output"].shape[0]):
            rows = step["where"][:, 0] == batch
            if not rows.any():
                continue
            logits, values = step["logits"][rows], step["values"][rows]
            weights = np.exp(logits - logits.max(axis=0))
            weights /= weights.sum(axis=0)
            rebuilt = (values.reshape(len(values), heads, -1) * weights[:, :, None]).sum(axis=0).reshape(-1)
            worst = max(worst, float(np.abs(rebuilt - step["output"][batch]).max()))
    return worst


def mean_shares(dicts):
    """Average share dictionaries over seeds, counting a missing key as zero, largest first."""
    totals = defaultdict(float)
    for shares in dicts:
        for key, share in shares.items():
            totals[key] += share / len(dicts)
    return dict(sorted(totals.items(), key=lambda item: -item[1]))


def read_names(path):
    with open(path, newline="") as handle:
        return {row["id"]: row["name"] for row in csv.DictReader(handle, delimiter="\t")}


def explain(recorder, drug, n_walks, lookup, graph, names):
    """Turn one forward pass's recordings into per-walk shares and drug -> disease stretches."""
    walk_ids, edge_types = recorder.walks
    groups, positions, logits = [], [], []
    for refinement, step in enumerate(recorder.occurrences):
        for batch, walk, position in step["where"].tolist():
            groups.append(refinement * walk_ids.shape[1] + batch)
            positions.append((refinement, batch, walk, position))
        logits.append(step["logits"])
    shares = occurrence_shares(groups, np.concatenate(logits))

    def label(node, kind_only=False):
        name = graph.entities[node]
        kind = name.split("::", 1)[0]
        return kind if kind_only else f"{names.get(name, name)} ({kind})"

    by_start, by_distance, by_metapath = defaultdict(float), defaultdict(float), defaultdict(float)
    stretches = defaultdict(float)
    on_walk, via_treating_compound = 0.0, 0.0
    treats = graph.relation_names.index(TREATS)
    for (refinement, batch, walk, position), share in zip(positions, shares, strict=True):
        by_start[START_KINDS[min(walk // n_walks, 2)]] += share
        nodes = walk_ids[refinement, batch, walk]
        stretch = nearest_stretch(nodes, drug, position)
        hops = None if stretch is None else len(stretch) - 1
        by_distance[distance_label(hops)] += share
        if stretch is None:
            continue
        on_walk += share
        if hops == 0 or hops > MAX_STRETCH_HOPS:
            continue
        types = edge_types[refinement, batch, walk]
        if stretch[0] < stretch[-1]:
            recorded = [int(types[p]) for p in stretch[1:]]
        else:
            recorded = [int(types[p + 1]) for p in stretch[1:]]
        stretch_nodes = [int(nodes[p]) for p in stretch]
        text, metapath = describe_stretch(stretch_nodes, recorded, lookup, graph.relation_names, label)
        stretches[text] += share
        by_metapath[metapath] += share
        if hops >= 2 and lookup.has(stretch_nodes[-2], treats, stretch_nodes[-1]):
            via_treating_compound += share
    return {
        "share_by_walk_start": dict(by_start),
        "share_on_walks_visiting_drug": on_walk,
        "share_via_other_treating_compound": via_treating_compound,
        "share_by_drug_distance": dict(by_distance),
        "share_by_metapath": dict(sorted(by_metapath.items(), key=lambda item: -item[1])),
        "stretches": dict(stretches),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deadline", type=float, help="Unix time after which no new pair starts")
    parser.add_argument("--edges", type=Path, default=EDGES_PATH)
    parser.add_argument("--nodes", type=Path, default=NODES_PATH)
    parser.add_argument("--split", type=Path, default=SPLIT_PATH)
    parser.add_argument("--walk-num", type=int, help="Override n, for smoke tests only")
    parser.add_argument("--ensemble", type=int, help="Override P, for smoke tests only")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    graph = encode_triples(read_edges(args.edges))
    split = read_split(args.split)
    names = read_names(args.nodes)
    index = {name: position for position, name in enumerate(graph.entities)}
    treats = graph.relation_names.index(TREATS)
    diseases = np.array([position for position, name in enumerate(graph.entities) if name.startswith("Disease::")])
    known = defaultdict(set)
    for drug, disease in zip(split.drugs.tolist(), split.diseases.tolist(), strict=True):
        known[drug].add(disease)

    device = torch.device(args.device)
    model, settings = load_flock("entity", device, args.walk_num, args.ensemble)
    module = sys.modules[type(model).__module__]
    heads = model.H
    seeds = [int(seed) for seed in args.seeds.split(",")]
    metadata = run_metadata(device, model="flock", **settings)
    results = {"metadata": metadata, "pairs": []}

    for drug_name, disease_name in PAIRS:
        if args.deadline is not None and time.time() > args.deadline:
            print(f"time budget reached before {drug_name} -> {disease_name}", flush=True)
            break
        row = np.flatnonzero((split.drugs == drug_name) & (split.diseases == disease_name))
        if row.size != 1:
            raise ValueError(f"{drug_name} -> {disease_name} is not a held-out treats edge")
        fold = int(split.folds[row[0]])
        held = split.folds == fold
        keep = training_mask(
            graph,
            [index[name] for name in split.drugs[held].tolist()],
            [index[name] for name in split.diseases[held].tolist()],
            treats,
        )
        lookup = EdgeLookup(
            graph.heads[keep], graph.relations[keep], graph.tails[keep], len(graph.entities), len(graph.relation_names)
        )
        data = build_data(graph, keep, "flock", device).to(device)
        drug, disease = index[drug_name], index[disease_name]
        position = {int(node): offset for offset, node in enumerate(diseases)}
        others = [position[index[name]] for name in known[drug_name] - {disease_name}]
        mask = candidate_mask(len(diseases), range(len(diseases)), others, position[disease])
        pair = {
            "drug": drug_name,
            "disease": disease_name,
            "drug_name": names.get(drug_name),
            "disease_name": names.get(disease_name),
            "fold": fold,
            "seeds": [],
        }
        for seed in seeds:
            seed_everything(1000 * seed + fold)
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with WalkRecorder(model, module, disease, len(graph.entities)) as recorder:
                scores = score_query(model, data, drug, treats, "tail", len(graph.entities), device)
            reciprocal_rank, hit = rank_metrics(scores[diseases], position[disease], mask)
            summary = explain(recorder, drug, model.N, lookup, graph, names)
            summary.update(
                seed=seed,
                reciprocal_rank=reciprocal_rank,
                hits10=hit,
                target_occurrences=int(sum(len(step["where"]) for step in recorder.occurrences)),
                consensus_max_abs_error=consensus_error(recorder.occurrences, heads),
                seconds=time.perf_counter() - start,
                peak_gpu_gb=peak_gpu_gb(device),
            )
            pair["seeds"].append(summary)
            print(
                f"{pair['drug_name']} -> {pair['disease_name']} (fold {fold}, seed {seed}): "
                f"RR {reciprocal_rank:.3f}, {summary['target_occurrences']} target visits, "
                f"{summary['share_on_walks_visiting_drug']:.0%} of weight on walks visiting the drug, "
                f"{summary['share_via_other_treating_compound']:.0%} via another compound that treats it, "
                f"consensus error {summary['consensus_max_abs_error']:.1e}, {summary['seconds']:.1f} s",
                flush=True,
            )
        totals = defaultdict(float)
        for summary in pair["seeds"]:
            for text, share in summary.pop("stretches").items():
                totals[text] += share / len(pair["seeds"])
        pair["mean_over_seeds"] = {
            key: mean_shares([summary[key] for summary in pair["seeds"]])
            for key in ("share_by_walk_start", "share_by_drug_distance", "share_by_metapath")
        }
        for key in ("share_on_walks_visiting_drug", "share_via_other_treating_compound"):
            pair["mean_over_seeds"][key] = float(np.mean([summary[key] for summary in pair["seeds"]]))
        pair["top_stretches"] = [
            {"stretch": text, "mean_share": share}
            for text, share in sorted(totals.items(), key=lambda item: -item[1])[:TOP_STRETCHES]
        ]
        results["pairs"].append(pair)
        (args.output_dir / "e4_walk_explanations.json").write_text(json.dumps(results, indent=2) + "\n")
        del data
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
