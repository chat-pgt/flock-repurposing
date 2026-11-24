"""E4: which of FLOCK's walks carry its drug -> disease predictions, and does deleting them matter?

Every case is scored zero-shot on the fold where its edge is held out, at the D9 setting. The runner records
the walks and the target disease's attention logits without changing upstream code, checks that the weights
rebuild the model's consensus output, and credits each walk with its share of the disease's state. Walks
that also visit the drug give drug -> disease stretches, named by the Hetionet facts behind each hop.

Two layers test whether the credited stretches matter. Deleting the facts of the top stretches is compared
with deleting as many random facts touching the drug or the disease. For the four showcase pairs, families of
the drug's facts and the disease's other treatments are also deleted one at a time.

Usage, from experiments/:
    python -m e4_explanations.explain_walks --output-dir e4
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
    drop_facts,
    incident_facts,
    nearest_stretch,
    occurrence_shares,
    stretch_category,
)
from gpu.graphs import encode_triples, training_mask
from gpu.models import build_data, load_flock, peak_gpu_gb, run_metadata, seed_everything
from gpu.run_entity import score_query

NODES_PATH = EDGES_PATH.parent / "hetionet-v1.0-nodes.tsv"
# Chosen before running (PROTOCOL.md, E4). The first three come from E1 and E2. Dasatinib is the held-out
# edge with the most drug -binds-> gene <-associates- disease paths among those where no compound that
# resembles the drug or shares its class treats the disease in training: the best case for a mechanism.
SHOWCASE = (
    ("Compound::DB00177", "Disease::DOID:10763"),  # valsartan -> hypertension: flagged in E2
    ("Compound::DB00284", "Disease::DOID:9352"),  # acarbose -> type 2 diabetes: first indication, FLOCK >> ULTRA
    ("Compound::DB00331", "Disease::DOID:9352"),  # metformin -> type 2 diabetes: best known
    ("Compound::DB01254", "Disease::DOID:2531"),  # dasatinib -> hematologic cancer: mechanism available
)
SURVEY_PER_FOLD = 8
SURVEY_SEED = 0
START_KINDS = ("drug", "random node", "relation-balanced edge")
MAX_STRETCH_HOPS = 4
TOP_STRETCHES = 10
TOP_FOR_DELETION = 5
DRUG_FAMILIES = {
    "drug resembles": ("CrC",),
    "drug's pharmacologic classes": ("PCiC",),
    "drug's genes": ("CbG", "CuG", "CdG"),
    "drug's side effects": ("CcSE",),
}


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
            self.occurrences.append(
                {
                    "where": mask.nonzero().cpu().numpy(),  # (pass, walk, position)
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


def read_names(path):
    with open(path, newline="") as handle:
        return {row["id"]: row["name"] for row in csv.DictReader(handle, delimiter="\t")}


def mean_shares(dicts):
    """Average share dictionaries over seeds, counting a missing key as zero, largest first."""
    totals = defaultdict(float)
    for shares in dicts:
        for key, share in shares.items():
            totals[key] += share / len(dicts)
    return dict(sorted(totals.items(), key=lambda item: -item[1]))


class Explainer:
    def __init__(self, args, device):
        self.args, self.device = args, device
        self.graph = encode_triples(read_edges(args.edges))
        self.split = read_split(args.split)
        self.names = read_names(args.nodes)
        self.index = {name: position for position, name in enumerate(self.graph.entities)}
        self.treats = self.graph.relation_names.index(TREATS)
        self.diseases = np.array(
            [position for position, name in enumerate(self.graph.entities) if name.startswith("Disease::")]
        )
        self.position = {int(node): offset for offset, node in enumerate(self.diseases)}
        self.known = defaultdict(set)
        for drug, disease in zip(self.split.drugs.tolist(), self.split.diseases.tolist(), strict=True):
            self.known[drug].add(disease)
        self.model, settings = load_flock("entity", device, args.walk_num, args.ensemble)
        self.module = sys.modules[type(self.model).__module__]
        self.metadata = run_metadata(device, model="flock", **settings)
        self.fold_cache = {}

    def label(self, node, kind_only=False):
        name = self.graph.entities[node]
        kind = name.split("::", 1)[0]
        return kind if kind_only else f"{self.names.get(name, name)} ({kind})"

    def fold_graph(self, fold):
        if fold not in self.fold_cache:
            held = self.split.folds == fold
            keep = training_mask(
                self.graph,
                [self.index[name] for name in self.split.drugs[held].tolist()],
                [self.index[name] for name in self.split.diseases[held].tolist()],
                self.treats,
            )
            graph = self.graph
            lookup = EdgeLookup(
                graph.heads[keep],
                graph.relations[keep],
                graph.tails[keep],
                len(graph.entities),
                len(graph.relation_names),
            )
            self.fold_cache = {fold: (keep, lookup)}
        return self.fold_cache[fold]

    def score(self, keep, drug_name, disease_name, fold, seeds, record=False):
        """Score (drug, treats, ?) on the kept graph with each seed. Returns per-seed results."""
        drug, disease = self.index[drug_name], self.index[disease_name]
        others = [self.position[self.index[name]] for name in self.known[drug_name] - {disease_name}]
        mask = candidate_mask(len(self.diseases), range(len(self.diseases)), others, self.position[disease])
        data = build_data(self.graph, keep, "flock", self.device).to(self.device)
        results = []
        for seed in seeds:
            seed_everything(1000 * seed + fold)
            recorder = WalkRecorder(self.model, self.module, disease, len(self.graph.entities))
            if record:
                with recorder:
                    scores = score_query(
                        self.model, data, drug, self.treats, "tail", len(self.graph.entities), self.device
                    )
            else:
                scores = score_query(self.model, data, drug, self.treats, "tail", len(self.graph.entities), self.device)
            reciprocal_rank, _ = rank_metrics(scores[self.diseases], self.position[disease], mask)
            results.append(
                {
                    "seed": seed,
                    "reciprocal_rank": reciprocal_rank,
                    "score": float(scores[disease]),
                    "recorder": recorder,
                }
            )
        del data
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        return results

    def explain(self, recorder, drug, lookup):
        """Turn one forward pass's recordings into weight shares and named drug -> disease stretches."""
        walk_ids, edge_types = recorder.walks
        groups, positions, logits = [], [], []
        for refinement, step in enumerate(recorder.occurrences):
            for batch, walk, position in step["where"].tolist():
                groups.append(refinement * walk_ids.shape[1] + batch)
                positions.append((refinement, batch, walk, position))
            logits.append(step["logits"])
        shares = occurrence_shares(groups, np.concatenate(logits)) if groups else []

        by_start, by_distance, by_metapath, by_category = (defaultdict(float) for _ in range(4))
        stretches, facts_of = defaultdict(float), {}
        on_walk = 0.0
        for (refinement, batch, walk, position), share in zip(positions, shares, strict=True):
            by_start[START_KINDS[min(walk // self.model.N, 2)]] += share
            nodes = walk_ids[refinement, batch, walk]
            stretch = nearest_stretch(nodes, drug, position)
            hops = None if stretch is None else len(stretch) - 1
            by_distance[distance_label(hops)] += share
            if stretch is None:
                by_category["drug not on walk"] += share
                continue
            on_walk += share
            if hops > MAX_STRETCH_HOPS:
                by_category["longer than 4 hops"] += share
                continue
            types = edge_types[refinement, batch, walk]
            if stretch[0] < stretch[-1]:
                recorded = [int(types[p]) for p in stretch[1:]]
            else:
                recorded = [int(types[p + 1]) for p in stretch[1:]]
            stretch_nodes = [int(nodes[p]) for p in stretch]
            text, metapath, facts = describe_stretch(
                stretch_nodes, recorded, lookup, self.graph.relation_names, self.label
            )
            ends_with_treatment = len(stretch_nodes) > 2 and lookup.has(
                stretch_nodes[-2], self.treats, stretch_nodes[-1]
            )
            kinds = [self.label(node, kind_only=True) for node in stretch_nodes]
            by_category[stretch_category(kinds, ends_with_treatment)] += share
            stretches[text] += share
            facts_of[text] = facts
            by_metapath[metapath] += share
        return {
            "share_by_walk_start": dict(by_start),
            "share_on_walks_visiting_drug": on_walk,
            "share_by_drug_distance": dict(by_distance),
            "share_by_category": dict(by_category),
            "share_by_metapath": dict(sorted(by_metapath.items(), key=lambda item: -item[1])),
            "stretches": dict(stretches),
            "facts_of": facts_of,
        }

    def attribute(self, drug_name, disease_name, fold, seeds):
        keep, lookup = self.fold_graph(fold)
        drug = self.index[drug_name]
        runs = self.score(keep, drug_name, disease_name, fold, seeds, record=True)
        per_seed, totals, facts_of = [], defaultdict(float), {}
        for run in runs:
            summary = self.explain(run["recorder"], drug, lookup)
            summary.update(
                seed=run["seed"],
                reciprocal_rank=run["reciprocal_rank"],
                score=run["score"],
                target_visits=int(sum(len(step["where"]) for step in run["recorder"].occurrences)),
                consensus_max_abs_error=consensus_error(run["recorder"].occurrences, self.model.H),
            )
            for text, share in summary.pop("stretches").items():
                totals[text] += share / len(runs)
            facts_of.update(summary.pop("facts_of"))
            per_seed.append(summary)
        ranked = sorted(totals.items(), key=lambda item: -item[1])
        mean = {
            key: mean_shares([summary[key] for summary in per_seed])
            for key in ("share_by_walk_start", "share_by_drug_distance", "share_by_category", "share_by_metapath")
        }
        mean["share_on_walks_visiting_drug"] = float(np.mean([s["share_on_walks_visiting_drug"] for s in per_seed]))
        top_facts = sorted({fact for text, _ in ranked[:TOP_FOR_DELETION] for fact in facts_of[text]})
        return {
            "per_seed": per_seed,
            "mean_over_seeds": mean,
            "reciprocal_rank": float(np.mean([run["reciprocal_rank"] for run in runs])),
            "score": float(np.mean([run["score"] for run in runs])),
            "top_stretches": [
                {"stretch": text, "mean_share": share, "facts": len(facts_of[text])}
                for text, share in ranked[:TOP_STRETCHES]
            ],
            "top_facts": top_facts,
        }

    def delete(self, drug_name, disease_name, fold, seeds, facts, baseline):
        """Rescore without `facts` and report the change from the baseline."""
        keep, _ = self.fold_graph(fold)
        runs = self.score(drop_facts(self.graph, keep, facts), drug_name, disease_name, fold, seeds)
        reciprocal_rank = float(np.mean([run["reciprocal_rank"] for run in runs]))
        score = float(np.mean([run["score"] for run in runs]))
        return {
            "facts_deleted": len(facts),
            "reciprocal_rank": reciprocal_rank,
            "reciprocal_rank_change": reciprocal_rank - baseline["reciprocal_rank"],
            "score_change": score - baseline["score"],
        }

    def random_facts(self, drug_name, disease_name, fold, count, draw):
        keep, _ = self.fold_graph(fold)
        pool = incident_facts(self.graph, keep, [self.index[drug_name], self.index[disease_name]])
        rng = np.random.default_rng([SURVEY_SEED, self.index[drug_name], self.index[disease_name], draw])
        chosen = rng.choice(len(pool), size=min(count, len(pool)), replace=False)
        return [pool[i] for i in chosen]

    def case(self, drug_name, disease_name, seeds, families):
        rows = np.flatnonzero((self.split.drugs == drug_name) & (self.split.diseases == disease_name))
        if rows.size != 1:
            raise ValueError(f"{drug_name} -> {disease_name} is not a held-out treats edge")
        fold = int(self.split.folds[rows[0]])
        start = time.perf_counter()
        result = {
            "drug": drug_name,
            "disease": disease_name,
            "drug_name": self.names.get(drug_name),
            "disease_name": self.names.get(disease_name),
            "fold": fold,
            "drug_has_other_indication": bool(self.split.has_other_indication[rows[0]]),
            **self.attribute(drug_name, disease_name, fold, seeds),
        }
        guided = result["top_facts"]
        result["deletions"] = {"top stretches": self.delete(drug_name, disease_name, fold, seeds, guided, result)}
        random_runs = [
            self.delete(
                drug_name,
                disease_name,
                fold,
                [summary["seed"]],
                self.random_facts(drug_name, disease_name, fold, len(guided), summary["seed"]),
                summary,  # compared with the same seed's baseline
            )
            for summary in result["per_seed"]
        ]
        result["deletions"]["random facts at drug or disease"] = {
            key: float(np.mean([run[key] for run in random_runs]))
            for key in ("facts_deleted", "reciprocal_rank", "reciprocal_rank_change", "score_change")
        }
        if families:
            keep, _ = self.fold_graph(fold)
            drug, disease = self.index[drug_name], self.index[disease_name]
            for family, relations in DRUG_FAMILIES.items():
                ids = [self.graph.relation_names.index(name) for name in relations if name in self.graph.relation_names]
                facts = incident_facts(self.graph, keep, [drug], ids)
                result["deletions"][family] = self.delete(drug_name, disease_name, fold, seeds, facts, result)
            treatments = [
                fact for fact in incident_facts(self.graph, keep, [disease], [self.treats]) if fact[2] == disease
            ]
            result["deletions"]["disease's other treatments"] = self.delete(
                drug_name, disease_name, fold, seeds, treatments, result
            )
        result["seconds"] = time.perf_counter() - start
        result["peak_gpu_gb"] = peak_gpu_gb(self.device)
        errors = [summary["consensus_max_abs_error"] for summary in result["per_seed"]]
        top = result["top_stretches"][0]["stretch"] if result["top_stretches"] else "none"
        print(
            f"{result['drug_name']} -> {result['disease_name']} (fold {fold}): RR {result['reciprocal_rank']:.3f}, "
            f"categories { ({k: round(v, 2) for k, v in result['mean_over_seeds']['share_by_category'].items()}) }, "
            f"RR change: top stretches {result['deletions']['top stretches']['reciprocal_rank_change']:+.3f}, "
            f"random {result['deletions']['random facts at drug or disease']['reciprocal_rank_change']:+.3f} "
            f"({len(guided)} facts) | consensus error {max(errors):.1e} | {result['seconds']:.0f} s | top: {top}",
            flush=True,
        )
        return result


def survey_edges(split, per_fold, seed):
    """A fixed random sample of held-out edges, stratified by fold."""
    rng = np.random.default_rng(seed)
    chosen = []
    for fold in np.unique(split.folds).tolist():
        rows = np.flatnonzero(split.folds == fold)
        chosen += sorted(rng.choice(rows, size=min(per_fold, len(rows)), replace=False).tolist())
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="0,1,2", help="Walk seeds for the showcase pairs")
    parser.add_argument("--survey-per-fold", type=int, default=SURVEY_PER_FOLD)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deadline", type=float, help="Unix time after which no new case starts")
    parser.add_argument("--edges", type=Path, default=EDGES_PATH)
    parser.add_argument("--nodes", type=Path, default=NODES_PATH)
    parser.add_argument("--split", type=Path, default=SPLIT_PATH)
    parser.add_argument("--walk-num", type=int, help="Override n, for smoke tests only")
    parser.add_argument("--ensemble", type=int, help="Override P, for smoke tests only")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    explainer = Explainer(args, device)
    seeds = [int(seed) for seed in args.seeds.split(",")]
    results = {"metadata": explainer.metadata, "showcase": [], "survey": []}
    path = args.output_dir / "e4_walk_explanations.json"

    plan = [("showcase", drug, disease, seeds, True) for drug, disease in SHOWCASE]
    for row in survey_edges(explainer.split, args.survey_per_fold, SURVEY_SEED):
        plan.append(("survey", explainer.split.drugs[row], explainer.split.diseases[row], seeds[:1], False))
    for part, drug, disease, case_seeds, families in plan:
        if args.deadline is not None and time.time() > args.deadline:
            print(f"time budget reached before {drug} -> {disease}", flush=True)
            break
        results[part].append(explainer.case(drug, disease, case_seeds, families))
        path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
