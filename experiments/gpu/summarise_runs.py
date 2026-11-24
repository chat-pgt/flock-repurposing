"""Print a quick summary of saved GPU runs. The final analysis uses the shared bootstrap instead.

Usage, from experiments/: python -m gpu.summarise_runs --runs runs
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarise_entity(paths):
    groups = defaultdict(list)
    for path in paths:
        groups[path.stem.rsplit("_fold", 1)[0]].append(path)
    for name, shards in sorted(groups.items()):
        loaded = [np.load(path) for path in shards]
        info = [json.loads(str(shard["metadata"])) for shard in loaded]
        rr = np.concatenate([shard["reciprocal_rank"] for shard in loaded])
        hits = np.concatenate([shard["hits10"] for shard in loaded])
        n_queries = sum(len(shard["query_entities"]) for shard in loaded)
        seconds = sum(row["seconds"] for row in info)
        peak = max((row["peak_gpu_gb"] or 0) for row in info)
        print(
            f"{name}: folds {len(shards)}/5, edges {len(rr)}, MRR {rr.mean():.3f}, Hits@10 {hits.mean():.3f}, "
            f"{seconds / n_queries:.2f} s/query, peak GPU {peak:.1f} GB"
        )


def summarise_relation(paths):
    for path in sorted(paths):
        shard = np.load(path)
        print(
            f"{path.stem}: triples {len(shard['heads'])}, 24-way MRR {shard['all_relations_rr'].mean():.3f}, "
            f"treats vs palliates Hits@1 {shard['treats_vs_palliates_hits1'].mean():.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args()
    summarise_relation(args.runs.glob("relation_*.npz"))
    summarise_entity(list(args.runs.glob("entity_*.npz")))


if __name__ == "__main__":
    main()
