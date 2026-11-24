"""E4: summarise the walk explanations and deletion tests. NumPy only.

The survey gives base rates over 40 held-out edges: where the disease's weight comes from, and whether
deleting the top stretches' facts lowers the disease's reciprocal rank more than deleting as many random
facts. That comparison is paired per edge, with a two-way bootstrap CI over drugs and diseases.

Usage, from experiments/:
    uv run python -m e4_explanations.analyze_explanations --raw path/to/e4/e4_walk_explanations.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common.hetionet import EXPERIMENTS
from gpu.analysis import interval

GUIDED, RANDOM = "top stretches", "random facts at drug or disease"
SUCCESS_RR = 1 / 3  # the disease ranked in the top 3


def mean_dict(dicts):
    totals = defaultdict(float)
    for shares in dicts:
        for key, share in shares.items():
            totals[key] += share / len(dicts)
    return dict(sorted(totals.items(), key=lambda item: -item[1]))


def survey_summary(cases):
    drugs = np.array([case["drug"] for case in cases])
    diseases = np.array([case["disease"] for case in cases])
    rr = np.array([case["reciprocal_rank"] for case in cases])
    guided = np.array([case["deletions"][GUIDED]["reciprocal_rank_change"] for case in cases])
    random = np.array([case["deletions"][RANDOM]["reciprocal_rank_change"] for case in cases])
    guided_score = np.array([case["deletions"][GUIDED]["score_change"] for case in cases])
    random_score = np.array([case["deletions"][RANDOM]["score_change"] for case in cases])
    success = rr >= SUCCESS_RR
    summary = {
        "n_edges": len(cases),
        "mrr": interval(rr, drugs, diseases),
        "share_by_category": mean_dict([case["mean_over_seeds"]["share_by_category"] for case in cases]),
        "share_by_drug_distance": mean_dict([case["mean_over_seeds"]["share_by_drug_distance"] for case in cases]),
        "share_by_walk_start": mean_dict([case["mean_over_seeds"]["share_by_walk_start"] for case in cases]),
        "deletion": {
            "facts_deleted_mean": float(np.mean([case["deletions"][GUIDED]["facts_deleted"] for case in cases])),
            "rr_change_top_stretches": interval(guided, drugs, diseases),
            "rr_change_random": interval(random, drugs, diseases),
            "rr_change_top_minus_random": interval(guided - random, drugs, diseases),
            "score_change_top_minus_random": interval(guided_score - random_score, drugs, diseases),
            "edges_where_top_lowers_rr_more": int((guided < random).sum()),
            "edges_where_random_lowers_rr_more": int((random < guided).sum()),
        },
    }
    if success.sum() >= 2:
        summary["top3_cases"] = {
            "n_edges": int(success.sum()),
            "share_by_category": mean_dict(
                [case["mean_over_seeds"]["share_by_category"] for case, ok in zip(cases, success, strict=True) if ok]
            ),
            "rr_change_top_minus_random": interval(guided - random, drugs, diseases, success),
        }
    return summary


def showcase_rows(cases):
    return [
        {
            "pair": f"{case['drug_name']} -> {case['disease_name']}",
            "fold": case["fold"],
            "reciprocal_rank": case["reciprocal_rank"],
            "share_by_category": case["mean_over_seeds"]["share_by_category"],
            "top_stretches": case["top_stretches"][:5],
            "deletions": {
                name: {key: row[key] for key in ("facts_deleted", "reciprocal_rank", "reciprocal_rank_change")}
                for name, row in case["deletions"].items()
            },
            "consensus_max_abs_error": max(seed["consensus_max_abs_error"] for seed in case["per_seed"]),
        }
        for case in cases
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=EXPERIMENTS / "results/e4_walk_explanations.json")
    args = parser.parse_args()
    raw = json.loads(args.raw.read_text())
    results = {"metadata": raw["metadata"], "showcase": showcase_rows(raw["showcase"])}
    if raw["survey"]:
        results["survey"] = survey_summary(raw["survey"])
    args.output.write_text(json.dumps(results, indent=2) + "\n")

    for row in results["showcase"]:
        deletions = ", ".join(f"{name} {d['reciprocal_rank_change']:+.2f}" for name, d in row["deletions"].items())
        print(f"{row['pair']}: RR {row['reciprocal_rank']:.2f} | RR change: {deletions}")
    if "survey" in results:
        survey = results["survey"]
        difference = survey["deletion"]["rr_change_top_minus_random"]
        print(
            f"survey ({survey['n_edges']} edges): MRR {survey['mrr']['mean']:.3f}, "
            f"categories { ({k: round(v, 3) for k, v in survey['share_by_category'].items()}) }, "
            f"top-stretch minus random deletion {difference['mean']:+.3f} "
            f"[{difference['ci95'][0]:+.3f}, {difference['ci95'][1]:+.3f}]"
        )


if __name__ == "__main__":
    main()
