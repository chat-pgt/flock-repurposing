"""Paired significance tests of FLOCK vs TRIX and ULTRA on the paper's per-dataset tables.

Usage: python significance.py paper_tables.json [--output ../results/a1_paper_significance.json]

The unit of analysis is the knowledge graph. Two tests are reported:
- Wilcoxon signed-rank across KGs (Demsar 2006), Holm-corrected over the 8 headline comparisons.
- An exact sign-flip test on source-family means, because KGs derived from the same source
  (e.g. FB-25 and FB-50) are not independent. With 12 families the smallest possible p is 0.0005.

These tests use single-run published numbers. They cannot capture pretraining-seed variance.
"""
import argparse
import itertools
import json

import numpy as np
from scipy import stats

TASKS = {"9": "Entity zero-shot", "10": "Entity fine-tuned", "11": "Relation zero-shot", "12": "Relation fine-tuned"}
BASELINES = {"T": "TRIX", "U": "ULTRA"}
SPLITS = ["All 54", "Inductive e, r", "Inductive e", "Transductive"]


def family(name):
    if name == "FBNELL":
        return "FBNELL"
    if name.startswith("FB"):
        return "Freebase"
    if name.startswith(("WK", "CoDEx", "WDsinger", "ILPC", "MT", "Wiki")):
        return "Wikidata"
    if name.startswith(("NL", "NELL")):
        return "NELL"
    if name.startswith("WN"):
        return "WordNet"
    if name.startswith("HM"):
        return "HM"
    return name


def holm(p_values):
    p_values = np.asarray(p_values)
    adjusted = np.empty(len(p_values))
    running_max = 0.0
    for rank, index in enumerate(np.argsort(p_values)):
        running_max = max(running_max, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def family_signflip(diff, families):
    families = np.asarray(families)
    means = np.array([diff[families == key].mean() for key in sorted(set(families))])
    signs = np.array(list(itertools.product([1, -1], repeat=len(means))))
    return float((np.abs((signs * means).mean(1)) >= abs(means.mean()) - 1e-12).mean()), len(means)


def bootstrap_ci(diff, rng, draws=20000):
    index = rng.integers(0, len(diff), (draws, len(diff)))
    return np.percentile(diff[index].mean(1), [2.5, 97.5]).tolist()


def wilcoxon(diff):
    nonzero = diff[diff != 0]
    return float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0


def compare(rows, base, rng, flock_key="F", base_key=None):
    base_key = base_key or base
    flock = np.array([row[flock_key] for row in rows])
    other = np.array([row[base_key] for row in rows])
    diff = np.round(flock - other, 3)
    family_p, n_families = family_signflip(diff, [family(row["ds"]) for row in rows])
    low, high = bootstrap_ci(diff, rng)
    return {
        "n": len(rows),
        "flock_mean": float(flock.mean()),
        "baseline_mean": float(other.mean()),
        "mean_diff": float(diff.mean()),
        "wins": int((diff > 0).sum()),
        "losses": int((diff < 0).sum()),
        "ties": int((diff == 0).sum()),
        "ci95": [low, high],
        "wilcoxon_p": wilcoxon(diff),
        "family_p": family_p,
        "n_families": n_families,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tables")
    parser.add_argument("--output")
    args = parser.parse_args()
    tables = json.load(open(args.tables))
    rng = np.random.default_rng(0)

    results = []
    for table, task in TASKS.items():
        rows = [row for row in tables[table] if row["split"] != "Pretrained"]
        for base, base_name in BASELINES.items():
            for split in SPLITS:
                subset = rows if split == "All 54" else [row for row in rows if row["split"] == split]
                results.append({"task": task, "baseline": base_name, "split": split, **compare(subset, base, rng)})

    headline = [row for row in results if row["split"] == "All 54"]
    for key in ("wilcoxon_p", "family_p"):
        for row, adjusted in zip(headline, holm([row[key] for row in headline])):
            row[key + "_holm"] = float(adjusted)

    print("task | baseline | dMRR | W/L/T | 95% CI | Wilcoxon (Holm) | family (Holm)")
    for row in headline:
        print(
            f"{row['task']} | {row['baseline']} | {row['mean_diff']:+.4f} | "
            f"{row['wins']}/{row['losses']}/{row['ties']} | [{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}] | "
            f"{row['wilcoxon_p']:.5f} ({row['wilcoxon_p_holm']:.5f}) | {row['family_p']:.4f} ({row['family_p_holm']:.4f})"
        )

    entity = tables["9"]
    gaps = {row["ds"]: round(row["F"] - row["T"], 3) for row in entity}
    robustness = {
        "entity_zero_shot_vs_trix_without_metafam": float(np.mean([v for k, v in gaps.items() if k != "Metafam"])),
        "entity_zero_shot_vs_trix_without_metafam_aristo": float(
            np.mean([v for k, v in gaps.items() if k not in ("Metafam", "AristoV4")])
        ),
        "entity_zero_shot_hits10_vs_trix": compare(entity, "T", rng, flock_key="Fh", base_key="Th"),
    }
    zero_shot = {row["ds"]: row for row in entity}
    fine_tuned = {row["ds"]: row for row in tables["10"] if row["split"] != "Pretrained"}
    robustness["fine_tuning_effect_entity"] = {}
    for key, name in (("U", "ULTRA"), ("T", "TRIX"), ("F", "FLOCK")):
        delta = np.array([round(fine_tuned[k][key] - zero_shot[k][key], 3) for k in zero_shot])
        robustness["fine_tuning_effect_entity"][name] = {
            "mean": float(delta.mean()),
            "worse": int((delta < 0).sum()),
            "unchanged": int((delta == 0).sum()),
        }
    print(json.dumps(robustness, indent=2))

    if args.output:
        with open(args.output, "w") as handle:
            json.dump({"comparisons": results, "robustness": robustness}, handle, indent=2)


if __name__ == "__main__":
    main()
