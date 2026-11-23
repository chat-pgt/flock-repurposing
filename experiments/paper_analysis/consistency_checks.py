"""Check the paper's reported numbers against each other, the released code and ULTRA's tables.

Usage:
    python consistency_checks.py path/to/flock.pdf path/to/ULTRA.pdf paper_tables.json \
        [--output ../results/a4_paper_consistency.json]

Fig. 4a has no table, so its values are the ones read off the plot. Everything else is parsed from the
PDFs, `paper_tables.json` or the released code.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIG_4A_THREE_GRAPHS = 0.396
FIG_4A_ONE_GRAPH = 0.390
FIG_4A_EIGHT_GRAPHS = 0.415
NUMBER = r"\d\.\d{3}"


def pdf_text(path):
    return subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True, check=True).stdout


def row_numbers(text, label):
    match = re.search(rf"^\s*{label}\s+(.*)$", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"row not found: {label}")
    return [float(value) for value in re.findall(NUMBER, match.group(1))]


def table_8a_baseline(flock):
    """Compare Table 8a's TRIX baseline with Table 2 and the per-KG means of Table 9."""
    table_8 = flock[flock.index("Table 8: Noise injection") :]
    reported = row_numbers(table_8, r"T RIX")[:2]
    with_noise = row_numbers(table_8, r"\+ noise")[:2]
    table_2_ultra = row_numbers(flock, r"U LTRA \(zero-shot\)")[6:8]
    table_2_trix = row_numbers(flock, r"T RIX \(zero-shot\)")[6:8]
    return {
        "table_8a_trix": reported,
        "table_8a_trix_with_noise": with_noise,
        "table_2_ultra": table_2_ultra,
        "table_2_trix": table_2_trix,
        "table_8a_equals_table_2_ultra": reported == table_2_ultra,
        "noise_effect_if_table_2_is_right": round(with_noise[0] - table_2_trix[0], 3),
    }


def per_kg_means(tables):
    rows = tables["9"]
    return {
        "ultra_mrr": round(float(np.mean([row["U"] for row in rows])), 3),
        "trix_mrr": round(float(np.mean([row["T"] for row in rows])), 3),
        "ultra_hits10": round(float(np.mean([row["Uh"] for row in rows])), 3),
        "trix_hits10": round(float(np.mean([row["Th"] for row in rows])), 3),
    }


def figure_4a(tables):
    """Compare Fig. 4a's 3-graph point with the same model's inductive results in Table 9."""
    rows = tables["9"]
    inductive = [row["F"] for row in rows if row["split"] in ("Inductive e, r", "Inductive e")]
    expected = float(np.mean(inductive))
    gap = expected - FIG_4A_THREE_GRAPHS
    total_gain = FIG_4A_EIGHT_GRAPHS - FIG_4A_ONE_GRAPH
    return {
        "n_kgs": len(inductive),
        "expected_from_table_9": round(expected, 3),
        "read_from_fig_4a": FIG_4A_THREE_GRAPHS,
        "gap": round(gap, 3),
        "gap_as_share_of_scaling_gain": round(gap / total_gain, 2),
    }


def config_value(path, key):
    match = re.search(rf"^\s*{key}:\s*(\S+)", (REPO / path).read_text(), flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"{key} not found in {path}")
    return int(match.group(1))


def paper_vs_code(flock):
    finetune = "src_entity/config/finetune_transductive/n1024_ensemble1.yaml"
    relation_run = (REPO / "src_relation/run.py").read_text()
    return {
        "training_walks": {
            "appendix_e_says_n_16": "fix F LOCK to n = 16 base walks" in flock,
            "released_pretraining_config_n": config_value("src_entity/config/pretrain_3g.yaml", "walk_num"),
        },
        "hetionet_finetuning_ensemble": {
            "table_18_says_p_1_n_1024": "(P, n) = (1, 1024)" in flock,
            "released_config_p": config_value(finetune, "test_samples"),
            "released_config_n": config_value(finetune, "walk_num"),
        },
        "relation_ranking": {
            "paper_says_filtered": "filtered ranking protocol" in flock,
            "released_code_is_unfiltered": "compute_ranking_relation(r_pred, pos_r_index, None)" in relation_run,
        },
    }


def ultra_finetuning_noise(ultra):
    """Summarise the 5-seed standard deviations in ULTRA's Tables 10 and 11."""
    section = ultra[ultra.index("Table 10: Full results") :]
    rows = [re.findall(r"\d(?:\.\d+)?\s*±\s*(\d(?:\.\d+)?)", line) for line in section.splitlines() if "±" in line]
    summary = {"n_kgs": len(rows)}
    for name, column in (("mrr", 0), ("hits10", 1)):
        deviations = [float(row[column]) for row in rows]
        summary[f"{name}_median_std"] = round(float(np.median(deviations)), 3)
        summary[f"{name}_p90_std"] = round(float(np.percentile(deviations, 90)), 3)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("flock_pdf")
    parser.add_argument("ultra_pdf")
    parser.add_argument("tables")
    parser.add_argument("--output")
    args = parser.parse_args()

    flock = pdf_text(args.flock_pdf)
    ultra = pdf_text(args.ultra_pdf)
    tables = json.loads(Path(args.tables).read_text())
    results = {
        "table_8a_baseline": table_8a_baseline(flock),
        "table_9_per_kg_means": per_kg_means(tables),
        "figure_4a_vs_table_9": figure_4a(tables),
        "paper_vs_released_code": paper_vs_code(flock),
        "ultra_finetuning_noise": ultra_finetuning_noise(ultra),
    }
    print(json.dumps(results, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
