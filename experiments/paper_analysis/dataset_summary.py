"""Summarise the 57 benchmark KGs by source, from Tables 13-15 of the FLOCK paper.

Usage: python dataset_summary.py path/to/flock.pdf
Requires `pdftotext`. Sizes use the inference graph for inductive KGs and the training graph for
transductive ones. Each group's ranges include any pretraining graph from that source.
"""

import re
import statistics
import subprocess
import sys

NUM = r"(\d+)"
TABLE_13 = re.compile(r"^\s+(\S.*?)\s{2,}" + r"\s+".join([NUM] * 11) + r"\s*$")
TABLE_14 = re.compile(r"^\s+(\S.*?)\s{2,}" + r"\s+".join([NUM] * 9) + r"\s*$")
TABLE_15 = re.compile(r"^\s+(\S.*?)\s{2,}" + r"\s+".join([NUM] * 5) + r"\s+(h/t|t)\s*$")


def source(name):
    if name.startswith(("FB15k-237", "FB-")):
        return "Freebase"
    if name == "FBNELL":
        return "FBNELL"
    if name.startswith(("WK", "CoDEx", "WDsinger", "ILPC", "Wiki")):
        return "Wikidata"
    if name.startswith(("NL", "NELL")):
        return "NELL"
    if name.startswith("WN"):
        return "WordNet"
    if name.startswith("HM"):
        return "HM"
    return name


def load(pdf):
    lines = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True, check=True).stdout.split(
        "\n"
    )
    start = {n: next(i for i, l in enumerate(lines) if re.match(rf"\s*Table {n}: ", l)) for n in (13, 14, 15, 16)}
    rows = []  # (name, nodes, relations, edges)
    for line in lines[start[13] : start[14]]:
        if match := TABLE_13.match(line):
            group = match.groups()
            rows.append((group[0], int(group[8]), int(group[9]), int(group[10])))
    for line in lines[start[14] : start[15]]:
        if match := TABLE_14.match(line):
            group = match.groups()
            rows.append((group[0], int(group[7]), int(group[1]), int(group[8])))
    for line in lines[start[15] : start[16]]:
        if match := TABLE_15.match(line):
            group = match.groups()
            rows.append((group[0], int(group[1]), int(group[2]), int(group[3])))
    return rows


if __name__ == "__main__":
    data = load(sys.argv[1])
    print(len(data), "KGs")
    groups = {}
    for row in data:
        groups.setdefault(source(row[0]), []).append(row)
    print("source | KGs | nodes | relations | edges | median edges/node")
    for key, rows in groups.items():
        nodes, relations, edges = zip(*[row[1:] for row in rows])
        density = statistics.median(row[3] / row[1] for row in rows)
        print(
            f"{key} | {len(rows)} | {min(nodes)}-{max(nodes)} | {min(relations)}-{max(relations)} | {min(edges)}-{max(edges)} | {density:.1f}"
        )
