"""Extract the per-dataset results (Tables 9-12) from the FLOCK paper PDF into JSON.

Usage: python parse_tables.py path/to/flock.pdf paper_tables.json
Requires `pdftotext` (poppler-utils).

Each row holds MRR and a Hits metric for ULTRA (U, Uh), TRIX (T, Th) and FLOCK (F, Fh).
Hits is Hits@10 in Tables 9-10 (entity) and Hits@1 in Tables 11-12 (relation).
"""
import argparse
import json
import re
import subprocess
from collections import Counter

ROW = re.compile(
    r"^\s*(\S.*?)\s{2,}(\d\.\d{3})\s+(\d\.\d{3})\s+(\d\.\d{3})\s+(\d\.\d{3})\s+(\d\.\d{3})\s+(\d\.\d{3})\s*$"
)
SPLIT = re.compile(r"^(Inductive|Transductive|Pretrain)")


def table_start(lines, number):
    pattern = re.compile(rf"\s*Table {number}: ")
    return next(i for i, line in enumerate(lines) if pattern.match(line))


def parse(pdf):
    text = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True, check=True).stdout
    lines = text.split("\n")
    tables = {}
    for number in (9, 10, 11, 12):
        split, rows = None, []
        for line in lines[table_start(lines, number) : table_start(lines, number + 1)]:
            stripped = line.strip()
            if SPLIT.match(stripped) and len(stripped) < 40:
                split = stripped
                continue
            match = ROW.match(line)
            if match:
                name, *values = match.groups()
                keys = ("U", "Uh", "T", "Th", "F", "Fh")
                rows.append({"ds": name, "split": split, **dict(zip(keys, map(float, values)))})
        tables[number] = rows
        print(number, len(rows), dict(Counter(row["split"] for row in rows)))
    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf")
    parser.add_argument("output")
    args = parser.parse_args()
    with open(args.output, "w") as handle:
        json.dump(parse(args.pdf), handle, indent=0)
