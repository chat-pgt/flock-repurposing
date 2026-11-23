# Analyses of the published paper

CPU-only scripts. Run `uv sync` in `experiments/`; the PDF scripts also need `pdftotext`.

| Script | What it does | Output |
|---|---|---|
| `parse_tables.py` | Extracts per-KG results from Tables 9–12 | `paper_tables.json` |
| `significance.py` | Paired tests of FLOCK vs TRIX and ULTRA | `../results/a1_paper_significance.json` |
| `dataset_summary.py` | Summarises the 57 KGs by source | stdout |
| `walk_coverage.py` | Simulates FLOCK's sampler on Hetionet (7–15 min) | `../results/a2_walk_coverage.txt` |
| `edge_lookup_cost.py` | Estimates the edge-type lookup cost per walk step | stdout |
| `consistency_checks.py` | Checks reported numbers against other tables, the released code and ULTRA's seed variance | `../results/a4_paper_consistency.json` |

```bash
bash ../common/get_hetionet.sh
uv run python parse_tables.py path/to/flock.pdf paper_tables.json
uv run python significance.py paper_tables.json --output ../results/a1_paper_significance.json
uv run python dataset_summary.py path/to/flock.pdf
uv run python walk_coverage.py ../data/hetionet/hetionet-v1.0-edges.sif.gz
uv run python edge_lookup_cost.py ../data/hetionet/hetionet-v1.0-edges.sif.gz
uv run python consistency_checks.py path/to/flock.pdf path/to/ULTRA.pdf paper_tables.json \
  --output ../results/a4_paper_consistency.json
```
