# E6: no-learning baselines for relation prediction

FLOCK reports a zero-shot relation-prediction MRR of 0.940 on Hetionet. This experiment measures how much
of that score simple counts can reach.

## Task

Each query gives two entities that are known to be linked, (h, ?, t), and asks which of Hetionet's 24
relations connects them. The evaluation matches the released code: one query per triple, unfiltered
ranking over forward relations, on the authors' 112,510 test triples. It covers all relation types. Only
56 of those triples are Compound–Disease, and the task does not test whether a link exists at all.

## Baselines

- **Baseline A (uniform)** picks at random among the relations seen in training between the entity types
  of h and t.
- **Baseline B (frequency)** ranks those same relations by how often each one occurs in training. It
  ignores which specific entities are in the query, so every Compound–Gene pair gets the same ranking.
- **Baseline C (graph signature)** uses no entity-type labels, which the models do not receive either.
  Relations that h has used as a head and t has used as a tail in training come first, and each group
  is ordered by training frequency.

## Examples

These are real test triples.

| Query | True relation | Candidates (training count) | B's rank | C's rank | Why |
|---|---|---|---|---|---|
| BRCA1 → double-strand break repair | *participates* | *participates* only | 1 | 1 | Easy: the entity types leave one option |
| Metformin → PSME1 | *downregulates* | *downregulates* 18,975; *upregulates* 16,936; *binds* 10,410 | 1 | 1 | Harder: correct because it is the most common relation |
| Metformin → COPB2 | *upregulates* | Same as above | 2 | 2 | Harder: both endpoints have used *up* and *down*, so C also falls back on frequency |
| Diclofenac → osteoarthritis | *palliates* | *treats* 682; *palliates* 355 | 2 | 1 | Harder: *palliates* is the only relation both endpoints have used in training |

## Reproduce

From `experiments/`:

```bash
bash e6_relation_baseline/get_authors_split.sh
uv run python -m e6_relation_baseline.relation_baselines
```

The output is `results/e6_relation_baselines.json`.
