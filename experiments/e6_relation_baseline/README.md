# E6: no-learning baselines for relation prediction

FLOCK reports a zero-shot relation-prediction MRR of 0.940 on Hetionet. This experiment measures how much
of that score simple counts can reach.

## Task

Each query gives two entities that are known to be linked, (h, ?, t), and asks which of Hetionet's 24
relations connects them. The evaluation matches the released code: one query per triple, unfiltered
ranking over forward relations, on the authors' 112,510 test triples. It covers all relation types. Only
56 of those triples are Compound–Disease, and the task does not test whether a link exists at all.

## Baselines

Both baselines look only at relations seen in training between the entity types of h and t.

- **Baseline A (uniform)** picks among those relations at random.
- **Baseline B (frequency)** ranks them by how often each one occurs in training. It ignores which
  specific entities are in the query, so every Compound–Gene pair gets the same ranking.

## Examples

These are real test triples.

| Query | True relation | Candidates (training count) | Baseline B | Difficulty |
|---|---|---|---|---|
| BRCA1 → double-strand break repair | *participates* | *participates* only | Rank 1 | Easy: the entity types leave one option |
| Metformin → PSME1 | *downregulates* | *downregulates* 18,975; *upregulates* 16,936; *binds* 10,410 | Rank 1 | Harder: correct only because it is the most common |
| Metformin → COPB2 | *upregulates* | Same as above | Rank 2 | Harder: needs information about this pair |
| Diclofenac → osteoarthritis | *palliates* | *treats* 682; *palliates* 355 | Rank 2 | Harder: needs information about this pair |

## Reproduce

From `experiments/`:

```bash
bash e6_relation_baseline/get_authors_split.sh
uv run python -m e6_relation_baseline.relation_baselines
```

The output is `results/e6_relation_baselines.json`.
