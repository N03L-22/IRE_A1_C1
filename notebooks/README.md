# Exploratory notebooks

One-off data exploration, run **2026-08-21** on the downloaded bundles. They are the
source of the dataset facts in `architecture.md` Part E (row counts, schema
asymmetries, null rates, the submission format).

**These are not part of the pipeline.** Nothing in `src/` imports them and no
reported metric comes from them — `make data && make store` rebuilds everything
they inspected. They use `polars`, which is why it is pinned in
`requirements.txt` while `src/` uses `pyarrow` directly.

| Notebook | Covers |
|---|---|
| `mind_analysis.ipynb` | MIND: TSV schema, 51K/42K/121K article splits, entity JSON, the unlabelled large test set |
| `ebnerd_analysis.ipynb` | EB-NeRD: parquet schema, 12M train impressions, the four columns removed from test, `is_beyond_accuracy` |

They also contain the first popularity-baseline submissions, which established the
Codabench format before `src/submit/` existed.
