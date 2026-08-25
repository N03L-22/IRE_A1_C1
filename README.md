# IRE Assignment 1 — Lexical & Semantic Retrieval (Component 1)

**CS4.406 Information Retrieval and Extraction · IIIT Hyderabad · Monsoon 2026**
Noel Alex Jacob · 2025201085

News-recommendation candidate generation on **MIND** (English) and **EB-NeRD** (Danish): a
reproducible data pipeline, lexical retrieval, and an offline evaluation harness with bootstrap
confidence intervals.

> **Status.** Q1, Q2, Q4 and Q9 are built and measured. **Q3 (semantic retrieval) is not yet
> built.** Prediction files are generated but **not yet submitted**, so no leaderboard score exists.
> Nothing in this repo is estimated — anything unmeasured is absent, not guessed.

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

make data && make store    # raw archives -> unified, temporally-split parquet store
make bm25                  # BM25 vs baselines, both datasets
make eval                  # full harness: metrics, slices, bootstrap CIs
make test                  # 72 tests, incl. the leakage mutation test
```

Everything downstream reads `data/store/`, so `make store` is the only step that touches raw files.
A full rebuild from the archives takes **~42 s**.

### Resource budget

Every stage takes `--n-jobs` and `--mem-gb`, capped at **28 cores / 28 GB all-inclusive**, and the
*resolved* values are written into every output file so a number always traces back to the budget
that produced it. Override per invocation:

```bash
make eval N_JOBS=8 MEM_GB=8
```

The budget is clamped to what is actually free at startup rather than to what was requested — a run
that would only swap refuses to start instead of dying three hours in.

---

## What this actually does

A user arrives at a news site. You are shown who they are, when they arrived, and a slate of
articles that could be displayed. Guess which they will click.

**The move that makes it a retrieval problem:** we do not know what the user wants, but we know what
they read recently. Paste their recent headlines together and treat that as a *search query*. "What
will they click?" becomes "which articles match this query?"

**This component is candidate generation, not ranking.** The question is only *did the clicked
article survive the cut* — `recall@K` for K ∈ {50, 100, 200}. Position 180 of 200 is a success;
being absent is fatal, because no downstream re-ranker can recover it. Ranking is Component 2.

---

## Layout

```
src/
  data/         extract, per-dataset readers, unified schema, temporal split
  retrieval/    Retriever protocol, tokeniser, BM25 (+ reference implementation)
  eval/         metrics (two regimes), bootstrap, slices, harness, runner
  submit/       Codabench prediction files
tests/          72 tests, incl. leakage mutation tests
plan/           phase plans + execution log (the F-numbered findings)
results/        every measured run, as JSON with CIs and resolved budget
report/         a1_report.tex — the Q6 design note source
```

### Documents

| File | Holds |
|---|---|
| [`foundations.md`](foundations.md) | Every concept from scratch — vectors, BM25, embeddings, and the evaluation metrics built up need → why → how |
| [`architecture.md`](architecture.md) | What the system **is**, plus the measured dataset facts |
| [`decisions.md`](decisions.md) | What we **chose and rejected**, open questions, and the cost of each choice — the design-note source |
| [`plan/execution_plan_log.md`](plan/execution_plan_log.md) | Findings F1–F33, dated. Also the architecture changelog |

---

## The findings that shaped the design

Cited as `Fnn` throughout the code and docs; full detail in the execution log.

**Recency dominates everything (F16, F21).** 94.3% of EB-NeRD clicks are on articles under 24 hours
old, while only ~1.1% of the corpus is that fresh. A retriever that *ignores the user entirely* and
returns the newest articles gets `recall@50 = 0.9050 [0.8850, 0.9237]`. Candidate generation here is
primarily a **recency-filtering** problem and only secondarily a text-matching one.

**BM25 is indistinguishable from popularity (F21, F25).** On EB-NeRD their CIs overlap. On MIND,
popularity *beats* BM25 with non-overlapping CIs at every K — and MIND has no publish times, so the
24-hour window that rescues BM25 on EB-NeRD is unavailable there.

**Tuning BM25 is nearly pointless compared with choosing the pool (F23).** Across a 54-cell sweep,
the recency window is worth **33×**; `k1` and `b` are worth ~0.01 — inside the CI.

**Abstracts are 75% of the index and buy nothing detectable (F28).** Q2.1 mandates title + abstract.
Measured: +0.011 recall against a CI half-width of ±0.030. Kept because the requirement is binding,
reported as the null result it is.

**Random ranking scores nDCG@10 = 0.42 (F25).** Slates average 11–12 items with exactly one click,
so even a random permutation looks respectable. **Any slate-regime table without a random baseline
row is misleading.**

**Coverage has no defensible bootstrap CI (F24).** It counts *distinct* articles, so it grows with
sample size; resampling with replacement makes ~37% duplicates and biases every resample low. The
first attempt put the point estimate **outside its own interval** (0.9783 vs [0.9035, 0.9235]).
Subsampling fails at every ratio tried. Coverage is reported as a point estimate marked `(no CI)` —
saying so beats shipping a plausible-looking number that is biased by construction.

**A 7-day test split is impossible on MIND (F27).** Its entire labelled range is 6.0 days. The split
*rule* is held constant; the realised spans differ (EB-NeRD 7 days, MIND 1 day).

**The submission path needed a different algorithm, not tuning (F32).** Full-corpus retrieval per
impression ran at 162 impressions/s (~4 h per file) because it discarded 99% of each result. Scoring
the slate directly runs at ~2,540/s — **16×**.

---

## The dataset asymmetry, and the rule it forces

Every richer feature EB-NeRD offers is one MIND lacks:

| | MIND | EB-NeRD |
|---|---|---|
| Body text, click timestamps, publish time, session ids | ❌ | ✅ |
| Provided embeddings | ❌ | ✅ 300-d w2v, 768-d mBERT |
| Entity embeddings | ✅ TransE 100-d | ❌ |

Using any one-sided feature for a headline number silently converts a **dataset** comparison into a
**feature-availability** comparison. So:

> The shared retrieval path uses only the intersection — title, abstract, category, click history.
> Everything one-sided becomes a clearly-labelled single-dataset ablation.

That one rule resolves four otherwise-independent decisions and is the most load-bearing choice in
the component.

---

## Correctness

**The leakage boundary (Q9).** For a test impression at time *t*, that user's history is truncated
to clicks strictly before *t*. `tests/test_no_leakage.py` asserts this on the built store **and
injects a deliberate post-boundary click to confirm the checker fails** — a test that passes both
before and after the boundary breaks proves nothing.

*Stated limitation:* the invariant is verifiable only where per-click timestamps exist, i.e.
**EB-NeRD**. MIND's history is untimestamped, so there the boundary rests on the authors'
construction. The store records this per row (`history_verifiable`) and in the manifest
(`history_boundary_verifiable: false`). Claiming the test covers MIND would be worse than the gap.

**Serving-unavailable features (Q9).** EB-NeRD's training rows carry `read_time`,
`scroll_percentage`, `next_read_time`, `next_scroll_percentage` — all recorded *after* the click.
Predicting a click from its own consequences is circular. The organisers settled the boundary for
us: those four columns are exactly what is deleted from the test set. They are excluded in
`clean.py` with a comment naming why.

**Two evaluation regimes, kept apart.** `recall@K` is computed over the **whole corpus** ("did the
click survive the cut?"); AUC/MRR/nDCG over the impression's **own slate** ("did you rank it above
the others shown?"). Conflating them produces numbers that look fine and mean nothing. Every result
row carries a `regime` column.

---

## Reproducing the numbers

```bash
make bm25-sweep    # 54-cell parameter sweep, val only, parallel
make eval          # the CI-bearing results in results/eval_*.json
python -m src.submit.codabench --dataset mind --tier large     # Q5
```

Artefacts: `results/eval_*.json` (every row with CI, *n*, regime, slice basis),
`results/bm25_sweep_*.json`, `data/store/*/manifest.json` (split spans, counts, resolved budget),
`submissions/*.meta.json`.

Parameters are chosen on **val** and reported on test — the discipline that keeps the test split
honest. Chosen before test was touched: EB-NeRD `k1=1.6, b=1.0, n=15`; MIND `k1=1.6, b=0.75, n=5`.

---

## Not in this repo

- `data/` — 30+ GB of archives and derived store, gitignored per Q8. `make data` rebuilds it.
- `submissions/*.txt`, `*.zip` — hundreds of MB, regenerable in ~15 min. The `.meta.json` beside
  each **is** tracked: it records the retriever, params, line count and budget, which is what makes
  a leaderboard score traceable.

## Known gaps

1. **Q3 (semantic retrieval) is not built** — so no lexical-vs-semantic comparison, the substance of
   Q3.5, exists yet.
2. **No leaderboard scores.** Prediction files are generated and format-validated; not submitted.
3. **n = 800 per dataset** for CI-bearing results — several reported differences are not
   statistically separable, and the report says so at each point.
4. **HNSW figures use random vectors** (near-orthogonal, the worst case) and are a pessimistic
   bound; they must be re-run on real embeddings.
5. **MIND's title-only ablation timed out** and is not reported.
