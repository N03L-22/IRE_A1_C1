# IRE Assignment 1 — Lexical & Semantic Retrieval (Component 1)

**CS4.406 Information Retrieval and Extraction · IIIT Hyderabad · Monsoon 2026**
Noel Alex Jacob · 2025201085

News-recommendation candidate generation on **MIND** (English) and **EB-NeRD** (Danish): a
reproducible data pipeline, lexical retrieval, and an offline evaluation harness with bootstrap
confidence intervals.

> **Status.** Q1–Q5, Q7 and Q9 complete; Q6 design note at 4 pages. Both leaderboards submitted.
>
> | MIND submission | AUC |
> |---|---|
> | BM25 | 0.5568 |
> | + RRF fusion with semantic | 0.5934 |
> | + 256-d truncation | **0.5938** |
>
> EB-NeRD (BM25) submitted, awaiting score. **Nothing here is estimated** — anything unmeasured is
> absent, not guessed, and the three leaderboard results above are also the evidence that our
> *offline* harness is an unreliable proxy for them (F34, F42, F58).

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

make data && make store    # raw archives -> unified, temporally-split parquet store
make bm25                  # BM25 vs baselines, both datasets
make eval                  # full harness: metrics, slices, bootstrap CIs
make test                  # 100 tests, incl. the leakage mutation test
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
  retrieval/    Retriever protocol, tokeniser, BM25 (+ own reference impl),
                encoder, semantic retriever, RRF fusion
  eval/         metrics (two regimes), bootstrap, slices, harness, sweeps
  submit/       Codabench prediction files
tests/          100 tests, incl. leakage mutation tests
plan/           phase plans + execution log (findings F1-F59)
results/        every measured run, as JSON with CIs and resolved budget
report/         a1_design_note.tex (the Q6 deliverable) + figs/ screenshots
notebooks/      one-off data exploration; NOT part of the pipeline
brief/          the assignment PDFs, v1 and v2
submissions/    prediction metadata (the .txt/.zip are gitignored)
configs/        per-dataset paths and resource budget
```

### Documents

| File | Holds |
|---|---|
| [`foundations.md`](foundations.md) | Every concept from scratch — vectors, BM25, embeddings, and the evaluation metrics built up need → why → how |
| [`architecture.md`](architecture.md) | What the system **is**, plus the measured dataset facts |
| [`decisions.md`](decisions.md) | What we **chose and rejected**, open questions, and the cost of each choice — the design-note source |
| [`plan/execution_plan_log.md`](plan/execution_plan_log.md) | Findings F1–F59, dated. Also the architecture changelog |
| [`mistakes.md`](mistakes.md) | Every defect found, in plain terms — eight of nine did not crash |
| [`ai-log.md`](ai-log.md) | Q7.4 deliverable: curated prompts, what worked, what failed |

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

1. **The offline harness is not a reliable proxy for the leaderboard** — three-for-three in
   disagreeing, and in both directions (F34 understated a real effect, F42 missed one entirely,
   F58 overstated one). It remains sound for what it was built for — leakage, slicing, the recency
   finding — but no parameter choice here is settled until it has been submitted. **This is the
   project's strongest methodological finding, and it is a limitation, not a result.**
2. **EB-NeRD's leaderboard score is pending**, so the cross-dataset comparison in the design note
   rests on offline numbers on that side — see gap 1 for what that is worth.
3. **Several differences are not statistically separable**, and each is labelled inconclusive at
   the point it is reported rather than rounded into a claim. Where a null result comes from an
   underpowered experiment rather than an absent effect, the number of impressions the two
   configurations actually differ on is given (F46: 4 of 800).
4. **MIND's title-only ablation timed out** and is not reported.
5. **The leakage invariant is machine-checked only on EB-NeRD.** MIND's history carries no
   per-click timestamps, so there the boundary rests on the authors' construction. Recorded per row
   as `history_verifiable` rather than claimed.

*Closed since the first draft:* Q3 semantic retrieval is built and compared (F39); three MIND
leaderboard scores are in (F58); the HNSW figures were re-run on real vectors, which reversed the
conclusion (F49 — random vectors are near-orthogonal and were the worst case).
