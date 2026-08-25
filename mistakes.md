---
type: note
kind: reference
title: A1 — Bugs found, and how each was caught
---

# Bugs found, and how each was caught

Every real defect in this component, in plain terms: **what broke, why nobody
noticed, and what actually caught it.** Written for the AI-usage log and the
design note, where the verification loop is worth more than the successes.

> [!important] The pattern worth noticing
> **Not one of these was found by reading the code.** Every single one was
> caught by a number that disagreed with a prediction — a rate, a memory
> figure, a score that was impossible, an interval that excluded its own
> estimate. The lesson is not "test more", it is **"predict the number before
> you run it, then check"**. A bug that produces plausible output is invisible
> to review and obvious to arithmetic.

The nastiest ones here share a shape: **they did not crash.** They returned
something reasonable-looking, and only a cross-check exposed them.

---

## 1. The retriever that silently returned nothing

**Where:** `src/retrieval/semantic.py`, `SemanticRetriever._query_vector`

**What it did.** The harness hands a retriever the user's click history as
*article text* ("Brøndby beat FCK", …). The base class looked those strings up
in `self._by_id` — a map of article **ids** ("9771627", …). Text never matches
an id, so every lookup missed, the query vector came out `None`, and the
retriever returned an empty list.

**Why nobody noticed.** No exception, no warning. It just returned nothing, and
"nothing retrieved" is a legitimate answer for a cold-start user. A subclass,
`HistoryIdRetriever`, happened to override the method correctly — and that
subclass was what every earlier run used, so all published numbers were fine.
The bug sat in the base class, unexercised.

**What caught it.** An ablation used the base class directly and scored
**recall = 0.0000 on every semantic row** — including EB-NeRD's *provided,
click-trained* vectors. Those are trained on real click data; they cannot
plausibly score zero. The impossibility of the number, not the code, gave it
away.

**The fix.** Build a text→row map in `index()`, try it first, keep the id
lookup as a fallback. The subclass override became identical to the base and
was deleted.

**Transferable lesson:** *a silent empty return is worse than a crash.* If
"found nothing" is a valid answer, you cannot distinguish it from "looked in
the wrong place" without an independent expectation of what the number should
be.

---

## 2. The confidence interval that excluded its own estimate

**Where:** `src/eval/bootstrap.py`, coverage metric

**What it did.** Reported `coverage = 0.9783, CI [0.9035, 0.9235]`. The point
estimate sits **outside** its own interval — arithmetically impossible for a
percentile bootstrap.

**Why it happened.** Coverage counts *distinct* articles across all results, so
it only grows as you evaluate more impressions. The bootstrap resamples n items
**with replacement**, which makes ~37% of draws duplicates — so every resample
sees *fewer* unique articles than the real sample. The whole distribution sits
below the true value.

**What caught it.** Reading the output table. `0.9783` is not inside
`[0.9035, 0.9235]`, and that is checkable at a glance.

**The fix — and the part that matters.** I tried subsampling without
replacement at ratios 0.5, 0.8, 0.9, 0.95, 0.99. **Every one still failed to
bracket the point.** The bias is inherent: any smaller sample sees fewer
distinct articles. So there is no honest percentile interval for coverage at
fixed n, and it is now reported as a point estimate marked `(no CI)` with the
reasoning recorded.

**Transferable lesson:** *when a fix does not work, say so.* Shipping a
subsample-based interval would have looked more rigorous and been just as
wrong. The requirement said "CI on every metric"; the honest answer was "one
metric cannot have one, here is the evidence".

---

## 3. Swap did not slow the merge down — it stopped it

**Where:** `src/submit/codabench.py`, the shard-merge step

**What it did.** The parallel run wrote all 51 shards correctly at 6,426
lines/s, then produced **zero output for over 20 minutes**. It looked like a
hang.

**Why it happened.** Two workers holding 13.6 GB and 10.9 GB were still alive,
because the merge ran *inside* the `with ProcessPoolExecutor(...)` block. The
merge then tried to build a `set` of 13.3M impression ids on top of that. Total
24 GB went to swap.

Scoring streams a row group at a time — sequential access, so paging costs a
constant factor. The merge hits a large set at **random**, where swap is
roughly five orders of magnitude slower than RAM. It did not degrade; it
stopped.

**What caught it.** Watching the output file size: shards complete, output file
0 bytes, and `/proc/*/status` showing 11.7 GB and 12.5 GB swapped.

**The fix.** Shut the pool down explicitly *before* merging; use `set[int]`
instead of `set[str]` (~1.5 GB saved); warn when free memory drops. The same
merge on the same shards afterwards took **7 seconds**.

**Transferable lesson:** *"it's slow" and "it stopped" are different failures
with different causes.* And peak memory is not the number that matters — **when**
you hold it is.

---

## 4. Two competitions, two filenames, one letter apart

**Where:** `src/submit/codabench.py`, archive member name

**What it did.** The first MIND upload was rejected:

```
FileNotFoundError: '/app/input/res/prediction.txt'
```

The 2,370,727 predictions were correct. The file inside the zip was named
`mind_prediction.txt`; the scorer opens `prediction.txt` literally.

**The second, worse half.** After fixing it I assumed both competitions agreed
and used one constant for both. Checking the actual sources found they do not:

| Competition | Required member |
|---|---|
| MIND (`evaluate.py`) | `prediction.txt` |
| EB-NeRD (`ebrec.utils._python`) | **`predictions.txt`** |

The EB-NeRD upload would have failed exactly the same way.

**What caught it.** The first: a rejected submission, costing one of ten daily
attempts. The second: **reading both upstream sources instead of generalising
from one.**

**The fix.** A per-dataset dict, plus tests asserting both the constant and the
built artefact.

**Transferable lesson:** *one confirmed fact is not a pattern.* Fixing MIND
taught me the name mattered; it did not tell me what EB-NeRD wanted. Two
similar systems agreeing is a hypothesis, not an inference.

---

## 5. A test fixture that tested nothing

**Where:** `tests/test_bm25.py`, the Unicode normalisation test

**What it did.** Asserted that NFC and NFD forms of a Danish word produce the
same tokens, using `brøndby`. The assertion passed — because **`ø` has no
decomposed form**. NFD left it unchanged, so the test compared a string to
itself.

**What caught it.** A guard assertion inside the test:
`assert composed != decomposed, "fixture is not exercising the two encodings"`.

**The fix.** Use `århus` — `å` *does* decompose (a + combining ring).

**Transferable lesson:** *a passing test proves nothing unless you know it can
fail.* The guard line cost nothing and caught a test that would have given
false confidence about the exact bug it was written for.

---

## 6. Two "BM25" scores that were not the same number

**Where:** `src/retrieval/bm25.py`, `score_subset` vs `retrieve`

**What it did.** The fast submission path recomputes BM25 directly over a
slate. I wrote it using the textbook formula and expected it to match `bm25s`.
It did not — by up to **106 in absolute score** on the real corpus.

**What caught it.** Comparing the two implementations on the same documents.

**The fix, and the honest part.** I tried to reproduce `bm25s`'s exact
arithmetic — its Lucene IDF variant, its document-count conventions — and
**failed**. Rather than keep guessing, I verified the property that actually
matters: **ranking agreement**, measured at 0 discordant pairs out of 780. The
docstring states plainly that absolute scores differ, that only the ordering is
ever written to a submission, and that this function must never be used for a
reported metric.

**Transferable lesson:** *verify the property you depend on, not the one that
is easiest to state.* A submission file records a permutation; identical scores
were never required.

---

## 7. Trusting a benchmark run on the wrong data

**Where:** the HNSW-vs-exact comparison (F31)

**What it did.** Benchmarked FAISS HNSW with **randomly generated vectors** and
measured recall 0.45–0.77 against exact search — poor enough to look like a
real limitation.

**Why it is wrong.** Random vectors in 768 dimensions are nearly orthogonal to
each other, which is the *worst possible case* for a proximity graph. Real
embeddings cluster, which is exactly the structure HNSW exploits.

**What caught it.** Noticing before publishing, and flagging it in the finding:
*"these are a worst case and must be re-run on real vectors"*.

**Transferable lesson:** *synthetic data can make a component look broken.* The
latency and memory numbers from that run were fine; only the recall number was
meaningless, and it was the one that would have gone in the report.

---

## 8. Measuring on a sample too small to decide anything

**Where:** the whole evaluation harness, early on

**What it did.** Reported MIND `AUC = 0.4981 [0.4776, 0.5190]` at n=800 —
indistinguishable from chance. The leaderboard, on 2.37M impressions, scored
**0.5568**. The true value was **outside** our interval.

**Why it happened.** n=800 with a CI half-width of ±0.021, and parameters were
being *chosen* on that sample.

**What caught it.** An external measurement disagreeing with an internal one.

**The fix.** The default evaluation sample is now 20,000 impressions, and every
comparison whose CIs overlap is labelled as inconclusive rather than reported
as a result.

**Transferable lesson:** *a confidence interval is a claim, and claims can be
wrong.* Nothing internal could have caught this — it took a number from
outside the system.

---

## 9. Optimising 11% of a pipeline and projecting a 1,200× speedup

**Where:** `decisions.md` Part 4c, the Polars + GPU costing

**What it did.** Costed a GPU rewrite of the semantic scoring path at **~1,200× on that stage** and
89 min → 37 min end-to-end. On that basis the work looked clearly worth 4–6 hours. Built on branch
`polars-gpu`; the measured end-to-end gain is **~4%**.

**Why it happened — two errors of the same shape.** Both are *timing a fragment and quoting it as
the whole*:

1. **The baseline was the wrong algorithm.** The projection compared the GPU against 623 µs/slate —
   *full-corpus retrieval*, which bug/finding F32 had replaced with direct slate scoring a week
   earlier. The real CPU path runs at **105 µs/slate**. I benchmarked against code that was no
   longer running.
2. **I never asked what fraction of the run the stage was.** Semantic slate scoring is **11%** of a
   MIND fusion run (249 s of 2,288 s). Query-vector construction — pure CPU, untouched by batching —
   is **67% of the batched path**, so even a *free* GPU stage caps the whole thing at **2.4×**.

**What caught it.** Building it and profiling end-to-end instead of per-stage. Nothing else would
have: the per-stage number was real, reproducible, and completely misleading.

**The fix.** The projection is corrected in place with the measurement beside it rather than
deleted, and the branch is not merged — 4% does not justify a CUDA dependency on the submission
path. The code is correct and its merge gate passes; **the premise failed, not the implementation.**

**The part that stings.** The stated reason for doing this was *"speed buys statistical power"* —
faster runs would rescue ablations left underpowered at n=800. At 4% that is worthless (CI width
falls as 1/√n). And the underpowered ablations were never compute-bound anyway: F46's dedup test
differed on **4 impressions out of 800**. It was short of *effect size*, not CPU time.

**Transferable lesson:** *profile the whole before optimising the part.* One `perf_counter` around
the existing loop — under a minute — would have shown 11% and saved the entire estimate. This is
bug 7 again in a new costume: **a measurement can be perfectly accurate and still answer a question
nobody asked.**

---

## 10. Dismissing a tool by assuming which step it optimises

**Where:** F60 in the execution log — "Polars replaces the reader, and the reader is 0.6% of setup,
so Polars cannot help"

**What it did.** After F59 profiled the run and found history loading was 58.5% of setup, I ruled
Polars out with a specific, confident-sounding argument: it optimises Parquet reading, reading is
1.5 s of 271 s, therefore it is irrelevant. Both halves are wrong.

**The measurement.** Same file, same machine:

| Step | pyarrow | polars |
|---|---|---|
| Read → columnar | **1.66 s** | 3.06 s — polars is **slower** |
| Export → Python objects | 28.43 s | **3.00 s — 9.5× faster** |

**Polars' entire advantage is in the export path — the step I asserted it could not touch** — and it
is *worse* at the step I assumed was its whole purpose. Staying columnar
(`explode().to_numpy()`) does better still: 116.8M clicks in **0.35 s and 0.47 GB**, against 158.6 s
and ~13 GB as Python objects.

**What caught it.** Being asked "if arrow to python is taking time, will polar take similar to and
fro times?" — a question about the specific step, which is exactly what I had not measured. The
argument was structurally sound and rested on an unchecked premise about what the library does.

**Why this one is worse than bug 9.** It is the *same mistake in the very next finding*. Bug 9 was
projecting a speedup without profiling the whole; F60 then dismissed a tool without profiling the
part. Both times I produced a precise number (1,200×; 0.6%) that made a guess look like a
measurement.

**Transferable lesson:** *a percentage attached to the wrong step is more misleading than no
number at all.* "The reader is 0.6% of setup" was true and irrelevant — Polars was never mainly a
reader optimisation. **Before claiming a tool will or will not help, measure the specific step you
believe it changes**, not the step its name suggests.

---

## What the failures have in common

| Bug | Crashed? | Caught by |
|---|---|---|
| 1. Empty semantic results | No | An impossible score (0.0000 for click-trained vectors) |
| 2. Coverage CI | No | Arithmetic — estimate outside its own interval |
| 3. Swap stall | No | Watching throughput and `/proc` |
| 4. Archive filename | **Yes** | A rejected submission |
| 5. Empty test fixture | No | A guard assertion inside the test |
| 6. BM25 score mismatch | No | Cross-checking two implementations |
| 7. Random-vector benchmark | No | Questioning the input before publishing |
| 8. Under-powered sample | No | An external number disagreeing |
| 9. 1,200× speedup projection | No | Profiling end-to-end instead of per-stage |
| 10. Ruling out Polars by assumption | No | Measuring the step, after being asked about it |

**Nine of ten did not crash.** The one that did — the filename — was the
cheapest to fix and the least interesting.

The working method, stated as a rule: **before running anything, write down
what the number should be. Then check.** Every bug above was found in the gap
between a prediction and a measurement, and none would have been found by
re-reading the code.

---
[[architecture|architecture]] · [[decisions|decisions]] ·
[[plan/execution_plan_log|execution log]] · [[ai-log|AI usage log]]
