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

## 13. Sizing a worker from the one component I had optimised

**Where:** `WORKER_GB_COLUMNAR` in `src/submit/codabench.py`

**What it did.** After measuring the columnar click arrays at 0.93 GB, I set the per-worker memory
constant to **2.9 GB**. Measured: **8.06 GB** — wrong by 2.8×.

**Why.** A worker holds more than histories: 125,541 article objects and the BM25 index, ~1.8 GB
together. I had measured the piece I changed and treated it as the whole.

**Why it mattered more than a wrong number.** That constant feeds the *preflight clamp* that decides
how many workers start. Too low, and it starts workers that do not fit — which per bug 3 does not
fail loudly, it swaps, and a swapping merge stalls rather than slows.

**What caught it.** Running one real worker and printing its RSS before trusting the constant.

**The second, more useful finding.** Chasing the gap showed peak RSS is set by peak *allocation*,
not by what survives: the one-shot `explode().to_numpy()` peaks at **4.93 GB to produce 0.94 GB of
arrays**, and `del df` returns **none** of it — glibc keeps freed pages. Streaming the 9 row groups
never reaches that peak: **4.93 → 2.85 GB**, worker **8.06 → 5.32 GB**.

**Transferable lesson:** *the size of a result says nothing about the footprint of producing it.*
F62's "0.47 GB of arrays" was accurate and did not predict a worker. When a number will gate a
resource decision, measure the thing that consumes the resource — not the artefact it leaves behind.

---

## 14. Loading 14.5 GB the code path never reads

**Where:** `src/submit/codabench.py`, the parent process before `build_predictions_parallel`

**What it did.** The parent loaded every EB-NeRD history — 807,677 users, 116,825,984 clicks,
**~14.5 GB of Python objects, ~158 s** — and then passed them only to the *serial* branch.
`build_predictions_parallel` builds its own per-worker copies in `_init_worker` and takes no
`histories` argument at all. On EB-NeRD, which always parallelises, the parent's copy was never read.

**Why nobody noticed.** It produced correct output. Wasted work that returns the right answer is
invisible to every test, and the line reads perfectly sensibly in isolation — you have to notice that
the *branch below it* doesn't use the variable.

**The part that actually hurt.** The parent holds that 14.5 GB for the entire run, so the preflight
measured a machine with far less free memory than it had and clamped **3 workers down to 1** — and
the survivor then requested parent + worker ≈ **36 GB on a 31 GB box**. The run went 15 minutes at
95% idle CPU with no output.

**What caught it.** The user saying "ebnerd didn't start". It had started 15 minutes earlier; the
symptom was that it was producing nothing. Reading `/proc/*/status` showed a 14.5 GB parent next to a
21.9 GB worker, and `vmstat` showed `si/so` at zero — so it was not swapping (bug 3), it was starved.

**The fix.** Load histories only on the serial path. Parent RSS 14.5 GB → **1.73 GB**, workers 1 → 3,
total footprint ~36 GB → **~13 GB**.

**Transferable lesson:** *dead weight is invisible when it is correct.* Every test passed, the output
was byte-perfect, and the cost showed up only as a resource decision made on bad information. When a
preflight reads free memory to size a run, anything the process is needlessly holding becomes a
correctness input — not just an efficiency one.

---

## 15. Passing `--allow-swap` on the argument that swap was safe here

**Where:** the EB-NeRD reproduction run, my own command line

**What it did.** The preflight refused 2 workers, saying they would swap. I overrode it with
`--allow-swap`, reasoning that scoring streams parquet row groups sequentially, and sequential access
under swap costs a constant factor rather than stalling — the distinction bug 3 established.

**Why that was wrong.** The reasoning was right about the *row groups* and wrong about everything
else in a worker. Each scored impression also hits the **BM25 index and the article dictionary at
random**, and those are gigabytes. Random access against swap is the exact pattern bug 3 says does
not degrade but **stops**. I applied a true fact about one data structure to a process dominated by
another.

**What it cost.** Parent 9.6 GB plus two workers at 19.3 GB each — **~48 GB of RSS on a 31 GB
machine**, 13.1 GB into swap and climbing, `si/so` at ~140,000 blocks/s in both directions, all 32
cores 75–100% idle. The machine was doing nothing but paging.

**What caught it.** The user sending a system-monitor screenshot: memory 98.2%, swap climbing, CPU
flat near zero. I had been reporting the run as healthy.

**The fix, and the better answer.** Killed it. The real problem was never swap tolerance — it was
that six copies of identical read-only data existed at all. F70 (build once, share through `fork`)
took a worker from 19.3 GB to 0.38 GB private, after which 6 workers fit in 10 GB with **zero** swap
I/O and the run finished in 22 minutes.

**Transferable lesson:** *an override exists because the check knows something you might not.* The
preflight was measuring the actual requirement; I was reasoning from a model of it. When a guard
says no and you can construct an argument for why it is wrong, the argument is the thing to check —
and reaching for the override is a signal to fix the underlying cost instead.

## 11. Quoting a microbenchmark as the cost of the feature

**Where:** F62 in the execution log — "116.8M clicks in 0.35 s and 0.47 GB, ~450× faster"

**What it did.** After measuring `explode().to_numpy()` in isolation, I reported those figures as
what a columnar history loader would cost. Building the loader gave **4.58 s and 1.40 GB — 34.9×**,
not 450×.

**Why.** The microbenchmark timed one call. The actual feature also builds per-user offsets,
converts the timestamp column, and maps 807,677 user ids to rows. All necessary; none measured.

**What caught it.** Building it and benchmarking the whole thing against the real loader.

**Why it is bug 9 again.** Bug 9 was projecting 1,200× from GPU kernel time that excluded query
construction and transfer. This is the same shape — a real number from a real measurement, describing
a fragment, presented as the whole. It is milder (34.9× is still worth having, where 1,200× → 4% was
not) but it is the same reasoning error, three findings later.

**The fix.** Both numbers are kept in F64 side by side, so the gap between the microbenchmark and the
feature stays visible rather than being quietly replaced.

**Transferable lesson:** *a microbenchmark of one call is not the cost of the feature it belongs to.*
The rule earned in bug 9 — profile the whole before optimising the part — applies just as much when
the news is good. Optimism needs the same verification as pessimism.

---

## 12. A memory optimisation that quietly moved the leakage boundary

**Where:** `src/data/columnar.py`, storing click times as int32 seconds

**What it did.** Times were the largest array in the columnar loader (0.93 GB against the ids'
0.47 GB), and measurement showed **0 of 116,825,984 EB-NeRD clicks have sub-second precision** — every
timestamp is a whole second. So storing seconds instead of microseconds is lossless, and halves the
biggest array. That reasoning is correct.

**What it missed.** The *clicks* are whole seconds. The **cutoffs are not.**
`np.datetime64(cutoff, "s")` floors toward the past, so a cutoff of `07:51:01.000001` becomes
`07:51:01` — and `t < cutoff` then **excludes a click at exactly `07:51:01`**, which is genuinely
before it. One click per affected user, dropped from the history that feeds the query.

**What caught it.** `test_matches_history_before_on_real_ebnerd_users`, on real user 40107:

```
truncation diverged for user 40107 at 2023-05-11 07:51:01.000001: 0 vs 1 clicks
```

The test probes each real click time and **±1 µs either side**, specifically because a boundary is
easiest to get wrong at its own edge. A fixture test with whole-second cutoffs passes either way.

**The fix.** Ceiling, not floor: since every stored time is a whole second, anything strictly before
`07:51:01.000001` is at `07:51:01` or earlier, hence strictly before `07:51:02`. One line —
`cut = -((-us) // 1_000_000)` — and the throughput cost (251K/s → 180K/s) is worth it.

**Why this is the most valuable bug in the file.** It is the case F36's rule was written for. Had I
*migrated* `History.before()` to the columnar form instead of *duplicating and pinning* it, there
would have been no original left to disagree with — the leak would have shipped, in the direction
nothing checks for (too *few* clicks, not too many), affecting only the uploaded file. **The
duplicate-and-pin pattern paid for itself the first time it was used.**

**Transferable lesson:** *a lossless change to the data is not automatically a lossless change to the
comparison.* I verified the precision of the stored values and never asked about the precision of
what they are compared *against*. When you narrow a type, the invariant to check is the operation,
not the storage.
---

## 16. A probe that condemned an encoder on the wrong property

**Where:** `danish_probe()` in `src/retrieval/encode.py`, and F37's conclusion

**What it did.** The probe measured `xlm-roberta-base` at margin **+0.0018** between related and
unrelated Danish pairs and F37 concluded it **"cannot separate related from unrelated"** and
**"carries no usable retrieval geometry"**. The first clause is true. The second is wrong, and it is
the one that got quoted in the design note.

**What was actually wrong.** The encoder is **anisotropic** — all its vectors sit in a narrow cone
around one dominant direction, so every cosine is ~0.99 regardless of content. Subtracting that mean
direction, one line, takes the margin from **+0.0018 to +0.3875 — a 215× improvement.** With
truncation as well, it **separates cleanly** (+0.5070 at 128-d). The information was there the whole
time.

**The part that makes it a real mistake, not just a missed optimisation.** Even at the collapsed
baseline, **4 of 5 related pairs still ranked in the top 5 by cosine.** Retrieval consumes *ranking*,
not absolute similarity — so the property the system depends on was largely intact while the property
the probe measured was destroyed.

**What caught it.** The user pushing back: *"the fact that XLM-R and MiniLM were not having good
related/unrelated values is bugging me — check deeper."* Nothing internal would have; the probe was
reporting its number correctly and the conclusion had already been written down as settled.

**Transferable lesson:** *verify the property you depend on, not the one that is easy to measure.*
This is bug 6 in a new costume — there I checked ranking agreement instead of absolute BM25 scores
and got it right; here I measured absolute cosine separation and drew a conclusion about retrieval
usability. **A smoke test can prove something unusable only if it tests the thing you use.**

Also worth recording: **whitening, the textbook fix for anisotropy, failed** (+0.0020). It removes
the signal along with the dominant direction. Centering alone is what worked.

---

## 17. Three semantic parameters that were never swept, and set wrong

**Where:** `tau`, `decay`, `lam` in `src/retrieval/semantic.py`

**What it did.** The semantic side has more tunable parameters than the lexical side. An audit of
which ones appear in any results file as a *varied* quantity found that **`tau`, `decay` and `lam`
never do** — `tau` shows up only inside retriever name strings, which reads like it was swept and
was not. All three shipped at their initial guesses.

Sweeping them on MIND val, paired against the shipped default:

| Change | Δ recall@100 | |
|---|---|---|
| `decay=flat` (no recency weighting) | **+0.0051** [+0.0004, +0.0098] | significant |
| `tau=0.20` (looser) | **+0.0026** [+0.0005, +0.0053] | significant |
| `tau=0.80` (stricter) | **−0.0079** [−0.0128, −0.0034] | significantly worse |

**The one that stings.** `decay=flat` — no recency weighting at all — **beats** the shipped log
decay. Recency decay was added to the semantic path because it "matches the lexical decay" (D3).
That analogy was the *entire* justification, and it was never tested on the semantic side. On MIND,
which has no publish timestamps, down-weighting older clicks just discards history that carries
signal.

**What caught it.** The user asking *"should we have a stricter tau? we should be getting more score
from the extra semantic params — check if we missed any."* The audit was a one-line grep over
`results/*.json`; nothing in the code or tests would have flagged it, because a default that is never
varied looks identical to a default that was chosen.

**Transferable lesson:** *a parameter justified by analogy is an untested parameter.* "It matches the
lexical side" is a reason to try something, not evidence that it works. And more knobs are not more
score — three of them were pointed the wrong way, which is worse than not having them.

---

## 18. A fallback strategy that was worse than not falling back

**Where:** `build_user_vector`'s `max_pool` rung, `src/retrieval/semantic.py`

**What it did.** When a user's history is too incoherent for a centroid, the ladder max-pools their
clicked vectors. The docstring justifies it: *"a noisy vector that points somewhere beats a smooth
one pointing at the corpus mean."*

**Measured on the 1,410 MIND impressions that actually reach that rung:**

| | MRR within slate |
|---|---|
| `max_pool` (shipped) | 0.2979 |
| plain mean (the thing it replaces) | **0.3165** |
| paired difference | **+0.0186 [+0.0059, +0.0312]** |

**The "meaningless" centroid beats the fallback designed to replace it**, on exactly the population
the fallback exists for.

**What caught it.** The user asking directly: *"should we replace max pool with mean pool?"* Two
independent routes then agreed — the τ sweep already showed looser thresholds scoring better
(because they route fewer users into `max_pool`), without testing pooling at all.

**Transferable lesson:** *a plausible sentence in a docstring is not a measurement.* That
justification reads well, survived code review, and was wrong. The rung was added to handle a case,
and nobody scored the case with and without it — the population it applies to (2–7% of users) was
small enough that aggregate metrics never revealed the loss.

---

## 19. Calling a result significant before checking it at another sample size

**Where:** F73 and F74, reported an hour before F75 corrected them

**What it did.** Swept the never-tested semantic parameters, found `decay=flat` at
**+0.0051 [+0.0004, +0.0098]** and dropping `max_pool` at **+0.0186 [+0.0059, +0.0312]**, and wrote
both up as **SIGNIFICANT** — CI excluding zero, paired test, the correct machinery.

**What was wrong.** Both were measured at **n ≈ 1,200–1,410**. Re-run at **n = 2,000 on both
datasets**, neither survives: +0.0026 [−0.0011, +0.0070] and +0.0012 [+0.0000, +0.0027]. **More data
made the effects less significant, not more** — which is the signature of an interval driven by
sample noise rather than a real effect of that size.

Two further things the re-test exposed that the originals had missed entirely:

- **The two "independent" fixes are the same fix.** Applied together they are byte-for-byte identical
  to τ=0.20 alone, because a loose τ already routes nobody into `max_pool`. F74 was measuring the
  tail of F73.
- **F74 does nothing at all on EB-NeRD** — +0.0000 exactly, because 99.9% of its users never reach
  that rung (F40). The original only tested MIND.

**What caught it.** The user saying *"test them"* rather than accepting the write-up. Nothing else
would have — the statistics were computed correctly, the code was right, and the conclusion was
still wrong.

**Transferable lesson:** *a significant result at one sample size is a hypothesis, not a finding.*
This project already knew that — bug 8 is the same error, where n=800 put the true value outside its
own CI. Knowing the lesson did not prevent repeating it, which is worth recording honestly: the
paired test felt rigorous enough that re-measuring did not seem necessary.

**What survives, stated at its real strength:** the *direction* is consistent 4/4 — every fixed
configuration beats shipped on two metrics across two datasets. Consistency across independent splits
is evidence. But ~0.002 is the same magnitude F58 showed does not transfer to a leaderboard, so it is
recorded as a measured design defect, not shipped as a tuning win.

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
| 11. Microbenchmark quoted as feature cost | No | Building the feature and benchmarking it whole |
| 12. int32-seconds moved the leakage boundary | No | Differential test vs `History.before()` on real users |

**Eleven of twelve did not crash.** The one that did — the filename — was the
cheapest to fix and the least interesting.

The working method, stated as a rule: **before running anything, write down
what the number should be. Then check.** Every bug above was found in the gap
between a prediction and a measurement, and none would have been found by
re-reading the code.

---
[[architecture|architecture]] · [[decisions|decisions]] ·
[[plan/execution_plan_log|execution log]] · [[ai-log|AI usage log]]
