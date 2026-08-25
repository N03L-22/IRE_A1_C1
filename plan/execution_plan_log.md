---
type: note
kind: reference
title: Execution plan log — A1 Component-1
---

# Execution plan log

Running log of **what we decided, why, what is done and what is not**. One task at a time; review and
correct before starting the next. Doc and data-layout work come before coding.

> [!abstract] Where things stand (2026-08-25)
> **Q1–Q4 built and measured; Q5 half-submitted.** MIND has a real leaderboard score. The
> semantic retriever exists but has not yet been scored through the harness.
>
> | Item | State |
> |---|---|
> | **Q1** pipeline, unified schema, temporal split | ✅ 42 s rebuild, both datasets |
> | **Q2** BM25 over title + abstract | ✅ + 54-cell sweep, val only |
> | **Q3** embeddings + ANN | ✅ scored both; dimension swept (F47); multi-query under test |
> | **Q4** harness, CIs, slices | ✅ 98 tests, + paired-difference test (F46) |
> | **Q5** Codabench | ✅ **MIND 0.5568 → fusion 0.5934**; EB-NeRD submitted |
> | **Q6** design note | ✅ **`report/a1_design_note.tex`, 3 pp** (≤4 limit met) |
> | **Q9** leakage test + serving features | ✅ incl. mutation tests |
> | Repo | ✅ `github.com/N03L-22/IRE_A1_C1`, pushed |
> | Leaderboard screenshots (Q7.3) | ⬜ MIND available, EB-NeRD pending |
> | Pair declaration (C2) | ⚠️ **deadline was 2026-08-15 — verify this is sorted** |
>
> **Two days to the 2026-08-27 deadline. Findings F1–F58.**
>
> **The three findings that reshaped the work:**
> 1. **F16/F21 — recency dominates.** A retriever that ignores the user entirely scores
>    recall@50 = 0.9050 on EB-NeRD. Candidate generation here is primarily a *freshness*
>    problem; BM25 on the full corpus is indistinguishable from popularity.
> 2. **F37 — XLM-RoBERTa cannot separate related from unrelated Danish** (margin +0.0018 vs
>    MiniLM's +0.6271). The brief names it; the probe proves it unusable. Reported as a
>    measured failure rather than quietly swapped.
> 3. **F34 — the offline harness was under-powered.** It reported MIND AUC 0.4981
>    [0.4776, 0.5190]; the leaderboard scored 0.5568, *outside* the CI. n = 800 was too small,
>    and parameters were being chosen on it. Default is now 20,000.
>
> **Remaining:** curate the AI usage log (Q7.4, still raw) and capture the leaderboard screenshots
> (Q7.3). Every named ablation is now run (F41, F43, F46, F47).
>
> **The three results worth leading with:**
> 1. **F47 — 256-d beats 384-d significantly at 34% less memory**, and only the paired test could
>    see it. The best efficiency result in the project.
> 2. **F42/F46 — the offline harness cannot rank retrievers at n=4,000.** Fusion was *not
>    significant* offline (differing on 55/800) and scored **+0.0366 AUC** on 2.37M impressions.
> 3. **F44 — we sit at the challenge's own "most clicks" baseline (59.70).** The winners at 87–89
>    used GBDT/transformer re-rankers, which is Component-2, and their ablations independently
>    identified the same two levers we found: popularity and article timeliness.

### F39 — Q3.5's answer flips between datasets, and fusion only helps where they disagree
The full harness on both datasets, n = 4,000 each, MiniLM encoder, all retrievers in one pass.

| recall@100 | EB-NeRD (Danish) | MIND (English) |
|---|---|---|
| Best overall | **recency 0.9434** [0.9361, 0.9505] | **popularity 0.0688** [0.0615, 0.0763] |
| Lexical (BM25) | **0.2375** [0.2244, 0.2490] | 0.0142 [0.0108, 0.0172] |
| Semantic (MiniLM) | 0.2307 [0.2175, 0.2432] | **0.0163** [0.0130, 0.0199] |
| RRF fusion | 0.0070 — **no gain** | **0.0181** [0.0146, 0.0218] — **best retriever** |

**On MIND semantic beats lexical and fusion beats both. On EB-NeRD lexical edges ahead and fusion
helps nothing.** A single "which is better" answer to Q3.5 would be wrong.

*Why fusion splits this way:* RRF exploits **disagreement**. On EB-NeRD the two retrievers agree —
their CIs overlap heavily — so there is nothing for fusion to combine, and it lands at the level of
its components. On MIND they disagree enough that combining ranks adds signal. This is a mechanism,
not a quirk, and it is the most useful thing fusion taught us.

**The cold/warm crossover is the sharper result** (EB-NeRD, recall@100, cold = history ≤ 97, the
q0.25 of observed):

| | cold (n=1,004) | warm (n=2,996) |
|---|---|---|
| semantic + 24h | **0.2468** [0.2198, 0.2742] | 0.2253 [0.2104, 0.2407] |
| bm25 + 24h | 0.2216 [0.1982, 0.2490] | **0.2428** [0.2276, 0.2588] |

**The ordering reverses.** Semantic wins cold users, lexical wins warm ones — precisely the Phase 3
D6 hypothesis. With few clicks, word overlap has too few terms while embeddings still place a sparse
history somewhere meaningful. **CIs overlap, so this is suggestive, not established** — but the
crossover is the shape Q3.5 asks about, and it is reported with that caveat.

### F40 — The conditional pooling branches on MIND and is dead weight on EB-NeRD
`strategy_counts` over 4,000 impressions each:

| Dataset | mean | recent_half | max_pool |
|---|---|---|---|
| EB-NeRD | 3,991 | 9 | 0 |
| **MIND** | 2,117 | **1,462** | **270** |

**46% of MIND users have histories too incoherent for a centroid, against 0.2% on EB-NeRD.**
Consistent with the median history lengths (MIND 19, EB-NeRD 92): shorter histories are more likely
to span unrelated topics without enough mass in any one.

*Consequence:* D-POOL's conditional ladder is **justified on MIND and unnecessary on EB-NeRD**. The
honest report is both counts — a mechanism that fires 0.2% of the time is complexity without effect,
and saying so is worth more than implying it helped everywhere.

### F41 — The recency window has an optimum at 24h, and both retrievers agree on it
Swept the window over {6, 12, 24, 48, 72}h for **both** BM25 and semantic, EB-NeRD, n=800.
Open questions O2/O6, now closed. `results/window_sweep_ebnerd.json`.

| Window | BM25 r@50 | BM25 r@200 | Semantic r@50 | Semantic r@200 |
|---|---|---|---|---|
| 6h | 0.1800 | 0.1800 | 0.1825 | 0.1825 |
| 12h | 0.2387 | 0.2387 | 0.2213 | 0.2213 |
| **24h** | **0.2475** | 0.2475 | **0.2325** | 0.2325 |
| 48h | 0.2100 | 0.2500 | 0.2150 | 0.2400 |
| 72h | 0.1437 | **0.2512** | 0.1594 | **0.2412** |

**Past 24h, recall@50 and recall@200 move in opposite directions.** A wider window admits *more*
clicked articles overall — recall@200 climbs monotonically to 0.2512 — while pushing them *out of
the top 50*, which collapses from 0.2475 to 0.1437. Stale-but-topically-similar articles outrank the
fresh target once they are eligible.

F22 saw one point of this at 72h; the full curve shows it is a genuine optimum, not a tuning artefact.

> [!important] The same shape appears under two completely different scoring functions
> BM25 scores word overlap; the semantic retriever scores embedding proximity. That both peak at 24h
> and both invert past it means **this is a property of the data, not of BM25** — news relevance
> decays on roughly a one-day scale, and any retriever searching a wider pool pays the same price.
> A single-retriever sweep could not have established that.

*Consequences:*

1. **24h was already the default, so no submission needs regenerating.** It is now a validated
   choice rather than a plausible guess — which is the difference between an assertion and an
   ablation row.
2. **Report window and K together.** Quoting recall@50 at one window hides an inversion that changes
   the conclusion; the 2-D surface is the honest presentation.
3. **The CIs overlap between 12h, 24h and 48h** (e.g. BM25 24h [0.2175, 0.2775] vs 48h). Report 24h
   as *best measured*, not as significantly better.

### F42 — Fusion scored 0.5934 on the leaderboard: the offline harness understated it 9x
Submitted the RRF fusion MIND file (901779) against the earlier BM25 one (901650).

| Submission | Retriever | Leaderboard AUC |
|---|---|---|
| 901650 | bm25(k1=1.6,b=0.75,n=5) | 0.5568 |
| **901779** | **rrf(bm25+semantic)** | **0.5934** |

**Gain: +0.0366.** The offline harness predicted **+0.004** (0.5095 vs 0.5057) — an order of
magnitude too small.

> [!warning] This is the second time the offline harness mis-ranked against the leaderboard
> F34 recorded the first: it put BM25's MIND AUC at 0.4981 [0.4776, 0.5190] where the leaderboard
> scored 0.5568 — *outside* the interval. Now it has understated a real improvement by 9x.
>
> Both failures point the same way: **at n = 4,000 with overlapping CIs, the harness cannot
> resolve differences of this size**, and using it to rank retrievers is unsound. It remains
> valuable for what it was built for — catching leakage, exposing the recency effect, slicing —
> but a retriever choice should be validated on the leaderboard, not settled offline.
>
> Stated plainly because the prediction was mine and it was wrong: I recommended the fusion
> submission as "+0.004, inside the CI, probably not worth it for the score". It was worth it.

*Consequence for the design note:* report the offline-vs-leaderboard disagreement as a
**methodological finding**, not a footnote. It is the clearest evidence in the project that an
under-powered offline proxy can invert a decision.

### F43 — The four remaining ablations, and three of them found nothing
Run on EB-NeRD, n=800, closing open questions O5 and O2 plus upgrades U4 and U7.
`results/ablations_ebnerd.json`.

| Ablation | recall@50 | recall@200 | Verdict |
|---|---|---|---|
| **O5** dedup=True (default) | 0.0075 [0.0025, 0.0138] | 0.0200 | — |
| **O5** dedup=False | 0.0050 [0.0013, 0.0100] | 0.0175 | **inside the CI** — no evidence either way |
| **U4** TF-IDF | 0.0063 [0.0013, 0.0125] | 0.0200 | **indistinguishable from BM25** |
| **U7** ours: MiniLM 384d | 0.0037 [0.0000, 0.0088] | 0.0213 | — |
| **U7** provided BERT 768d | 0.0013 [0.0000, 0.0037] | 0.0175 | **overlaps ours** — undecided |
| **O2** exact (brute force) | 0.0037 [0.0000, 0.0088] | 0.0213 | the ceiling |
| **O2** HNSW ef=128 / 256 | 0.0037 [0.0000, 0.0088] | 0.0213 | **identical to exact** |

Four results worth stating:

1. **O5 (dedup) is inside the CI**, exactly as F23's pattern predicted. Keep dedup on principle —
   `k1` saturates repetition *within a document*, and repetition within the *query* is a different
   axis the formula was not designed around — and report that the data does not decide it.
2. **U4: TF-IDF matches BM25** on this data (0.0063 vs 0.0075, overlapping). So BM25's two knobs buy
   nothing measurable here, which is consistent with F23 finding `k1`/`b` worth ~0.01. On a
   full-corpus news task the *pool*, not the weighting scheme, is what matters.
3. **O2: HNSW on REAL vectors matches exact search exactly** at ef ≥ 128 — recall@200 0.0213 for
   both. This corrects F31, which used **random** vectors and measured only 0.45–0.77 recall
   vs exact. Random vectors in 768 dimensions are near-orthogonal, the worst possible case for a
   proximity graph; real embeddings cluster, which is the structure HNSW exploits. **The
   pessimistic figure should not be reported.**
4. **U7: the provided click-trained BERT vectors are *indistinguishable* from our generic MiniLM**
   — 0.0013 [0.0000, 0.0037] against 0.0037 [0.0000, 0.0088]. The intervals overlap, so neither is
   shown to be better. (An earlier draft of this finding said the provided vectors scored *below*
   ours; that over-claimed a ranking the data does not support.)

> [!important] U7 was a correctness check, and it did not come out as designed
> Phase 3 D1 predicted the provided vectors *should* beat ours, and said explicitly: **"if ours
> win, suspect a bug"**. Ours won. Two readings, and the honest position is that we cannot yet
> distinguish them:
>
> - **Benign:** both sit in a regime where full-corpus semantic retrieval barely works at all
>   (recall@50 under 0.008 for everything here), so the comparison is between two numbers that are
>   both nearly zero and whose CIs overlap.
> - **A real bug:** the provided vectors are joined by `article_id` and matched 20,738/20,738, but
>   nothing verifies their *orientation* — they were not run through the Danish probe that gated
>   MiniLM.
>
> **The check that would separate these is running `danish_probe()` against the provided vectors**,
> which is not yet done. Until then this row is reported as unresolved, not as a finding that our
> encoder is better.

### F44 — Calibrated against the challenge's own baselines: we are at the "most clicks" level
The RecSys 2024 Challenge paper (arXiv 2409.20483) publishes official reference points for EB-NeRD:

| System | AUC |
|---|---|
| :D (1st) — transformers + GBDT, 3-stage, time-aware features | **89.24** |
| BlackPearl (2nd) — hierarchical long/short-term interest modelling | 88.15 |
| Tom3TK (3rd) — GBDT ensemble with **article timeliness** features | 87.07 |
| **Most-clicks editorial baseline** | **59.70** |
| Read-time baseline | 59.49 |
| In-view-rate baseline | 54.50 |
| Random | 49.98 |

**Our MIND fusion scored 0.5934 — essentially level with the "most clicks" baseline (0.5970)**,
above in-view-rate and well above random.

> [!warning] This is an analogy, not a like-for-like comparison
> Those baselines are **EB-NeRD**; our 0.5934 is **MIND**. Different corpus, different slate sizes,
> different scorer. It is a useful order-of-magnitude calibration and nothing more. The classmate
> cluster on the EB-NeRD board (0.43–0.60) suggests the two scales behave similarly, which is why
> the comparison is worth making at all.

**What the winners did, and why the gap is not closable here.** All three top teams used GBDT
ensembles and/or transformers in multi-stage pipelines with engineered temporal features — i.e. a
**re-ranker**, which is explicitly Component-2. The 0.59 → 0.87 gap is a different class of system,
not a better-tuned candidate generator. F23 measured the entire BM25 parameter space as worth ~0.01;
this gap is ~0.28.

> [!important] Two of our findings are independently confirmed by the winners
> - *"Ablation studies confirmed that features capturing article popularity significantly impacted
>   performance"* — matches F25/F39, where popularity beat BM25 outright on MIND at every K.
> - Tom3TK *"integrates article timeliness features"*; the paper stresses *"the time-sensitive
>   relevance of news articles"* — matches F16/F41, where recency dominated everything and the
>   window peaked at 24h.
>
> The challenge winners independently identified recency and popularity as the load-bearing signals.
> That is meaningful validation of the analysis even though our absolute scores are far lower — we
> found the right levers and stopped at the point where the next lever is a re-ranker.

**No answer key exists.** The test labels are held by the organisers, which is precisely why the
leaderboard is the only external check available — and why F42's offline/leaderboard disagreement
matters so much.

### F45 — The ablation script dropped its own confidence intervals
`src/eval/ablations.py` built each result row as
`{..., **{m: rec[m][0] for m in rec}, "ci50": ...}`. The `**` spread ran **after** the `ci50` key was
set and overwrote it, so `results/ablations_ebnerd.json` stored point estimates with `ci50: null`.

*Consequence:* every "inside the CI" claim in the first write-up of F43 was **asserted from memory
of the console output, not read from the saved data**. The conclusions happened to survive when the
intervals were recovered from the log — but that was luck, not method, and it is the same class of
error as the coverage-CI bug: a number that looks complete and is not.

*Also corrected in F43:* the provided-BERT row was described as scoring *below* our MiniLM. Their
intervals overlap (0.0013 [0.0000, 0.0037] vs 0.0037 [0.0000, 0.0088]), so the honest statement is
**indistinguishable**. The original wording claimed a ranking the data does not support.

*Fix:* every metric's CI is now stored under a `ci` key alongside the point estimates, and the
ablations were re-run so the artefact matches what was reported.

> [!warning] The ablations sit in a weak measurement regime, and it is worth saying so
> Every recall figure in F43 is **below 0.008** — this is full-corpus semantic retrieval, which F16
> showed is close to hopeless on EB-NeRD without the recency window. Comparing numbers that are all
> nearly zero, at n=800, resolves very little. The ablations should have been run *inside* the 24h
> window where the retrievers actually work. Their conclusions are therefore weak evidence of "no
> detectable effect", not evidence of "no effect".

### F46 — Re-tested every overlapping comparison with a paired bootstrap; two claims were wrong
Prompted by the observation that "do the CIs overlap?" is a **conservative approximation, not the
test**. Two marginal intervals can overlap while the difference is significant, because both
retrievers are scored on the *same* impressions and that shared noise cancels when you subtract.
`bootstrap.paired_difference_ci()` now does it properly.

Audited every comparison reported with overlapping CIs. Paired results:

| Comparison | Point gap | Paired diff, 95% CI | Differ on | Verdict |
|---|---|---|---|---|
| MIND fusion vs bm25, nDCG@10 | +0.0015 | +0.0033 [−0.0029, +0.0105] | **55/800** | not significant |
| MIND fusion vs bm25, recall@100 | +0.0039 | +0.0047 [−0.0021, +0.0118] | 11/800 | not significant |
| EB-NeRD semantic vs bm25, nDCG@10 | **+0.0061** | **−0.0055** [−0.0277, +0.0155] | **470/800** | not significant, and **sign flips** |
| EB-NeRD semantic vs bm25, MRR | +0.0061 | −0.0057 [−0.0312, +0.0202] | 515/800 | not significant |
| O5 dedup on/off, recall@50 | +0.0025 | +0.0025 [−0.0025, +0.0075] | **4/800** | not significant |

**Two corrections to previously reported claims:**

1. **The design note said semantic "edges ahead on ranking" on EB-NeRD** (nDCG@10 0.4750 vs 0.4689).
   The paired difference is **−0.0055** — the sign is *negative*. The marginal point estimates
   favoured semantic; the paired test does not. Corrected in `report/a1_design_note.tex`.
2. **F43's dedup conclusion was right but for an unstated reason.** The two configurations produce
   different results on only **4 of 800** impressions, because dedup only bites when a user's recent
   titles repeat a term — rare at `last_n=15`. The experiment was underpowered by construction, not
   merely inconclusive.

> [!important] "Differ on n/800" separates two very different null results
> A non-significant result means either *no effect* or *no power to see one*, and the count
> distinguishes them:
>
> - **dedup: 4/800** — nothing to measure. No statistical method could resolve this.
> - **EB-NeRD semantic vs lexical: 470/800** — abundant signal, genuinely equal performance. This is
>   a real finding: the two retrieval families disagree constantly and score the same.
>
> Every future null result in this project reports that count alongside the CI.

**And the sharpest consequence, combining with F42.** MIND fusion vs BM25 differed on **55 of 800**
impressions offline and was not significant — yet on the leaderboard's 2.37M impressions fusion
scored **+0.0366 AUC**. The offline harness did not merely understate the effect; **it had no power
to detect it at all**. An offline proxy at n=4,000 can rule out *large* differences and cannot rank
retrievers whose gaps are this size. That is now the strongest methodological statement the project
can make about its own evaluation.

### F47 — Truncating the embedding to 256-d is *significantly better* than 384-d, at 34% less memory
Sweep of embedding dimension inside the 24h window, EB-NeRD, n=800. Truncate the cached 384-d MiniLM
vectors and re-normalise, so **only the width changes** — same model, same training, same text.
`results/dim_sweep_ebnerd.json`.

| Dim | Vectors | recall@50 | nDCG@10 | Paired vs 384-d (recall@50) |
|---|---|---|---|---|
| 384 | 31.9 MB | 0.2325 [0.2025, 0.2613] | 0.4603 | — |
| **256** | **21.2 MB** | **0.2500** [0.2206, 0.2806] | 0.4690 | **+0.0175 [+0.0025, +0.0338] SIGNIFICANT** |
| **128** | **10.6 MB** | 0.2437 [0.2137, 0.2750] | 0.4651 | +0.0112 [−0.0112, +0.0338] — no worse |
| 64 | 5.3 MB | 0.2175 [0.1888, 0.2494] | 0.4550 | −0.0150 [−0.0425, +0.0150] — degrades |

**256-d beats full width on both recall@50 and nDCG@10 with the paired CI excluding zero**, and
**128-d is statistically indistinguishable from 384-d at a third of the memory**. The floor sits
between 128 and 64.

*Why truncation helps rather than merely costing little:* the tail dimensions of a
non-Matryoshka-trained embedding carry mostly noise for this task. Dropping them removes variance
without removing signal — the same mechanism that makes PCA often *improve* retrieval rather than
just compress it.

> [!important] Only the paired test could see this
> The marginal CIs overlap heavily — 384-d [0.2025, 0.2613] against 256-d [0.2206, 0.2806] — so the
> overlap heuristic would have called this "undecided" and we would have shipped the larger, worse
> vector. The paired difference is significant because both configurations are scored on the *same*
> impressions and that shared noise cancels. **This is the concrete payoff of F46's method change.**

*Consequences:*

1. **Prefer 256-d.** Better quality, 34% less memory, marginally faster search. There is no axis on
   which 384-d wins.
2. **128-d is the right choice if memory is tight** — a third of the footprint for no measurable
   loss, which matters at the large tier where vectors reach 514 MB at 1024-d.
3. **Dimension is not a proxy for capacity.** F37 already showed XLM-R's 768-d losing to MiniLM's
   384-d by 348× on the Danish probe; this shows *the same model* improving when narrowed. Both
   point the same way: **the training objective and the noise floor matter, the width does not.**

*Caveat:* n=800, one dataset, one window. The 256-vs-384 gap is significant but small, and the
result should be confirmed at a larger sample before being treated as settled.

### F48 — Multi-query retrieval does not help, and costs 19x the query time
D3 named multi-query (cluster the history, retrieve per centroid, merge) as *considered, not built*
on cost grounds. F40's finding that **46% of MIND users fall back** from the mean made it look worth
revisiting, so it was built and measured. `results/multiquery.json`.

| Dataset | Variant | recall@50 | nDCG@10 | Query time | Paired diff (recall@50) |
|---|---|---|---|---|---|
| MIND | single vector | 0.0103 [0.0043, 0.0169] | 0.2767 | **3.0 s** | — |
| MIND | multi-query, c=3 | 0.0112 [0.0053, 0.0182] | 0.2795 | **57.5 s** | +0.0009 [−0.0055, +0.0077] |
| EB-NeRD | single vector | 0.0037 [0.0000, 0.0088] | 0.4270 | **1.5 s** | — |
| EB-NeRD | multi-query, c=3 | 0.0063 [0.0013, 0.0125] | 0.4260 | **53.9 s** | +0.0025 [−0.0037, +0.0100] |

**Not significant on either dataset or either metric, at 19–36× the query cost.**

*The mechanism did fire.* Cluster counts: MIND `{3: 1518, 1: 16, 2: 30}`, EB-NeRD `{3: 1600}` — so
nearly every user was genuinely split into three interest centroids. This is not a case of the
clustering silently collapsing back to the mean.

> [!important] The surprising part is *how little* it changes
> The two variants differ on only **16/800** impressions on MIND and **8/800** on EB-NeRD at
> recall@50. Even for users with three distinct interest clusters, RRF-merging three rankings lands
> on almost the same top-50 as the single blended vector.
>
> That contradicts D3's stated motivation — *"a user who reads football and recipes gets a centroid
> between the two, matching neither"*. The blended centroid is evidently not as useless as the
> theory assumed: it still ranks each interest's articles highly enough to reach the top-50, because
> **recall@K at K=50 is a forgiving target**. The blur would matter far more for precision at rank 1,
> which is a re-ranker's problem, not candidate generation's.

*Consequence:* **D3's original judgement was right, and is now evidence rather than a guess.** Keep
the conditional single vector. Report multi-query as built, measured, and rejected — which is a
stronger design-note entry than having named it and moved on.

### F49 — HNSW is 60-70x faster at 0.99+ recall: F31's pessimism was a synthetic-data artefact
Re-benchmarked FAISS HNSW against exact search on **clustered** vectors (50 centroids + noise, the
structure real embeddings have) rather than the uniform-random vectors F31 used.

| Corpus | n | exact | HNSW ef=128 | Speed-up | recall vs exact | build |
|---|---|---|---|---|---|---|
| EB-NeRD small | 20,738 | 98.8 ms | **1.6 ms** | **62.7×** | **0.9997** | 0.2 s |
| MIND small | 65,238 | 300.7 ms | **4.4 ms** | **69.1×** | 0.9962 | 1.2 s |
| EB-NeRD large | 125,541 | 580.1 ms | **9.0 ms** | **64.5×** | 0.9875 | 4.0 s |

**This corrects F31, and the correction is large.** F31 measured HNSW losing 11–66% of the exact
answer and I flagged that it "must be re-run on real vectors" — but kept quoting the pessimistic
figure as though it were a genuine trade-off. On clustered vectors the recall loss is
**0.03–1.25%**, not 45%. Uniform-random vectors in 384 dimensions are near-orthogonal, which is the
one input for which a proximity graph has no structure to exploit; it is the worst case, not a
neutral one.

**It also corrects the framing of D4 (Phase 3).** "Brute force is affordable at this scale" is true
*per query* — 98.8 ms is nothing. But the submission run issues **2.37M queries** on MIND and 13.3M
on EB-NeRD. At 62× that is the difference between hours and minutes, which is exactly the cost F32
attacked from the other direction.

*Consequence:* **HNSW should be the default for any full-corpus pass, not just the large tier.**
Brute force keeps one job — it is the *exact* reference the ANN recall is measured against, and
F43's n=800 run confirmed both return identical recall@200 on real data. Keep it as the ceiling,
ship HNSW as the workhorse.

> [!warning] The general lesson, stated twice now
> F31 and F7's random-vector benchmark are the same mistake: **synthetic data made a component look
> broken.** The latency and memory figures from F31 were fine; only the recall number was
> meaningless, and it was the one that would have gone in the report. Any benchmark whose input is
> generated rather than measured needs its distribution justified before its numbers are quoted.

### F50 — At 10x scale HNSW recall drops to 0.83: the concrete "where it breaks" answer
Extending F49's benchmark to a million vectors, clustered, 384-d:

| n | exact | HNSW ef=128 | Speed-up | recall vs exact | build |
|---|---|---|---|---|---|
| 125,541 | 580 ms | 9.0 ms | 64× | 0.9875 | 4.0 s |
| **1,000,000** | **4,603 ms** | **28.7 ms** | **160×** | **0.8332** | 155.9 s |

**Both sides of the trade-off move at 10×, in opposite directions.** Exact search costs **22.8 ms
per query** (the 4,603 ms figure is for a batch of 200), so 2.37M queries would take ~15 hours,
while HNSW's speed advantage grows to 160× but it starts **losing 17% of the true answer** at the
`ef` that was lossless at 125K.

> [!warning] Correction — an earlier version of this finding said "4.6 s per query"
> That was the batch time for 200 queries read as a per-query cost, a 200× error. Exact search at
> 1M vectors is 22.8 ms/query: slow enough to matter over millions of impressions, not slow enough
> to be absurd. The conclusion survives — see F52, which shows the `ef` dial recovers the recall
> cheaply.

*This is the Q6 answer, measured rather than projected:* the pipeline does not break at 10× because
of memory or the model — it breaks because **the ANN index's recall/latency trade-off stops being
free**. The fix is a higher `efSearch` (which costs latency back) or a different index family
(IVF-PQ, which trades memory). Build time also grows super-linearly, 4 s → 156 s.

### F51 — `last_n` inside the 24h window peaks at 20, and is still not significant
F23 swept `last_n` over the *full corpus*, a regime where everything scores under 0.008. Re-run
inside the 24h window — the regime we actually ship. `results/lastn_sweep_ebnerd.json`.

| last_n | recall@50 | nDCG@10 | Paired vs 15 (recall@50) |
|---|---|---|---|
| 5 | 0.2288 [0.2013, 0.2575] | 0.4649 | — |
| 10 | 0.2313 [0.2013, 0.2600] | 0.4644 | — |
| **15** *(shipped)* | 0.2475 [0.2175, 0.2775] | 0.4657 | baseline |
| **20** | **0.2531** [0.2225, 0.2825] | **0.4703** | +0.0056 [−0.0156, +0.0250] **not significant** |
| 30 | 0.2456 [0.2150, 0.2737] | 0.4630 | −0.0019 — not significant |
| 50 | 0.2338 [0.2037, 0.2637] | 0.4543 | −0.0138 — not significant |

**A clear inverted U peaking at 20**, and *nothing* significant against the shipped 15. The shape
matches F23's full-corpus result, so the window does not change the conclusion — which is itself
worth knowing, since F22/F41 showed window and K *do* interact.

*Consequence:* **D4's `last_n` is confirmed as inside the noise in both regimes.** The curve peaks
near 20 while we ship 15 — a candidate for change, not a demonstrated improvement, and the honest
report is the curve plus the non-significance rather than a claim that 20 is better.

### F52 — `efSearch` must scale with the corpus; a fixed 128 loses a third of the answer at 1M
F50 reported HNSW at 1M vectors recovering only 0.83 of the exact answer, which read as a hard limit
of the method. It is not — it is a **default that stops fitting**. Swept `efSearch` at 1M, M=16,
clustered vectors. `results/ann_sweep_1m.json`.

Exact search baseline: **22.8 ms/query**.

| efSearch | recall vs exact | ms/query | vs exact |
|---|---|---|---|
| 64 | 0.5195 | 0.05 | 474× |
| **128** *(old default)* | **0.6782** | 0.08 | 286× |
| 256 | 0.8108 | 0.14 | 167× |
| 512 | 0.9071 | 0.22 | 105× |
| **1024** | **0.9588** | 0.34 | **68×** |

**Recall is bought back almost free.** Going 0.68 → 0.96 costs **0.26 ms/query** and is *still 68×
faster than exact search*. The "17% loss at 10× scale" was a tuning failure, not a ceiling.

*Why a fixed value cannot work:* HNSW explores `efSearch` candidates **regardless of corpus size**,
so the *fraction* of the corpus inspected shrinks as it grows. ef=128 inspects ~0.1% of 125K and
~0.013% of 1M. A value that is lossless at one scale is guaranteed to degrade at ten times it.

*Fix:* `default_ef_search(n)` derives the budget from the corpus — 128 up to 50K, 256 to 200K, 512
to 500K, 1024 above. An explicit value still wins so sweeps can pin what they measure.

> [!important] This changes the Q6 answer
> F50 said the pipeline "breaks at 10× because the ANN trade-off stops being free". With `ef` scaled
> it does **not** break: 0.96 recall at 68× the speed of exact, on a 1.54 GB index built in 105 s.
> The honest 10× statement is now: *the defaults break, the method does not* — and the cost of
> fixing them is a quarter of a millisecond per query.

**Also corrected in F50:** it quoted exact search at "4.6 s per query". That was the batch time for
200 queries; the per-query cost is **22.8 ms**, a 200× error in the pessimistic direction.

### F53 — A denser HNSW graph wins at fixed latency; shipped M=64 / ef=512 = 0.9947 recall at 45x
F52 swept `efSearch` at M=16. Sweeping `M` as well shows the *graph* matters more than the search
budget. 1M clustered vectors, 384-d, exact baseline **22.8 ms/query**. `results/ann_sweep_1m.json`.

| M | ef | recall | ms/query | vs exact | graph mem |
|---|---|---|---|---|---|
| 16 | 1024 | 0.9588 | 0.34 | 68× | 0.13 GB |
| 32 | 512 | 0.9793 | 0.34 | 67× | 0.26 GB |
| **64** | **256** | 0.9759 | **0.31** | **73×** | 0.52 GB |
| **64** | **512** | **0.9947** | 0.51 | **45×** | 0.52 GB |
| 64 | 1024 | 0.9990 | 0.58 | 39× | 0.52 GB |

**Read the frontier at fixed latency.** At ~0.34 ms, M=32 gives 0.9793 where M=16 gives 0.9588. At
~0.31 ms, M=64 gives 0.9759 — better than M=16 managed at 0.34 ms. **More links means fewer dead
ends, so the walk reaches the same recall while exploring fewer candidates.**

*Shipped:* **M=64, `ef` derived from corpus size, 512 at the top end.** That is **0.9947 recall at
45× the speed of exact search** — within half a percent of lossless, far below the noise in every
downstream metric we measure. ef=1024 buys the last 0.43% for 14% more latency and is available when
near-exactness is worth it.

*What it costs:* 0.52 GB of graph against 1.54 GB of vectors, and a 183 s one-off build. Index
memory is not the binding constraint here, which is why the densest graph tested is also the default.

> [!important] The Q6 story is now complete, and it is not "it breaks"
> F50 reported the pipeline breaking at 10× because ANN recall fell to 0.83. F52 showed that was a
> stale `efSearch`; F53 shows a denser graph does better still. **At 1M vectors the shipped
> configuration retrieves 99.5% of the exact answer 45× faster.** The honest 10× statement is:
> *the defaults break, the method does not* — and both dials that fix them were found by sweeping,
> not by reasoning.

### F54 — Where each index actually runs, and why HNSW never reaches a submission
Worth stating explicitly, because it was easy to get wrong from the inside:

| Stage | Index used | Why |
|---|---|---|
| Q3.2 requirement | **HNSW implemented** (`IndexHNSWFlat`, M=64, ef by corpus size) | Q3.2 asks for an ANN index; brute force alone would leave none |
| Q4 harness ablation | **both** — HNSW at ef=64/128/256 *and* exact, side by side | F43's ANN-vs-exact comparison |
| Q3/Q4 default (`auto`) | brute force | All four corpora (20,738 / 65,238 / 120,961 / 125,541) are **below** `BRUTE_FORCE_LIMIT = 200,000` |
| **Q5 submissions** | **neither** | `score_subset()` scores the ~37 slate candidates directly |

**Two independent reasons HNSW cannot change a submitted file.** First, every corpus we have is under
the auto threshold. Second, and more fundamental: the submission format is a *permutation of the
slate*, so only relative order among the candidates shown matters. An article's rank among 120,961
adds nothing to its rank among 37 — which is why `score_subset()` is 16× faster (F32) and why the
index is bypassed entirely.

*Consequence for how the sweeps should be read:* **F49–F53 are Q6 scale analysis, not tuning.** They
corrected a wrong published number (F31's 11–66% recall loss was a random-vector artefact) and
turned the Q6 answer from *"the pipeline breaks at 10×"* into *"the defaults break, the method does
not"*. They moved no leaderboard score, and the log should not imply otherwise.

> [!warning] The cost, recorded honestly
> The M × efSearch sweep at 1M vectors took ~25 minutes of compute for a configuration that touches
> no corpus we have. The right sequence was to establish *"this cannot affect a submission"* first
> and then decide how much evidence Q6 needed — not to measure first and notice afterwards.

### F55 — Only one recent finding changes a submitted file: F47's 256-d truncation
Audited the recent findings against what the submission path actually consumes:

| Finding | Changes a submission? |
|---|---|
| **F47** 256-d beats 384-d (+0.0175 paired, 34% less memory) | **Yes** — the semantic vectors are truncated before scoring |
| F51 `last_n` peaks at 20 | No — the semantic side already used 20, and it was not significant |
| F52/F53 `efSearch` / `M` | No — see F54 |
| F48 multi-query rejected | No — not shipped |
| F42 fusion > BM25 on the leaderboard | Already applied (submission 901779) |

`DEFAULT_TRUNCATE_DIM = 256` is now applied at index time in `SemanticRetriever`, so both the
harness and the submission path use it. Set `truncate_dim=None` to keep the encoder's native width.

### F56 — Semantic scoring is only 17% slower than lexical, not 2.5x
Measured on the submission path (`score_subset`, 300 real MIND slates), which is the only timing
that affects a submission:

| Retriever | ms/slate | slates/s |
|---|---|---|
| BM25 | 0.531 | 1,883 |
| Semantic (MiniLM 256-d) | **0.623** | 1,605 |

**A 17% gap, not the large one implied earlier.** Both reduce to a small arithmetic op over ~37
candidates — a sparse dot product for BM25, a dense one for the embeddings — and at that size the
dense operation is barely more expensive.

*What is actually 2.5x slower is **fusion*** (1,042/s against BM25's 2,540/s in the real submission
runs), because it evaluates **both** retrievers per slate and then merges ranks. That cost is
inherent to fusion, not to semantic retrieval.

*Correction:* earlier commentary attributed fusion's runtime to the semantic side being slow. It is
not; the semantic retriever is competitive with BM25 per slate, and would run at essentially the
same speed as a standalone submission.

### F57 — What BM25 alternatives remain untested, and which is worth it
The brief names BM25 and TF-IDF on the lexical axis. Both are built and compared (U4: 0.0063 vs
0.0075, indistinguishable). The wider lexical family:

| Approach | Status | Assessment |
|---|---|---|
| BM25 | ✅ shipped | the baseline |
| TF-IDF | ✅ tested (U4) | indistinguishable from BM25 here |
| **BM25F** (per-field weights for title vs abstract) | ❌ untested | **The one real gap.** F28 measured abstracts as 75% of the index buying +0.011 (inside the CI), so a title-weighted variant is a genuine hypothesis rather than a guess |
| Query expansion / RM3 | ❌ untested | The classic BM25 upgrade, but adds a pseudo-relevance-feedback round per query |
| SPLADE / learned sparse | ❌ | Requires training — outside Q3's "compute or load" |
| Per-language stemming | rejected (D3) | Different stemmers per dataset would confound the cross-dataset comparison Q3.5 asks for |

*Decision:* **not changing BM25.** The brief names it explicitly, F23 measured its whole parameter
space as worth ~0.01, and the measured lever on this data is the candidate *pool* (F16/F41: 33x),
not the weighting scheme. BM25F is recorded as the strongest untested lexical idea, with the
evidence that motivates it, rather than being tried at the deadline.

### F58 — 256-d transfers to the leaderboard, but at 1/44th the offline effect size
Submission 901876, MIND fusion at 256-d. Screenshot in `report/figs/leaderboard_mind.png`.

| Submission | Config | Leaderboard AUC |
|---|---|---|
| 901650 | bm25, 384-d | 0.5568 |
| 901779 | fusion, 384-d | 0.5934 |
| **901876** | **fusion, 256-d** | **0.5938** |

**+0.0004 over the 384-d fusion.** F47 measured 256-d beating 384-d by **+0.0175 recall@50** offline
(paired, CI excluding zero) on EB-NeRD inside the 24h window. On the MIND leaderboard the gain is
**1/44th of that**.

*Three readings, and honesty requires holding all of them:*

1. **The direction is right.** 256-d did not hurt, which is the claim that matters for shipping it —
   a third less memory for no loss, and nominally a small gain.
2. **The magnitude did not transfer.** Different dataset (EB-NeRD → MIND), different metric
   (recall@50 → AUC), different regime (inside a 24h window → full slate ranking). Any of those
   could explain it; we cannot separate them from one submission.
3. **+0.0004 is almost certainly inside the leaderboard's own noise.** It should be reported as
   *"no measurable change, and cheaper"*, not as a confirmed improvement.

> [!important] The offline/leaderboard relationship is now three-for-three unreliable
> - **F34:** offline said AUC 0.4981 [0.4776, 0.5190]; leaderboard 0.5568 — *outside* the interval.
> - **F42:** offline said fusion was **not significant** (differing on 55/800); leaderboard +0.0366.
> - **F58:** offline said 256-d was **significant at +0.0175**; leaderboard +0.0004.
>
> It has now understated a real effect, missed one entirely, and overstated one. **The offline
> harness at n=4,000 is not a reliable proxy for leaderboard rank in either direction.** It remains
> valuable for what it was built for — catching leakage, exposing the recency effect, slicing — but
> a retriever or parameter choice is only settled by submitting it.
>
> This is the strongest methodological finding in the project, and it took three leaderboard
> submissions to establish.

## Findings

Numbered so they can be cited from the phase files and the design note.

### F1 — MIND and EB-NeRD have incompatible history representations
MIND stores click history **inline** in `behaviors.tsv` col 4 as space-separated ids, with **no
timestamps**. EB-NeRD stores it in a **separate parquet** with parallel `*_fixed` list columns
including `impression_time_fixed`. *Consequence:* the unified schema must make timestamps optional,
and history truncation is provable on EB-NeRD but only assumed on MIND. → [[1-Data-Pipeline]] D2.

### F2 — MIND ships two different `news.tsv` files
51,282 articles in train, 42,416 in dev, and they differ. *Consequence:* the corpus must be the
union, or dev-only articles are unretrievable and recall is capped below 1.0 for reasons unrelated to
the model. → [[1-Data-Pipeline]] D3.

### F3 — MIND's official split is already temporal, and dev is a single day
Train spans 9–14 Nov 2019; dev is **15 Nov only**. *Consequence:* honouring the official split leaves
**no validation set** — val must be carved from the tail of train regardless of which split strategy
is chosen. → [[1-Data-Pipeline]] D1.

### F4 — A literal 80/10/10 does not fit both datasets
MIND's labelled range is 7 days with a 1-day official test; EB-NeRD's is 14 days split evenly in half.
Forcing 80/10/10 on EB-NeRD means discarding ~40% of its labelled impressions. *Decision:* apply the
**rule** (hold out the official test period; last 10% of the train window becomes val), which yields
≈80/10/10 on MIND and ≈45/5/50 on EB-NeRD. Report actual per-dataset proportions rather than claiming
a uniform ratio. → [[1-Data-Pipeline]] D1.

### F5 — MIND train and dev are largely different users
Only **5,943 of 50,000** users (12%) appear in both. *Consequence:* MIND's dev split tests
generalisation to *new users* more than to *new time*, which is a genuine difference from EB-NeRD and
a design-note observation. Also means cold-start behaviour dominates MIND's dev metrics more than
expected.

### F6 — EB-NeRD ships three post-click fields that are direct leaks
`read_time`, `scroll_percentage`, and especially `next_read_time` / `next_scroll_percentage` are
measured *after* the click, in the same row as the label. *Decision:* exclude explicitly in
`clean.py` with a comment naming why. → [[4-Evaluation-Harness]] D6.

### F7 — 99.5% of EB-NeRD impressions have exactly one click
Measured: 231,731 of 232,887 train impressions are single-click (max 9). *Consequence:* the
"multiple clicks = multiple positives" decision barely moves any number, and MRR ≈ recall@1. Decide
once, state it, move on — it is not worth agonising over.

### F8 — EB-NeRD history is long; MIND-style query construction will not transfer
Mean **160 clicks/user** (min 5, max 1,896). Concatenating all titles gives a query of thousands of
tokens — far outside BM25's design regime, and a bland centroid for the semantic retriever.
*Consequence:* history truncation is not an optimisation, it is required for the method to work at
all. → [[2-Lexical-BM25]] D4, [[3-Semantic-Embeddings]] D3.

### F9 — EB-NeRD has no zero-history users
Minimum history length is **5**. *Consequence:* a "cold-start = fewer than 5 clicks" threshold selects
**nobody** on EB-NeRD. The threshold must be chosen per dataset from the measured distribution.
→ [[4-Evaluation-Harness]] D4.

### F10 — MIND-small has no body text
`news.tsv` has title and abstract only. EB-NeRD has a real `body`. *Consequence:* "title + abstract"
is the only field pair available on both — a second, independent reason to follow Q2.1 over the
brief's intro wording. → [[2-Lexical-BM25]] D2.

### F11 — Neither small tier has a test split, and MIND has no small test set at all
Both official test sets are separate downloads; MIND's only exists at large tier. *Consequence:*
Q5 for MIND means running the small-trained pipeline over the 1.5 GB large test set, and Q5 for
EB-NeRD needs a download not yet started. → [[5-Submission-and-Note]].

### F12 — Local hardware is sufficient; Colab/Kaggle unnecessary
i9-14900HX (24c/32t), 31 GB RAM, RTX 4060 8 GB, torch 2.5.1+cu121. Beats free Colab on CPU and RAM;
loses only on VRAM, which affects one step (the encoder forward pass, fine at batch 32–64 fp16).
*Caveat:* the machine is shared — observed load average 26.5 with 9.5 GB swap in use while another job
ran. Hence resource limits as arguments, not constants. → [[Pipeline]], [[../architecture|arch]] 7b.

### F13 — The provided embedding artifacts cover the **full** corpus, and small is a strict subset
Both artifacts hold vectors for **125,541 articles** — the entire large corpus, not a small-tier
subset. Verified: `small (20,738) ⊆ large (125,541) ⊆ artifacts (125,541)`, **zero missing ids**.

| Artifact | Dim | Rows |
|---|---|---|
| `Ekstra_Bladet_word2vec` | 300 | 125,541 |
| `google_bert_base_multilingual_cased` | 768 | 125,541 |

*Consequence:* the provided-vectors baseline is a plain join on `article_id` — **no subsetting logic,
no coverage gap, and it keeps working unchanged if we later move to the large tier**. Cheaper than
[[3-Semantic-Embeddings]] D1 assumed. **The decision itself does not change**: own embeddings stay
primary, because MIND ships no equivalent and mixing encoders across datasets makes Q3.5
uninterpretable. Only the cost of the baseline row went down.

### F14 — MIND-large test is unlabelled and covers a *later* 7-day week
`MINDlarge_test/behaviors.tsv` col 5 contains **bare article ids with no `-1`/`-0` suffixes** —
confirmed over 20,000 sampled rows, zero labelled. It spans **16–22 Nov 2019**, a clean 7-day period
*after* dev (15 Nov).

| | train | dev | test |
|---|---|---|---|
| Impressions | 2,232,748 | 376,471 | 2,370,727 |
| Users | 711,222 | 255,990 | 702,005 |
| Articles | 101,527 | 72,023 | **120,961** |
| Labels | ✅ | ✅ | ❌ |

*Consequence:* test is leaderboard-only and can never contribute to an offline metric — it has no
ground truth. Its `news.tsv` carries **19% more articles than train**, so submission-time inference
faces a genuine cold-article population that no training data ever saw. Worth a design-note sentence,
and it is the strongest argument for keeping a content-based (rather than purely popularity-based)
retriever in the submission path. → [[5-Submission-and-Note]].

### F15 — Large tiers add users, not time; the bottleneck is the join, not the model
Both datasets keep the *same date windows* at large scale and simply add users.

| | small | large | factor |
|---|---|---|---|
| MIND impressions | 156,965 | 2,232,748 | **14×** |
| EB-NeRD impressions | 232,887 | 12,063,890 | **52×** |
| EB-NeRD users | 15,143 | 788,090 | 52× |
| EB-NeRD articles | 20,738 | 125,541 | **6×** |
| EB-NeRD extracted | ~93 MB | **~3.6 GB** | 39× |

EB-NeRD large's `history.parquet` is **1.24 GB (train) + 1.13 GB (validation)** — larger than
behaviors, because mean history is 160 clicks/user across 788K users.

*Consequence, and this is the Q6 answer:* **encoding does not break at scale** — 125K articles is only
6× small, still minutes on the 4060, and for EB-NeRD no encoding is needed at all thanks to F13.
**Impression-to-history joining is what breaks.** At 12M impressions against 2.3 GB of history, the
`--mem-gb` ceiling stops being advisory and the history must be streamed or chunked rather than
loaded. Say "the join, not the model" in the note — it is a more specific and more defensible answer
than "it gets slower". → [[5-Submission-and-Note]] Q6.

### F16 — Recency dominates everything: a recency-only baseline gets recall@50 = 0.92
Measured on EB-NeRD demo, 500 evaluated impressions, via the walking skeleton.

| Baseline | recall@50 | recall@100 | recall@200 |
|---|---|---|---|
| **Recency only** (K most recently published, user ignored) | **0.9160** | 0.9420 | 0.9580 |
| Popularity (train clicks) | 0.0500 | 0.0500 | 0.0833 |
| Token overlap (lexical stand-in) | 0.0000 | 0.0000 | 0.0000 |
| Hashed bag-of-words (semantic stand-in) | 0.0000 | 0.0000 | 0.0167 |

The cause: **94.3% of clicks are on articles less than 24 hours old** (96.0% within 48h), while
only **1.1% of the corpus (125 of 11,777 articles)** was published in any given prior-24h window.
Median clicked-article age at impression time is **0.1 days**.

*Consequence — this reshapes Phases 2 and 3.* Candidate generation on this dataset is
**primarily a recency-filtering problem and only secondarily a text-matching one**. Retrieving from
the full corpus by text similarity is close to hopeless, because the target is one of ~125 fresh
articles hiding among ~11,650 stale ones that are *more* textually similar to the user's history
(the history is itself mostly older articles).

**What this does not mean:** that BM25 and embeddings are useless. It means they must be applied
**within** a recency-constrained candidate pool, not against the whole corpus. The interesting
question shifts from *"lexical or semantic?"* to *"given the ~125 fresh articles, which does the
user pick?"* — which is exactly the question a re-ranker answers, and it is a strong Q3.5 finding.

**Actions:**
1. Add a **recency-only retriever** as a first-class baseline in Phase 2 D6 — it is currently the
   strongest thing measured and every real retriever must beat it.
2. Add a **publish-time window** to the `Retriever` contract. The `at_time` argument already exists
   in the protocol for exactly this; it is now load-bearing rather than defensive.
3. Report both regimes: full-corpus retrieval **and** recency-windowed retrieval. The gap between
   them is a genuine design-note finding, not a tuning detail.
4. **MIND has no `published_time` (F1/D3)**, so this filter cannot be built there from article
   metadata. First-seen-in-impressions time is the available proxy — needs its own decision, and
   the cross-dataset asymmetry is worth reporting.

> [!warning] Do not let this become an accidental leak
> "Most recently published before *t*" is legitimate — publish time is known at serving time. But
> *"most clicked in the next hour"* is not. The recency baseline must filter on `published_time < t`
> only, never on future engagement. → [[4-Evaluation-Harness]] D6.

### F17 — Corpus truncation is a measurement artefact, caught by the skeleton
The skeleton's first run capped the corpus at "first 5,000 articles by id" and returned recall 0.0000
everywhere. Cause: **372 of 379 clicked articles fell outside the kept slice**, so recall was
structurally capped near zero for a reason unrelated to any retriever.

*Consequence:* any corpus subsetting must **keep every article referenced by an evaluated
impression** and pad with distractors, never truncate arbitrarily. Fixed in `src/skeleton.py`. This
is the same argument [[1-Data-Pipeline]] D3 makes for taking the *union* of MIND's two `news.tsv`
files — and it is now demonstrated rather than argued. Worth one line in the design note as a
methodology point: a recall ceiling imposed by the harness is invisible in the output.

### F18 — EB-NeRD's history files are *already* boundary-partitioned by the authors
Truncation drops **0.0% of clicks** on every EB-NeRD split. That looked like a bug; it is not.
Measured on demo:

| | window |
|---|---|
| train **history** | 2023-04-27 07:00:05 .. **2023-05-18 06:59:51** |
| train **impressions** | **2023-05-18 07:18:10** .. 2023-05-25 06:58:46 |
| validation history ends | 2023-05-25 06:59:54 |
| validation impressions start | 2023-05-25 07:02:58 |

The history window closes **8 minutes before** the train impressions open, and validation ships a
**separate** history file closing 3 minutes before its own impressions. Checked directly: **0 of
3,000** impressions have any history click at or after the impression time.

*Consequence — this changes how the Q9 claim must be worded.* Our truncation is correct and it runs,
but on EB-NeRD it is **enforcing a boundary the authors already enforced**, not repairing a violation.
Reporting "we removed post-boundary clicks" would overstate what happened. The honest claim is:
**"the boundary was verified to hold; the shipped history required no truncation."** That is a
stronger statement than a drop percentage, because it is checkable — and `check_no_leakage` proves
it rather than assuming it.

The truncation code still earns its place: it is what makes the property *verified* rather than
*assumed*, it is what the Q9 mutation test exercises, and it is required the moment we re-draw the
split boundary ourselves (any val cut inside the train window makes history straddle it).

### F19 — MIND's realised split is 61/7/32, not the ~80/10/10 F4 predicted
Built proportions, both datasets, small tier:

| dataset | train | val | test | span |
|---|---|---|---|---|
| **MIND** | 141,265 (**61.4%**) | 15,700 (6.8%) | 73,152 (**31.8%**) | 9–15 Nov 2019 |
| **EB-NeRD** | 209,597 (43.9%) | 23,290 (4.9%) | 244,647 (51.2%) | 18 May – 1 Jun 2023 |

**F4's estimate for MIND was wrong.** It reasoned from *days* (6 train : 1 dev ≈ 86/14) but the
split is over *impressions*, and MIND's single dev day carries 73,152 impressions — far denser than
an average train day (~26K). So test is 32% of the labelled data, not 10%.

*Consequence:* none for correctness — the rule (hold out the official period, carve val from the
train tail) is unchanged and still right. But **the design note must report realised proportions, not
intended ones**, and F4's "≈80/10/10 on MIND" should not be quoted. Supersedes that estimate.
Reinforces F4's actual point: a literal 80/10/10 fits neither dataset, and saying so with measured
numbers is the observation.

### F20 — MIND's article union is 65,238, larger than either split alone
Train `news.tsv` has 51,282 articles, dev has 42,416, and the de-duplicated union is **65,238** — so
the two files share only 28,460 articles and **dev contributes 13,956 that train never saw** (21% of
the corpus). Confirms [[1-Data-Pipeline]] D3: per-split corpora would leave those unretrievable and
cap recall for a reason unrelated to any retriever, exactly the artefact F17 caught by accident.

Also measured: **0% of MIND articles carry a publish time** vs **100% of EB-NeRD's**. This is the
blocker for applying F16's recency filter to MIND, and it is now confirmed on the real store rather
than inferred from the schema.

### F21 — Real BM25 confirms F16 rather than overturning it
Phase 2's `bm25s`-backed retriever, run on the **small** tier through the skeleton harness. Not
deliverable numbers — no CIs, single configuration, exploratory split — but the shape is unambiguous.

**EB-NeRD small** (800 evaluated impressions, 20,738-article corpus):

| Retriever | recall@50 | recall@100 | recall@200 |
|---|---|---|---|
| Recency only (user ignored) | **0.9050** | **0.9463** | **0.9688** |
| BM25 + 24h window | 0.2387 | 0.2387 | 0.2387 |
| Popularity (train clicks) | 0.0063 | 0.0100 | 0.0262 |
| BM25, full corpus | 0.0063 | 0.0112 | 0.0200 |

**MIND small** (782 evaluated impressions, 65,238-article corpus):

| Retriever | recall@50 | recall@100 | recall@200 |
|---|---|---|---|
| Popularity (train clicks) | **0.0652** | **0.0934** | **0.1918** |
| BM25, full corpus | 0.0128 | 0.0179 | 0.0243 |

Three things this settles:

1. **Real BM25 ≈ the token-overlap stand-in on the full corpus** (0.006 vs 0.000 on EB-NeRD). The
   skeleton predicted replacing it "should visibly improve recall"; it did not. F16's diagnosis was
   right and the stand-in was not the limiting factor — **the full-corpus regime is**.
2. **The 24h window is worth ~38× on EB-NeRD** (0.0063 → 0.2387 at K=50), reproducing F16's
   0.00 → 0.28 with a real scorer. Recency filtering, not the scoring function, is what moves recall.
3. **BM25 loses to popularity on MIND at every K** — and on MIND there is no recency filter available
   (F20: 0% publish times), so the rescue that works on EB-NeRD is unavailable there. This is the
   sharpest cross-dataset asymmetry measured so far.

### F22 — The recency window caps recall@K, making K irrelevant above the window size
The EB-NeRD windowed row above is **identical at K = 50, 100 and 200** (0.2387). Not a bug in K
handling — the plan's pitfall table lists `recall@200 < recall@50` as impossible, and this is the
adjacent case: recall *flat* in K.

Measured cause: a 24h window over this corpus admits **~132 of 20,738 articles (0.6%)**, and after
BM25 scoring the returned lists average **7.0 results at K=50 and 27.0 at K=200** — far below K. The
window, not K, is the binding constraint. Widening it restores monotonicity and confirms the
mechanism:

| Window | recall@50 | recall@100 | recall@200 |
|---|---|---|---|
| 24h | 0.2387 | 0.2387 | 0.2387 |
| 72h | 0.1462 | 0.2400 | 0.2412 |

Note 72h is *worse* at K=50 — a wider pool admits more stale-but-similar distractors that outrank the
fresh target, which is F16's mechanism seen from the other side.

*Consequence for Phase 4:* **the window size and K interact, so they must be swept together.**
Reporting recall@K at a single window is reporting one cell of a 2-D surface. Also worth a design-note
sentence: a retriever that returns fewer than K results makes recall@K and recall@(K+n) identical,
which looks like a bug and is not — the harness should log realised list length alongside recall.

### F23 — The k1/b sweep confirms D5: the window dominates, the BM25 knobs barely matter
Full grid `k1 ∈ {0.9, 1.2, 1.6} × b ∈ {0.3, 0.75, 1.0} × last_n ∈ {5, 15, 50} × window ∈ {24h, none}`,
run on **val only**, 800 evaluated impressions. 54 cells on EB-NeRD in **44 s** across 12 workers;
27 cells on MIND in **31 s** (no window arm — F20). Raw output in `results/bm25_sweep_*.json`.

**The effect sizes are ordered, and the ordering is the finding:**

| Factor | Range across the sweep (EB-NeRD, recall@100) | Verdict |
|---|---|---|
| **Recency window** (24h vs none) | 0.0075 → 0.2475 — **33×** | Dominates everything |
| `last_n` (5/15/50) | 0.2162 → 0.2475 within the 24h arm | Modest; 15 is the peak, 50 is worst |
| `b` (0.3/0.75/1.0) | 0.2313 → 0.2475 | Small, and in D5's predicted direction |
| `k1` (0.9/1.2/1.6) | 0.2362 → 0.2475 | **Smallest — as D5 predicted** |

D5 predicted `k1` "barely matters — titles+abstracts rarely repeat a term" and that `b` would matter
more because empty abstracts make lengths bimodal (measured: **8.2% of EB-NeRD articles**). Both held.
The honest summary is that **tuning BM25 is nearly pointless here compared to deciding what pool it
searches** — which is a better design-note sentence than any parameter table.

**Chosen on val, before test is touched** (the D5 discipline):

| Dataset | k1 | b | last_n | window |
|---|---|---|---|---|
| EB-NeRD | 1.6 | 1.0 | 15 | 24h |
| MIND | 1.6 | 0.75 | 5 | none (unavailable) |

> [!warning] These margins are inside the noise — do not over-claim them
> Top-to-bottom spread within the EB-NeRD 24h arm is 0.2162–0.2475 on 800 impressions. No CIs have
> been computed yet, and a 0.03 gap at n=800 is very plausibly noise. **The parameter choice is
> defensible as a procedure, not as a finding.** Phase 4 must re-run the top few cells with bootstrap
> CIs before any of this appears in the note.

Note `last_n=5` winning on MIND while `last_n=15` wins on EB-NeRD is consistent with F8 (EB-NeRD
histories are ~5× longer), but at these margins it is a hypothesis, not a result.

### F24 — Coverage has no defensible bootstrap CI, and the first attempt produced an impossible one
D5 requires a bootstrap CI on **every** number. Coverage cannot have one, and the failure was caught
by the harness's own output rather than by reasoning.

**The symptom:** BM25's coverage came out as `0.3933 [0.3432, 0.3671]` — the point estimate **outside
its own confidence interval**, which is impossible for a percentile bootstrap.

**The cause:** coverage counts *distinct* articles across the result set, so it is monotonically
increasing in the number of impressions evaluated. Resampling n units with replacement makes ~37% of
draws duplicates, so every resample sees fewer unique articles than the full sample. The entire
bootstrap distribution sits below the point estimate by construction.

Measured, on 400 synthetic impressions over a 20,738-article corpus (full-sample coverage 0.9790):

| Scheme | Result |
|---|---|
| Resample n **with** replacement | 0.9783 vs CI [0.9035, 0.9235] — **excludes the point** |
| Subsample m/n = 0.50 without replacement | mean 0.8532, CI [0.8502, 0.8564] — still excludes |
| m/n = 0.80 | mean 0.9540 — still excludes |
| m/n = 0.90 | mean 0.9688 — still excludes |
| m/n = 0.95 | mean 0.9745 — still excludes |
| m/n = 0.99 | mean 0.9782 — still excludes |
| m = n without replacement | the original sample: zero variance, zero-width interval |

**No ratio works.** The bias vanishes only as m → n, at which point there is no resampling left.

*Decision:* **coverage is reported as a point estimate with explicit `NaN` bounds and a `(no CI)`
marker in the table.** Emitting NaN rather than omitting the row makes the absence visible instead of
looking like a formatting loss. `src/eval/bootstrap.py:point_only()` carries the full reasoning, and a
test asserts the NaN bounds so nobody "fixes" it back into a manufactured interval.

> [!important] This is a design-note paragraph, not a footnote
> D5 says "bootstrap CI for each metric" and the honest answer is that one metric cannot have one.
> Saying so — with the measurements above — is a stronger answer than shipping a plausible interval
> that is biased by construction. Coverage remains comparable **across retrievers on the same
> impression sample**, which is how the harness reports it.

### F25 — The first CI-bearing numbers: BM25 beats nothing on either dataset
Full harness, small tier, 800 evaluated impressions per dataset (from a 4,000-impression val slice),
B = 1000 seeded bootstrap, BM25 at the F23 val-chosen parameters. `results/eval_*.json`.

**EB-NeRD — candidate generation (corpus regime):**

| Retriever | recall@50 | recall@100 | recall@200 |
|---|---|---|---|
| **Recency** (ignores the user) | **0.9050 [0.8850, 0.9237]** | **0.9463 [0.9300, 0.9600]** | **0.9688 [0.9562, 0.9800]** |
| BM25 + 24h window | 0.2475 [0.2175, 0.2775] | 0.2475 [0.2175, 0.2775] | 0.2475 [0.2175, 0.2775] |
| Popularity | 0.0063 [0.0013, 0.0125] | 0.0100 [0.0037, 0.0163] | 0.0262 [0.0150, 0.0375] |
| BM25, full corpus | 0.0075 [0.0025, 0.0138] | 0.0125 [0.0050, 0.0200] | 0.0200 [0.0100, 0.0300] |
| Random | 0.0000 | 0.0000 | 0.0000 |

**MIND — candidate generation:**

| Retriever | recall@50 | recall@100 | recall@200 |
|---|---|---|---|
| **Popularity** | **0.0468 [0.0340, 0.0609]** | **0.0682 [0.0531, 0.0861]** | **0.1415 [0.1187, 0.1649]** |
| BM25 | 0.0116 [0.0054, 0.0189] | 0.0159 [0.0086, 0.0240] | 0.0217 [0.0136, 0.0310] |
| Random | 0.0004 [0.0000, 0.0013] | 0.0007 [0.0000, 0.0017] | 0.0007 [0.0000, 0.0017] |

With CIs attached, F21's claims sharpen into non-overlapping statements:

1. **BM25's full-corpus CI overlaps popularity's on EB-NeRD** — they are statistically
   indistinguishable. The 24h window is the only thing that separates BM25 from the floor, and its CI
   is far clear of both.
2. **On MIND, popularity beats BM25 with non-overlapping CIs at every K** — and MIND has no publish
   times (F20), so the window that rescues EB-NeRD is unavailable. This is now a measured claim, not
   an observation.
3. **Random is not quite zero on MIND** (0.0004) but is exactly zero on EB-NeRD, because EB-NeRD's
   corpus is 3× smaller *and* its clicks concentrate on ~0.6% of articles.

**The slate regime tells a completely different and much duller story:**

| EB-NeRD | AUC | nDCG@10 |
|---|---|---|
| BM25 + 24h | 0.5243 [0.5002, 0.5467] | 0.4657 [0.4465, 0.4861] |
| Recency | 0.5047 [0.4841, 0.5240] | 0.4281 [0.4095, 0.4487] |
| Random | 0.4810 [0.4591, 0.5022] | 0.4191 [0.4019, 0.4373] |

**Every AUC is within noise of 0.5, and random scores nDCG@10 = 0.42.** That is not a bug: EB-NeRD
slates average 11–12 items with exactly one click (F7), so nDCG@10 covers nearly the whole slate and
even a random permutation puts the single positive in a decent position often enough. The plan
predicted this ("nDCG@10 ≈ nDCG@5 exactly — slates are ~11 items; expected, not a bug").

*Consequence:* **recall@K in the corpus regime is the only metric currently discriminating between
retrievers.** The slate metrics are what Codabench scores, so they cannot be dropped — but reporting
them without the "random gets 0.42" reference row would badly mislead. Every slate table must carry
the random baseline.

### F26 — Beyond-accuracy exposes the popularity-collapse trade-off cleanly
Same run. The D3 prediction that these oppose accuracy is visible in one table:

| EB-NeRD retriever | diversity | novelty | coverage |
|---|---|---|---|
| BM25, full corpus | 0.6979 [0.6907, 0.7046] | 12.61 [12.60, 12.61] | **0.5037** (no CI) |
| BM25 + 24h | 0.8080 [0.8056, 0.8104] | 11.67 [11.63, 11.71] | 0.0110 (no CI) |
| Recency | 0.8313 [0.8308, 0.8319] | 11.30 [11.27, 11.33] | 0.0147 (no CI) |
| Popularity | 0.7924 (zero width) | **8.64** (zero width) | 0.0096 (no CI) |
| Random | 0.8650 (zero width) | 12.71 (zero width) | 0.0096 (no CI) |

Three things worth the design note:

- **Popularity's novelty (8.64) is 4 bits below everything else** — it recommends exactly the
  articles everyone already clicked, which is the definition of un-novel. Its zero-width CI is
  correct, not a bug: it returns the *same list to every user*, so there is no between-impression
  variation to bootstrap.
- **The recency filter costs 46× coverage** (0.5037 → 0.0110) while buying 33× recall. That is the
  sharpest accuracy-vs-coverage trade-off measured, and it is exactly the D3 exhibit.
- **Random has the highest diversity (0.8650)**, confirming diversity alone is not a quality signal —
  it must always be read against an accuracy column.

## Decisions taken

| # | Decision | Rationale | Where |
|---|---|---|---|
| 1 | Small tier only for now | Comparability; brief names it; fast iteration. Large stays idle until small works end to end | [[Pipeline]] |
| 2 | Own embeddings, provided as baseline | MIND ships no provided vectors — mixing sources makes cross-dataset claims uninterpretable | [[3-Semantic-Embeddings]] D1 |
| 3 | Hybrid temporal split (official test, val from train tail) | Leaderboard-faithful *and* gives a real val split on both | [[1-Data-Pipeline]] D1 |
| 4 | Brute force for headline, HNSW as ablation | Q3.2 sanctions it; exact ceiling + the gap answers Q6 with measurements | [[3-Semantic-Embeddings]] D4 |
| 5 | Title + abstract for the index | Q2.1 mandates; only pair available on both datasets | [[2-Lexical-BM25]] D2 |
| 6 | Parquet + Polars | List columns survive round-trip; multithreaded within the memory ceiling | [[1-Data-Pipeline]] D4, D5 |
| 7 | Resource limits as CLI/config args (26/26 default) | Shared machine; must be tunable and logged with results | [[Pipeline]] |
| 8 | SBERT-family encoder, not vanilla BERT | `[CLS]` from an MLM-trained model is a poor similarity vector | [[3-Semantic-Embeddings]] D2 |
| 9 | No LLM in the data path | Non-deterministic under one-command rebuild; corrupts the corpus BM25 scores against | [[../architecture\|arch]] 9 |

## Open — needs a decision before or during the phase

| Question | Blocks | Lean |
|---|---|---|
| Cold-start threshold per dataset | Q4 slices | Bottom quartile of history length, measured per dataset |
| MIND recency weighting under no timestamps | Q3 user vectors | Plain mean on both for the headline; positional decay as a MIND-only ablation |
| Empty MIND abstracts — drop or title-only | Q1 cleaning | Title-only; log how many |
| Does EB-NeRD `body` contain HTML | Q1 cleaning | Check on first extract |
| MIND `entity_embedding.vec` — use? | Q3 ablation | MIND-only ablation, never in the headline comparison |

## Task queue

Next action is the top unchecked item.

- [x] Verify all raw archives without extracting
- [x] Measure both datasets' schemas, row counts, date ranges, distributions
- [x] Write the plan set (Pipeline + phases 1–5)
- [x] Measure the large tiers and the embedding artifacts (F13–F15)
- [x] Phase 1 step 1–2: extract + per-dataset readers
- [x] Phase 1 step 3: unified schema (`src/data/clean.py`)
- [x] Phase 1 step 4–5: temporal split + history truncation (`src/data/split.py`)
- [x] Phase 1 step 7: leakage test, with the mutation test that gives it teeth
- [ ] **Review the built store, then start Phase 2** ← *current*
- [ ] Confirm C2 pair declaration status (deadline passed)
- [ ] Phase 1 step 6: feature-store polish — article/user feature columns beyond
      what retrieval needs (Q1.4 reading of "small, reusable feature store")
- [ ] Phase 2: BM25 + baselines
- [ ] Phase 3: embeddings + ANN
- [ ] Phase 4: harness
- [ ] Phase 5: submissions + note

## Log

### 2026-08-25 — Phase 2 + Phase 4 built; decision round; docs split
**Built:** `src/retrieval/` (BM25 via `bm25s`, own reference implementation, tokeniser, parallel
sweep runner) and `src/eval/` (metrics in both regimes, bootstrap, slices, harness, runner).
69 tests pass. Findings **F21–F26**.

**Docs restructured.** `architecture.md` had grown to ~1180 lines mixing *what the system is* with
*what we debated*. Split:

| File | Holds | Status |
|---|---|---|
| `architecture.md` | current architecture, the machinery, measured dataset facts | Part D marked **superseded** |
| **`decisions.md`** (new) | brief options + combinational effects, decisions with rejected alternatives, open questions, drawbacks | **The Q6 design-note source** |
| `execution_plan_log.md` (this file) | measurements and dated changes | Also the **architecture changelog** — no separate one |

Decided: architecture changes are logged here as dated entries rather than in a changelog inside
`architecture.md`, so there is one chronology, not two.

**Decisions taken this round** (full reasoning + costs in `decisions.md`):

| ID | Decision | Notable because |
|---|---|---|
| D-SPLIT | Keep the existing rule → 7 days test on EB-NeRD, 1 day on MIND | "Last 7 days" is **impossible on MIND** — see F27 |
| D-LEX-QUERY | `last_n = 20`, logarithmic position decay | Mild decay by request; positional not temporal on MIND (F1) |
| D-LEX-FIELDS | Title + abstract kept, title-only as a measured ablation | F28 — abstracts are 75% of the index for an effect inside the CI |
| D-COLD | Cold start `< 7` clicks; rising+popular fallback, `α = 0.7`, train-only | The fallback flag **is** the Q9 with/without pair |
| D-ENC | Own `xlm-roberta-base` (768) primary; large (1024) + MiniLM (384) ablations | Provided mBERT is **768**, so same-dim head-to-head |
| D-POOL | Conditional: coherence ≥ τ → mean; else recent-half mean, else max-pool | Avoids the meaningless centroid |
| D-ANN | Brute force on demo/small (exact ceiling), FAISS HNSW on large | ScaNN rejected; **HNSW is CPU** — GPU is for the encoder |
| D-BUDGET | **28 cores / 28 GB, all-inclusive** (was 26/26 per-stage) | Applied to `src/resources.py` and the Makefile |
| D-STORE | No change — the store is already parquet/polars-native | Vectors will join as `vectors.parquet` on `article_id` |
| D-C2 | Define the `Reranker` seam now, ship only an identity implementation | Building a real ranker is C-2 and costs C-1's marks |

### F27 — A 7-day test split is physically impossible on MIND
Measured from the built store manifests:

| Dataset | Total labelled span | Realised test |
|---|---|---|
| MIND | **6.0 days** (2019-11-09 00:00 → 11-15 23:58) | 1 day |
| EB-NeRD | 14.0 days (2023-05-18 07:00 → 06-01 06:59) | **7 days** |

MIND's *entire* labelled range is under seven days, so a 7-day test window would leave no training
data. *Consequence:* the split **rule** stays constant across datasets (hold out the official test
period, carve val from the train tail) and the **realised spans differ** because the datasets do —
which is F4's argument, now with the specific numbers that make it unarguable. Report realised spans
per dataset, never a single "N days" claim.

Secondary consequence worth a note sentence: MIND's test is a *single day* of news, so one day's
topical anomaly moves every MIND number. EB-NeRD's 7-day test averages over a week.

### F28 — Abstracts are 75% of the lexical index and buy an effect inside the CI
Q2.1 mandates indexing title + abstract. Measured what that actually contributes.

**Index composition:**

| Corpus | Title tokens (mean) | Abstract tokens (mean) | Abstract share of indexed tokens | Empty abstracts |
|---|---|---|---|---|
| MIND | 11.2 | 36.1 | **76.3%** | 3,415 (5.2%) |
| EB-NeRD | 6.8 | 18.3 | **72.8%** | 1,709 (8.2%) |

**Contribution** (EB-NeRD, BM25 k1=1.6 b=1.0 last_n=15, 24h window, n=800):

| Indexed text | recall@50 |
|---|---|
| Title + abstract | 0.2475 |
| Title only | 0.2362 |

**+0.011, against a CI half-width of ~0.030 (F25) — statistically indistinguishable.** Three quarters
of the index buys an effect too small to detect at this sample size.

*Decision:* keep title+abstract (Q2.1 is binding) and **report title-only as an ablation row**. A null
result reported *as* a null result is a finding: "the mandated field pair is not measurably better
than titles alone, despite being 75% of the index." The MIND title-only arm timed out and is still to
be run — do not report a MIND figure until it is.

### F29 — Session context exists only on EB-NeRD, and is thinner than expected
From the built store:

| Dataset | Rows | Non-null `session_id` | Unique sessions | Mean impressions/session |
|---|---|---|---|---|
| MIND | 141,265 | **0 (0.0%)** | — | — |
| EB-NeRD | 209,597 | 209,597 (100%) | 108,976 | **1.9** (max 24) |

Two consequences. First, session context is **structurally unavailable on MIND** — it is one of the
brief's three named behavioural signals and it cannot enter any cross-dataset claim. Second, even on
EB-NeRD a "session" averages **1.9 impressions**, so session context means roughly *one prior
impression* — a much thinner signal than the phrase suggests.

*Decision:* **defer to C-2** (open question O6). Building it is a day of work for a signal that may be
undetectable, and C-2 is explicitly click-log modelling, so it lands there naturally. Named as
considered-and-deferred rather than overlooked.

### F30 — `faiss-cpu` 1.15.0 is broken on this stack; pin lower
`import faiss` raises `NameError: name 'SuperKMeans' is not defined` from faiss's own
`class_wrappers.py` — a packaging defect in 1.15.0, not a usage error. Pinned to **1.11.0**.

Also settled while choosing: **`faiss-gpu` is not on PyPI** for current versions (conda-only and
CUDA-version-sensitive), so `faiss-cpu` is the practical choice. This costs nothing — **HNSW graph
search is CPU-bound by design**, and the 28-core budget suits it better than 8 GB of VRAM. The RTX
4060 earns its place on the *encoder forward pass*, which is the genuinely VRAM-bound step. Worth
stating explicitly so nobody assumes the GPU accelerates retrieval.

### F31 — HNSW is fast but loses 11–66% of the exact answer; the ANN-vs-exact gap must be reported
Benchmarked FAISS 1.11.0, `IndexHNSWFlat(M=32, efConstruction=200)`, inner product on L2-normalised
vectors, 28 threads, 200 queries, measured against `IndexFlatIP` as ground truth. **Random vectors**
— see the caveat below.

| Corpus | n | d | exact | HNSW ef=128 | speedup | **recall@200 vs exact** |
|---|---|---|---|---|---|---|
| EB-NeRD small | 20,738 | 768 | 190 ms | 14.8 ms | 12.8× | **0.7700** |
| MIND small | 65,238 | 768 | 888 ms | 36.3 ms | 24.5× | **0.5778** |
| EB-NeRD large | 125,541 | 1024 | 1,677 ms | 60.1 ms | 27.9× | **0.4461** |

`efSearch` trades recall for latency, and the trade is steep:

| Corpus | ef=64 | ef=128 | ef=256 |
|---|---|---|---|
| EB-NeRD small | 0.6197 | 0.7700 | 0.8890 |
| MIND small | 0.4469 | 0.5778 | 0.7178 |
| EB-NeRD large | 0.3402 | 0.4461 | 0.5777 |

Build cost is not free either: 4.1 s / 23.3 s / **74.4 s**, and 514 MB of vectors at the large tier.

> [!warning] These numbers are a worst case — the benchmark used *random* vectors
> Uniform-random vectors in 768–1024 dimensions are close to mutually orthogonal, which is the
> hardest possible case for a proximity graph: there is no cluster structure for HNSW to exploit.
> **Real embeddings are strongly clustered, so measured recall on actual article vectors should be
> substantially higher.** This must be re-run on the real vectors before any of it is reported —
> quoting 0.4461 as *our* ANN recall would be wrong in the pessimistic direction.
>
> What the benchmark does establish regardless of vector distribution: the **latency** figures, the
> **build times**, the **memory**, and the *shape* of the `efSearch` trade-off.

*Consequences:*

1. **The D-ANN decision is confirmed, and for a sharper reason than convenience.** Brute force on
   demo/small is not merely permitted by Q3.2 — at 190 ms/query on EB-NeRD small it is perfectly
   affordable, and it is *exact*, so it is the only defensible source of a headline recall number.
2. **Never report ANN recall without the exact baseline beside it.** An `efSearch` left at a low
   default silently caps recall, and the result looks like a weak encoder rather than an
   under-tuned index. This is the Q6 "where it breaks at 10×" story with real numbers: recall falls
   from 0.77 → 0.45 as the corpus grows 6× and the dimension grows 1.33×.
3. **`efSearch` must be swept and reported**, not left at a default (open question O2).

### F32 — Full-corpus retrieval is the wrong algorithm for the submission path (16× fix)
The first submission run produced predictions at **162 impressions/s** on MIND-large test — about
**4 hours** for one file. Cause: for every impression it ran a top-500 retrieval over all 120,961
articles, then discarded everything outside that impression's ~37-item slate.

The ranking only ever needs the slate scored. Three approaches, measured on the real MIND-large
corpus:

| Approach | Rate | Verdict |
|---|---|---|
| `retrieve(k=500)` over the full corpus | **162/s** | 4 hours per file |
| `bm25s` `weight_mask` restricted to the slate | **173/s** | Numerically exact, but the mask filters *selection*, not the scan — no real gain |
| **Doc-major scoring of the slate only** | **~2,560/s** | **16× faster**; ~15 min per file |

*Decision:* `BM25Retriever.score_subset()` scores the slate directly, doc-major, and the submission
path uses it. `retrieve()` is unchanged and remains the source of every measured metric.

> [!warning] `score_subset()` and `retrieve()` do NOT agree numerically — and that is deliberate
> `retrieve()` delegates to `bm25s` (Lucene IDF variant); `score_subset()` uses the textbook
> Robertson formula. Reproducing `bm25s`'s exact scores was attempted and abandoned: a careful
> replication of its documented formula still disagreed by up to **106 in absolute score** on the
> real corpus, so the library is doing something further not worth reverse-engineering here.
>
> **What was verified instead: the two produce equivalent rankings.** Measured 0 discordant pairs out
> of 780 on the toy corpus — every apparent list-order difference is a score *tie* broken differently.
> A regression test asserts zero discordant pairs rather than list equality, because a tie is not a
> disagreement.
>
> This is acceptable **only** because the submission format is a permutation: nothing but the ordering
> is ever written. `score_subset()` must never be used for a reported metric.

### F33 — Both large test sets are now extractable; EB-NeRD's arrived separately
`ebnerd_large.zip` was verified to contain **train/ and validation/ only — no test member**, so the
EB-NeRD leaderboard submission was blocked on `ebnerd_testset.zip`, a separate download. It has since
been provided and staged at `data/work/ebnerd/testset/` (1.8 GB: `articles.parquet` + `test/`).

Added a **`large` tier** to `src/data/extract.py` (`mind/large_test`, `ebnerd/large`) and a `test`
entry to `SPLIT_NAMES` for both datasets, marked in-code as the unlabelled leaderboard split that can
never contribute an offline metric (F14).

> [!warning] The submission corpus must be the test split's own article file
> MIND-large test ships **120,961 articles** against small-train's 51,282 (F14). Indexing train's
> corpus would leave most test candidates unscored, so they would sink to the bottom of every slate —
> the same recall-ceiling trap as F17 and D-CORPUS, in the submission path rather than the eval path.
> `codabench.py` therefore requests `articles(splits=(test_split,))`.

### F34 — First leaderboard score: MIND AUC 0.5568, and it disagrees with our offline number
Submitted BM25 (`k1=1.6, b=0.75, n=5`) to the MIND Codabench leaderboard, submission 901650.

| Metric | Leaderboard (n = 2,370,727) | Our offline val (n = 800) |
|---|---|---|
| **AUC** | **0.5568** | 0.4981 [0.4776, 0.5190] |
| MRR | 0.2646 | 0.2467 [0.2278, 0.2680] |
| nDCG@5 | 0.2778 | 0.2172 [0.1959, 0.2392] |
| nDCG@10 | 0.3331 | 0.2712 [0.2505, 0.2933] |

Rank 75. Peers visible on the same page span AUC 0.5571–0.6544.

**The disagreement is the finding.** Our offline harness said BM25's AUC was indistinguishable from
chance on MIND; the leaderboard says it is clearly above it. Three candidate explanations, in
descending order of likelihood:

1. **Sample size.** 800 impressions against 2.37M. Our CI half-width was ±0.021, and the gap is
   0.059 — larger than the CI, so sampling noise alone does not explain it, but a 2,963× larger
   sample is far more trustworthy.
2. **Different data.** Our val is one day carved from MIND-small train (Nov 14); the leaderboard is
   the large test week (Nov 16–22) with a different corpus (120,961 vs 65,238 articles) and
   different users.
3. **Different slate composition.** The large test set averages 39.3 candidates per impression
   against small's 37.2, and slate size directly changes AUC's denominator.

*Consequence:* **the offline harness is under-powered at n = 800 and should not be the sole basis
for a retriever decision.** Raise the evaluated slice before comparing lexical against semantic —
otherwise Q3.5's headline comparison inherits the same weakness. This is the most actionable
methodological finding so far.

*Also settled:* the submission path is validated end to end — format, rank alignment, memory,
scorer compatibility — by a real score rather than by our own assertions.

### F35 — The submission failed once on a filename, not on content
First upload was rejected:

```
FileNotFoundError: '/app/input/res/prediction.txt'
```

The scorer opens that path literally; our archive contained `mind_prediction.txt`. **The 2,370,727
predictions were correct** — only the member name inside the zip was wrong. Repacking with
`arcname="prediction.txt"` scored on the next attempt.

*Consequence:* `SUBMISSION_MEMBER = "prediction.txt"` is now a named constant, and
`tests/test_submit.py` asserts both the constant and the built artefact. A wrong filename costs a
submission from the daily quota of 10 plus ~15 minutes of regeneration, which is a disproportionate
price for a string.

### F36 — Worker-local int arrays cut memory 39% and *raised* throughput 45%
The parallel submission path was unusable at EB-NeRD test scale: one worker peaked at **15.08 GB**
against 20.1 GB free, so exactly one fitted. Measured breakdown:

| Component | Cumulative RSS |
|---|---|
| Articles (125,541) | 1.34 GB |
| + BM25 index | 1.82 GB |
| + histories (807,677 users, **116,825,984 clicks**) | **15.08 GB** |

**88% of the footprint was click ids held as Python `str`** — roughly 57 bytes of interpreter
overhead apiece for a value parquet stores as int32.

`CompactHistories` stores them as `array('i')` indices into a shared article-text table, built
inside `_init_worker` only:

| | Before | After |
|---|---|---|
| Per-worker RSS | 15.08 GB | **9.25 GB** (−39%) |
| Throughput | ~2,540/s | **3,681/s** (+45%) |
| Serial estimate | ~90 min | **~61 min** |
| Workers fitting in 23.1 GB | 1 | **2** (→ ~30 min) |

The speed-up was not the goal and is worth noting: indexing a contiguous list by integer beats
hashing strings through a dict.

> [!warning] The saving is smaller than predicted, and the reason matters
> ~2.5 GB was projected; 9.25 GB was measured. The residual is the per-user
> `impression_time_fixed` lists — 116.8M Python `datetime` objects, which the int-array change did
> not touch. Converting those to epoch ints would likely reach the original projection, but it was
> not done: the truncation comparison `t < cutoff` is the leakage boundary, and changing the type
> on both sides of that comparison is a bigger risk than the remaining GB is worth.

**Why worker-local rather than a schema change.** `History.clicked_ids` is also read by
`src/eval/harness.py` and by `tests/test_no_leakage.py` — `History.before()` *is* the Q9 boundary.
Changing its element type would put a memory optimisation on Q9's correctness surface for no
benefit, since only the submission path is memory-bound. Here the ids never escape the loop.

**The duplicated truncation is pinned to the original.**
`test_submit.py::test_compact_histories_match_history_before` asserts `texts_before()` agrees with
`History.before()` across six cutoffs, plus the no-timestamp (MIND) case and the
click-outside-corpus case. Without that test this optimisation would not be defensible.

### F37 — XLM-RoBERTa fails the Danish probe outright: it cannot separate related from unrelated
The anisotropy risk named in D-ENC before any code was written is **confirmed, and it is severe**.

`danish_probe()` embeds 5 obviously-related Danish headline pairs and 5 obviously-unrelated ones,
then compares mean cosine similarity:

| Encoder | Related | Unrelated | Margin | Verdict |
|---|---|---|---|---|
| `xlm-roberta-base` (768-d) | **0.9972** | **0.9954** | **+0.0018** | **OVERLAPS** |

Every pair scores ~0.996 regardless of content. "Brøndby beat FCK" and "a new apple cake recipe"
are as similar to each other as two reports of the same football match. **The representations are
collapsed into a cone so narrow that the encoder carries no usable retrieval geometry for Danish.**

*Why this matters more than a bad number:* a retriever built on these vectors would return
effectively arbitrary articles **while producing perfectly plausible metrics**. There would be no
crash, no warning, and the natural misreading is "semantic retrieval does not work on news" rather
than "the encoder was never trained to make cosine similarity mean anything". This is precisely the
silent failure the probe exists to catch, and it justifies gating Q3 on it.

*What it does not mean:* XLM-R is not a bad model — it is a masked-language model, trained to
predict hidden tokens, not to place similar texts near each other. Mean-pooling and L2-normalising
(both applied) mitigate but do not fix that; the geometry is absent, not merely mis-scaled.

**Consequence for D-ENC.** The brief names BERT and XLM-RoBERTa, so `xlm-roberta-base` remains the
brief-sanctioned primary *and is now reported as a measured failure* rather than an assumption —
which is a stronger design-note paragraph than quietly picking a different model. The
similarity-trained MiniLM row, kept in the ladder as a control precisely for this, becomes the
working semantic retriever if it passes the same probe.

> [!important] This turns a design disagreement into a measured finding
> The plan (D2) argued from theory that MLM-trained encoders would underperform for retrieval and
> chose SBERT; the user chose XLM-R because the brief names it. **Both were right to hold their
> position, and the probe settles it with a number.** Report both rows: it is the clearest available
> demonstration of *why* an encoder's training objective matters more than its size or dimension.

### F38 — Swap does not slow the merge down, it stops it: 20+ minutes vs 7 seconds
The EB-NeRD parallel run wrote **all 51 shards correctly** at 6,426 lines/s, then produced *zero*
output for over 20 minutes. It looked like a hang. It was swap.

Measured at the point of the stall:

| | |
|---|---|
| Worker 1 | 13.6 GB RSS, **11.7 GB in swap** |
| Worker 2 | 10.9 GB RSS |
| Parent (merging) | 2.6 GB RSS, **12.5 GB in swap** |
| Free RAM | 6.1 GB of 31.1 |
| Merge output after 20 min | **0 bytes** |

After killing the processes and re-running the identical merge on the same shards with RAM
available: **7 seconds**, 13,395,569 rows read → **13,336,711 kept**, 58,858 duplicates dropped.

> [!important] The asymmetry that matters
> **Scoring streams and degrades gracefully under swap; the merge does not.** Scoring reads one row
> group at a time — sequential access, so paging costs a constant factor. The merge hits a
> `set` of 13.3M impression ids at random, and random access against swap is ~5 orders of magnitude
> slower than RAM. The same operation went from *never finishing* to 7 seconds purely on memory
> availability.
>
> This is why the fix is not "use less memory" but "**do not hold worker memory during the merge**".

**Three root causes, all fixed in `src/submit/codabench.py`:**

1. **The worker pool was still alive during the merge.** Concatenation ran inside
   `with ProcessPoolExecutor(...)`, so two workers holding 24.5 GB stayed resident for a phase that
   does not need them. The pool is now shut down with an explicit `finally: pool.shutdown(wait=True)`
   before the merge starts, and the freed memory is logged.
2. **`set[str]` over 13.3M ids costs ~1.5 GB** in Python string objects. Impression ids are integers,
   so `set[int]` is used, with a string fallback for non-numeric ids on other datasets.
3. **Nothing warned.** `_warn_if_swapping()` now fires during scoring, and a preflight clamps
   `--n-jobs` to what actually fits (`WORKER_GB = 9.5`, `MERGE_HEADROOM_GB = 3.0`).

**`--allow-swap` exists but does not cover the merge.** The user's position — swap is acceptable if
it buys speed — is right for scoring and wrong for the merge, and the 20-minutes-vs-7-seconds
measurement is why. The flag relaxes the worker clamp; the merge still runs only after every worker
has exited.

*Also worth noting:* the shards were never at risk. All 51 were complete and valid on disk, so the
run was recovered by merging them directly rather than re-scoring 13.3M impressions.

### 2026-08-18 — Phase 1 steps 3–5 built and run
- `src/data/clean.py` (unify + drive), `src/data/split.py` (temporal split, truncation, leakage
  checker), `tests/test_no_leakage.py` (Q9, with mutation tests).
- **Store built for both datasets, small tier, in 42 s.** MIND 9.0 s, EB-NeRD 32.1 s. 272 MB of
  parquet under `data/store/`, with a `manifest.json` per dataset recording counts, realised
  proportions, spans, verifiability and the resolved resource budget.
- **38 tests pass**, including the real-store leakage checks read back off disk.
- Findings **F18–F20**. Two of them correct earlier claims:
  - **F18:** EB-NeRD's history is already boundary-partitioned by the authors (history window closes
    8 min before impressions open), so truncation drops 0.0%. Correct behaviour, but the Q9 claim must
    be *"verified to hold"*, not *"we removed post-boundary clicks"*.
  - **F19 supersedes F4's MIND estimate:** realised split is **61/7/32**, not ~80/10/10. F4 reasoned
    from days; the split is over impressions, and MIND's single dev day is ~3× denser than an average
    train day. The rule is unchanged; the reported numbers must be the realised ones.
  - **F20:** MIND's article union is 65,238 — dev contributes 13,956 articles train never saw (21%).
    Confirms D3. And 0% of MIND articles carry a publish time vs 100% of EB-NeRD's, which is what
    blocks F16's recency filter on MIND.
- Fixed a `.gitignore` bug: a bare `data/` pattern matches at any depth and had silently untracked
  `src/data/` — the three modules the previous commit was mostly about. Anchored to `/data/`.

### 2026-08-18 — large-tier measurement
- Measured MIND-large and EB-NeRD-large from the archives → F13–F15.
- **F13 is the useful one:** the provided embedding artifacts cover all 125,541 EB-NeRD articles, and
  small is a strict subset with zero missing ids, so the baseline row is a plain join.
- **F14:** MIND-large test confirmed unlabelled (0 of 20,000 sampled rows carry `-1`/`-0`), spanning
  16–22 Nov with 19% more articles than train.
- **F15:** large adds users, not time. EB-NeRD 52× impressions, 3.6 GB extracted, history parquet
  larger than behaviors. Gives Q6 a specific answer: the join breaks before the model does.
- **No scope change.** Small tier remains the working and headline tier.

### 2026-08-18 — planning session
- Verified all 10 archives intact (CRC over every member, central-directory read, no extraction).
  `ebnerd_large.zip` and both embedding artifacts completed downloading since 15 Aug.
- Measured both datasets directly from the archives → findings F1–F11.
- Scope decision: **small tier only**; prepare models and pipeline on it, revisit large afterwards.
- Wrote this plan set. Every design decision carries alternatives + rationale, per the request, so the
  phase files convert directly into the Q6 design note's "alternatives considered" section.
- **Not done:** no pipeline code. Deliberate — review the plan first.

### 2026-08-15 — docs and git
- First two commits: docs + Q8 ignore policy, then the data/scale refresh.
- Decided own-embeddings-primary; recorded the resource-budget rule.

---
[[Pipeline|architecture]] · [[1-Data-Pipeline|Phase 1]] · [[../Assignment-1-Lexical-Semantic-Retrieval|tracking note]]
