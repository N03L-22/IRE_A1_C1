---
type: note
kind: reference
title: Phase 4 — Evaluation harness (Q4) and the leakage test (Q9)
status: done
---

# Phase 4 — Evaluation harness (Q4)

> [!success] As built (2026-08-25) — `src/eval/`
> One `evaluate()` over any `Retriever`, so Q4.5 holds by construction. Both regimes kept
> strictly apart, bootstrap CIs at impression level, slices with per-dataset thresholds.
>
> **D5 has one documented exception (F24).** Coverage counts *distinct* articles, so it grows
> with sample size and every resampling scheme is biased low. The first attempt put the point
> estimate **outside its own interval** (0.9783 vs [0.9035, 0.9235]); subsampling fails at every
> ratio tried. Coverage is reported as a point estimate marked `(no CI)` — saying so beats
> shipping a plausible-looking number that is biased by construction.
>
> **The harness was under-powered and the leaderboard proved it (F34).** At n = 800 it reported
> MIND AUC 0.4981 [0.4776, 0.5190]; the leaderboard scored **0.5568** — outside the interval.
> `src/eval/run_all.py` now defaults to **20,000 impressions**, and no retriever decision should
> be made on less.
>
> **Also measured:** random ranking scores nDCG@10 = 0.42 on EB-NeRD, because slates are ~11
> items with one click. **Any slate table without a random row misleads.**


One harness scoring both retrievers. Reads what [[2-Lexical-BM25]] and [[3-Semantic-Embeddings]]
produce. Also home to the **leakage test** (Q9) and the **with/without serving-features comparison**
(Q9) — correctness is proven here, not just measured.

> [!abstract] What this phase commits to
> **AUC · MRR · nDCG@5 · nDCG@10 · recall@{50,100,200}**, plus **diversity · novelty · coverage**,
> reported over **at least one slice**, with a **bootstrap 95% CI on every number**, resampled at the
> **impression** level. Plus the two Q9 artifacts: a leakage test that fails when the boundary breaks,
> and metrics computed with and without serving-unavailable features.

> [!important] The harness is the deliverable, not the retrievers
> Q4.5 requires running it on **both** BM25 and embedding results. A harness that only works for one
> retriever fails the requirement no matter how good that retriever is. It takes a `Retriever` and a
> split, and knows nothing else about either.

## Q4 requirements

| # | Requirement | Where |
|---|---|---|
| Q4.1 | AUC, MRR, nDCG@5, nDCG@10 | D2 |
| Q4.2 | Intra-list diversity, novelty, coverage | D3 |
| Q4.3 | At least one slice (cold vs. warm, or head vs. tail) | D4 |
| Q4.4 | Bootstrap 95% CI for each metric | D5 |
| Q4.5 | Run on **both** BM25 and embedding results | D1 |
| Q9 | Metrics with **and without** serving-unavailable features | D6 |
| Q9 | Leakage test asserting the behaviour-window boundary | D7 |

## Design decisions

### D1 — Harness shape

| Option | Buys | Costs |
|---|---|---|
| Per-retriever eval scripts | Quick to write the first one | Guarantees divergence; Q4.5 becomes a manual comparison; **fails the requirement in spirit** |
| **One harness taking a `Retriever`** ✅ | Q4.5 satisfied by construction; adding a baseline is one line | Requires the interface to be right (settled in Phase 2) |
| A metrics DSL / config-driven framework | Very general | Overkill; the doc's own "overkill" column |

**Chosen: one function, `evaluate(retriever, split, config) -> ResultTable`.** Everything —
BM25, semantic, popularity, TF-IDF, random, fusion — goes through it. The output is a tidy frame
(one row per metric × slice × retriever) so the design-note tables are a groupby, not a copy-paste job.

### D2 — Accuracy metrics: two evaluation regimes, and why both exist

> [!warning] The most common conceptual error in this assignment
> **recall@K and AUC/MRR/nDCG are computed over different candidate sets**, and conflating them
> produces numbers that look fine and mean nothing.
>
> - **recall@K** — retrieve from the **whole corpus** (51K/21K articles). Question: *did the clicked
>   article survive the cut?* This is the candidate-generation metric.
> - **AUC / MRR / nDCG** — score the **impression's own slate** (`article_ids_inview`, mean 11–12
>   items). Question: *did you rank the clicked one above the others shown?* This is the leaderboard's
>   metric and how Codabench scores you.
>
> A retriever produces a corpus-wide ranking; to get AUC you restrict-and-rerank onto the slate. Say
> which regime every reported number belongs to.

| Metric | Regime | What it answers | Weakness |
|---|---|---|---|
| **recall@K** | corpus | Did the clicked item make the shortlist? | Ignores position entirely — correct here, a re-ranker fixes order |
| **AUC** | slate | Would a random clicked item outrank a random ignored one? | Weights the whole ranking equally; deep improvements count as much as top ones |
| **MRR** | slate | How far down is the first hit? | Ignores subsequent hits — but 99.5% of impressions have exactly one click, so this barely matters here |
| **nDCG@5/@10** | slate | Are good items near the top? | With binary relevance $2^{rel}-1$ reduces to 1/0 — worth stating so the graded form is unambiguous |

**Note on slate size:** EB-NeRD slates average 11–12 items, so **nDCG@10 covers almost the whole
slate** and nDCG@5 is the more discriminating of the two. Expect them to correlate strongly; say so
rather than presenting them as independent evidence.

### D3 — Beyond-accuracy metrics

| Metric | Definition | Choice to make |
|---|---|---|
| **Intra-list diversity** | $\frac{2}{k(k-1)}\sum_{i<j}(1 - \text{sim}(d_i,d_j))$ | Similarity from **embeddings** (continuous, fine-grained) or **category** (discrete, interpretable). **Chosen: embeddings for the number, category as a readable cross-check.** Caveat: using embedding similarity to score a retriever that optimises embedding similarity is circular — state it |
| **Novelty** | $-\log_2 p(d)$, $p(d)$ = train-split popularity | Popularity must come from **train only** — using test-period popularity is leakage |
| **Coverage** | fraction of catalogue appearing in *any* user's top-K | Over which corpus — full, or train-visible only? **Chosen: full corpus**, stated |

> [!tip] The trade-off is the finding
> These genuinely oppose accuracy: recommending the ten most popular articles to everyone scores well
> on nDCG and near-zero on coverage. **One table of nDCG@10 against coverage across all retrievers is
> a strong design-note exhibit** — and the popularity baseline is what makes it land.

### D4 — Slices

Q4.3 requires at least one. Cheap to add more, and slices are where findings live.

| Slice | Split on | Why it's interesting |
|---|---|---|
| **Cold vs. warm users** ✅ primary | history length | Directly tests the D6 hypothesis in Phase 3 (semantic should win cold) |
| **Head vs. tail articles** ✅ second | article click frequency in train | Tests whether the system is a recommender or a popularity list |
| Fresh vs. stale articles | `published_time` age | **EB-NeRD only** — MIND has no publish time |
| Session length | `session_id` | EB-NeRD only |
| Danish vs. English | dataset | The headline cross-dataset comparison |

> [!warning] The cold-start threshold is a decision, not a given
> Q4.3 says "few clicks" with no number. **EB-NeRD's minimum history length is 5** (measured) — so on
> EB-NeRD there are *no* zero-history users, and a "< 5 clicks" threshold selects nobody. MIND's
> history distribution is different and must be measured before choosing. **Pick per dataset, justify
> by the actual distribution (e.g. bottom quartile), and report the threshold with every cold-start
> number.** A slice boundary chosen after seeing the results is not a finding.

### D5 — Bootstrap confidence intervals

| Option | Buys | Costs |
|---|---|---|
| No CI | — | **Fails Q4.4** |
| Normal approximation | Instant | Assumes normality; wrong for bounded, skewed metrics like recall |
| **Percentile bootstrap** ✅ | Assumption-free; what the brief names | B× the metric computation |
| BCa bootstrap | Corrects bias and skew | More code; unnecessary at n ≈ 70–240K impressions |

**Chosen: percentile bootstrap, B = 1000, seeded.**

```python
def bootstrap_ci(per_impression_values, B=1000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)          # seeded — the CI must be reproducible
    n = len(per_impression_values)
    idx = rng.integers(0, n, size=(B, n))      # vectorised; no Python loop
    means = per_impression_values[idx].mean(axis=1)
    return np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])
```

> [!warning] Resample impressions, not predictions
> Predictions within one impression are correlated. Resampling them independently **understates the
> interval** — the classic symptom is suspiciously narrow CIs. Every metric must therefore be
> expressible as a mean over per-impression values, which constrains how coverage (a global quantity)
> is handled: bootstrap the *impression sample*, recompute coverage per resample.

**What the CI does and does not claim:** it quantifies **sampling noise only**. A leaking pipeline
produces beautifully tight intervals around a wrong number. **The bootstrap cannot detect bias** —
that is what D7 is for.

### D6 — Q9: serving-unavailable features

Q9 requires reporting metrics **with and without** features unavailable at serving time. The
requirement only means something once the set is decided — and deciding it late means re-running
everything.

**The candidate set, with a verdict on each:**

| Feature | Available at serving? | Verdict |
|---|---|---|
| `article_ids_inview` (the slate) | ✅ yes — it is the input | Keep. Not a leak |
| Train-split popularity | ✅ yes — computable from the past | Keep, but **must be train-only** |
| **Test-period popularity / `total_inviews`** | ❌ **no** — aggregates the future | **Exclude from headline; the with/without pair** |
| **`read_time`, `scroll_percentage`** | ❌ no — measured *after* the click | **Exclude.** Using them to predict the click is circular |
| **`next_read_time`, `next_scroll_percentage`** | ❌ no — literally the next event | **Exclude.** The most obvious leak in the EB-NeRD schema |
| Article `published_time` | ✅ yes | Keep |
| User age/gender/postcode | ✅ yes (profile) | Keep — but a fairness caveat is worth a sentence |
| Cold-start popularity fallback | ⚠️ depends | Flagged per result in Phase 2 D4 → **this is the cheapest with/without pair available** |

> [!important] EB-NeRD hands you three leaks by name
> `read_time`, `scroll_percentage`, and especially `next_read_time` / `next_scroll_percentage` are
> post-click measurements sitting in the same row as the label. They are not needed by either
> retriever, so the risk is not that we plan to use them — it is that a future feature-store change
> quietly includes them. **Exclude them explicitly in `clean.py`, with a comment naming why**, and let
> the with/without comparison use the popularity-fallback flag instead.

### D7 — The leakage test (Q9)

The single most important test in the assignment.

| Option | Buys | Costs |
|---|---|---|
| Assert on the produced store | Cheap; catches real violations | Passes trivially if the store happens to be correct — proves little about the *code* |
| **Assert + mutation test** ✅ | Proves the test has teeth | Slightly more setup |
| Full property-based testing | Thorough | Overkill for one invariant |

**Chosen: an invariant assertion plus a deliberate-violation test.**

```
test_no_leakage_in_store()          # every test impression: max(history_time) < impression_time
test_leakage_test_catches_violation()   # inject a future click → the assertion MUST fail
```

> [!warning] A test that passes both before and after you break the boundary proves nothing
> This is exactly what Q9 guards against. The second test is the one that matters: it deliberately
> injects a post-boundary click into a copy of the store and asserts the checker **fails**. Without
> it, a checker with an inverted comparison or an empty loop looks identical to a working one.

**Coverage limit, stated honestly:** this test verifies the *behaviour-window* boundary, for which
timestamps exist — i.e. **EB-NeRD**. On MIND, history has no timestamps ([[1-Data-Pipeline]] D2), so
the invariant is unverifiable from the data and rests on the authors' construction. **Say this in the
design note.** Claiming a test covers MIND when it cannot is worse than the gap itself.

## Output shape

One tidy table, everything downstream is a groupby:

| dataset | retriever | slice | metric | value | ci_low | ci_high | n_impressions | features |
|---|---|---|---|---|---|---|---|---|
| mind | bm25 | all | recall@100 | … | … | … | 73152 | serving_only |
| ebnerd | semantic | cold | ndcg@10 | … | … | … | … | serving_only |

The `features` column carries the Q9 with/without dimension. Writing it as a column rather than two
separate runs means the comparison is a filter, not a re-run.

> [!important] Never write a bare number in a note
> Per IRE conventions: `nDCG@10 = 0.34 [0.31, 0.37], n = 73,152`. The CI and the sample size travel
> with the number, always. And **never record a metric that was not actually run** — a target or a
> guess must be labelled as such.

## Acceptance criteria

- [ ] One harness runs **all** retrievers — BM25, semantic, popularity, TF-IDF, random
- [ ] Both regimes implemented and labelled: corpus-wide recall@K, slate-restricted AUC/MRR/nDCG
- [ ] Diversity, novelty, coverage computed; novelty popularity from **train only**
- [ ] ≥ 2 slices; cold-start threshold chosen from the measured distribution and reported
- [ ] Bootstrap CI on every number, **resampled at impression level**, seeded
- [ ] Q9 with/without table produced from the `features` column
- [ ] `test_no_leakage.py` passes clean **and** fails on injected violation
- [ ] Post-click fields explicitly excluded in `clean.py`, with a comment
- [ ] Every result row carries `n_impressions` and the resolved resource budget

## Expected pitfalls

| Symptom | Likely cause |
|---|---|
| CIs suspiciously narrow | Resampling predictions instead of impressions (D5) |
| AUC ≈ 1.0 | Scoring the corpus instead of the slate, or leakage |
| recall@200 < recall@50 | Impossible — bug in K handling |
| nDCG@10 ≈ nDCG@5 exactly | Slates are ~11 items; expected, not a bug |
| Coverage ≈ 0 for a good retriever | It is ranking, not recommending — the popularity-collapse finding |
| MRR ≈ recall@1 | Expected with single-click impressions; not an error |
| Cold-start slice empty on EB-NeRD | Min history is 5; threshold too low (D4) |

---
[[3-Semantic-Embeddings|← Phase 3]] · [[5-Submission-and-Note|next: Phase 5 →]] · [[execution_plan_log|log]]
