---
type: note
kind: reference
title: Phase 2 — Lexical retrieval, BM25 (Q2)
status: done
---

# Phase 2 — Lexical retrieval (Q2)

> [!success] As built (2026-08-25) — `src/retrieval/`
> `bm25s`-backed BM25 over title + abstract, an independent textbook implementation as the
> D1 correctness check, NFC-safe tokeniser, and a 54-cell parallel sweep (44 s on 12 workers).
>
> **The headline finding contradicts the phase's own framing (F21–F23).** BM25 on the full
> corpus is *statistically indistinguishable from popularity* on EB-NeRD, and **loses to it on
> MIND** with non-overlapping CIs. The 24-hour recency window is worth **33×**; the `k1`/`b`
> knobs are worth ~0.01, inside the CI. So D5's sweep mattered far less than F16's pool choice.
>
> **Abstracts (D2) measured (F28):** they are ~75% of the index and buy **+0.011 recall against
> a ±0.030 CI** — not measurably better than titles alone. Kept because Q2.1 is binding;
> reported as the null result it is.
>
> **D1 answered: we did both.** `bm25s` is the workhorse for every reported number and submission;
> `ReferenceBM25` is ~50 lines of hand-written Robertson BM25 (IDF, k1 saturation, b length
> normalisation) used as the correctness check. `test_reference_agrees_with_library` pins them
> together. That test earned its place twice — it revealed the two disagree on *absolute* scores
> (Lucene vs Robertson IDF) while ranking identically, and it later caught that `score_subset`'s
> arithmetic could not be made to match `bm25s` exactly (off by up to 106), which is why that path
> is verified on **ranking agreement, 0 discordant pairs in 780**, and is barred from reported
> metrics.
>
> **Alternatives still untested (F57):** BM25F with per-field weights is the strongest remaining
> lexical idea — F28 showed abstracts are 75% of the index for an effect inside the CI, so
> down-weighting them is a motivated hypothesis. Query expansion/RM3 is the other classic. Neither
> is being tried: the brief names BM25, F23 measured its whole parameter space at ~0.01, and the
> lever on this data is the candidate pool (33x), not the weighting scheme.
>
> **Added since planning:** `score_subset()` for the submission path — scores a slate directly
> instead of retrieving top-K over the corpus and discarding 99%. **16× faster** (162 → 2,570
> impressions/s), ranking-equivalent (0 discordant pairs), never used for a reported metric (F32).


Build the first of two retrievers. [[1-Data-Pipeline]] produces the feature store this reads;
[[4-Evaluation-Harness]] scores what this produces. Architecture context in [[Pipeline]].

> [!abstract] What this phase commits to
> An **inverted index over title + abstract**, a **query built from the user's recent click titles**,
> **BM25** scoring, and **recall@K for K ∈ {50, 100, 200}** on both datasets. Plus a TF-IDF and a
> popularity baseline — cheap, and they are what make the BM25 number mean something.

> [!important] The framing that makes this work at all
> There is no query in a recommendation dataset. **We manufacture one**: concatenate the titles of the
> user's recently clicked articles and treat that text as a search query. Everything in this phase
> follows from that one move — including its main weakness, that the resulting "query" is 10× longer
> than anything BM25 was designed for.

## Q2 requirements

| # | Requirement | Where |
|---|---|---|
| Q2.1 | Inverted index over article text (title + abstract) | D1, D2 |
| Q2.2 | Query from click history (e.g. concatenated titles) | D3, D4 |
| Q2.3 | Retrieve top-K with BM25 | D5 |
| Q2.4 | Report recall@K for K ∈ {50, 100, 200} | Phase 4 harness |

## Design decisions

### D1 — Build the index, or use a library?

| Option | Buys | Costs |
|---|---|---|
| **`rank_bm25`** | Three lines to working BM25; pure Python; zero setup | Slow (recomputes IDF per query); memory-hungry; no persistence — rebuilds every run |
| **`bm25s`** | 100–500× faster than `rank_bm25`; sparse-matrix scoring; saves/loads the index | Newer, smaller community |
| **Own inverted index** | Full control; demonstrates understanding; the data structure is genuinely simple | A day of work + testing; easy to get IDF edge cases subtly wrong |
| Elasticsearch / Lucene | Production-grade; battle-tested | A service to run; enormous overkill; obscures the algorithm being graded |

**Chosen: `bm25s` as the workhorse, with a small own-implementation used as a correctness check on a
toy corpus.** Rationale: the brief grades *pipeline correctness and design*, not whether you
hand-rolled a posting list — but A-2 requires implementing ANN from scratch, so demonstrating the
same capability here cheaply (a 50-line index, verified against the library on ~100 documents) buys
the credibility without the timeline risk. If the check disagrees with the library, that is a finding
worth a paragraph.

### D2 — What text goes in the index?

| Option | Buys | Costs |
|---|---|---|
| Title only | Sharpest signal; news titles are dense and information-rich | Very short documents (5–15 tokens); many near-ties; low recall |
| **Title + abstract** | **Q2.1 specifies it**; the only pair available on *both* datasets; ~30–60 tokens is a reasonable document length | Abstract quality varies; some MIND abstracts are empty |
| Title + abstract + body | Most text, best coverage of rare terms | **MIND-small has no body** → breaks cross-dataset comparability; also 100× the index size |

**Chosen: title + abstract.** Two independent reasons converge — Q2.1 mandates it, and it is the only
option MIND can satisfy. EB-NeRD's `body` is noted in the design note as an available-but-unused
field, with the comparability argument as the reason.

> [!warning] Empty abstracts are a real case, not an edge case
> Some MIND articles have a blank abstract. Falling back to title-only for those is fine, but it makes
> document length bimodal, which interacts with BM25's length normalisation (`b`). Log how many
> articles are affected — if it is more than a few percent, it is worth a sentence in the note.

### D3 — Tokenisation, and the Danish problem

| Option | Buys | Costs |
|---|---|---|
| Whitespace + lowercase | Trivial; language-agnostic | Punctuation attaches to tokens; no stemming |
| **Regex word tokens + lowercase + NFC** | Handles punctuation; keeps Danish `æ ø å` intact under Unicode normalisation | No morphological reduction |
| + English stemming (Porter/Snowball) | Better English recall | **Wrong for Danish** — applying an English stemmer to Danish is actively harmful |
| + per-language stemming (Snowball has Danish) | Correct per language | Two code paths; makes the datasets less comparable |
| + stopword removal | Smaller index; less noise from "the"/"og" | BM25's IDF already down-weights common terms; removal is largely redundant and risks dropping meaningful words |

**Chosen: regex word tokens + lowercase + NFC normalisation, no stemming, no stopword removal — the
same pipeline for both datasets.** Rationale: IDF already handles stopwords, and using *different*
stemmers per dataset would confound the cross-dataset comparison that Q3.5 asks for. **Per-language
stemming is the obvious ablation** — run it on both, report the delta, and let the number decide.
That is one cheap experiment producing a real finding.

> [!important] Unicode normalisation is not optional for EB-NeRD
> Danish `æ ø å` can be encoded as single code points or as base+combining forms. Two encodings of the
> same word are two different index terms, silently halving recall on affected queries. Normalise to
> NFC on **both** the corpus and the query, and assert it in a test.

### D4 — Query construction from click history

The underdetermined part of the assignment, and where the interesting choices live.

**How many clicks?**

| Option | Buys | Costs |
|---|---|---|
| All history | Maximum signal | EB-NeRD averages **160 clicks/user** (max 1,896) → a query of thousands of tokens. BM25 degenerates: hundreds of terms each contribute a little, common words dominate the sum, scores become diffuse and near-identical across documents |
| **Last N (N ≈ 10–20)** | Recent interest is what predicts the next click in news; query length stays in a regime BM25 handles | Discards long-term interests |
| Last N + decay weighting | Recency without a hard cutoff | One more parameter to justify |

**Chosen: last N = 15 as the default, swept over {5, 10, 20, 50} as the phase's main ablation.**
The sweep is nearly free — no re-indexing, only re-scoring — and directly produces the
"ablation rigour" the rubric names.

> [!warning] The BM25 trap specific to this assignment
> BM25 was tuned for 2–5 word queries. A concatenation of 15 news titles is ~120 tokens. Long queries
> mean many terms contribute, IDF-weak terms creep in, and every document starts to look moderately
> relevant. **This is the single most important thing to say about lexical retrieval here** — it is a
> genuine insight about applying a model outside its design regime, and it explains any
> disappointing BM25 numbers far better than "BM25 is weak".

**Deduplicate query terms?**

| Option | Buys | Costs |
|---|---|---|
| No dedup | Term repeated across titles gets higher weight — arguably correct, it is a repeated interest | Inflates TF for whatever the user reads most; interacts with `k1` saturation |
| **Dedup** | Each distinct term contributes once; query behaves more like a real query | Loses the "read this topic five times" signal |
| Dedup + count as weight | Keeps the signal explicitly | Needs a weighted-query BM25 variant |

**Chosen: dedup as default, no-dedup as an ablation row.** BM25's `k1` already saturates repetition
*within a document*; repetition within the *query* is a different axis and is not what the formula was
designed around.

**Cold-start users (no history)?**

| Option | Buys | Costs |
|---|---|---|
| Return nothing | Honest; the retriever genuinely has no signal | recall = 0 for that slice; drags the headline number |
| **Most-popular fallback** | Non-zero recall; matches what a real system does | Popularity is arguably a serving-time-unavailable feature → **interacts with Q9** |
| Category prior | Slightly smarter | Needs a category to guess from — cold users have none |
| Random | A true floor | Useless in production, but a legitimate baseline row |

**Chosen: most-popular fallback, computed on the train split only**, with an explicit `is_fallback`
flag on every such result so the harness can report metrics with and without it. That flag is exactly
the Q9 "with and without serving-unavailable features" comparison, obtained almost for free.

### D5 — BM25 parameters

Defaults `k1 = 1.2`, `b = 0.75` are TREC-era ad-hoc conventions, not laws.

| Parameter | What it controls | Expected behaviour here |
|---|---|---|
| `k1` | TF saturation | **Barely matters** — titles+abstracts rarely repeat a term. Confirm empirically, don't assume |
| `b` | Length normalisation | **Matters** — empty-abstract articles make lengths bimodal |

**Plan: sweep `k1 ∈ {0.9, 1.2, 1.6}` × `b ∈ {0.3, 0.75, 1.0}`** on the val split only, pick by
recall@100, report the surface. Tuning on val and reporting on test is the discipline that keeps the
test split honest — say so in the note.

### D6 — Baselines to run alongside

| Baseline | Cost | Why it earns its place |
|---|---|---|
| **Recency** (K most recently published before *t*) | ~15 lines | **The strongest baseline measured, by a wide margin — recall@50 = 0.92 (F16).** This is the number BM25 has to beat. |
| **Popularity** (top-K most-clicked in train) | ~20 lines | The classic floor, and it turns out to be a weak one here (recall@50 = 0.03) |
| **TF-IDF** (`sklearn.TfidfVectorizer` + cosine) | ~1 hour | Isolates what BM25's two knobs actually buy over plain TF-IDF |
| Random | ~5 lines | True chance floor; makes recall@200 interpretable |

> [!warning] Recency changes what this phase is for — measured, not predicted (F16)
> A walking-skeleton run on EB-NeRD demo found that **94.3% of clicks are on articles under 24 hours
> old**, while only **1.1% of the corpus** is that fresh. Consequences, measured:
>
> | Retriever | recall@50 |
> |---|---|
> | Recency only (user ignored) | **0.9250** |
> | Token overlap, full corpus | 0.0000 |
> | Token overlap, 24h window | **0.2750** |
> | Hashed vectors, 24h window | 0.4750 |
>
> Full-corpus lexical retrieval is close to hopeless here: the user's history is mostly *older*
> articles, so the most textually similar documents are stale ones. **The same scoring function
> inside a 24-hour window goes from 0.00 to 0.28.**
>
> **So BM25 must operate within a recency-constrained pool, not over the whole corpus.** The
> `at_time` argument in the `Retriever` protocol — added defensively — is now load-bearing.
> Report both regimes; the gap between them is a genuine finding, not a tuning detail.
>
> **This is legitimate, not a leak.** `published_time` is known at serving time. Ranking by *future*
> engagement would be a leak; filtering by publish date is not.
>
> **MIND has no `published_time`.** The filter cannot be built there from article metadata —
> first-seen-in-impressions is the available proxy, and it needs its own decision. The asymmetry is
> worth reporting.

All three go through the same `Retriever` interface and the same harness. "BM25 recall@100 = 0.42" is
a number; "0.42 vs TF-IDF 0.36 vs popularity 0.19 vs random 0.01" is a finding.

## The `Retriever` interface

Settled here because Phase 3 must satisfy the same contract.

```python
class Retriever(Protocol):
    name: str
    def index(self, articles: pl.DataFrame) -> None: ...
    def retrieve(self, history: list[str], k: int,
                 at_time: datetime) -> list[tuple[str, float]]: ...
```

`at_time` is deliberately in the signature even though BM25 does not use it: it makes the temporal
boundary **visible at the call site** for every retriever, so a future implementation that filters by
publish time cannot forget it. Returning `(id, score)` pairs rather than bare ids lets the harness do
fusion and score analysis without a second retrieval pass.

## Acceptance criteria

- [ ] Index builds from the feature store for both datasets; build time and memory logged
- [ ] Own toy implementation agrees with `bm25s` on a ~100-document corpus
- [ ] `retrieve()` returns exactly K results, descending by score, no duplicates
- [ ] recall@{50,100,200} computed on both datasets via the Phase 4 harness
- [ ] Popularity + TF-IDF + random baselines run through the same interface
- [ ] `k1`/`b` sweep run on **val only**; chosen values recorded before touching test
- [ ] Cold-start fallback flagged per result, so Q9's with/without comparison is available
- [ ] NFC normalisation asserted on both corpus and query

## Expected pitfalls

| Symptom | Likely cause |
|---|---|
| recall@200 < recall@50 | Impossible by construction — bug in K handling or hit counting |
| Every user gets near-identical results | Query too long → diffuse scores; or the popularity fallback firing for everyone |
| EB-NeRD far worse than MIND | Tokenisation mangling Danish characters — check NFC first |
| BM25 barely beats popularity | Expected if the query is over-long; shorten N before concluding BM25 is weak |
| Suspiciously high recall | Future-click leakage from Phase 1 — check the boundary before celebrating |

---
[[1-Data-Pipeline|← Phase 1]] · [[3-Semantic-Embeddings|next: Phase 3 →]] · [[execution_plan_log|log]]
