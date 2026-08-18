---
type: note
kind: reference
title: Execution plan log — A1 Component-1
---

# Execution plan log

Running log of **what we decided, why, what is done and what is not**. One task at a time; review and
correct before starting the next. Doc and data-layout work come before coding.

> [!abstract] Where things stand (2026-08-18)
> **Q1 is essentially done.** Both datasets build into one unified, temporally-split store in 42 s,
> with the leakage boundary verified by a test that is itself proven to fail when the boundary breaks.
>
> | Item | State |
> |---|---|
> | Raw data (small tier) | ✅ downloaded, all archives CRC-verified |
> | Git repo | ✅ `main`, 5 commits, `data/` correctly ignored |
> | Plan docs | ✅ this set |
> | **Q1.1** download | ✅ small tier complete |
> | **Q1.2** unified schema | ✅ `src/data/clean.py`, both datasets |
> | **Q1.3** temporal split | ✅ `src/data/split.py`, ordering asserted |
> | **Q1.4** feature store | 🟡 parquet store exists; feature columns beyond retrieval pending |
> | **Q1.5** one-command rebuild | ✅ `make clean && make data && make store` |
> | **Q9** leakage test | ✅ 38 tests, incl. mutation tests that prove the checker bites |
> | Phases 2–4 | ⬜ next |
> | `ebnerd_testset.zip` | ❌ not downloaded — **blocks Q5 only** |
> | GitHub Classroom repo | ❌ not accepted yet |
> | Pair declaration (C2) | ⚠️ **deadline was 2026-08-15 — verify this is sorted** |
>
> Built store: 272 MB parquet, `data/store/{mind,ebnerd}/` + a manifest per dataset.
>
> Nine days to the 2026-08-27 deadline. Findings are recorded below as F1–F20.
>
> **Large tiers measured (F13–F15) but still deliberately idle.** The headline numbers stay on small;
> what the measurement bought is a concrete Q6 scale answer and a cheaper baseline, not a change of
> scope.

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
