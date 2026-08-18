---
type: note
kind: reference
title: Execution plan log — A1 Component-1
---

# Execution plan log

Running log of **what we decided, why, what is done and what is not**. One task at a time; review and
correct before starting the next. Doc and data-layout work come before coding.

> [!abstract] Where things stand (2026-08-18)
> **Planning complete; no pipeline code written yet.** All raw data for the small tier is downloaded
> and verified. The plan set ([[Pipeline]] + phases 1–5) is written, with every design decision
> carrying its alternatives and rationale.
>
> | Item | State |
> |---|---|
> | Raw data (small tier) | ✅ downloaded, all archives CRC-verified |
> | Git repo | ✅ `main`, 2 commits, `data/` correctly ignored |
> | Plan docs | ✅ this set |
> | Pipeline code | ⬜ none — next step |
> | `ebnerd_testset.zip` | ❌ not downloaded — **blocks Q5 only** |
> | GitHub Classroom repo | ❌ not accepted yet |
> | Pair declaration (C2) | ⚠️ **deadline was 2026-08-15 — verify this is sorted** |
>
> Nine days to the 2026-08-27 deadline. Findings are recorded below as F1–F12.

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
- [ ] **Review this plan set and correct it** ← *current*
- [ ] Confirm C2 pair declaration status (deadline passed)
- [ ] Phase 1 step 1–2: extract + per-dataset readers
- [ ] Phase 1 step 3: unified schema
- [ ] Phase 1 step 4–5: temporal split + history truncation
- [ ] Phase 1 step 6–7: feature store + leakage test
- [ ] Phase 2: BM25 + baselines
- [ ] Phase 3: embeddings + ANN
- [ ] Phase 4: harness
- [ ] Phase 5: submissions + note

## Log

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
