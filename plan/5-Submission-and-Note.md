---
type: note
kind: reference
title: Phase 5 — Codabench submission and design note (Q5–Q7)
status: in-progress
---

# Phase 5 — Submission and design note (Q5, Q6, Q7)

> [!success] As built (2026-08-25) — `src/submit/codabench.py`, `report/a1_report.tex`
> **MIND submitted and scored: AUC 0.5568, MRR 0.2646, nDCG@5 0.2778, nDCG@10 0.3331**
> (rank 75/84). EB-NeRD prediction file generating at the time of writing.
>
> **Two failures worth keeping, both found by running the real thing:**
> - The first upload was **rejected on a filename** — the scorer opens `prediction.txt`
>   literally and our archive held `mind_prediction.txt`. The 2.37M predictions were correct
>   (F35). Now a named constant with tests guarding both it and the built artefact.
> - The first EB-NeRD run **crashed on the unlabelled test schema** (F33).
>
> **Both leaderboards now have scores.** MIND BM25 **0.5568** (901650), MIND fusion **0.5934**
> (901779) — a real +0.0366 that the offline harness had no power to see. EB-NeRD submitted.
>
> **A format bug caught before it cost a second submission.** The two competitions want archive
> members whose names differ by one letter — MIND `prediction.txt`, EB-NeRD **`predictions.txt`** —
> verified against both upstream sources. Submission filenames are now
> `{dataset}_{retriever}_{paramhash}_i{n}` so no run can overwrite another.
>
> **Q9 results.** *(a)* The leakage test suite passes — **15 tests**, including mutation tests that
> inject a post-boundary click and confirm the checker fails. Verifiable on EB-NeRD only; MIND's
> history is untimestamped, and the store records `history_boundary_verifiable: false` rather than
> implying otherwise. *(b)* With/without serving-unavailable features: **the gap is ~0.002**
> (MIND recall@100, popularity 0.0688 → 0.0706 without the cold-start fallback). Our retrievers
> never touch the post-click fields, so the only serving-unavailable feature in play is the
> fallback, and it fires too rarely to matter. **EB-NeRD has no `no_fallback` row at all** — F9
> established it has no zero-history users, so the fallback never fires there. That is a data
> property, not a gap.
>
> **Tuned resubmission.** Only F47 (256-d truncation) changes a submitted file; F51/F52/F53 do not
> (F54, F55). MIND regenerated as fusion at 256-d.
>
> **Cohort context:** median AUC 0.6121 across 50 classmate submissions; we are 0.055 below it.
> The gap is not parameter tuning (measured at ~0.01) — it is the missing popularity signal and
> the semantic/fusion rows, which are built but not yet scored.
>
> **Still open:** EB-NeRD submission, leaderboard screenshots (Q7.3), and the final report.
> The draft is `report/a1_report.tex` (9 pages, compiles) and is findings-centred.


**Outline only, deliberately.** The content of this phase is determined by what phases 1–4 actually
produce; writing it in detail now would mean inventing results. It is expanded once the harness has
run.

> [!abstract] What this phase commits to
> Predictions submitted to **both** Codabench leaderboards, a **≤4-page design note**, screenshots,
> and a curated **AI usage log**. Grading is *never* on leaderboard rank — it is on pipeline
> correctness, system design, ablation rigour, scale analysis, and note clarity.

## Q5 — Codabench

| | MIND | EB-NeRD |
|---|---|---|
| Competition | [13967](https://www.codabench.org/competitions/13967/) | [2469](https://www.codabench.org/competitions/2469/) |
| Test data | `MINDlarge_test.zip` ✅ downloaded | `ebnerd_testset.zip` ❌ **not downloaded** |
| Registered? | ⬜ verify | ⬜ verify |

> [!warning] Two asymmetries that bite late
> 1. **MIND has no small test set.** Submitting means running the small-trained pipeline over the
>    *large* unlabelled test set (1.5 GB). Feasible, but it is the one place large-tier data is
>    unavoidable — do not discover this on 24 Aug.
> 2. **EB-NeRD's test set is a separate 1.5 GB download** that is not yet started, and download
>    throughput was measured at ~19 KB/s single-stream. Start it well before the deadline.

Submission format: verify against each competition's example file before generating. EB-NeRD ships
`predictions_large_random.zip` (220 MB) as a known-good format reference — worth fetching purely as
insurance against a malformed upload.

## Q6 — Design note (≤4 pages)

Four required sections. What goes in each is already accumulating in the phase files:

| Section | Source |
|---|---|
| What you built + key choices | The D-decisions across phases 1–4 |
| **Alternatives considered and why** | Every D-decision's alternatives table — this is why they are written that way |
| Observations (lexical vs. semantic, dataset differences) | Phase 3 D6 hypotheses vs. what the harness actually found |
| **Where it breaks at 10×** | Phase 3 HNSW ablation, encoder throughput, index memory, demo-vs-small timings |

**The 10× question, concretely:** measure at two scales already available (demo and small), extrapolate,
and name where the curve bends — inverted-index RAM, ANN build time, whether the pipeline becomes
I/O- or compute-bound. Having the large tiers downloaded means one real anchor point is optionally
available, which beats pure projection.

## Q7 — Deliverables checklist

- [ ] **Code** on GitHub Classroom — pipeline, models, harness, prediction files, `README.md` with
      one-command reproduce. No large files.
- [ ] **Design note** ≤4 pages → Moodle
- [ ] **Leaderboard screenshots** from both competitions
- [ ] **AI usage log** — curated with `/ai-log`, marking AI-generated vs. hand-written code

> [!info] The repo does not exist yet
> GitHub Classroom has not been accepted (as of 2026-08-18). The local repo is on `main` with two
> commits. When the Classroom repo appears: `git remote add origin <url>` and push — history
> transfers intact. Per IRE conventions, **clone and extend the Classroom repo rather than inventing
> a parallel layout** if it ships with a structure.

## Q8 — Git policy

- [x] `.gitignore` covers `*.zip`, `*.pt`, `*.ckpt`, `__pycache__/`, `data/` — verified
- [ ] Frequent, meaningful commits (2 so far)
- [ ] No force-pushes after the deadline

---
[[4-Evaluation-Harness|← Phase 4]] · [[Pipeline|architecture]] · [[execution_plan_log|log]]
