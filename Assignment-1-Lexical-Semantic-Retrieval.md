---
type: assignment
title: Assignment 1 — Lexical & Semantic Retrieval on EB-NeRD and MIND
status: todo
due: 2026-08-27
priority: high
---

# Assignment 1 — Lexical & Semantic Retrieval on EB-NeRD and MIND

📄 Brief: `brief/Assignment1_v1.pdf` · 🎓 [[foundations|Startup foundations]] · 🏗️ [[architecture|Architecture & design decisions]]

> [!tip] Reading order
> New to retrieval / no SMAI or iNLP? **[[foundations|foundations.md]] first** — it builds every term
> from scratch and lists what you don't need to learn. Then [[architecture|architecture.md]], which
> now runs easy → hard (Part A the idea · B the shape · C the machinery · D the judgement).

**This folder is A1 · Component-1 only** — 2026-08-27 (at Quiz-1) · **Individual** · 5%

> [!info] A-1 is 10%, split across two separately-graded components
> | | Scope | Mode | Due | Weight |
> |---|---|---|---|---|
> | **C1** (this folder) | Lexical + semantic retrieval, temporal split, offline metrics | Individual | 2026-08-27 | 5% |
> | **C2** | Click-log modelling, re-rank, **baseline beaten** (ablation), serving/cost analysis | **Pairs** | 2026-09-10 | 5% |
>
> `brief/Assignment1_v1.pdf` is the **C1** brief — its Q1–Q9 covers C1's scope only. The C2 spec is a
> separate document (course proposal references `revisions/assignment-1-revision.md`); it isn't here yet.
>
> **C1 feeds C2:** "baseline beaten" is a C2 deliverable, and C1's numbers are that baseline. Tag the
> commit you submit on 27 Aug and make every C1 metric regenerable from one command — otherwise C2
> starts by rebuilding C1.
>
> ⚠️ **Pair must be declared by 2026-08-15** — blocks C2 *and* Assignment-2. See [[_syllabus#Teams]].

| | |
|---|---|
| Code submission | GitHub Classroom (link on Moodle) |
| Report submission | Moodle (design note, ≤4 pages) |
| MIND competition | <https://www.codabench.org/competitions/13967/> |
| RecSys 2024 (EB-NeRD) | <https://www.codabench.org/competitions/2469/> |
| Starter code | <https://github.com/jppol-ai/ebnerd-benchmark> |

> [!warning] Grading is **never** on leaderboard rank
> Grade = pipeline correctness · system design · ablation rigour · scale analysis · design-note clarity.
> Registering and submitting to **both** leaderboards is mandatory; topping them is not.

## Brief

Rank the candidate articles in an impression by click likelihood, using click history, session
context, and article content. Three modelling axes: **lexical** (BM25/TF-IDF), **semantic**
(embeddings), **behavioural** (click history, recency/decay).

## Submissions
- [ ] Register on both Codabench competitions 📅 2026-08-10
- [ ] Submit MIND predictions to leaderboard 📅 2026-08-24
- [ ] Submit EB-NeRD predictions to leaderboard 📅 2026-08-24
- [ ] Push code to GitHub Classroom 📅 2026-08-27
- [ ] Submit design note + leaderboard screenshots to Moodle 📅 2026-08-27

## Sections
- [ ] Q1 — Reproducible data pipeline (download → clean → temporal split → feature store) 📅 2026-08-12
- [ ] Q2 — BM25 lexical candidate generation + recall@K 📅 2026-08-16
- [ ] Q3 — Semantic candidate generation (embeddings + ANN) + recall@K 📅 2026-08-20
- [ ] Q4 — Offline evaluation harness (metrics, slices, bootstrap CIs) 📅 2026-08-23
- [ ] Q5 — Codabench submission (both) 📅 2026-08-24
- [ ] Q6 — Design note ≤4 pages 📅 2026-08-26
- [ ] Anti-gaming: leakage test asserting behaviour-window boundary 📅 2026-08-23
- [ ] AI usage log assembled 📅 2026-08-26

## Deliverables (Q7)
1. **Code** (GitHub Classroom) — pipeline, model code, eval harness, prediction files, `README.md`
   with one-command reproduce. No large files — use `.gitignore`.
2. **Design note** (≤4 pages, Moodle) — what you built, choices, observations, where it breaks at 10×.
3. **Leaderboard screenshots** from both competitions.
4. **AI usage log** — all prompts, chat history exports, marking AI-generated vs. human-written code.
   → Auto-captured to `ai-log.md` in this folder whenever Claude is run from `Subjects/IRE/`.
   Curate it with **`/ai-log`** before submitting. See [[CLAUDE|IRE conventions]].

## Policies
- **Git (Q8):** commit frequently with meaningful messages; ignore `*.zip`, `*.pt`, `*.ckpt`,
  `__pycache__/`, `data/`; no force-pushes after the deadline.
- **Anti-gaming (Q9):** report metrics with *and without* features unavailable at serving time;
  enforce the behaviour-window boundary — **no future-click leakage**, with a test asserting it.

## Datasets
| | Scale | Notes |
|---|---|---|
| **EB-NeRD** (Ekstra Bladet) | ~2.7M users, 600M+ impressions, 120K+ articles | Danish. Demo/small/large bundles. Provided article embeddings. |
| **MIND** (Microsoft) | ~1M users, 160K+ articles, 15M+ impressions | English. Entity annotations. MIND-small for fast iteration. |

**Compute:** free-tier GPUs (Colab, Kaggle, Lightning AI). MIND-small and EB-NeRD demo run on a
single free GPU in a few hours.

## Files
`brief/Assignment1_v1.pdf` — the brief. Code goes here once the GitHub Classroom repo is cloned.
A local git repo is initialised here; `.gitignore` excludes `data/`, `*.zip`, checkpoints (Q8 policy).

## Raw data — status

In `data/raw/` (gitignored). All present files pass `unzip -t`. **Nothing is extracted yet** —
extraction is a Q1 pipeline step.

| File | Size | Status |
|---|---|---|
| `ebnerd/ebnerd_demo.zip` | 20.5 MB | ✅ smoke-test tier |
| `ebnerd/ebnerd_small.zip` | 80.2 MB | ✅ **headline tier** |
| `mind/MINDsmall_train.zip` | 50.5 MB | ✅ **headline tier** |
| `mind/MINDsmall_dev.zip` | 29.5 MB | ✅ headline tier |
| `mind/MINDlarge_train.zip` | 505.6 MB | ✅ Q6 scale story only |
| `mind/MINDlarge_dev.zip` | 98.7 MB | ✅ Q6 scale story only |
| `mind/MINDlarge_test.zip` | 576.6 MB | ✅ unlabelled — Q5 leaderboard only |
| `ebnerd_testset.zip` | 1.5 GB | ⏳ **needed for Q5** — see below |
| `ebnerd_large.zip` | 3.0 GB | ⏳ downloading — optional, Q6 only |
| one embeddings artifact | ~344 MB | ⏳ baseline row only — see architecture decision 4 |

MIND totals 1.3 GB zipped → ~3.4 GB extracted.

> [!warning] EB-NeRD has no test split until `ebnerd_testset.zip` lands
> Both `ebnerd_demo` and `ebnerd_small` ship **`train/` and `validation/` only**. The EB-NeRD
> Codabench submission (Q5, mandatory) needs `ebnerd_testset.zip`:
> `https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip`
> Q1–Q4 are unaffected. Get it before ~24 Aug.

> [!info] Artifact URLs need the `artifacts/` prefix
> The embedding zips are **not** at the bucket root — a bare
> `wget .../Ekstra_Bladet_word2vec.zip` returns 404 (this is why the earlier queued downloads
> silently failed). The correct form is
> `https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/<name>.zip`, e.g.
> `artifacts/google_bert_base_multilingual_cased.zip` (344.5 MB).
>
> Only **one** is needed, as the baseline row against our own computed embeddings.

---
[[_syllabus|IRE course plan]] · [[Claude-Code-Toolkit/README|Claude Code Toolkit]] · [[Dashboard]]
