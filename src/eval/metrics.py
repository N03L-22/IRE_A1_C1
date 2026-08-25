"""The metrics, in two regimes (Q4.1, Q2.4/Q3.4).

plan/4-Evaluation-Harness.md D2 makes the distinction this module is built
around, and it is the most common conceptual error in the assignment:

    recall@K          retrieve from the WHOLE CORPUS (21K-65K articles).
                      "Did the clicked article survive the cut?"
                      The candidate-generation metric.

    AUC/MRR/nDCG      score the impression's OWN SLATE (mean 11-12 items).
                      "Did you rank the clicked one above the others shown?"
                      The leaderboard's metric.

A retriever produces a corpus-wide ranking; to get AUC you restrict-and-rerank
onto the slate. Conflating the two produces numbers that look fine and mean
nothing, so every function here is named for its regime and the harness labels
every reported row with it.

All slate metrics return **per-impression values**, never a pre-averaged
scalar. That is what lets the bootstrap resample at impression level (D5) --
the constraint that keeps the CIs honest.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Corpus regime -- did the clicked article survive the cut?
# ---------------------------------------------------------------------------


def recall_at_k(retrieved: list[str], clicked: set[str], k: int) -> float:
    """Fraction of clicked articles present in the top-K.

    Q2.4 names K in {50, 100, 200}. With 99.5% single-click impressions (F7)
    this is usually 1.0 or 0.0, but the fractional form is correct for the
    multi-click tail and costs nothing.

    Returns 0.0 for an impression with no clicks -- the caller should filter
    unlabelled impressions rather than relying on that.
    """
    if not clicked:
        return 0.0
    hits = sum(1 for aid in retrieved[:k] if aid in clicked)
    return hits / len(clicked)


# ---------------------------------------------------------------------------
# Slate regime -- did you rank the clicked one above the others shown?
# ---------------------------------------------------------------------------


def rank_slate(
    scores: dict[str, float], slate: list[str], fallback_order: list[str] | None = None
) -> list[str]:
    """Order one impression's slate by retriever score, best first.

    Candidates the retriever never scored get -inf and fall to the bottom in
    their original slate order. That is deliberate: an *arbitrary* tie-break
    over unscored items is what made the popularity baseline look like a
    popularity ranker when it was really a slate-order ranker. Preserving the
    given order at least makes the fallback explicit and deterministic.
    """
    order = fallback_order if fallback_order is not None else slate
    position = {aid: i for i, aid in enumerate(order)}
    return sorted(slate, key=lambda aid: (-scores.get(aid, -math.inf), position.get(aid, 0)))


def auc(ranked: list[str], clicked: set[str]) -> float | None:
    """Probability a random clicked item outranks a random ignored one.

    Computed by the rank-sum identity rather than by enumerating pairs::

        AUC = (sum of ranks of positives - n_pos*(n_pos+1)/2) / (n_pos * n_neg)

    Returns ``None`` when the slate is all-positive or all-negative -- AUC is
    undefined there, and averaging a fabricated 0.5 into the mean would be a
    quiet lie. The harness drops those impressions and reports how many.
    """
    n = len(ranked)
    pos = [i for i, aid in enumerate(ranked) if aid in clicked]
    n_pos, n_neg = len(pos), n - len(pos)
    if n_pos == 0 or n_neg == 0:
        return None
    # Ranks are 1-indexed ascending; our list is best-first, so a low index is
    # a good rank. Sum of positive ranks under that convention:
    rank_sum = sum(i + 1 for i in pos)
    # Probability that a positive is ranked *worse* than a negative, inverted.
    return 1.0 - (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def mrr(ranked: list[str], clicked: set[str]) -> float:
    """Reciprocal rank of the first hit; 0.0 if none.

    With single-click impressions (F7) this reduces to 1/rank of the one
    positive, and MRR == recall@1 when the click is at position 1. Expected,
    not an error -- the pitfall table says so.
    """
    for i, aid in enumerate(ranked):
        if aid in clicked:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list[str], clicked: set[str], k: int) -> float:
    """nDCG with binary relevance.

    With binary relevance the usual (2^rel - 1) gain reduces to 1/0, so this
    is sum(1/log2(i+2)) over hits in the top-K, divided by the same sum over
    the ideal ordering. Stated explicitly because the graded form should be
    unambiguous (D2).

    Note EB-NeRD slates average 11-12 items, so nDCG@10 covers nearly the
    whole slate and correlates strongly with nDCG@5. Expected.
    """
    if not clicked:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, aid in enumerate(ranked[:k]) if aid in clicked)
    ideal_hits = min(len(clicked), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


# ---------------------------------------------------------------------------
# Beyond-accuracy (Q4.2)
# ---------------------------------------------------------------------------


def intra_list_diversity(items: list[str], category: dict[str, str]) -> float | None:
    """Mean pairwise dissimilarity within one result list, by category.

    D3 chose embedding similarity for the headline number with category as a
    readable cross-check. Category is implemented here because it is
    dependency-free and, more importantly, **not circular**: scoring an
    embedding retriever with embedding similarity measures the retriever
    against its own objective. The circularity caveat belongs in the note.

    Returns None for lists shorter than 2 -- no pairs, no diversity.
    """
    cats = [category.get(aid) for aid in items if aid in category]
    n = len(cats)
    if n < 2:
        return None
    same = sum(
        1 for i in range(n) for j in range(i + 1, n) if cats[i] == cats[j]
    )
    pairs = n * (n - 1) / 2
    return 1.0 - same / pairs


def novelty(items: list[str], train_popularity: dict[str, float]) -> float | None:
    """Mean self-information -log2 p(d) of the recommended items.

    ``train_popularity`` must be built from the **train split only** -- using
    test-period popularity is leakage (D3), and it is the kind that produces a
    better-looking number, which is exactly why it needs saying.

    Unseen articles are maximally novel. Rather than an infinite value, they
    get the information content of a single hypothetical occurrence, which is
    the finite ceiling implied by the training sample.
    """
    if not items:
        return None
    floor = min(train_popularity.values()) if train_popularity else 1.0
    vals = [-math.log2(train_popularity.get(aid, floor) or floor) for aid in items]
    return sum(vals) / len(vals)


def coverage(all_retrieved: list[list[str]], corpus_size: int) -> float:
    """Fraction of the catalogue that appears in *any* user's top-K.

    A global quantity, not a per-impression one, which is why the bootstrap
    has to recompute it per resample rather than averaging per-impression
    values (D5). Computed over the **full corpus** (D3), stated.
    """
    if corpus_size <= 0:
        return 0.0
    seen: set[str] = set()
    for items in all_retrieved:
        seen.update(items)
    return len(seen) / corpus_size
