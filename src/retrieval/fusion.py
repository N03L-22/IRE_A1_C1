"""Hybrid fusion: combining retrievers that fail differently (Q3.5).

Q3.5 asks "which works better, on which slices?" -- and the natural follow-up
when the answer is *"they disagree"* is to use both. Lexical retrieval misses
when the user's vocabulary does not overlap the article's; semantic retrieval
misses when topical similarity is not what the user wanted. A document that
both rank highly is a stronger candidate than one either ranks alone.

**Reciprocal Rank Fusion**, over ranks rather than scores::

    RRF(d) = sum_r  weight_r / (k + rank_r(d))

Why ranks and not a weighted score sum: BM25 scores are unbounded and
corpus-dependent, cosine similarities live in [-1, 1], and popularity is a
probability. Normalising three incomparable scales into one sum requires
choosing a normalisation, and that choice silently becomes a hyperparameter
with more influence than the retrievers. Ranks are already comparable, which
is exactly why RRF is the standard answer.

``k`` (default 60, the value from the original RRF paper) damps the influence
of the very top ranks so a single retriever cannot dominate the fusion by
being confidently wrong.

> [!warning] Fusion is not automatically better, and must be measured
> If one component is much weaker, fusion drags the stronger one down. The
> harness scores it as just another retriever, so the comparison against each
> component is direct -- and a fusion row that loses to its own best component
> is a finding worth reporting, not a bug to hide.
"""

from __future__ import annotations

import logging
from datetime import datetime

log = logging.getLogger(__name__)

#: From the original RRF paper (Cormack et al. 2009). Rarely worth tuning --
#: the ranking is insensitive to it over a wide range.
DEFAULT_RRF_K = 60


class RRFusion:
    """Fuse any number of `Retriever`s by reciprocal rank.

    Satisfies the same protocol as its components, so the harness scores it
    identically and no special case is needed anywhere.
    """

    def __init__(
        self,
        retrievers: list,
        weights: list[float] | None = None,
        k: int = DEFAULT_RRF_K,
        overfetch: int = 3,
        name: str | None = None,
    ) -> None:
        if not retrievers:
            raise ValueError("fusion needs at least one retriever")
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        if len(self.weights) != len(retrievers):
            raise ValueError("weights must match retrievers")
        self.k = k
        #: Each component is asked for more than K, because a document ranked
        #: 150th by one and 3rd by another should still be able to surface.
        #: Fetching exactly K would discard precisely the disagreements
        #: fusion exists to exploit.
        self.overfetch = overfetch
        self.name = name or "rrf(" + "+".join(r.name.split("(")[0] for r in retrievers) + ")"

    def index(self, articles) -> None:
        for r in self.retrievers:
            r.index(articles)

    def retrieve(
        self, history_text: list[str], k: int, at_time: datetime | None = None
    ) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for retriever, weight in zip(self.retrievers, self.weights):
            results = retriever.retrieve(history_text, k * self.overfetch, at_time)
            for rank, (aid, _score) in enumerate(results, start=1):
                scores[aid] = scores.get(aid, 0.0) + weight / (self.k + rank)

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]


    def score_subset(
        self, history_text: list[str], subset: list[str]
    ) -> dict[str, float]:
        """Fuse over a slate directly -- the submission path.

        RRF needs only *ranks*, and each component can rank a slate without a
        full-corpus retrieval. On a 13.3M-impression run that is the difference
        between minutes and hours (F32).

        > [!warning] This is NOT a drop-in equivalent of ``retrieve()``
        > ``retrieve()`` ranks each component over the **whole corpus** and
        > fuses those global ranks. This ranks each component **within the
        > slate**. A document lying 3rd of 20,738 globally but 1st of 30
        > in-slate receives a different RRF weight, so the two orderings can
        > differ: measured **15 discordant pairs out of 435** (3.4%) on a
        > 400-document toy corpus.
        >
        > The in-slate frame is the correct one *for a submission* -- the
        > output is a permutation of the candidates shown, and a global rank
        > for a document that is not in the slate is irrelevant to it. But do
        > not use this for a reported metric, and do not assume a fusion
        > submission reproduces the harness's fusion row exactly.
        """
        scores: dict[str, float] = {}
        for retriever, weight in zip(self.retrievers, self.weights):
            for aid, rank in _subset_ranks(retriever, history_text, subset).items():
                scores[aid] = scores.get(aid, 0.0) + weight / (self.k + rank)
        return scores


class PopularityPrior:
    """Blend a retriever with train-split popularity.

    Motivated by measurement, not intuition: on MIND, popularity **beat** BM25
    at every K with non-overlapping CIs (recall@100 0.0682 vs 0.0159, F25). A
    retriever that ignores that signal is leaving the stronger baseline on the
    table.

    > [!warning] Popularity must come from the TRAIN split only
    > Popularity computed over the evaluation window is future knowledge, and
    > it is the kind that flatters you. Built once from train and frozen, this
    > is a legitimate serving-time feature; recomputed over test it is a leak.
    > `alpha` is reported so the with/without comparison (Q9) is available.
    """

    def __init__(self, retriever, popularity: dict[str, float], alpha: float = 0.3,
                 k: int = DEFAULT_RRF_K, overfetch: int = 3) -> None:
        self.inner = retriever
        self.popularity = popularity
        self.alpha = alpha
        self.k = k
        self.overfetch = overfetch
        self.name = f"{retriever.name}+pop({alpha:g})"
        self._top: list[str] = [
            aid for aid, _ in sorted(popularity.items(), key=lambda kv: -kv[1])
        ]

    def index(self, articles) -> None:
        self.inner.index(articles)

    def retrieve(
        self, history_text: list[str], k: int, at_time: datetime | None = None
    ) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}

        for rank, (aid, _s) in enumerate(
            self.inner.retrieve(history_text, k * self.overfetch, at_time), start=1
        ):
            scores[aid] = scores.get(aid, 0.0) + (1 - self.alpha) / (self.k + rank)

        for rank, aid in enumerate(self._top[: k * self.overfetch], start=1):
            scores[aid] = scores.get(aid, 0.0) + self.alpha / (self.k + rank)

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]


def _subset_ranks(retriever, history_text: list[str], subset: list[str]) -> dict[str, int]:
    """Rank a slate with one retriever, returning id -> 1-based rank.

    RRF works on ranks, so a component that can score a subset directly does
    not need a full retrieval -- which is what keeps fusion affordable on a
    13.3M-impression submission run.
    """
    scored = retriever.score_subset(history_text, subset)
    if not scored:
        return {}
    order = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    return {aid: i for i, (aid, _s) in enumerate(order, start=1)}
