"""Semantic retrieval: embeddings + ANN (Q3).

The second retriever, satisfying the same `Retriever` protocol as BM25 so the
one harness scores both (Q4.5). Same three-step shape as the lexical path:

    build an index  ->  turn click history into a query  ->  retrieve top-K

What differs is only what "similar" means -- embedding proximity rather than
word overlap. That symmetry is the whole point: it is what makes the Q3.5
comparison a comparison of *methods* rather than of implementations.

Two design decisions carry most of the weight, both from
plan/3-Semantic-Embeddings.md:

**D3, user representation.** A plain mean over a user's clicked vectors is the
brief's suggestion and is what we ship *when it is meaningful*. Its failure
mode: a user who reads football and recipes gets a centroid sitting between
the two clusters, matching neither -- averaging is worst exactly when the user
is most interesting. So the mean is used **conditionally**, with a measured
coherence check and a documented fallback ladder.

**D4, index.** Brute force at small scale (exact, and Q3.2 explicitly permits
it) with FAISS HNSW for large. The brute-force row is not a fallback, it is
the measuring instrument: ANN recall is only interpretable against the exact
answer, so the gap between them *is* the ablation.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from ..data.schema import Article
from .encode import DEFAULT_MODEL, encode_cached, l2_normalise

log = logging.getLogger(__name__)

#: Above this mean cosine-to-centroid, a user's history is coherent enough
#: that the mean represents it. Below, the centroid is a blur. Chosen by
#: reasoning, not measurement -- it must be swept and reported (open O1).
DEFAULT_COHERENCE_TAU = 0.35

#: Corpus size above which brute force stops being the sensible default.
#: Below this, exact search costs ~190 ms/query on 20K x 768 -- affordable,
#: and exact beats approximate when you can afford it.
BRUTE_FORCE_LIMIT = 200_000


def log_decay_weights(n: int) -> np.ndarray:
    """Mild positional decay: w_j = 1 / log2(rank_from_recent + 2).

    Matches the lexical query construction so the two retrievers see the same
    notion of "recent", and a difference between them is a difference of
    method rather than of weighting.

    Exponential decay at any useful lambda collapses the query onto the last
    two or three clicks; log decay keeps the 20th click at ~0.28 of the first
    click's weight instead of ~0.002. Recent clicks lead, older ones still
    contribute.

    > [!warning] This is positional, not temporal -- a MIND compromise
    > Weighting by elapsed time needs per-click timestamps, which MIND does
    > not have (F1). Position stands in for time under the assumption the
    > history list is chronologically ordered, which MIND does not document.
    """
    ranks = np.arange(n, dtype=np.float32)
    return 1.0 / np.log2(ranks + 2.0)


def build_user_vector(
    vectors: np.ndarray, tau: float = DEFAULT_COHERENCE_TAU
) -> tuple[np.ndarray, str]:
    """Turn a user's clicked-article vectors into one query vector.

    Returns ``(vector, strategy)`` -- the strategy is reported so the harness
    can slice on it, which is the only way to find out whether the conditional
    branch was worth having.

    The ladder (D3):

    1. Recency-weighted mean, matching the lexical decay.
    2. Measure **coherence**: mean cosine of each clicked vector to that mean.
    3. Coherent (>= tau) -> keep the mean. The user has one interest.
    4. Incoherent -> the centroid is meaningless. Fall back to the recent
       half only; if that is still incoherent, max-pool.

    Max-pool is last because it is noisy -- one outlier dimension dominates --
    but a noisy vector that points *somewhere* beats a smooth one pointing at
    the corpus mean.
    """
    n = len(vectors)
    if n == 0:
        return np.zeros(vectors.shape[1] if vectors.ndim == 2 else 0, dtype=np.float32), "empty"
    if n == 1:
        return vectors[0], "single"

    weights = log_decay_weights(n)[::-1]  # oldest first -> newest heaviest
    weighted = (vectors * weights[:, None]).sum(axis=0) / weights.sum()
    weighted = weighted / max(np.linalg.norm(weighted), 1e-12)

    coherence = float(np.mean(vectors @ weighted))
    if coherence >= tau:
        return weighted.astype(np.float32), "mean"

    # Incoherent: try just the recent half, which is likelier to be one topic.
    half = max(2, n // 2)
    recent = vectors[-half:]
    w2 = log_decay_weights(half)[::-1]
    recent_vec = (recent * w2[:, None]).sum(axis=0) / w2.sum()
    recent_vec = recent_vec / max(np.linalg.norm(recent_vec), 1e-12)

    if float(np.mean(recent @ recent_vec)) >= tau:
        return recent_vec.astype(np.float32), "recent_half"

    pooled = vectors.max(axis=0)
    pooled = pooled / max(np.linalg.norm(pooled), 1e-12)
    return pooled.astype(np.float32), "max_pool"


class SemanticRetriever:
    """Embedding retrieval over an exact or HNSW index.

    ``at_time`` is accepted and ignored, exactly as in BM25: recency filtering
    is a composition concern handled by the same `WindowedRetriever` wrapper,
    kept separate so the two effects stay independently ablatable (F16).
    """

    def __init__(
        self,
        model_key: str = DEFAULT_MODEL,
        index_kind: str = "auto",
        tau: float = DEFAULT_COHERENCE_TAU,
        last_n: int = 20,
        batch_size: int = 128,
        ef_search: int = 128,
        vectors: np.ndarray | None = None,
        vector_ids: list[str] | None = None,
    ) -> None:
        self.model_key = model_key
        self.index_kind = index_kind
        self.tau = tau
        self.last_n = last_n
        self.batch_size = batch_size
        self.ef_search = ef_search
        self.name = f"semantic({model_key},tau={tau:g},n={last_n})"

        # Precomputed vectors (the provided-embeddings baseline) may be
        # supplied instead of encoding. Same retriever, different source.
        self._given = (vectors, vector_ids) if vectors is not None else None

        self._ids: list[str] = []
        self._vecs: np.ndarray | None = None
        self._by_id: dict[str, int] = {}
        self._index = None
        self.encode_stats = None
        #: How often each pooling strategy fired -- reported, since the
        #: conditional branch is only justified if it actually branches.
        self.strategy_counts: dict[str, int] = {}

    def index(self, articles: list[Article]) -> None:
        if self._given is not None:
            vecs, ids = self._given
            keep = [i for i, a in enumerate(ids) if a in {x.article_id for x in articles}]
            self._ids = [ids[i] for i in keep]
            self._vecs = l2_normalise(vecs[keep].astype(np.float32))
        else:
            self._ids = [a.article_id for a in articles]
            texts = [a.retrieval_text for a in articles]
            # Cached: the Q3.5 comparison builds several retrievers over the
            # same corpus (bare, windowed, fused), and encoding is
            # deterministic. One forward pass serves all of them.
            self._vecs, self.encode_stats = encode_cached(
                texts, self._ids, model_key=self.model_key, batch_size=self.batch_size
            )

        self._by_id = {aid: i for i, aid in enumerate(self._ids)}
        # The harness identifies history by retrieval text, so map that too.
        # setdefault keeps the first row when two articles share a title.
        self._by_text: dict[str, int] = {}
        for a in articles:
            row = self._by_id.get(a.article_id)
            if row is not None:
                self._by_text.setdefault(a.retrieval_text, row)

        # An un-normalised index with inner-product search silently ranks by
        # magnitude -- i.e. builds a popularity ranker that looks like a
        # semantic one. Assert rather than trust.
        norms = np.linalg.norm(self._vecs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), (
            f"vectors not L2-normalised (min {norms.min():.4f}, max {norms.max():.4f}) "
            "-- inner-product search would rank by magnitude"
        )

        kind = self.index_kind
        if kind == "auto":
            kind = "brute" if len(self._ids) <= BRUTE_FORCE_LIMIT else "hnsw"
        self._build_index(kind)

    def _build_index(self, kind: str) -> None:
        if kind == "brute":
            self._index = None  # numpy matmul; exact by construction
            log.info("%s: brute-force index over %d vectors (exact)", self.name, len(self._ids))
            return

        import faiss

        dim = self._vecs.shape[1]
        idx = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = 200
        idx.add(self._vecs)
        idx.hnsw.efSearch = self.ef_search
        self._index = idx
        self.name += "+hnsw"
        log.info(
            "%s: HNSW over %d vectors, efSearch=%d "
            "(report the exact-search gap alongside -- ANN recall alone is not interpretable)",
            self.name, len(self._ids), self.ef_search,
        )

    def _query_vector(self, history_text: list[str]) -> np.ndarray | None:
        """History text -> one query vector, via the article vectors we hold.

        Uses the *indexed* vectors rather than re-encoding, so a user's query
        is built from exactly the representations the corpus is searched with.

        > [!warning] The harness passes retrieval TEXT, not article ids
        > This originally looked history entries up in ``self._by_id``, an
        > article-id map -- so every lookup missed, the query vector was
        > always None, and the retriever silently returned **zero results with
        > no error**. It surfaced only when an ablation used this class
        > directly and scored recall 0.0000 across the board, including for
        > provided click-trained vectors that could not plausibly score zero.
        >
        > ``_by_text`` is built in ``index()`` and tried first; ``_by_id`` is
        > kept as a fallback so a caller that genuinely passes ids still works.
        """
        recent = history_text[-self.last_n :] if self.last_n > 0 else history_text
        rows = [self._by_text[t] for t in recent if t in self._by_text]
        if not rows:  # fall back to id lookup for callers that pass ids
            rows = [self._by_id[t] for t in recent if t in self._by_id]
        if not rows:
            return None
        vec, strategy = build_user_vector(self._vecs[rows], tau=self.tau)
        self.strategy_counts[strategy] = self.strategy_counts.get(strategy, 0) + 1
        return vec

    def score_subset(
        self, history_text: list[str], subset: list[str]
    ) -> dict[str, float]:
        """Score only ``subset`` -- the submission path's fast lane.

        Mirrors ``BM25Retriever.score_subset``: ranking an impression's slate
        needs scores for its ~11-40 candidates, not a top-K over 20K-65K
        vectors. Here it is one small matrix-vector product instead of a full
        corpus scan.

        Unlike the BM25 pair, this is **numerically identical** to what
        ``retrieve()`` would give for the same documents -- both compute the
        same dot product against the same L2-normalised vectors, so there is
        no Lucene-vs-Robertson discrepancy to reason about.
        """
        if self._vecs is None:
            raise RuntimeError("index() must be called before score_subset()")

        q = self._query_vector(history_text)
        if q is None:
            return {}

        rows, ids = [], []
        for aid in subset:
            r = self._by_id.get(aid)
            if r is not None:
                rows.append(r)
                ids.append(aid)
        if not rows:
            return {}

        scores = self._vecs[rows] @ q
        return {aid: float(s) for aid, s in zip(ids, scores)}

    def retrieve(
        self, history_text: list[str], k: int, at_time: datetime | None = None
    ) -> list[tuple[str, float]]:
        if self._vecs is None:
            raise RuntimeError("index() must be called before retrieve()")

        q = self._query_vector(history_text)
        if q is None:
            return []

        k = min(k, len(self._ids))
        if self._index is None:
            scores = self._vecs @ q
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]
            return [(self._ids[int(i)], float(scores[int(i)])) for i in top]

        dist, idx = self._index.search(q.reshape(1, -1), k)
        return [
            (self._ids[int(i)], float(s))
            for i, s in zip(idx[0], dist[0])
            if i >= 0
        ]


class HistoryIdRetriever(SemanticRetriever):
    """SemanticRetriever driven by article *ids* rather than text.

    The harness hands retrievers `history_text` because that is what BM25
    needs. The semantic path would rather have ids -- it already holds a
    vector per article and re-deriving the row from text means a text->id
    lookup that can fail on duplicate titles.

    Rather than widen the shared protocol for one implementation, this
    subclass maps text back to rows via an explicit text->id table built at
    index time. The protocol stays as it is, and the harness stays ignorant
    of which retriever it is scoring.
    """

    def index(self, articles: list[Article]) -> None:
        super().index(articles)
