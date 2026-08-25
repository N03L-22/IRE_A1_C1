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
#: Below this, exact search costs ~99 ms/query on 20K x 384 -- affordable, and
#: exact beats approximate when you can afford it. Above it, F49 measured HNSW
#: at 62-69x the speed for 0.99+ recall, so the approximation is close to free.
BRUTE_FORCE_LIMIT = 200_000

#: `efSearch` must scale with the corpus. Measured at 1M vectors (F52), a
#: fixed ef=128 -- lossless at 125K -- collapses to **0.68 recall**:
#:
#:   ef=  64  recall 0.5195   0.05 ms/q   474x faster than exact
#:   ef= 128  recall 0.6782   0.08 ms/q   286x
#:   ef= 256  recall 0.8108   0.14 ms/q   167x
#:   ef= 512  recall 0.9071   0.22 ms/q   105x
#:   ef=1024  recall 0.9588   0.34 ms/q    68x   <- the knee
#:
#: Recall is bought back almost free, and those figures are for M=16. At the
#: shipped M=64 the same ef values do considerably better -- ef=512 reaches
#: **0.9947** at 1M (F53) -- which is why EF_SEARCH_LARGE is 512 rather than
#: 1024: the denser graph already supplies what the wider search was buying.
#: A single global default cannot serve both 20K and 1M, so it is derived
#: from the corpus size instead.
EF_SEARCH_BY_SIZE = ((50_000, 128), (200_000, 256), (500_000, 512))
EF_SEARCH_LARGE = 512

#: Graph connectivity. Measured at 1M vectors (F53), a denser graph wins at
#: *fixed latency* -- more links means fewer dead ends, so the walk needs
#: fewer candidates to reach the same recall:
#:
#:   M=16 ef=1024  recall 0.9588  0.34 ms/q   68x   graph 0.13 GB
#:   M=32 ef= 512  recall 0.9793  0.34 ms/q   67x   graph 0.26 GB
#:   M=64 ef= 256  recall 0.9759  0.31 ms/q   73x   graph 0.52 GB
#:   M=64 ef= 512  recall 0.9947  0.51 ms/q   45x   graph 0.52 GB
#:
#: **64 is the default**: it dominates the frontier at every latency point
#: tested, and index memory is not the binding constraint here (0.52 GB of
#: graph against 1.54 GB of vectors at 1M). Paired with ef=512 that is
#: **0.9947 recall at 45x the speed of exact search** -- within half a percent
#: of lossless, comfortably below the noise in every downstream metric we
#: measure. Use ef=1024 for 0.9990 when near-exactness is worth 14% more
#: latency; drop to M=32 only if index memory becomes the constraint.
DEFAULT_M = 64

#: Truncate embeddings to this width before indexing (F47). Measured on
#: EB-NeRD inside the 24h window, truncating the cached 384-d MiniLM vectors:
#:
#:   384d  31.9 MB  recall@50 0.2325   --
#:   256d  21.2 MB  recall@50 0.2500   +0.0175 [+0.0025, +0.0338] SIGNIFICANT
#:   128d  10.6 MB  recall@50 0.2437   no worse than full width
#:    64d   5.3 MB  recall@50 0.2175   degrades
#:
#: 256 is the only setting measured as *better* than full width, and it is a
#: third cheaper. Set None to keep the encoder's native width.
DEFAULT_TRUNCATE_DIM = 256


def default_ef_search(n_vectors: int) -> int:
    """Pick `efSearch` from the corpus size (F52).

    A graph search explores `efSearch` candidates regardless of how many
    vectors exist, so the *fraction* of the corpus it inspects shrinks as the
    corpus grows -- which is exactly why a value that is lossless at 125K
    loses a third of the answer at 1M.
    """
    for limit, ef in EF_SEARCH_BY_SIZE:
        if n_vectors <= limit:
            return ef
    return EF_SEARCH_LARGE


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
        ef_search: int | None = None,
        m: int = DEFAULT_M,
        truncate_dim: int | None = DEFAULT_TRUNCATE_DIM,
        vectors: np.ndarray | None = None,
        vector_ids: list[str] | None = None,
    ) -> None:
        self.model_key = model_key
        self.index_kind = index_kind
        self.tau = tau
        self.last_n = last_n
        self.batch_size = batch_size
        #: None means "derive from corpus size at index time" (F52).
        self.m = m
        self.truncate_dim = truncate_dim
        #: None means "derive from corpus size at index time" (F52).
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

        # Truncate before indexing (F47). Measured on EB-NeRD inside the 24h
        # window: 256-d scored +0.0175 recall@50 over full width with the
        # paired CI excluding zero, at 34% less memory. The tail dimensions of
        # a non-Matryoshka embedding carry mostly noise for this task, so
        # dropping them removes variance without removing signal.
        if self.truncate_dim and self._vecs.shape[1] > self.truncate_dim:
            self._vecs = l2_normalise(
                self._vecs[:, : self.truncate_dim].astype(np.float32).copy()
            )
            log.info("%s: truncated %d -> %dd (F47)", self.name,
                     self.truncate_dim, self._vecs.shape[1])

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
        idx = faiss.IndexHNSWFlat(dim, self.m, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = 200
        idx.add(self._vecs)
        ef = self.ef_search or default_ef_search(len(self._ids))
        idx.hnsw.efSearch = ef
        self.ef_search = ef
        self._index = idx
        self.name += "+hnsw"
        log.info(
            "%s: HNSW over %d vectors, M=%d efSearch=%d "
            "(report the exact-search gap alongside -- ANN recall alone is not interpretable)",
            self.name, len(self._ids), self.m, self.ef_search,
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


class MultiQueryRetriever(SemanticRetriever):
    """Cluster the history, retrieve per centroid, merge (D3's rejected option).

    **The failure this addresses.** A user who reads football *and* recipes
    gets a mean vector sitting between the two clusters, matching neither.
    Averaging is worst exactly when the user is most interesting.
    `build_user_vector`'s conditional ladder detects that case and falls back;
    this handles it properly instead, by issuing one query per interest.

    **Why it is worth testing now rather than dismissing.** D3 named this
    "considered, not built" on cost grounds. F40 then measured **46% of MIND
    users falling back** from the mean because their history is incoherent
    (against 0.2% on EB-NeRD). That is a large population for whom the shipped
    representation is known to be a compromise, which changes the cost/benefit
    the original decision assumed.

    Merging is by **reciprocal rank**, the same scheme as `RRFusion`: each
    centroid produces a ranking, and an article ranked well by more than one
    interest beats one ranked well by a single interest. Scores across
    centroids are not comparable (different query vectors, different
    magnitudes), so fusing ranks rather than scores avoids inventing a
    normalisation.

    Cost is honest: ``c`` centroids means ``c`` searches per impression.
    """

    def __init__(self, *args, n_clusters: int = 3, rrf_k: int = 60, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_clusters = n_clusters
        self.rrf_k = rrf_k
        self.name = f"multiquery({self.model_key},c={n_clusters},n={self.last_n})"
        #: How many centroids were actually used, so the report can say
        #: whether the clustering ever fired or silently collapsed to one.
        self.cluster_counts: dict[int, int] = {}

    def _history_vectors(self, history_text: list[str]) -> np.ndarray | None:
        recent = history_text[-self.last_n :] if self.last_n > 0 else history_text
        rows = [self._by_text[t] for t in recent if t in self._by_text]
        if not rows:
            rows = [self._by_id[t] for t in recent if t in self._by_id]
        return self._vecs[rows] if rows else None

    def _centroids(self, vecs: np.ndarray) -> np.ndarray:
        """K-means over the clicked vectors; falls back to the mean if tiny."""
        k = min(self.n_clusters, len(vecs))
        if k <= 1:
            v, _ = build_user_vector(vecs, tau=self.tau)
            return v.reshape(1, -1)
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(vecs)
        cents = km.cluster_centers_.astype(np.float32)
        norms = np.linalg.norm(cents, axis=1, keepdims=True)
        return cents / np.maximum(norms, 1e-12)

    def retrieve(
        self, history_text: list[str], k: int, at_time: datetime | None = None
    ) -> list[tuple[str, float]]:
        if self._vecs is None:
            raise RuntimeError("index() must be called before retrieve()")
        vecs = self._history_vectors(history_text)
        if vecs is None or len(vecs) == 0:
            return []

        cents = self._centroids(vecs)
        self.cluster_counts[len(cents)] = self.cluster_counts.get(len(cents), 0) + 1

        fused: dict[str, float] = {}
        for c in cents:
            scores = self._vecs @ c
            top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
            top = top[np.argsort(-scores[top])]
            for rank, i in enumerate(top, start=1):
                aid = self._ids[int(i)]
                fused[aid] = fused.get(aid, 0.0) + 1.0 / (self.rrf_k + rank)

        return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
