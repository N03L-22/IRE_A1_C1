"""Walking skeleton: Q1 -> Q2 -> Q3 -> Q4 end to end, thinly.

The point is NOT to produce good numbers. It is to prove the *interfaces* fit
together before any component is built properly, so that each piece can then be
replaced without redesigning its neighbours:

    readers  ->  temporal split  ->  Retriever  ->  harness  ->  recall@K

Every stage here is the simplest thing that satisfies its contract:

    ============  ==========================  ============================
    stage         skeleton implementation     phase-file replacement
    ============  ==========================  ============================
    split         timestamp percentile        src/data/split.py (Phase 1)
    lexical       token-overlap scoring       BM25 via bm25s (Phase 2)
    semantic      hashed bag-of-words vecs    encoder + ANN (Phase 3)
    metrics       recall@K only               full harness + CIs (Phase 4)
    ============  ==========================  ============================

Each replacement is a drop-in: same Retriever protocol, same evaluate() call.
If swapping one requires touching another, the interface was wrong -- which is
exactly what this file exists to find out early.

NOT a deliverable. Numbers from here are meaningless and must never appear in
the design note.

Usage::

    python3 -m src.skeleton --dataset ebnerd --tier demo --limit 2000
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .data.readers import SPLIT_NAMES, get_reader, normalise
from .data.schema import Article, History, Impression
from .resources import add_arguments, from_args

log = logging.getLogger("skeleton")


# --------------------------------------------------------------------------
# Stage 1 -- temporal split  (replaced by src/data/split.py)
# --------------------------------------------------------------------------


def temporal_split(
    impressions: list[Impression], val_fraction: float = 0.1
) -> tuple[list[Impression], list[Impression]]:
    """Split by time: the last ``val_fraction`` of the window becomes val.

    Never random -- Q1.3. The real implementation also holds out the official
    test period and truncates history; this proves only the ordering rule.
    """
    ordered = sorted(impressions, key=lambda x: x.time)
    cut = int(len(ordered) * (1 - val_fraction))
    return ordered[:cut], ordered[cut:]


# --------------------------------------------------------------------------
# Stage 2 -- retrievers  (replaced by Phase 2 / Phase 3)
# --------------------------------------------------------------------------


class OverlapRetriever:
    """Stand-in for BM25: score by raw query-term overlap, IDF-weighted.

    Genuinely a lexical retriever, just a bad one -- no TF saturation, no
    length normalisation. Those two absences are precisely what BM25 adds, so
    replacing this with bm25s should visibly improve recall. If it does not,
    something upstream is wrong.
    """

    name = "overlap"

    def __init__(self) -> None:
        self._docs: dict[str, Counter[str]] = {}
        self._idf: dict[str, float] = {}

    def index(self, articles: list[Article]) -> None:
        df: Counter[str] = Counter()
        for a in articles:
            terms = Counter(normalise(a.retrieval_text).lower().split())
            self._docs[a.article_id] = terms
            df.update(terms.keys())
        n = max(1, len(self._docs))
        self._idf = {t: math.log(1 + n / (1 + c)) for t, c in df.items()}

    def retrieve(
        self, history_text: list[str], k: int, at_time: datetime | None = None
    ) -> list[tuple[str, float]]:
        query = Counter(" ".join(history_text).lower().split())
        if not query:
            return []
        scored: list[tuple[str, float]] = []
        for aid, terms in self._docs.items():
            s = sum(self._idf.get(t, 0.0) for t in query if t in terms)
            if s > 0:
                scored.append((aid, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


class HashedVectorRetriever:
    """Stand-in for embeddings: hashed bag-of-words, cosine similarity.

    Not semantic at all -- it cannot see that "car" and "automobile" are
    related, which is the entire point of the real thing. What it *does* share
    with the real retriever is the shape: fixed-width vectors, mean-pooled user
    representation, L2 normalisation, cosine top-K. So the plumbing Phase 3
    needs is exercised, including the normalisation assertion that catches the
    most common bug in that phase.
    """

    name = "hashed-vec"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self._ids: list[str] = []
        self._vecs: list[list[float]] = []

    def _embed(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for token in text.lower().split():
            v[hash(token) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else v

    def index(self, articles: list[Article]) -> None:
        for a in articles:
            self._ids.append(a.article_id)
            self._vecs.append(self._embed(a.retrieval_text))
        # The Phase 3 assertion, in miniature: un-normalised vectors with
        # inner-product search silently build a popularity ranker.
        for v in self._vecs:
            n = math.sqrt(sum(x * x for x in v))
            assert n == 0 or abs(n - 1.0) < 1e-6, f"vector not normalised: {n}"

    def retrieve(
        self, history_text: list[str], k: int, at_time: datetime | None = None
    ) -> list[tuple[str, float]]:
        q = self._embed(" ".join(history_text))
        if not any(q):
            return []
        scored = [
            (aid, sum(a * b for a, b in zip(q, v))) for aid, v in zip(self._ids, self._vecs)
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


class RecencyRetriever:
    """Return the K most recently published articles before ``at_time``.

    Ignores the user completely -- and on EB-NeRD it beats every content-based
    retriever by an order of magnitude (finding F16: recall@50 = 0.92, because
    94% of clicks land on articles under a day old while only ~1% of the corpus
    is that fresh).

    Legitimate, not a leak: publish time is known at serving time. What would
    be a leak is ranking by *future* engagement, which this does not touch.

    Keep this as a baseline in the real harness. Any retriever that cannot beat
    "show the newest articles" has not earned its complexity.
    """

    name = "recency"

    def __init__(self) -> None:
        self._by_time: list[tuple[datetime, str]] = []

    def index(self, articles: list[Article]) -> None:
        self._by_time = sorted(
            ((a.published_time, a.article_id) for a in articles if a.published_time),
            key=lambda x: -x[0].timestamp(),
        )

    def retrieve(
        self, history_text: list[str], k: int, at_time: datetime | None = None
    ) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for published, aid in self._by_time:
            if at_time is not None and published >= at_time:
                continue  # never return an article that did not exist yet
            out.append((aid, 1.0 / (1 + len(out))))
            if len(out) >= k:
                break
        return out


class WindowedRetriever:
    """Wrap any retriever so it only ever scores recently-published articles.

    This is the composition F16 argues for: recency decides *which* articles
    are eligible, the inner retriever decides *which of those* the user wants.
    Neither alone is enough -- recency has no personalisation, and text
    matching over the full corpus drowns in stale but topically-similar
    articles.

    Written as a wrapper rather than baked into each retriever so the two
    concerns stay separable and independently ablatable, which is what the
    design note needs.
    """

    def __init__(self, inner, window_hours: float = 24.0) -> None:
        self.inner = inner
        self.window_hours = window_hours
        self.name = f"{inner.name}+{window_hours:g}h"
        self._published: dict[str, datetime] = {}

    def index(self, articles: list[Article]) -> None:
        self._published = {a.article_id: a.published_time for a in articles if a.published_time}
        self.inner.index(articles)

    def retrieve(
        self, history_text: list[str], k: int, at_time: datetime | None = None
    ) -> list[tuple[str, float]]:
        if at_time is None:
            return self.inner.retrieve(history_text, k, at_time)
        # Over-fetch, then keep only what falls inside the window. Crude, but
        # the real implementation filters the posting list / ANN index itself.
        cutoff = at_time.timestamp() - self.window_hours * 3600
        out: list[tuple[str, float]] = []
        for aid, score in self.inner.retrieve(history_text, k * 20, at_time):
            p = self._published.get(aid)
            if p is not None and cutoff <= p.timestamp() < at_time.timestamp():
                out.append((aid, score))
                if len(out) >= k:
                    break
        return out


class PopularityRetriever:
    """The floor. If a real retriever cannot beat this, something is wrong.

    Ignores the user entirely. Kept in the skeleton because it is the cheapest
    possible sanity check and stays in the final harness as a baseline row.
    """

    name = "popularity"

    def __init__(self) -> None:
        self._top: list[tuple[str, float]] = []

    def index_from_clicks(self, impressions: list[Impression]) -> None:
        counts: Counter[str] = Counter()
        for imp in impressions:
            counts.update(imp.clicked)
        total = max(1, sum(counts.values()))
        self._top = [(aid, c / total) for aid, c in counts.most_common()]

    def index(self, articles: list[Article]) -> None:  # protocol compatibility
        pass

    def retrieve(
        self, history_text: list[str], k: int, at_time: datetime | None = None
    ) -> list[tuple[str, float]]:
        return self._top[:k]


# --------------------------------------------------------------------------
# Stage 3 -- evaluation  (replaced by Phase 4)
# --------------------------------------------------------------------------


@dataclass
class Result:
    retriever: str
    k: int
    recall: float
    n_impressions: int
    seconds: float

    def __str__(self) -> str:
        return (
            f"  {self.retriever:12s} recall@{self.k:<4d} {self.recall:.4f}   "
            f"n={self.n_impressions:<6d} {self.seconds:6.1f}s"
        )


def evaluate(
    retriever, impressions: list[Impression], histories: dict[str, History],
    articles: dict[str, Article], ks: tuple[int, ...] = (50, 100, 200),
) -> list[Result]:
    """recall@K over the whole corpus, per Q2.4/Q3.4.

    Phase 4 replaces this with the full harness: AUC/MRR/nDCG on the slate,
    beyond-accuracy metrics, slices, and bootstrap CIs. The signature stays --
    it takes a retriever and a split and knows nothing else about either.
    """
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    total = 0
    started = time.perf_counter()

    for imp in impressions:
        if not imp.is_labelled:
            continue
        hist = histories.get(imp.user_id)
        if hist is None:
            continue
        # The leakage boundary. Exact on EB-NeRD; on MIND `before` returns
        # everything because there are no timestamps (F1).
        past = hist.before(imp.time)
        texts = [articles[a].retrieval_text for a in past if a in articles]
        if not texts:
            continue
        got = [aid for aid, _ in retriever.retrieve(texts, max_k, imp.time)]
        truth = set(imp.clicked)
        total += 1
        for k in ks:
            if truth & set(got[:k]):
                hits[k] += 1

    elapsed = time.perf_counter() - started
    return [
        Result(retriever.name, k, hits[k] / total if total else 0.0, total, elapsed) for k in ks
    ]


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", choices=["mind", "ebnerd"], default="ebnerd")
    p.add_argument("--tier", default="demo", help="demo (smoke) or small")
    p.add_argument("--work-dir", type=Path, default=Path("data/work"))
    p.add_argument("--limit", type=int, default=2000, help="impressions to evaluate")
    p.add_argument("--max-articles", type=int, default=20000)
    p.add_argument(
        "--window-hours",
        type=float,
        default=24.0,
        help="recency window for the windowed retrievers (F16)",
    )
    add_arguments(p)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    budget = from_args(args)
    log.info("%s\n", budget)

    reader = get_reader(args.dataset, args.work_dir, args.tier)
    train_split = SPLIT_NAMES[args.dataset]["train"]

    t0 = time.perf_counter()
    articles = {a.article_id: a for a in reader.articles()}
    log.info("articles     %6d  (%.1fs)", len(articles), time.perf_counter() - t0)

    t0 = time.perf_counter()
    impressions = []
    for imp in reader.impressions(train_split):
        impressions.append(imp)
        if len(impressions) >= args.limit * 2:
            break
    log.info("impressions  %6d  (%.1fs)", len(impressions), time.perf_counter() - t0)

    # Corpus subsetting, done safely.
    #
    # Truncating the corpus arbitrarily (e.g. "first N by id") silently drops
    # the articles users actually clicked, capping recall near zero for a
    # reason that has nothing to do with the retriever. That is a measurement
    # artefact, not a result -- and it is exactly the trap the real pipeline
    # must avoid too (plan/1-Data-Pipeline.md D3 makes the same argument about
    # MIND's train/dev article union).
    #
    # So: keep every article that appears in any evaluated impression, then
    # pad with others up to the cap as distractors.
    if len(articles) > args.max_articles:
        needed = {a for imp in impressions for a in imp.candidates + imp.clicked}
        kept = {aid: articles[aid] for aid in needed if aid in articles}
        for aid, art in articles.items():
            if len(kept) >= args.max_articles:
                break
            kept.setdefault(aid, art)
        log.info(
            "corpus       %6d  (capped from %d; kept all %d referenced)",
            len(kept),
            len(articles),
            len(needed & articles.keys()),
        )
        articles = kept

    t0 = time.perf_counter()
    histories = {h.user_id: h for h in reader.histories(train_split)}
    verifiable = sum(1 for h in histories.values() if h.is_verifiable)
    log.info(
        "histories    %6d  (%.1fs)  boundary verifiable: %s",
        len(histories),
        time.perf_counter() - t0,
        "yes" if verifiable else "NO -- authors' construction (F1)",
    )

    train, val = temporal_split(impressions)
    log.info(
        "split        train=%d val=%d  boundary=%s\n",
        len(train),
        len(val),
        val[0].time if val else "n/a",
    )

    val = val[: args.limit]
    article_list = list(articles.values())

    results: list[Result] = []
    retrievers = [
        RecencyRetriever(),
        PopularityRetriever(),
        OverlapRetriever(),
        HashedVectorRetriever(),
        # F16: text matching within a recency-constrained pool, which is the
        # regime the real Phase 2/3 retrievers will have to operate in.
        WindowedRetriever(OverlapRetriever(), window_hours=args.window_hours),
        WindowedRetriever(HashedVectorRetriever(), window_hours=args.window_hours),
    ]

    for retriever in retrievers:
        t0 = time.perf_counter()
        if isinstance(retriever, PopularityRetriever):
            retriever.index_from_clicks(train)  # train only -- never val
        else:
            retriever.index(article_list)
        log.info("%-22s indexed in %.1fs", retriever.name, time.perf_counter() - t0)
        results.extend(evaluate(retriever, val, histories, articles))

    log.info("\n=== recall@K (skeleton -- NOT deliverable numbers) ===")
    for r in results:
        log.info("%s", r)

    log.info("\nInterfaces exercised: reader -> split -> Retriever -> evaluate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
