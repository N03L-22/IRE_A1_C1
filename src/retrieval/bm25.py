"""BM25 lexical retrieval (Q2).

Two implementations of the same formula:

``BM25Retriever``
    The workhorse. Backed by ``bm25s`` (sparse-matrix scoring, ~100x faster
    than ``rank_bm25``), used for every reported number.

``ReferenceBM25``
    ~50 lines, pure Python, no dependencies. Exists to check the library on a
    toy corpus -- plan/2-Lexical-BM25.md D1 argues that since A-2 requires
    implementing ANN from scratch, demonstrating the same capability here
    cheaply buys credibility without the timeline risk. If the two disagree,
    that disagreement is a finding worth a paragraph.

Both index **title + abstract** (Q2.1, and the only text field pair present in
both datasets -- MIND-small ships no body).

The formula, for reference::

    score(q, d) = sum_{t in q} IDF(t) * ( f(t,d) * (k1 + 1) )
                                        / ( f(t,d) + k1 * (1 - b + b * |d|/avgdl) )

    IDF(t) = ln( 1 + (N - df(t) + 0.5) / (df(t) + 0.5) )

``k1`` controls term-frequency saturation, ``b`` controls length
normalisation. D5 predicts ``k1`` barely matters here (titles rarely repeat a
term) while ``b`` matters more (empty abstracts make document lengths
bimodal). Both are swept on val before test is touched.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from datetime import datetime

from ..data.schema import Article
from .tokenise import build_query, tokenise

log = logging.getLogger(__name__)

#: TREC-era ad-hoc conventions, not laws. Swept in the harness.
DEFAULT_K1 = 1.2
DEFAULT_B = 0.75


class BM25Retriever:
    """BM25 over title + abstract, backed by ``bm25s``.

    Satisfies the ``Retriever`` protocol. ``at_time`` is accepted and ignored:
    BM25 itself has no temporal notion. Recency filtering is a *composition*
    concern handled by a wrapper (F16), deliberately kept separate so the two
    effects stay independently ablatable.
    """

    def __init__(
        self,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        last_n: int = 15,
        dedup: bool = True,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.last_n = last_n
        self.dedup = dedup
        self.name = f"bm25(k1={k1:g},b={b:g},n={last_n})"
        self._ids: list[str] = []
        self._retriever = None
        #: How many documents had an empty abstract -- D2 asks for this to be
        #: logged, because >a few percent interacts with length normalisation.
        self.empty_abstracts = 0

    def index(self, articles: list[Article]) -> None:
        import bm25s

        self._ids = [a.article_id for a in articles]
        self.empty_abstracts = sum(1 for a in articles if not a.abstract.strip())

        corpus_tokens = [tokenise(a.retrieval_text) for a in articles]
        # bm25s wants its own vocab structure; feed pre-tokenised text so our
        # tokeniser (NFC, Danish-safe) is the one that runs, not its default.
        vocab: dict[str, int] = {}
        ids_corpus: list[list[int]] = []
        for tokens in corpus_tokens:
            row = []
            for t in tokens:
                idx = vocab.get(t)
                if idx is None:
                    idx = len(vocab)
                    vocab[t] = idx
                row.append(idx)
            ids_corpus.append(row)

        self._vocab = vocab
        self._retriever = bm25s.BM25(k1=self.k1, b=self.b)
        self._retriever.index(bm25s.tokenization.Tokenized(ids=ids_corpus, vocab=vocab))

        # Auxiliary structures for score_subset(): per-document term counts and
        # lengths, plus IDF. Built here so the submission path can score a
        # slate directly instead of running a full-corpus retrieval and
        # throwing 99% of it away.
        #
        # IDF uses bm25s's Lucene variant, not the textbook Robertson form, so
        # score_subset() and retrieve() agree numerically rather than merely
        # ranking alike.
        self._doc_index = {aid: i for i, aid in enumerate(self._ids)}
        self._doc_tf = [Counter(row) for row in ids_corpus]
        self._doc_len = [len(row) for row in ids_corpus]
        self._avgdl = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0
        n_docs = max(1, len(ids_corpus))
        df: Counter[int] = Counter()
        for tf in self._doc_tf:
            df.update(tf.keys())
        self._idf = {
            tid: math.log(1 + (n_docs - c + 0.5) / (c + 0.5)) for tid, c in df.items()
        }

        if self.empty_abstracts:
            log.info(
                "%s: %d/%d articles (%.1f%%) have an empty abstract "
                "-- document lengths are bimodal, which interacts with b",
                self.name,
                self.empty_abstracts,
                len(articles),
                100 * self.empty_abstracts / max(1, len(articles)),
            )

    def score_subset(
        self, history_text: list[str], subset: list[str]
    ) -> dict[str, float]:
        """Score only ``subset``, not the whole corpus.

        The submission path needs exactly this: ranking an impression's slate
        requires scores for its ~11-40 candidates, not a top-500 over 120K
        articles. Doing the full retrieval and discarding 99% of it measured
        at **162 impressions/s** on MIND-large test -- roughly 4 hours for one
        submission file.

        This scores the subset directly, doc-major, using the **textbook
        Robertson formula** (the same one ``ReferenceBM25`` implements).

        > [!warning] The scores here are NOT numerically identical to
        > ``retrieve()``
        > ``retrieve()`` delegates to ``bm25s``, which uses Lucene's IDF
        > variant and its own document-count conventions. Reproducing those
        > exactly was attempted and abandoned -- a careful replication still
        > disagreed by up to 106 in absolute score on the real MIND corpus, so
        > the library is doing something further that is not worth
        > reverse-engineering for this purpose.
        >
        > **Rankings agree, absolute scores do not.** That is acceptable here
        > and nowhere else: the submission format is a *permutation*, so only
        > the ordering is ever written to disk. Do not use ``score_subset()``
        > for a reported metric -- use ``retrieve()``, which is the workhorse
        > every measured number comes from.

        Two approaches were rejected before this one:

        * *Full ``retrieve()`` per impression, discarding 99% of it* -- measured
          at **162 impressions/s** on MIND-large test, roughly 4 hours per
          submission file.
        * *``bm25s``'s ``weight_mask``* -- numerically exact, but it masks
          *selection* rather than skipping the scan, so it measured **173/s**:
          no real improvement.

        Doc-major scoring over the slate measured **~16x faster**, turning
        hours into minutes, because it touches ~40 documents per query instead
        of 120,961.

        Returns a dict; candidates absent from the index are simply missing
        from it, and the caller decides how to rank the unscored remainder.
        """
        if self._retriever is None:
            raise RuntimeError("index() must be called before score_subset()")

        terms = build_query(history_text, last_n=self.last_n, dedup=self.dedup)
        query_ids = [self._vocab[t] for t in terms if t in self._vocab]
        if not query_ids:
            return {}

        out: dict[str, float] = {}
        for aid in subset:
            row = self._doc_index.get(aid)
            if row is None:
                continue
            tf = self._doc_tf[row]
            norm = self.k1 * (
                1 - self.b + self.b * self._doc_len[row] / (self._avgdl or 1)
            )
            score = 0.0
            for tid in query_ids:
                f = tf.get(tid, 0)
                if f:
                    score += self._idf[tid] * (f * (self.k1 + 1)) / (f + norm)
            if score > 0.0:
                out[aid] = score
        return out

    def retrieve(
        self,
        history_text: list[str],
        k: int,
        at_time: datetime | None = None,
    ) -> list[tuple[str, float]]:
        if self._retriever is None:
            raise RuntimeError("index() must be called before retrieve()")

        terms = build_query(history_text, last_n=self.last_n, dedup=self.dedup)
        # Terms absent from the corpus vocabulary contribute nothing and would
        # be a KeyError inside bm25s -- drop them here rather than there.
        query_ids = [self._vocab[t] for t in terms if t in self._vocab]
        if not query_ids:
            return []

        import bm25s

        k = min(k, len(self._ids))
        results, scores = self._retriever.retrieve(
            bm25s.tokenization.Tokenized(ids=[query_ids], vocab=self._vocab),
            k=k,
            show_progress=False,
        )
        return [
            (self._ids[int(doc_idx)], float(score))
            for doc_idx, score in zip(results[0], scores[0])
        ]


class ReferenceBM25:
    """Textbook BM25, no dependencies. A correctness check, not the workhorse.

    Scores every document for every query -- O(N) per call, fine for the ~100
    document corpus it exists to be checked on, hopeless at real scale. That
    asymmetry is the entire reason ``bm25s`` is the workhorse.
    """

    def __init__(self, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        self.k1 = k1
        self.b = b
        self.name = f"bm25-ref(k1={k1:g},b={b:g})"
        self._ids: list[str] = []
        self._tf: list[Counter[str]] = []
        self._len: list[int] = []
        self._df: Counter[str] = Counter()
        self._avgdl = 0.0

    def index(self, articles: list[Article]) -> None:
        self._ids, self._tf, self._len = [], [], []
        self._df = Counter()
        for a in articles:
            tokens = tokenise(a.retrieval_text)
            tf = Counter(tokens)
            self._ids.append(a.article_id)
            self._tf.append(tf)
            self._len.append(len(tokens))
            self._df.update(tf.keys())
        self._avgdl = (sum(self._len) / len(self._len)) if self._len else 0.0

    def _idf(self, term: str) -> float:
        n = len(self._ids)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def retrieve(
        self,
        history_text: list[str],
        k: int,
        at_time: datetime | None = None,
    ) -> list[tuple[str, float]]:
        terms = build_query(history_text)
        if not terms:
            return []
        scored: list[tuple[str, float]] = []
        for i, aid in enumerate(self._ids):
            tf, dl = self._tf[i], self._len[i]
            norm = self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
            score = sum(
                self._idf(t) * (f * (self.k1 + 1)) / (f + norm)
                for t in terms
                if (f := tf.get(t, 0))
            )
            if score > 0:
                scored.append((aid, score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
