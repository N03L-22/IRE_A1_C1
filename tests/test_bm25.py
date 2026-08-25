"""BM25 correctness (Q2).

The acceptance criteria from plan/2-Lexical-BM25.md that can be asserted
rather than eyeballed:

  - the own implementation agrees with the library on a toy corpus
  - retrieve() returns <= K results, descending, no duplicates
  - NFC normalisation is applied to corpus *and* query
  - an empty history returns nothing rather than raising
"""

from __future__ import annotations

import itertools
import random
import unicodedata

import pytest

from src.data.schema import Article
from src.retrieval.bm25 import BM25Retriever, ReferenceBM25
from src.retrieval.tokenise import build_query, tokenise


@pytest.fixture
def toy_corpus() -> list[Article]:
    """~100 documents over a small vocabulary, with empty abstracts mixed in."""
    rng = random.Random(0)
    words = "election sport finance royal crime weather car automobile court fire storm".split()
    return [
        Article(
            article_id=f"A{i}",
            title=" ".join(rng.choices(words, k=6)),
            # Every 7th article has no abstract -- the bimodal-length case D2
            # calls out, present in the fixture so it is exercised by default.
            abstract=" ".join(rng.choices(words, k=12)) if i % 7 else "",
        )
        for i in range(100)
    ]


def test_reference_agrees_with_library(toy_corpus: list[Article]) -> None:
    """The D1 credibility check: our BM25 and bm25s rank the same way.

    Two things differ by design and are deliberately not asserted:

    absolute scores
        bm25s defaults to Lucene's IDF variant, ReferenceBM25 uses the
        textbook Robertson form. Ranking is what a retriever is judged on.

    the order *within* a score tie
        ReferenceBM25 breaks ties by article id for determinism; bm25s does
        not. So agreement is asserted on the score *sequence* and on set
        membership at each distinct score, not on raw id order -- otherwise
        this test fails on a tie, which is not a defect.
    """
    query = ["election royal crime"]

    lib = BM25Retriever()
    lib.index(toy_corpus)
    ref = ReferenceBM25()
    ref.index(toy_corpus)

    got_lib = lib.retrieve(query, 10)
    got_ref = ref.retrieve(query, 10)

    assert len(got_lib) == len(got_ref)

    # Group ids by rank position, allowing ties to be in either order.
    def tie_groups(results: list[tuple[str, float]]) -> list[set[str]]:
        groups: list[set[str]] = []
        last: float | None = None
        for aid, score in results:
            if last is None or abs(score - last) > 1e-9:
                groups.append({aid})
                last = score
            else:
                groups[-1].add(aid)
        return groups

    assert tie_groups(got_lib) == tie_groups(got_ref), (
        "library and reference implementations disagree on ranking beyond "
        "score ties -- if this fires, one of them is wrong and the "
        "disagreement is a finding worth a paragraph"
    )


def test_retrieve_contract(toy_corpus: list[Article]) -> None:
    """<= K results, strictly descending, no duplicates."""
    r = BM25Retriever()
    r.index(toy_corpus)
    got = r.retrieve(["election sport"], 20)

    assert len(got) <= 20
    ids = [i for i, _ in got]
    assert len(ids) == len(set(ids)), "duplicate article in one result list"
    scores = [s for _, s in got]
    assert scores == sorted(scores, reverse=True), "results not descending by score"


def test_k_larger_than_corpus_is_clamped(toy_corpus: list[Article]) -> None:
    """recall@200 over a 100-document corpus must not raise or pad."""
    r = BM25Retriever()
    r.index(toy_corpus)
    assert len(r.retrieve(["election"], 200)) <= len(toy_corpus)


def test_empty_history_returns_nothing(toy_corpus: list[Article]) -> None:
    """Cold start is a legitimate answer, not an exception."""
    r = BM25Retriever()
    r.index(toy_corpus)
    assert r.retrieve([], 50) == []
    assert r.retrieve(["   "], 50) == []


def test_nfc_normalisation_is_applied() -> None:
    """Danish letters in two encodings must produce one index term.

    Without this, EB-NeRD recall is silently halved on affected queries and
    nothing crashes -- the failure mode D3 exists to prevent.

    Note only 'aa' decomposes: 'oe' (U+00F8) and 'ae' (U+00E6) are atomic code
    points with no NFD expansion, so a fixture built on those tests nothing.
    The word below is chosen to actually exercise the two encodings.
    """
    composed = unicodedata.normalize("NFC", "århus")
    decomposed = unicodedata.normalize("NFD", "århus")
    assert composed != decomposed, "fixture is not exercising the two encodings"
    assert tokenise(composed) == tokenise(decomposed)


def test_nfc_applied_to_query_and_corpus() -> None:
    """The normalisation has to run on both sides or it achieves nothing."""
    corpus = [Article(article_id="A1", title=unicodedata.normalize("NFD", "Århus vandt"))]
    r = BM25Retriever()
    r.index(corpus)
    got = r.retrieve([unicodedata.normalize("NFC", "Århus")], 5)
    assert got and got[0][0] == "A1", "NFC mismatch between corpus and query"


def test_build_query_last_n_and_dedup() -> None:
    """The two D4 knobs do what they claim."""
    history = [f"title{i} shared" for i in range(30)]

    assert "title29" in build_query(history, last_n=5)
    assert "title0" not in build_query(history, last_n=5), "last_n did not truncate"

    deduped = build_query(history, last_n=10, dedup=True)
    assert deduped.count("shared") == 1
    assert build_query(history, last_n=10, dedup=False).count("shared") == 10


def test_empty_abstracts_are_counted(toy_corpus: list[Article]) -> None:
    """D2 asks for this to be logged; it has to be measured to be logged."""
    r = BM25Retriever()
    r.index(toy_corpus)
    assert r.empty_abstracts == sum(1 for a in toy_corpus if not a.abstract.strip())


def test_score_subset_ranks_like_full_retrieval(toy_corpus: list[Article]) -> None:
    """The fast submission path must order a slate the same way retrieve() does.

    Absolute scores deliberately differ -- score_subset() uses the textbook
    Robertson formula while retrieve() delegates to bm25s's Lucene variant.
    Only the ordering is ever written to a submission file, so ordering is
    what is asserted (and what must not regress).
    """
    r = BM25Retriever()
    r.index(toy_corpus)
    query = ["election royal crime"]

    slate = [a.article_id for a in toy_corpus[:40]]
    full = dict(r.retrieve(query, len(toy_corpus)))
    sub = r.score_subset(query, slate)

    common = [a for a in slate if a in full and a in sub]
    assert len(common) >= 5, "fixture did not produce enough scored overlap"

    # Compare by discordant pairs rather than by list equality. Two documents
    # tied under one formula can be separated under the other, which reorders
    # the list without disagreeing about anything -- the same tie artefact as
    # test_reference_agrees_with_library. A *discordant* pair is a real
    # disagreement: one formula says a > b and the other says a < b.
    discordant = sum(
        1
        for a, b in itertools.combinations(common, 2)
        if (full[a] - full[b]) * (sub[a] - sub[b]) < 0
    )
    assert discordant == 0, (
        f"{discordant} pairs ordered differently by the Lucene and Robertson "
        "formulas -- submission rankings would not match retrieve()"
    )


def test_score_subset_ignores_unknown_ids(toy_corpus: list[Article]) -> None:
    """Candidates absent from the index are simply unscored, never an error."""
    r = BM25Retriever()
    r.index(toy_corpus)
    got = r.score_subset(["election"], ["A0", "NOT_IN_CORPUS", "A1"])
    assert "NOT_IN_CORPUS" not in got


def test_score_subset_empty_query_returns_empty(toy_corpus: list[Article]) -> None:
    r = BM25Retriever()
    r.index(toy_corpus)
    assert r.score_subset([], ["A0", "A1"]) == {}
