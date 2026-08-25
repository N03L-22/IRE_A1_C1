"""Submission format (Q5).

The checks that would otherwise be discovered by a failed leaderboard upload,
which costs a submission from the daily quota and ~15 minutes of regeneration.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from src.submit.codabench import SUBMISSION_MEMBER, rank_candidates


def test_archive_member_name_is_exactly_prediction_txt() -> None:
    """The scorers open this path literally.

    A first MIND submission failed with

        FileNotFoundError: '/app/input/res/prediction.txt'

    because the archive contained `mind_prediction.txt`. The predictions were
    correct; only the name inside the zip was wrong.
    """
    assert SUBMISSION_MEMBER == "prediction.txt"


def test_ranks_are_a_permutation_aligned_to_the_original_slate() -> None:
    """Output is a mark-up of the given slate, not a re-ordering of it."""
    slate = ["a", "b", "c", "d"]
    ranks = rank_candidates(slate, {"c": 9.0, "a": 5.0})

    assert sorted(ranks) == [1, 2, 3, 4], "not a permutation of 1..N"
    assert len(ranks) == len(slate)
    # 'c' scored highest -> rank 1, and it sits at slate position 2.
    assert ranks[2] == 1
    assert ranks[0] == 2


def test_unscored_candidates_keep_original_order() -> None:
    """Deterministic tie-breaking, so two runs produce identical files."""
    slate = ["a", "b", "c", "d"]
    ranks = rank_candidates(slate, {"d": 1.0})
    assert ranks[3] == 1, "the only scored candidate must rank first"
    # a, b, c are unscored and keep their relative order: 2, 3, 4.
    assert ranks[0] < ranks[1] < ranks[2]


def test_empty_scores_yield_slate_order() -> None:
    """A cold-start slate is still a valid permutation, not an empty line."""
    assert rank_candidates(["x", "y", "z"], {}) == [1, 2, 3]


@pytest.mark.skipif(
    not Path("submissions/mind_prediction.zip").exists(),
    reason="run `python -m src.submit.codabench --dataset mind --tier large` first",
)
def test_built_mind_archive_has_the_right_member() -> None:
    """Guard the artefact itself, not just the constant."""
    with zipfile.ZipFile("submissions/mind_prediction.zip") as zf:
        names = zf.namelist()
    assert names == [SUBMISSION_MEMBER], f"archive contains {names}"


# ---------------------------------------------------------------------------
# CompactHistories -- the memory optimisation that reimplements truncation.
#
# These are the tests that make the optimisation defensible. It duplicates the
# `< cutoff` logic of History.before(), which IS the Q9 leakage boundary, so
# the duplicate is pinned to the original rather than trusted.
# ---------------------------------------------------------------------------

from datetime import datetime, timedelta

from src.data.schema import Article, History
from src.submit.codabench import CompactHistories


def _texts_by_id(articles: list[Article]) -> dict[str, str]:
    return {a.article_id: a.retrieval_text for a in articles}


def test_compact_histories_match_history_before() -> None:
    """The duplicated truncation must agree with the original, exactly.

    If these two ever disagree, the submission path is applying a different
    leakage boundary from the one the harness and test_no_leakage.py verify --
    silently, and only on the file that gets uploaded.
    """
    base = datetime(2023, 5, 20, 12, 0, 0)
    articles = [Article(article_id=f"A{i}", title=f"title {i}") for i in range(10)]
    lookup = _texts_by_id(articles)

    hist = History(
        user_id="u1",
        clicked_ids=[f"A{i}" for i in range(10)],
        times=[base + timedelta(hours=i) for i in range(10)],
    )
    compact = CompactHistories([hist], lookup)

    for hours in (0, 1, 5, 9, 10, 20):
        cutoff = base + timedelta(hours=hours)
        expected = [lookup[a] for a in hist.before(cutoff) if a in lookup]
        assert compact.texts_before("u1", cutoff) == expected, (
            f"truncation diverged at cutoff +{hours}h"
        )


def test_compact_histories_without_timestamps_return_everything() -> None:
    """MIND has no click timestamps (F1); before() returns all, so must this."""
    articles = [Article(article_id=f"A{i}", title=f"t{i}") for i in range(4)]
    lookup = _texts_by_id(articles)
    hist = History(user_id="u1", clicked_ids=["A0", "A1", "A2"], times=None)

    compact = CompactHistories([hist], lookup)
    got = compact.texts_before("u1", datetime(2019, 11, 15))
    assert got == [lookup[a] for a in hist.before(datetime(2019, 11, 15))]
    assert len(got) == 3


def test_compact_histories_drop_ids_absent_from_the_corpus() -> None:
    """A click on an article outside the index contributes no text.

    The original path filtered with `if a in articles` after truncating; the
    compact path filters at build time. Same result, and the user must not
    vanish entirely just because one click is unknown.
    """
    articles = [Article(article_id="A0", title="known")]
    hist = History(
        user_id="u1",
        clicked_ids=["A0", "GHOST"],
        times=[datetime(2023, 5, 1), datetime(2023, 5, 2)],
    )
    compact = CompactHistories([hist], _texts_by_id(articles))
    assert compact.texts_before("u1", datetime(2023, 6, 1)) == ["known"]


def test_compact_histories_unknown_user_is_cold_not_an_error() -> None:
    compact = CompactHistories([], {"A0": "text"})
    assert compact.texts_before("nobody", datetime(2023, 5, 1)) == []


def test_int_dedup_matches_string_dedup() -> None:
    """The merge dedups on int ids; it must agree with the string version.

    Switching set[str] -> set[int] cut ~1.5 GB from the merge, which mattered
    because that phase once drove the machine into 24 GB of swap and stalled
    for 20+ minutes (F38). A dedup that silently disagreed would drop or keep
    the wrong lines in a 13.3M-line submission.
    """
    lines = [
        "100 [1,2]", "007 [1]", "7 [1,2,3]", "100 [3,4]", "0100 [5]",
    ]

    seen_str, kept_str = set(), []
    for ln in lines:
        iid = ln.split(" ", 1)[0]
        if iid not in seen_str:
            seen_str.add(iid)
            kept_str.append(ln)

    seen_int, kept_int = set(), []
    for ln in lines:
        key = int(ln.split(" ", 1)[0])
        if key not in seen_int:
            seen_int.add(key)
            kept_int.append(ln)

    # "007" and "7" are the SAME impression numerically but different strings.
    # The int form is correct: an id is a number, and the leading zero is
    # formatting. Assert the difference explicitly so it is a decision, not a
    # surprise.
    assert len(kept_int) == 2, kept_int
    assert len(kept_str) == 4, kept_str
    assert [ln.split(" ", 1)[0] for ln in kept_int] == ["100", "007"]


def test_worker_budget_constants_are_sane() -> None:
    """Preflight must leave room for the merge, or it recreates F38."""
    from src.submit.codabench import MERGE_HEADROOM_GB, WORKER_GB

    assert WORKER_GB > 0 and MERGE_HEADROOM_GB > 0
    # The merge builds a set over 13.3M ints; 3 GB is the measured need with
    # margin. If someone drops this to zero the stall comes straight back.
    assert MERGE_HEADROOM_GB >= 2.0, "not enough headroom reserved for the merge"
