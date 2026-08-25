"""Submission format (Q5).

The checks that would otherwise be discovered by a failed leaderboard upload,
which costs a submission from the daily quota and ~15 minutes of regeneration.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from src.submit.codabench import SUBMISSION_MEMBER, rank_candidates


def test_archive_member_names_differ_between_competitions() -> None:
    """The two scorers open different filenames -- one letter apart.

    A first MIND submission failed with

        FileNotFoundError: '/app/input/res/prediction.txt'

    because the archive contained `mind_prediction.txt`. The predictions were
    correct; only the name inside the zip was wrong (F35).

    Checking upstream afterwards found the competitions do NOT agree:
    MIND's evaluate.py opens "prediction.txt", while EB-NeRD's
    ebrec.utils._python.write_submission_file defaults to "predictions.txt".
    Using one name for both would have failed the EB-NeRD upload the same way.
    """
    assert SUBMISSION_MEMBER["mind"] == "prediction.txt"
    assert SUBMISSION_MEMBER["ebnerd"] == "predictions.txt"
    assert SUBMISSION_MEMBER["mind"] != SUBMISSION_MEMBER["ebnerd"], (
        "if these ever match, verify upstream rather than assuming"
    )


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
    assert names == [SUBMISSION_MEMBER["mind"]], f"archive contains {names}"


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


def test_build_retriever_accepts_all_three_kinds() -> None:
    """The submission path must be able to ship any scored retriever.

    The first MIND submission was BM25 only, because Q3 did not exist yet.
    Once fusion measured best on both leaderboard metrics (F39), hardcoding
    one retriever became the thing stopping us acting on our own results.
    """
    from src.submit.codabench import build_retriever

    for kind in ("bm25", "semantic", "fusion"):
        r = build_retriever(kind, "mind")
        assert hasattr(r, "score_subset"), f"{kind} cannot take the fast path"
        assert hasattr(r, "index") and hasattr(r, "retrieve")

    with pytest.raises(ValueError):
        build_retriever("nonsense", "mind")


def test_rank_convention_matches_ebnerd_upstream() -> None:
    """Our ranks must equal ebrec's rank_predictions_by_score.

    Upstream (ebrec/utils/_python.py) computes:

        np.argsort(np.argsort(arr)[::-1]) + 1

    i.e. 1-based ranks, highest score = rank 1, aligned to the ORIGINAL
    candidate order. Pinned here against their exact expression, including the
    examples from their own docstring, so a refactor of rank_candidates cannot
    silently diverge from the scorer.
    """
    import numpy as np

    def upstream(scores):
        a = np.array(scores)
        return (np.argsort(np.argsort(a)[::-1]) + 1).tolist()

    cases = [
        [0.2, 0.1, 0.3],          # from their docstring
        [0.1, 0.2],               # from their docstring
        [0.4, 0.2, 0.1, 0.3],     # from their docstring
        [5.0, 1.0, 3.0, 9.0],
        [1.0],
    ]
    for scores in cases:
        cands = [f"c{i}" for i in range(len(scores))]
        ours = rank_candidates(cands, dict(zip(cands, scores)))
        assert ours == upstream(scores), f"{scores}: ours {ours} vs upstream {upstream(scores)}"


def test_submission_filenames_are_namespaced_by_retriever() -> None:
    """Two retrievers must not overwrite each other's submission.

    Comparing a BM25 submission against a fusion one on the same leaderboard
    is a reportable result (F39) -- and impossible if the second run silently
    replaces the first. The outer zip name is free (Codabench tracks by
    submission id, not filename); only the member inside is constrained.
    """
    src = Path("src/submit/codabench.py").read_text()
    assert "stem = submission_stem(" in src
    for suffix in ("_prediction.txt", "_prediction.zip", "_prediction.meta.json"):
        assert f'f"{{stem}}{suffix}"' in src, f"{suffix} not namespaced"
    # The member name must still come from the per-competition table.
    assert "SUBMISSION_MEMBER[args.dataset]" in src


def test_submission_stem_is_unique_per_run(tmp_path) -> None:
    """Four components, and each one has to matter.

    Naming by retriever alone was not enough: re-running the SAME
    configuration is normal (after a bug fix, or on another machine) and
    produces a genuinely different artefact with its own leaderboard row.
    Without the iteration counter the second run silently replaces the first,
    which is exactly what this project already got wrong once when every run
    wrote to a fixed {dataset}_prediction.* path.
    """
    from src.submit.codabench import submission_stem

    p = {"k1": 1.6, "b": 0.75, "last_n": 5, "retriever": "fusion"}

    # Same params, repeated runs -> i1, i2, i3 (the zip is what marks a run
    # as taken, so create it between calls).
    seen = []
    for expected in (1, 2, 3):
        stem = submission_stem("mind", "fusion", p, tmp_path)
        assert stem.endswith(f"_i{expected}"), stem
        seen.append(stem)
        (tmp_path / f"{stem}_prediction.zip").write_text("")
    assert len(set(seen)) == 3, "iterations collided"

    # Any param change -> different hash, so the counter restarts cleanly.
    other = submission_stem("mind", "fusion", {**p, "window_hours": 24.0}, tmp_path)
    assert other.endswith("_i1")
    assert other.split("_")[2] != seen[0].split("_")[2], "param change did not alter the hash"

    # Dataset and retriever stay in the name -- they are the comparison axes.
    assert submission_stem("ebnerd", "bm25", p, tmp_path).startswith("ebnerd_bm25_")
