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
