"""The differential tests that make the columnar fast path defensible (F64).

`ColumnarHistories.before()` duplicates the `t < cutoff` comparison that IS the
Q9 leakage boundary. Following F36's rule for `CompactHistories`: the duplicate
is **pinned to `History.before()`**, never trusted because it looks equivalent.

The important test here is `test_matches_history_before_on_real_ebnerd_users`.
Fixture tests confirm the logic; only real data confirms the *assumption*
`searchsorted` rests on -- that each user's click times are ascending. If they
are not, binary search returns a wrong truncation silently, which is the exact
failure mode this file exists to prevent.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from src.data.columnar import ColumnarHistories
from src.data.schema import History

EBNERD = Path("data/work/ebnerd/testset/test/history.parquet")


# ---------------------------------------------------------------------------
# Fixture-level: the logic
# ---------------------------------------------------------------------------


def _build(clicked: list[int], times: list[datetime] | None) -> ColumnarHistories:
    offsets = np.array([0, len(clicked)], dtype=np.int64)
    flat_t = (
        np.array([np.datetime64(t, "us") for t in times]).astype(np.int64)
        if times is not None
        else None
    )
    return ColumnarHistories(
        np.array([1], dtype=np.uint32),
        offsets,
        np.array(clicked, dtype=np.int32),
        flat_t,
    )


def _build_seconds(clicked: list[int], times: list[datetime]) -> ColumnarHistories:
    """As `_build`, but with times stored as int32 seconds -- the narrow path."""
    flat_t = np.array(
        [np.datetime64(t, "s").astype(np.int64) for t in times], dtype=np.int32
    )
    return ColumnarHistories(
        np.array([1], dtype=np.uint32),
        np.array([0, len(clicked)], dtype=np.int64),
        np.array(clicked, dtype=np.int32),
        flat_t,
        "s",
    )


def test_boundary_is_strict_less_than() -> None:
    """A click exactly at impression time is NOT available -- `<`, not `<=`.

    This is the one-click-per-user leak: small enough to pass a smoke test,
    large enough to inflate a metric.
    """
    base = datetime(2023, 5, 20, 12, 0, 0)
    ch = _build([10, 20, 30], [base, base + timedelta(hours=1), base + timedelta(hours=2)])

    assert ch.before(1, base).tolist() == [], "click at exactly the cutoff leaked"
    assert ch.before(1, base + timedelta(microseconds=1)).tolist() == [10]
    assert ch.before(1, base + timedelta(hours=2)).tolist() == [10, 20]
    assert ch.before(1, base + timedelta(days=99)).tolist() == [10, 20, 30]


def test_second_resolution_cutoff_does_not_shift_the_boundary() -> None:
    """A sub-second *cutoff* must not lose the clicks in the truncated fraction.

    Storing times as int32 seconds means the cutoff is converted with
    `np.datetime64(cutoff, "s")`, which floors toward the past. For a cutoff at
    12:00:00.5, flooring gives 12:00:00 -- so a click at exactly 12:00:00 would
    be dropped even though it is genuinely *before* the cutoff.

    Since every stored time is a whole second, the correct behaviour is that
    such a click is KEPT. This test exists because the flooring is invisible in
    the common case where cutoffs are also whole seconds.
    """
    base = datetime(2023, 5, 20, 12, 0, 0)
    ch = _build_seconds([10, 20], [base, base + timedelta(seconds=10)])

    assert ch.before(1, base).tolist() == [], "click at exactly the cutoff leaked"
    assert ch.before(1, base + timedelta(milliseconds=500)).tolist() == [10], (
        "a click before a sub-second cutoff was lost to flooring"
    )


def test_without_timestamps_returns_everything() -> None:
    """MIND has no click timestamps (F1); before() returns all, so must this."""
    ch = _build([7, 8, 9], None)
    assert ch.before(1, datetime(2019, 11, 15)).tolist() == [7, 8, 9]


def test_unknown_user_returns_empty() -> None:
    ch = _build([1, 2], None)
    assert ch.before(999, datetime(2023, 1, 1)).tolist() == []


def test_length_mismatch_is_rejected() -> None:
    """Misaligned parallel arrays make truncation meaningless -- fail loudly."""
    with pytest.raises(ValueError, match="length mismatch"):
        ColumnarHistories(
            np.array([1], dtype=np.uint32),
            np.array([0, 3], dtype=np.int64),
            np.array([1, 2, 3], dtype=np.int32),
            np.array([0, 1], dtype=np.int64),  # too short
        )


# ---------------------------------------------------------------------------
# The one that matters: agreement with History.before() on real data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not EBNERD.exists(), reason="needs the built EB-NeRD store")
def test_matches_history_before_on_real_ebnerd_users() -> None:
    """Exact agreement with the original, on real histories, at many cutoffs.

    If these two ever disagree, the fast path applies a different leakage
    boundary from the one `test_no_leakage.py` verifies -- and it would do so
    only on the file that gets uploaded.
    """
    import polars as pl

    n_users = 300
    df = pl.read_parquet(EBNERD).head(n_users)
    ch = ColumnarHistories.from_parquet(EBNERD)

    id_col = next(c for c in df.columns if "article_id" in c)
    t_col = next(c for c in df.columns if "time" in c)
    u_col = next(c for c in df.columns if "user" in c)

    checked = 0
    for row in df.iter_rows(named=True):
        ids, times = row[id_col], row[t_col]
        if not ids:
            continue
        hist = History(
            user_id=str(row[u_col]),
            clicked_ids=[str(a) for a in ids],
            times=list(times),
        )
        # Probe the boundary where it is easy to get wrong: exactly on each
        # click time, and just either side of it.
        for t in times[:: max(1, len(times) // 8)]:
            for delta in (timedelta(0), timedelta(microseconds=-1), timedelta(microseconds=1)):
                cutoff = t + delta
                expected = hist.before(cutoff)
                got = [str(a) for a in ch.before(row[u_col], cutoff).tolist()]
                assert got == expected, (
                    f"truncation diverged for user {row[u_col]} at {cutoff}: "
                    f"{len(got)} vs {len(expected)} clicks"
                )
                checked += 1

    assert checked > 500, f"only {checked} comparisons -- too few to be meaningful"


@pytest.mark.skipif(not EBNERD.exists(), reason="needs the built EB-NeRD store")
def test_real_click_times_are_ascending() -> None:
    """`searchsorted` is only valid on sorted times -- verify, do not assume.

    An unsorted user would get a silently wrong truncation from binary search.
    This asserts the precondition directly rather than inferring it from the
    differential test passing.
    """
    import polars as pl

    df = pl.read_parquet(EBNERD).head(5000)
    t_col = next(c for c in df.columns if "time" in c)

    violations = 0
    for times in df[t_col].to_list():
        if times and any(b < a for a, b in zip(times, times[1:])):
            violations += 1

    assert violations == 0, (
        f"{violations}/5000 users have non-ascending click times -- "
        "searchsorted would return wrong truncations for them"
    )


# ---------------------------------------------------------------------------
# ColumnarTexts -- the drop-in for CompactHistories in the submission path.
# Same rule as everything else touching the boundary: pinned to the original.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not EBNERD.exists(), reason="needs the built EB-NeRD store")
def test_columnar_texts_matches_compact_histories() -> None:
    """The two worker-path implementations must return identical text lists.

    `CompactHistories` is what produced every submitted file. If the faster
    replacement disagrees anywhere, the submission changes -- so agreement is
    asserted on real users at real cutoffs, not on a fixture.
    """
    import polars as pl

    from src.data.columnar import ColumnarTexts
    from src.data.schema import Article
    from src.submit.codabench import CompactHistories

    df = pl.read_parquet(EBNERD).head(200)
    id_col = next(c for c in df.columns if "article_id" in c)
    t_col = next(c for c in df.columns if "time" in c)
    u_col = next(c for c in df.columns if "user" in c)

    # A corpus covering the clicks these users actually made, plus a gap: some
    # clicked ids are deliberately absent, because both implementations must
    # drop unknown articles the same way.
    seen: set[int] = set()
    for ids in df[id_col].to_list():
        seen.update(int(a) for a in (ids or []))
    keep = sorted(seen)[: int(len(seen) * 0.8)]
    texts_by_id = {str(a): f"article {a} text" for a in keep}

    hists = []
    for row in df.iter_rows(named=True):
        if not row[id_col]:
            continue
        hists.append(
            type("H", (), {
                "user_id": str(row[u_col]),
                "clicked_ids": [str(a) for a in row[id_col]],
                "times": list(row[t_col]),
            })()
        )

    compact = CompactHistories(hists, texts_by_id)
    columnar = ColumnarTexts(EBNERD, texts_by_id)

    checked = 0
    for row in df.iter_rows(named=True):
        times = row[t_col]
        if not times:
            continue
        uid = str(row[u_col])
        for t in times[:: max(1, len(times) // 6)]:
            for delta in (timedelta(0), timedelta(microseconds=1)):
                cutoff = t + delta
                assert columnar.texts_before(row[u_col], cutoff) == compact.texts_before(uid, cutoff), (
                    f"ColumnarTexts diverged from CompactHistories for {uid} at {cutoff}"
                )
                checked += 1

    assert checked > 200, f"only {checked} comparisons -- too few to be meaningful"
