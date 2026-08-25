"""Columnar click histories -- the Q9 boundary without the Python objects (F64).

F60 measured the EB-NeRD submission run: **58.5% of setup is turning the
history parquet into Python objects**, 158.6 s and ~13 GB for 116,825,984
clicks. F62 then measured why, and corrected an assumption:

    read   -> columnar        pyarrow 1.66 s   polars 3.06 s   (polars slower)
    export -> python objects  pyarrow 28.4 s   polars 3.00 s   (polars 9.5x)
    stay columnar -> numpy                     polars 0.35 s, 0.47 GB

The win is not the reader. It is **never leaving columnar form**: the same
116.8M clicks are 0.47 GB of int32 rather than ~13 GB of Python ints, and
arrive ~450x faster.

Why this file exists rather than a change to ``History``
--------------------------------------------------------
``History.before()`` **is** the Q9 leakage boundary -- the single comparison
``t < cutoff`` that every reported number depends on being correct.
``tests/test_no_leakage.py`` verifies it, and ``src/eval/harness.py``,
``src/data/split.py`` and ``src/skeleton.py`` all call it.

So this does **not** modify it. It is a second implementation, and the rule
from F36 (which did the same thing for the worker path with
``CompactHistories``) applies unchanged:

> A duplicate of the leakage boundary is pinned to the original by a
> differential test, never trusted because it looks equivalent.

``tests/test_columnar.py`` asserts agreement with ``History.before()`` on real
EB-NeRD users across many cutoffs, plus the MIND no-timestamp case. If the two
ever diverge, the fast path is applying a *different* leakage boundary from the
one the harness verifies -- silently, and only on the file that gets uploaded.

> [!warning] The comparison is ``<``, not ``<=``
> A click exactly at impression time is *not* available when the impression is
> served. Getting this wrong leaks one click per user, which is small enough to
> pass a smoke test and large enough to inflate a metric. ``searchsorted`` with
> ``side="left"`` gives ``<``; ``side="right"`` would give ``<=``.

What this does not solve
------------------------
The 39% of setup spent streaming 13.5M impressions is untouched -- that is
I/O-bound, not allocation-bound. And a GPU cannot help either stage: this is
memory movement and interpreter overhead, not arithmetic. F63 covers where the
GPU *does* pay (MinHash over the corpus, 200,788 docs/s).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: Sentinel row for a user with no history, so lookups never branch on None.
_EMPTY = np.empty(0, dtype=np.int32)


class ColumnarHistories:
    """Every user's clicks as flat numpy arrays, with a per-user offset index.

    The layout is the standard "ragged array" one: all clicks concatenated
    into a single flat array, plus offsets marking where each user's slice
    begins and ends. One allocation for the corpus instead of 807,677 lists.

        ids      [a b c | d e | f g h i]      flat, int32
        offsets  [0     3     5         9]    len = n_users + 1
        user 1's clicks = ids[offsets[1]:offsets[2]]

    Times are stored as int64 epoch microseconds -- the unit
    ``timestamp[us]`` already uses, so the conversion is a reinterpret rather
    than arithmetic, and comparisons stay exact integer comparisons.
    """

    def __init__(
        self,
        user_ids: np.ndarray,
        offsets: np.ndarray,
        flat_ids: np.ndarray,
        flat_times: np.ndarray | None,
        time_unit: str = "us",
    ) -> None:
        # Checked here, not in from_parquet: the invariant belongs to the
        # object, and a caller building one directly must not skip it.
        # Misaligned parallel arrays pair a click with another click's time,
        # which makes every truncation wrong without raising anything.
        if flat_times is not None and len(flat_times) != len(flat_ids):
            raise ValueError(
                f"times/ids length mismatch ({len(flat_times)} vs {len(flat_ids)}) -- "
                "the per-click parallel arrays must align or truncation is meaningless"
            )
        if len(offsets) != len(user_ids) + 1:
            raise ValueError(
                f"offsets must have one more entry than users "
                f"({len(offsets)} vs {len(user_ids)} + 1)"
            )
        self._row = {int(u): i for i, u in enumerate(user_ids)}
        self._offsets = offsets
        self._ids = flat_ids
        self._times = flat_times
        # "s" or "us" -- the cutoff is converted to match, never the array.
        self._unit = time_unit

    # -- construction --------------------------------------------------

    @classmethod
    def from_parquet(cls, path: str | Path, *, with_times: bool = True) -> "ColumnarHistories":
        """Load without materialising a single Python click id.

        ``with_times=False`` is the MIND case (F1: no click timestamps), where
        ``History.before()`` returns everything and so must this.
        """
        import polars as pl

        df = pl.read_parquet(path)
        cols = df.columns
        id_col = next(c for c in cols if "article_id" in c)
        user_col = next(c for c in cols if "user" in c)

        lens = df[id_col].list.len().fill_null(0).to_numpy()
        # int64 for the cumsum (no overflow during accumulation), then narrowed:
        # the final offset is 116,825,984, comfortably inside int32.
        offsets = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens, out=offsets[1:])
        if offsets[-1] < np.iinfo(np.int32).max:
            offsets = offsets.astype(np.int32)
        # empty_as_null=True is the current default and Polars 2.0 flips it. Pin it:
        # an empty click list must explode to nothing, not to a null row, or every
        # downstream offset shifts by one.
        flat_ids = (
            df[id_col].explode(empty_as_null=True).fill_null(0)
            .to_numpy().astype(np.int32, copy=False)
        )
        user_ids = df[user_col].to_numpy()

        flat_times, time_unit = None, "us"
        if with_times:
            time_col = next((c for c in cols if "time" in c), None)
            if time_col is not None:
                t = df[time_col].explode(empty_as_null=True).to_numpy()
                # datetime64[us] -> int64 microseconds is a reinterpret, not a
                # conversion: same bytes, integer comparison semantics.
                us = t.astype("datetime64[us]").astype(np.int64)
                # Narrow to int32 *seconds* -- halves the largest array in the
                # object (times are 2x the ids). Only safe because EB-NeRD
                # records whole seconds: measured, 0 of 116,825,984 clicks carry
                # sub-second precision. Verified at load time rather than
                # assumed, because truncating a sub-second click to its second
                # would move it across a cutoff and change the boundary.
                if (us % 1_000_000 == 0).all() and abs(us // 1_000_000).max() < np.iinfo(np.int32).max:
                    flat_times = (us // 1_000_000).astype(np.int32)
                    time_unit = "s"
                else:
                    flat_times = us
                    time_unit = "us"

        log.info(
            "columnar histories: %d users, %d clicks, %.2f GB",
            len(user_ids), len(flat_ids),
            (flat_ids.nbytes + (flat_times.nbytes if flat_times is not None else 0)) / 1e9,
        )
        return cls(user_ids, offsets, flat_ids, flat_times, time_unit)

    # -- the boundary --------------------------------------------------

    def before(self, user_id: int | str, cutoff: datetime) -> np.ndarray:
        """Clicks strictly before ``cutoff``. Mirrors ``History.before()``.

        Returns int32 article ids as a numpy view -- no list, no Python ints.

        Each user's times are already ascending in the source data, so the
        truncation is a binary search rather than a scan: O(log k) instead of
        O(k) over a 160-click history.
        """
        i = self._row.get(int(user_id))
        if i is None:
            return _EMPTY
        lo, hi = self._offsets[i], self._offsets[i + 1]
        if self._times is None:
            # No timestamps (MIND): return everything, exactly as
            # History.before() does when times is None.
            return self._ids[lo:hi]

        times = self._times[lo:hi]
        if self._unit == "s":
            # CAREFUL: np.datetime64(cutoff, "s") floors toward the past, so a
            # cutoff of 12:00:00.5 becomes 12:00:00 -- and `<` would then drop a
            # click at exactly 12:00:00, which is genuinely *before* the cutoff.
            #
            # Every stored time is a whole second, so the correct integer cutoff
            # is the CEILING: any click strictly before 12:00:00.5 is at
            # 12:00:00 or earlier, i.e. strictly before 12:00:01.
            #
            # A real EB-NeRD user (40107, cutoff 07:51:01.000001) caught this in
            # the differential test -- flooring silently lost one click.
            us = np.datetime64(cutoff, "us").astype(np.int64)
            cut = -((-us) // 1_000_000)  # ceiling division, exact for negatives
        else:
            cut = np.datetime64(cutoff, self._unit).astype(np.int64)
        # side="left" gives strictly-less-than: a click exactly at impression
        # time is NOT available when the impression is served.
        n = int(np.searchsorted(times, cut, side="left"))
        return self._ids[lo : lo + n]

    def __len__(self) -> int:
        return len(self._row)

    def __contains__(self, user_id: int | str) -> bool:
        return int(user_id) in self._row
