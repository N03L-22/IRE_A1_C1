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
        import pyarrow.parquet as pq

        # Stream row groups rather than exploding the whole column at once.
        # Measured: the one-shot path peaks at 4.93 GB RSS to produce 0.94 GB
        # of arrays, and `del df` does not give it back -- glibc keeps freed
        # pages. Streaming 9 row groups peaks at 2.85 GB for the same result.
        # Peak allocation is what sets a worker's footprint, so this is the
        # difference between 2 workers fitting and 6 (F67).
        f = pq.ParquetFile(str(path))
        names = f.schema_arrow.names
        id_col = next(c for c in names if "article_id" in c)
        user_col = next(c for c in names if "user" in c)
        time_col = next((c for c in names if "time" in c), None) if with_times else None
        cols = [c for c in (user_col, id_col, time_col) if c]

        id_parts, len_parts, user_parts, time_parts = [], [], [], []
        for g in range(f.metadata.num_row_groups):
            tb = f.read_row_group(g, columns=cols)
            lists = tb.column(id_col).combine_chunks()
            len_parts.append(np.asarray(lists.value_lengths(), dtype=np.int64))
            id_parts.append(
                lists.flatten().to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
            )
            user_parts.append(tb.column(user_col).to_numpy())
            if time_col is not None:
                tt = tb.column(time_col).combine_chunks().flatten().to_numpy(zero_copy_only=False)
                time_parts.append(tt.astype("datetime64[us]").astype(np.int64))
                del tt
            del tb, lists

        flat_ids = np.concatenate(id_parts) if id_parts else np.empty(0, np.int32)
        del id_parts
        user_ids = np.concatenate(user_parts) if user_parts else np.empty(0, np.int64)
        del user_parts
        lens = np.concatenate(len_parts) if len_parts else np.empty(0, np.int64)
        del len_parts

        offsets = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens, out=offsets[1:])

        flat_times, time_unit = None, "us"
        if time_parts:
            us = np.concatenate(time_parts)
            del time_parts
            # Narrow to int32 *seconds*: times are the largest array, and
            # measurement shows 0 of 116,825,984 EB-NeRD clicks carry
            # sub-second precision. Verified rather than assumed -- truncating
            # a sub-second click to its second would move it across a cutoff.
            if (us % 1_000_000 == 0).all() and abs(us // 1_000_000).max() < np.iinfo(np.int32).max:
                flat_times = (us // 1_000_000).astype(np.int32)
                time_unit = "s"
            else:
                flat_times = us
            del us

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


class ColumnarTexts:
    """Drop-in for ``CompactHistories``: same ``texts_before`` interface (F67).

    ``CompactHistories`` (F36) already avoids Python string ids by holding
    ``array('i')`` indices -- 15.08 GB -> 9.25 GB per worker. What it still
    pays is the *construction*: iterating 807,677 ``History`` objects that the
    reader materialised from parquet, which F60 measured at 158.6 s and F64
    replaced with a 4.6 s columnar load.

    This joins the two: load columnar (F64), then expose the exact method the
    worker loop already calls, so ``codabench.py`` changes by one line.

    > [!important] The interface is texts, not ids
    > The retrievers take article *text* -- the manufactured-query move. So the
    > id array is mapped through a shared text table at the end, which is the
    > one place Python objects are unavoidable. Crucially that cost is
    > proportional to the *truncated* history (~15 recent clicks), not to all
    > 116.8M stored clicks.
    """

    def __init__(self, path: str | Path, texts_by_id: dict[str, str], *, with_times: bool = True) -> None:
        self._cols = ColumnarHistories.from_parquet(path, with_times=with_times)
        # Article id -> text, keyed by the int32 ids the columnar store holds.
        # Ids absent from the corpus map to None and are dropped, matching
        # CompactHistories' `if a in index` filter.
        self._text: dict[int, str] = {}
        for aid, txt in texts_by_id.items():
            try:
                self._text[int(aid)] = txt
            except (TypeError, ValueError):
                # Non-numeric article ids (MIND's "N12345") never reach this
                # path -- MIND has no click timestamps, so it uses the
                # CompactHistories route. Skip rather than guess an encoding.
                continue

    def texts_before(self, user_id: str, cutoff) -> list[str]:
        """Retrieval text of clicks strictly before ``cutoff``.

        Signature-compatible with ``CompactHistories.texts_before`` so the two
        are interchangeable at the call site.
        """
        get = self._text.get
        out = []
        for aid in self._cols.before(user_id, cutoff):
            t = get(int(aid))
            if t is not None:
                out.append(t)
        return out

    def __len__(self) -> int:
        return len(self._cols)

    def __contains__(self, user_id) -> bool:
        return user_id in self._cols
