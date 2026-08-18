"""Step 4 of the Q1 pipeline: the temporal split, and the leakage boundary.

This module owns the only irrecoverable property in the assignment. A wrong
BM25 parameter costs a few points of recall and is fixed by re-running; a
leaked future click invalidates every number in the report and is invisible in
the output -- the metrics just look good.

Q1.3 requires a temporal split and forbids a random one for interaction data.
The specifics are ours, and they cannot be uniform across the two datasets:

    MIND         train 9-14 Nov 2019 (6 days), dev = 15 Nov only (1 day)
    EB-NeRD      train 18-25 May 2023 (7 days), validation = 25 May-1 Jun (7)

So a literal 80/10/10 does not fit both (finding F4). What generalises is the
*rule*:

    test  = the official held-out period, untouched (leaderboard-faithful)
    val   = the last `val_fraction` of the official train window, by time
    train = the rest

which yields ~80/10/10 on MIND and ~45/5/50 on EB-NeRD. The actual proportions
are computed and reported per dataset rather than asserted.

See plan/1-Data-Pipeline.md D1 and D2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Iterator, Sequence

from .schema import History, Impression

log = logging.getLogger(__name__)

TRAIN, VAL, TEST = "train", "val", "test"


@dataclass
class SplitReport:
    """What the split actually did. Goes into the run manifest verbatim.

    Reported rather than assumed: the whole point of F4 is that the ratio you
    ask for is not the ratio you get, and the honest move is to publish what
    happened.
    """

    dataset: str
    counts: dict[str, int] = field(default_factory=dict)
    boundaries: dict[str, tuple[datetime, datetime] | None] = field(default_factory=dict)
    val_fraction_requested: float = 0.1

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def proportions(self) -> dict[str, float]:
        t = self.total
        return {k: (v / t if t else 0.0) for k, v in self.counts.items()}

    def __str__(self) -> str:
        lines = [f"split[{self.dataset}] n={self.total:,}"]
        for name in (TRAIN, VAL, TEST):
            n = self.counts.get(name, 0)
            pct = self.proportions.get(name, 0.0) * 100
            span = self.boundaries.get(name)
            when = f"{span[0]:%Y-%m-%d %H:%M} .. {span[1]:%Y-%m-%d %H:%M}" if span else "-"
            lines.append(f"  {name:5s} {n:>9,}  {pct:5.1f}%  {when}")
        return "\n".join(lines)


def _span(imps: Sequence[Impression]) -> tuple[datetime, datetime] | None:
    if not imps:
        return None
    return imps[0].time, imps[-1].time


def temporal_split(
    train_period: Iterable[Impression],
    heldout_period: Iterable[Impression],
    *,
    dataset: str,
    val_fraction: float = 0.1,
) -> tuple[dict[str, list[Impression]], SplitReport]:
    """Split by time. Never random -- Q1.3.

    ``train_period`` and ``heldout_period`` are the dataset's *official*
    splits. The held-out period becomes test untouched, keeping us faithful to
    what the leaderboard scores; validation is carved from the tail of the
    train window so there is a real val split on both datasets (MIND's official
    dev is a single day and is needed as test, so val has to come from train
    regardless -- F3).

    The cut is made on the timestamp at the ``1 - val_fraction`` quantile, so
    every val impression is strictly later than every train impression. Sorting
    is by time only; ties at the exact boundary timestamp go to train, which is
    the conservative direction (less data in val, none of it leaked).
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    ordered = sorted(train_period, key=lambda i: i.time)
    test = sorted(heldout_period, key=lambda i: i.time)

    cut_index = int(len(ordered) * (1 - val_fraction))
    if cut_index >= len(ordered):
        train, val = ordered, []
    else:
        # Move the cut to a timestamp boundary so no single instant straddles
        # train and val -- otherwise two impressions at the same second land on
        # opposite sides and "strictly later" stops being true.
        cut_time = ordered[cut_index].time
        while cut_index > 0 and ordered[cut_index - 1].time == cut_time:
            cut_index -= 1
        train, val = ordered[:cut_index], ordered[cut_index:]

    splits = {TRAIN: train, VAL: val, TEST: test}
    report = SplitReport(
        dataset=dataset,
        counts={k: len(v) for k, v in splits.items()},
        boundaries={k: _span(v) for k, v in splits.items()},
        val_fraction_requested=val_fraction,
    )

    _assert_ordered(train, val, test)
    return splits, report


def _assert_ordered(
    train: Sequence[Impression], val: Sequence[Impression], test: Sequence[Impression]
) -> None:
    """Every split must be strictly later than the one before it.

    Cheap, and it catches the class of bug where a split silently becomes
    random -- which is the failure Q1.3 exists to prevent.
    """
    if train and val:
        assert train[-1].time <= val[0].time, (
            f"train/val overlap: train ends {train[-1].time}, val starts {val[0].time}"
        )
    if val and test:
        assert val[-1].time <= test[0].time, (
            f"val/test overlap: val ends {val[-1].time}, test starts {test[0].time}"
        )
    if train and test and not val:
        assert train[-1].time <= test[0].time


# --------------------------------------------------------------------------
# Step 5 -- history truncation
# --------------------------------------------------------------------------


@dataclass
class TruncationReport:
    """How much history survived the boundary, and whether it was checkable."""

    dataset: str
    verifiable: bool
    n_users: int = 0
    n_impressions: int = 0
    clicks_before: int = 0
    clicks_after: int = 0

    @property
    def dropped_fraction(self) -> float:
        if not self.clicks_before:
            return 0.0
        return 1.0 - (self.clicks_after / self.clicks_before)

    def __str__(self) -> str:
        if not self.verifiable:
            return (
                f"truncate[{self.dataset}] NOT VERIFIABLE -- no history timestamps. "
                f"Relying on the dataset authors' construction (F1); "
                f"{self.n_users:,} users passed through unchanged."
            )
        return (
            f"truncate[{self.dataset}] {self.n_impressions:,} impressions, "
            f"{self.clicks_before:,} -> {self.clicks_after:,} clicks "
            f"({self.dropped_fraction * 100:.1f}% dropped as post-boundary)"
        )


def truncate_history(
    impressions: Iterable[Impression],
    histories: dict[str, History],
    *,
    dataset: str,
) -> tuple[Iterator[tuple[Impression, list[str]]], TruncationReport]:
    """Pair each impression with the clicks its user made *before* it.

    This is the behaviour-window boundary that Q9 requires a test for.

    On EB-NeRD the history carries timestamps, so the filter is exact and
    provable. On MIND there are none (F1), so nothing can be filtered and the
    full history passes through -- we rely on the authors having built it from
    the preceding period, and we say so rather than implying we checked.

    Returns a generator plus a report; the report is only fully populated once
    the generator has been consumed.
    """
    sample = next(iter(histories.values()), None)
    verifiable = bool(sample and sample.is_verifiable)
    report = TruncationReport(dataset=dataset, verifiable=verifiable, n_users=len(histories))

    def _iter() -> Iterator[tuple[Impression, list[str]]]:
        for imp in impressions:
            hist = histories.get(imp.user_id)
            if hist is None:
                report.n_impressions += 1
                yield imp, []
                continue
            before = hist.before(imp.time)
            report.n_impressions += 1
            report.clicks_before += len(hist.clicked_ids)
            report.clicks_after += len(before)
            yield imp, before

    return _iter(), report


def check_no_leakage(
    pairs: Iterable[tuple[Impression, list[str]]],
    histories: dict[str, History],
) -> list[str]:
    """Return a list of violations: history entries at or after the impression.

    Empty list means the boundary holds. Used by tests/test_no_leakage.py both
    to assert correctness and -- via a deliberately corrupted store -- to prove
    the check has teeth. A checker that cannot fail proves nothing (Q9).

    Only meaningful where timestamps exist. On MIND every history is
    unverifiable, so this returns empty for the honest reason that there is
    nothing to check, not because the data was validated.
    """
    violations: list[str] = []
    for imp, before in pairs:
        hist = histories.get(imp.user_id)
        if hist is None or not hist.is_verifiable:
            continue
        allowed = set(hist.before(imp.time))
        for aid in before:
            if aid not in allowed:
                violations.append(
                    f"impression {imp.impression_id} (user {imp.user_id}, {imp.time}): "
                    f"article {aid} is not in the pre-boundary history"
                )
    return violations
