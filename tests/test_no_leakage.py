"""Q9: the behaviour-window boundary, and a test that proves it is enforced.

The brief requires "a test asserting this". The important half is the second
test in each pair: a checker that passes on clean data proves nothing on its
own, because a checker with an inverted comparison or an empty loop looks
exactly the same. So every invariant here is tested twice --

    1. it holds on the real store
    2. it FAILS when the boundary is deliberately broken

If (2) ever stops failing, the leakage guarantee has quietly evaporated.

Coverage limit, stated honestly: this verifies the boundary where timestamps
exist, i.e. EB-NeRD. MIND history has no timestamps (F1), so the invariant is
unverifiable from the data and rests on the dataset authors' construction. The
tests below assert that MIND is *correctly reported as unverifiable* rather
than pretending it was checked.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.data.schema import History, Impression
from src.data.split import check_no_leakage, temporal_split, truncate_history

STORE = Path("data/store")
HAS_EBNERD = (STORE / "ebnerd" / "impressions_train.parquet").exists()
HAS_MIND = (STORE / "mind" / "impressions_train.parquet").exists()


def _imp(iid: str, user: str, when: datetime) -> Impression:
    return Impression(
        impression_id=iid, user_id=user, time=when, candidates=["a", "b"], clicked=["a"]
    )


# ------------------------------------------------------- synthetic pairs ---


class TestBoundaryEnforcement:
    """The invariant, on data we control completely."""

    def test_clean_history_passes(self):
        t = datetime(2023, 5, 20, 12, 0)
        hist = {
            "u1": History(
                user_id="u1",
                clicked_ids=["old1", "old2"],
                times=[t - timedelta(days=1), t - timedelta(hours=1)],
            )
        }
        pairs, _ = truncate_history([_imp("i1", "u1", t)], hist, dataset="synthetic")
        assert check_no_leakage(list(pairs), hist) == []

    def test_future_click_is_caught(self):
        """The mutation test. If this passes, the checker is broken."""
        t = datetime(2023, 5, 20, 12, 0)
        hist = {
            "u1": History(
                user_id="u1",
                clicked_ids=["past", "FUTURE"],
                times=[t - timedelta(hours=1), t + timedelta(hours=1)],
            )
        }
        imp = _imp("i1", "u1", t)

        # Truncation must drop the future click...
        pairs, _ = truncate_history([imp], hist, dataset="synthetic")
        kept = list(pairs)
        assert kept[0][1] == ["past"], "truncation let a future click through"

        # ...and the checker must reject it if something puts it back.
        corrupted = [(imp, ["past", "FUTURE"])]
        violations = check_no_leakage(corrupted, hist)
        assert violations, "checker did not catch an injected future click"
        assert "FUTURE" in violations[0]

    def test_click_exactly_at_impression_time_is_excluded(self):
        """The boundary is strict: a click at t is not before t."""
        t = datetime(2023, 5, 20, 12, 0)
        hist = {"u1": History(user_id="u1", clicked_ids=["sim"], times=[t])}
        pairs, _ = truncate_history([_imp("i1", "u1", t)], hist, dataset="synthetic")
        assert list(pairs)[0][1] == []

    def test_untimestamped_history_is_reported_unverifiable(self):
        """MIND (F1): nothing can be filtered, and we must not pretend it was."""
        t = datetime(2019, 11, 15, 12, 0)
        hist = {"u1": History(user_id="u1", clicked_ids=["a", "b"], times=None)}
        pairs, report = truncate_history([_imp("i1", "u1", t)], hist, dataset="mind")
        rows = list(pairs)
        assert rows[0][1] == ["a", "b"]  # passed through unfiltered
        assert report.verifiable is False
        assert "NOT VERIFIABLE" in str(report)
        # And the checker stays silent -- honestly, because there is nothing
        # to check, not because the data was validated.
        assert check_no_leakage(rows, hist) == []


class TestTemporalOrdering:
    """Q1.3: never a random split."""

    def test_splits_are_time_ordered(self):
        base = datetime(2023, 5, 18)
        train_period = [_imp(str(i), f"u{i}", base + timedelta(hours=i)) for i in range(100)]
        heldout = [_imp(f"h{i}", f"u{i}", base + timedelta(days=10, hours=i)) for i in range(20)]
        splits, report = temporal_split(
            train_period, heldout, dataset="synthetic", val_fraction=0.1
        )
        assert splits["train"][-1].time <= splits["val"][0].time
        assert splits["val"][-1].time <= splits["test"][0].time
        assert report.total == 120

    def test_shuffled_input_still_splits_by_time(self):
        """A random split would pass a naive test; this one would not."""
        import random

        base = datetime(2023, 5, 18)
        imps = [_imp(str(i), f"u{i}", base + timedelta(hours=i)) for i in range(100)]
        random.Random(0).shuffle(imps)
        splits, _ = temporal_split(imps, [], dataset="synthetic", val_fraction=0.2)
        times = [i.time for i in splits["train"]] + [i.time for i in splits["val"]]
        assert times == sorted(times), "split did not restore temporal order"

    def test_boundary_timestamp_does_not_straddle(self):
        """Impressions sharing the cut instant must not land on both sides."""
        base = datetime(2023, 5, 18, 12, 0)
        # 20 impressions all at the same instant, around the 90% cut.
        imps = [_imp(str(i), f"u{i}", base) for i in range(20)]
        imps += [_imp(f"late{i}", f"u{i}", base + timedelta(hours=1)) for i in range(5)]
        splits, _ = temporal_split(imps, [], dataset="synthetic", val_fraction=0.1)
        if splits["train"] and splits["val"]:
            assert splits["train"][-1].time < splits["val"][0].time


# ---------------------------------------------------------- the real store ---


@pytest.mark.skipif(not HAS_EBNERD, reason="run `make store` first")
class TestEbnerdStore:
    """EB-NeRD has timestamps, so the boundary is provable end to end."""

    @pytest.fixture(scope="class")
    def store(self):
        return STORE / "ebnerd"

    def test_manifest_reports_verifiable(self, store):
        m = json.loads((store / "manifest.json").read_text())
        assert m["history_boundary_verifiable"] is True

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_no_split_overlap(self, store, split):
        path = store / f"impressions_{split}.parquet"
        if not path.exists():
            pytest.skip(f"{split} not built")
        t = pq.read_table(path, columns=["split"])
        assert set(t["split"].to_pylist()) == {split}

    def test_splits_are_time_ordered_on_disk(self, store):
        spans = {}
        for split in ("train", "val", "test"):
            path = store / f"impressions_{split}.parquet"
            if not path.exists():
                continue
            times = pq.read_table(path, columns=["time"])["time"].to_pylist()
            spans[split] = (min(times), max(times))
        if "train" in spans and "val" in spans:
            assert spans["train"][1] <= spans["val"][0]
        if "val" in spans and "test" in spans:
            assert spans["val"][1] <= spans["test"][0]

    def test_stored_history_predates_every_impression(self, store):
        """The headline invariant, read back off disk."""
        t = pq.read_table(
            store / "impressions_train.parquet",
            columns=["impression_id", "time", "history", "history_verifiable"],
        )
        assert all(t["history_verifiable"].to_pylist()), "EB-NeRD should be verifiable"
        # The store holds ids, not timestamps, so re-derive from the reader.
        from src.data.readers import get_reader

        reader = get_reader("ebnerd", Path("data/work"), "small")
        hist = {h.user_id: h for h in reader.histories("train")}
        rows = pq.read_table(
            store / "impressions_train.parquet",
            columns=["impression_id", "user_id", "time", "history"],
        ).to_pylist()
        checked = 0
        for r in rows[:5000]:
            h = hist.get(r["user_id"])
            if h is None or not h.is_verifiable:
                continue
            allowed = set(h.before(r["time"]))
            assert set(r["history"]) <= allowed, f"leak in {r['impression_id']}"
            checked += 1
        assert checked > 0, "no verifiable rows were actually checked"


@pytest.mark.skipif(not HAS_MIND, reason="run `make store` first")
class TestMindStore:
    """MIND cannot be verified -- assert that we say so, rather than implying
    a guarantee we do not have."""

    def test_manifest_reports_unverifiable(self):
        m = json.loads((STORE / "mind" / "manifest.json").read_text())
        assert m["history_boundary_verifiable"] is False

    def test_rows_flagged_unverifiable(self):
        t = pq.read_table(
            STORE / "mind" / "impressions_train.parquet", columns=["history_verifiable"]
        )
        assert not any(t["history_verifiable"].to_pylist())
