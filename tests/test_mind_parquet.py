"""The differential tests that make MIND's parquet path defensible (F76).

`mind_parquet.convert_split` writes a *second copy* of the impressions in a
different format, and the submission path reads that copy. If the two ever
disagree -- a dropped row, a mis-parsed slate, a timezone shift -- every
submission built from the parquet is silently wrong while looking fine.

Same rule as F36's `CompactHistories` and F64's `ColumnarHistories`: **the
duplicate is pinned to the original by a differential test, never trusted
because it looks equivalent.**
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.data.mind_parquet import _parse_slate, convert_split
from src.data.readers import MindReader

MIND_DEV = Path("data/work/mind/dev")


# ---------------------------------------------------------------------------
# Slate parsing -- the one piece of logic the converter reimplements
# ---------------------------------------------------------------------------


def test_parse_slate_matches_the_reader() -> None:
    """The converter's parser must agree with MindReader's, token for token."""
    cases = [
        "N1-1 N2-0 N3-0",           # ordinary labelled row
        "N1-0 N2-0",                # no click
        "N1-1 N2-1",                # multiple clicks
        "N1 N2 N3",                 # unlabelled test row (F14)
        "N1-1 N2 N3-0",             # mixed, which the test split can produce
        "N-weird-id-1",             # id containing hyphens -- rpartition matters
        "",                         # empty slate
    ]
    for raw in cases:
        assert _parse_slate(raw) == MindReader._parse_slate(raw), f"diverged on {raw!r}"


def test_parse_slate_keeps_hyphenated_ids_intact() -> None:
    """`rpartition` splits on the LAST hyphen, so ids may contain hyphens.

    A naive `split("-")` would turn "N-weird-1" into candidate "N" -- a silent
    corruption that still produces a well-formed submission.
    """
    cands, clicks = _parse_slate("N-weird-id-1 N-other-0")
    assert cands == ["N-weird-id", "N-other"]
    assert clicks == ["N-weird-id"]


def test_unlabelled_rows_yield_no_clicks() -> None:
    """Test rows are unlabelled (F14); `is_labelled` depends on this."""
    cands, clicks = _parse_slate("N1 N2 N3")
    assert cands == ["N1", "N2", "N3"]
    assert clicks == []


# ---------------------------------------------------------------------------
# The one that matters: parquet == TSV on real rows
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (MIND_DEV / "behaviors.tsv").exists(),
                    reason="needs the built MIND store")
def test_parquet_impressions_match_tsv(tmp_path: Path) -> None:
    """Field-by-field equality against the TSV reader, on real MIND rows.

    If this fails, the fast path is reading different impressions from the ones
    every published number was computed on.
    """
    import shutil

    split = tmp_path / "dev"
    split.mkdir()
    # Copy a bounded prefix: the whole dev split is large and this test runs
    # in CI, but a few thousand real rows exercise every branch.
    src = MIND_DEV / "behaviors.tsv"
    with open(src, encoding="utf-8") as fh:
        head = [next(fh) for _ in range(5000)]
    (split / "behaviors.tsv").write_text("".join(head), encoding="utf-8")
    for aux in ("news.tsv",):
        if (MIND_DEV / aux).exists():
            shutil.copy(MIND_DEV / aux, split / aux)

    convert_split(split)
    reader = MindReader(tmp_path)

    from_tsv = list(reader.impressions("dev"))
    n_groups = reader.n_row_groups("dev")
    from_pq = [imp for g in range(n_groups) for imp in reader._impressions_row_group("dev", g)]

    assert len(from_pq) == len(from_tsv), (
        f"row count differs: parquet {len(from_pq)} vs tsv {len(from_tsv)}"
    )
    for a, b in zip(from_tsv, from_pq):
        assert a.impression_id == b.impression_id
        assert a.user_id == b.user_id
        assert a.time == b.time, f"timestamp drift on {a.impression_id}"
        assert a.candidates == b.candidates, f"slate differs on {a.impression_id}"
        assert a.clicked == b.clicked, f"clicks differ on {a.impression_id}"


@pytest.mark.skipif(not (MIND_DEV / "behaviors.tsv").exists(),
                    reason="needs the built MIND store")
def test_row_groups_partition_the_split_exactly(tmp_path: Path) -> None:
    """Every impression appears in exactly one row group -- no gaps, no repeats.

    Workers take one group each (F70). A gap silently drops impressions from the
    submission; an overlap emits duplicates the merge would have to catch.
    """
    split = tmp_path / "dev"
    split.mkdir()
    with open(MIND_DEV / "behaviors.tsv", encoding="utf-8") as fh:
        head = [next(fh) for _ in range(3000)]
    (split / "behaviors.tsv").write_text("".join(head), encoding="utf-8")

    convert_split(split)
    reader = MindReader(tmp_path)

    seen: list[str] = []
    for g in range(reader.n_row_groups("dev")):
        seen.extend(i.impression_id for i in reader._impressions_row_group("dev", g))

    expected = [i.impression_id for i in reader.impressions("dev")]
    assert len(seen) == len(set(seen)), "an impression appeared in two row groups"
    assert sorted(seen) == sorted(expected), "row groups do not partition the split"


@pytest.mark.skipif(not (MIND_DEV / "behaviors.tsv").exists(),
                    reason="needs the built MIND store")
def test_fast_path_is_gated_on_the_file_existing(tmp_path: Path) -> None:
    """No parquet -> no `impressions_row_group` -> the serial path is used.

    The submission path branches on `hasattr`. If the attribute existed
    unconditionally, an un-converted checkout would take a path that cannot
    serve it and fail mid-run rather than falling back.
    """
    split = tmp_path / "dev"
    split.mkdir()
    (split / "behaviors.tsv").write_text(
        "1\tU1\t11/15/2019 1:00:00 PM\tN1 N2\tN3-1 N4-0\n", encoding="utf-8"
    )
    reader = MindReader(tmp_path)
    assert not hasattr(reader, "impressions_row_group")

    convert_split(split)
    assert hasattr(reader, "impressions_row_group")
