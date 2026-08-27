"""MIND behaviours as parquet, so MIND can use the fast path too (F76).

**The asymmetry this closes.** Every optimisation from F64--F70 -- columnar
histories, streamed row groups, fork-shared workers -- is gated on
``hasattr(reader, "impressions_row_group")``, which only ``EbnerdReader``
implements, because only EB-NeRD ships parquet. MIND ships a 1.4 GB TSV with
no row groups, so it took the **serial** path with **one** worker.

Measured cost of that gap:

===============  ==============  ===========  ==============
dataset          lines           time         throughput
===============  ==============  ===========  ==============
EB-NeRD (bm25)   13,336,711      22 min       10,235 lines/s
MIND (fusion)     2,370,727      38 min        1,036 lines/s
===============  ==============  ===========  ==============

**EB-NeRD does 5.6x more lines in 1.7x less time -- 9.9x the throughput** --
and it is not because MIND is harder. It is because MIND never got ported.

> [!important] Row groups are the unit of parallelism, so they are the point
> A TSV can be split by byte offset, but not safely: a slate field may contain
> anything, and a naive split can land mid-record. Parquet row groups are
> independent by construction, already contiguous on disk, and carry their own
> row counts -- which is what lets a worker take group *i* without coordinating
> with any other worker (F70).

**What this does NOT change.** MIND's history still has no click timestamps
(F1), so ``History.times`` stays ``None`` and the leakage boundary still rests
on the authors' construction rather than being machine-checkable. Converting
the *container* does not manufacture information the dataset never had.

> [!warning] The conversion must be verified, not trusted
> This writes a second copy of the impressions in a different format. If it
> disagrees with the TSV in any way -- a dropped row, a mis-parsed slate, a
> timezone shift -- every submission built from it is silently wrong.
> ``tests/test_mind_parquet.py`` asserts field-by-field equality against
> ``MindReader.impressions()`` on real rows, which is the same duplicate-and-pin
> rule F36/F64 followed for the leakage boundary.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

MIND_TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"

#: Rows per row group. 200K over MIND-large's 2.37M test impressions gives
#: ~12 groups -- enough to keep 6-8 workers busy without making each group so
#: small that per-group overhead (open, seek, decode headers) dominates.
ROWS_PER_GROUP = 200_000

#: Rows buffered before writing a group. Kept equal to ROWS_PER_GROUP so one
#: buffer flush is exactly one row group; decoupling them buys nothing and
#: makes the group count depend on flush timing.
BATCH = ROWS_PER_GROUP

SCHEMA = pa.schema([
    ("impression_id", pa.string()),
    ("user_id", pa.string()),
    ("time", pa.timestamp("us")),
    # Stored as parallel lists rather than the raw "N123-1 N456-0" string:
    # parsing once at convert time beats re-parsing per worker per run, and it
    # makes the slate structure explicit in the schema.
    ("candidates", pa.list_(pa.string())),
    ("clicked", pa.list_(pa.string())),
])


def _parse_slate(raw: str) -> tuple[list[str], list[str]]:
    """Parse ``"N55689-1 N35729-0"`` into candidates and clicks.

    Deliberately identical to ``MindReader._parse_slate``. Test rows carry bare
    ids with no suffix (F14) and yield an empty clicked list -- which is what
    ``is_labelled`` tests, so the distinction must survive the conversion.
    """
    candidates: list[str] = []
    clicked: list[str] = []
    for token in raw.split():
        if "-" in token:
            aid, _, label = token.rpartition("-")
            if label in ("0", "1"):
                candidates.append(aid)
                if label == "1":
                    clicked.append(aid)
                continue
        candidates.append(token)
    return candidates, clicked


def convert_split(split_dir: Path, *, overwrite: bool = False) -> Path:
    """Write ``behaviors.parquet`` beside an existing ``behaviors.tsv``.

    Streams the TSV rather than loading it: MIND-large's test behaviours are
    1.4 GB of text, and the point of this file is to make the submission path
    cheaper, not to move the cost earlier.
    """
    split_dir = Path(split_dir)
    tsv = split_dir / "behaviors.tsv"
    out = split_dir / "behaviors.parquet"
    if not tsv.exists():
        raise FileNotFoundError(f"no behaviors.tsv in {split_dir}")
    if out.exists() and not overwrite:
        log.info("%s already exists; skipping", out)
        return out

    ids: list[str] = []
    users: list[str] = []
    times: list[datetime] = []
    cands: list[list[str]] = []
    clicks: list[list[str]] = []
    n = 0

    writer = pq.ParquetWriter(out, SCHEMA, compression="zstd")
    try:
        with open(tsv, encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(row) < 5:
                    continue
                c, k = _parse_slate(row[4])
                ids.append(row[0])
                users.append(row[1])
                times.append(datetime.strptime(row[2], MIND_TIME_FORMAT))
                cands.append(c)
                clicks.append(k)
                n += 1
                if len(ids) >= BATCH:
                    writer.write_table(
                        pa.table([ids, users, times, cands, clicks], schema=SCHEMA),
                        row_group_size=ROWS_PER_GROUP,
                    )
                    ids, users, times, cands, clicks = [], [], [], [], []
        if ids:
            writer.write_table(
                pa.table([ids, users, times, cands, clicks], schema=SCHEMA),
                row_group_size=ROWS_PER_GROUP,
            )
    finally:
        writer.close()

    groups = pq.ParquetFile(out).metadata.num_row_groups
    log.info(
        "%s: %d impressions -> %d row groups, %.1f MB (from %.1f MB tsv)",
        out.name, n, groups, out.stat().st_size / 1e6, tsv.stat().st_size / 1e6,
    )
    return out
