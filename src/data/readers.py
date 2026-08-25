"""Step 2 of the Q1 pipeline: per-dataset readers producing unified records.

One reader per dataset, each knowing exactly one thing -- how its own files are
laid out. Everything they emit conforms to src/data/schema.py, so no code past
this module contains a branch on which dataset it is looking at.

Deliberately dependency-light: stdlib ``csv`` for MIND's TSV, ``pyarrow`` for
EB-NeRD's parquet. No dataframe library, because these readers stream records
and never need one. That also keeps the polars-vs-pandas question out of the
interface -- a future bulk implementation can replace the internals without
touching a caller.

See plan/1-Data-Pipeline.md step 2.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from .schema import Article, History, Impression

log = logging.getLogger(__name__)

# MIND's history column can be very long; the default field limit truncates it.
csv.field_size_limit(sys.maxsize)

#: MIND timestamps: "11/11/2019 9:05:58 AM"
MIND_TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def normalise(text: str) -> str:
    """NFC-normalise and collapse whitespace.

    NFC matters for EB-NeRD: Danish 'ae/oe/aa' can be encoded as single code
    points or base+combining pairs, and two encodings of one word are two
    index terms -- silently halving recall on affected queries. Applied to the
    corpus *and* the query, or it achieves nothing.
    """
    return " ".join(unicodedata.normalize("NFC", text).split())


class MindReader:
    """MIND: TSV, no header, history inline in behaviors column 4.

    Layout after extraction (top-level dir stripped)::

        mind/train/behaviors.tsv   impression_id, user, time, history, impressions
        mind/train/news.tsv        id, cat, subcat, title, abstract, url, ents, rel_ents
        mind/dev/...

    Two things to know. First, train and dev ship *different* news.tsv files
    (51,282 vs 42,416 articles, F2) -- so ``articles()`` reads the union and
    de-duplicates, or dev-only articles would be unretrievable. Second, the
    history column has no timestamps (F1), so History.times is None and the
    leakage boundary rests on the authors' construction.
    """

    name = "mind"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _split_dir(self, split: str) -> Path:
        d = self.root / split
        if not d.exists():
            raise FileNotFoundError(f"mind: no such split {split!r} at {d}")
        return d

    def articles(self, splits: tuple[str, ...] = ("train", "dev")) -> Iterator[Article]:
        """Union of every split's news.tsv, de-duplicated by article_id (F2)."""
        seen: set[str] = set()
        for split in splits:
            path = self._split_dir(split) / "news.tsv"
            with open(path, encoding="utf-8", newline="") as fh:
                for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                    if len(row) < 8:
                        continue
                    aid = row[0]
                    if aid in seen:
                        continue
                    seen.add(aid)
                    yield Article(
                        article_id=aid,
                        title=normalise(row[3]),
                        abstract=normalise(row[4]),
                        body="",  # MIND-small ships none (F10)
                        category=row[1],
                        subcategory=row[2],
                        entities=self._entities(row[6]),
                        published_time=None,  # MIND has no publish time
                    )

    @staticmethod
    def _entities(raw: str) -> list[str]:
        """Pull surface forms out of MIND's entity JSON. Malformed rows skipped."""
        if not raw or raw == "[]":
            return []
        try:
            return [e["Label"] for e in json.loads(raw) if "Label" in e]
        except (json.JSONDecodeError, TypeError):
            return []

    def impressions(self, split: str) -> Iterator[Impression]:
        path = self._split_dir(split) / "behaviors.tsv"
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(row) < 5:
                    continue
                candidates, clicked = self._parse_slate(row[4])
                yield Impression(
                    impression_id=row[0],
                    user_id=row[1],
                    time=datetime.strptime(row[2], MIND_TIME_FORMAT),
                    candidates=candidates,
                    clicked=clicked,
                    session_id=None,  # MIND has no sessions
                )

    @staticmethod
    def _parse_slate(raw: str) -> tuple[list[str], list[str]]:
        """Parse "N55689-1 N35729-0" into candidates and clicks.

        Test-split rows carry bare ids with no suffix (F14) -- those yield
        candidates with an empty clicked list, which is what is_labelled tests.
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
            candidates.append(token)  # unlabelled test row
        return candidates, clicked

    def histories(self, split: str) -> Iterator[History]:
        """History is inline per impression, so the same user recurs.

        We emit one record per *impression* rather than per user, because the
        history MIND gives is the one valid at that moment. De-duplicating by
        user would mean picking one arbitrarily.
        """
        path = self._split_dir(split) / "behaviors.tsv"
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(row) < 5:
                    continue
                yield History(
                    user_id=row[1],
                    clicked_ids=row[3].split() if row[3] else [],
                    times=None,  # no timestamps (F1)
                )


class EbnerdReader:
    """EB-NeRD: parquet, history in its own file, far richer than MIND.

    Layout after extraction::

        ebnerd/small/articles.parquet
        ebnerd/small/train/{behaviors,history}.parquet
        ebnerd/small/validation/{behaviors,history}.parquet

    Post-click fields (read_time, scroll_percentage, next_read_time,
    next_scroll_percentage) are deliberately NOT read. They are measured after
    the click sits in the same row as the label, so using them to predict the
    click is circular -- the most direct leak in the schema (F6). Not reading
    them at all is stronger than reading and remembering to drop them.
    """

    name = "ebnerd"

    #: One shared articles file, not per split.
    ARTICLES = "articles.parquet"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _split_dir(self, split: str) -> Path:
        d = self.root / split
        if not d.exists():
            raise FileNotFoundError(f"ebnerd: no such split {split!r} at {d}")
        return d

    def articles(self) -> Iterator[Article]:
        path = self.root / self.ARTICLES
        cols = [
            "article_id",
            "title",
            "subtitle",
            "body",
            "category_str",
            "topics",
            "ner_clusters",
            "published_time",
        ]
        table = pq.read_table(path, columns=cols)
        for batch in table.to_batches():
            d = batch.to_pydict()
            for i in range(batch.num_rows):
                topics = d["topics"][i] or []
                yield Article(
                    article_id=str(d["article_id"][i]),
                    title=normalise(d["title"][i] or ""),
                    # EB-NeRD's subtitle plays MIND's abstract role.
                    abstract=normalise(d["subtitle"][i] or ""),
                    body=normalise(d["body"][i] or ""),
                    category=d["category_str"][i] or "",
                    subcategory=topics[0] if topics else "",
                    entities=list(d["ner_clusters"][i] or []),
                    published_time=d["published_time"][i],
                )

    def impressions(self, split: str) -> Iterator[Impression]:
        """Stream impressions, tolerating the unlabelled test schema.

        The test split has **14 columns against train's 17** (F14): the
        organisers removed ``article_ids_clicked`` and ``article_id`` (the
        labels) and ``next_read_time`` / ``next_scroll_percentage`` (future
        information). Asking for a column that is not there is a hard
        ParquetFile error, so the column list is intersected with the actual
        schema rather than hardcoded.

        An impression with no ``clicked`` is exactly what ``is_labelled``
        reports as False -- the harness skips those, and the submission path
        does not need them.
        """
        path = self._split_dir(split) / "behaviors.parquet"
        pf = pq.ParquetFile(path)
        cols = self._impression_columns(pf)
        for batch in pf.iter_batches(columns=cols, batch_size=50_000):
            yield from self._impressions_from_batch(batch)

    def _impression_columns(self, pf) -> list[str]:
        """Columns to read, intersected with what the file actually has."""
        available = set(pf.schema_arrow.names)
        required = ["impression_id", "user_id", "impression_time", "article_ids_inview"]
        missing = [c for c in required if c not in available]
        if missing:
            raise KeyError(f"behaviors.parquet missing required columns: {missing}")
        return required + [
            c for c in ("article_ids_clicked", "session_id") if c in available
        ]

    def _impressions_from_batch(self, batch) -> Iterator[Impression]:
        d = batch.to_pydict()
        clicked_col = d.get("article_ids_clicked")
        session_col = d.get("session_id")
        for i in range(batch.num_rows):
            yield Impression(
                impression_id=str(d["impression_id"][i]),
                user_id=str(d["user_id"][i]),
                time=d["impression_time"][i],
                candidates=[str(a) for a in (d["article_ids_inview"][i] or [])],
                clicked=(
                    [str(a) for a in (clicked_col[i] or [])]
                    if clicked_col is not None
                    else []
                ),
                session_id=str(session_col[i]) if session_col is not None else None,
            )

    def impressions_row_group(self, split: str, rg_index: int) -> Iterator[Impression]:
        """Impressions from ONE parquet row group.

        Row groups are the natural unit of parallelism for the submission
        path: they are independent, already materialised as contiguous byte
        ranges, and there are ~51 of them in EB-NeRD's test set. Each worker
        reads its own group and writes its own shard, so nothing is shared
        and nothing is pickled between processes.
        """
        path = self._split_dir(split) / "behaviors.parquet"
        pf = pq.ParquetFile(path)
        cols = self._impression_columns(pf)
        yield from self._impressions_from_batch(pf.read_row_group(rg_index, columns=cols))

    def histories(self, split: str) -> Iterator[History]:
        """One record per user, with timestamps -- so truncation is provable."""
        path = self._split_dir(split) / "history.parquet"
        cols = ["user_id", "article_id_fixed", "impression_time_fixed"]
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=cols, batch_size=10_000):
            d = batch.to_pydict()
            for i in range(batch.num_rows):
                yield History(
                    user_id=str(d["user_id"][i]),
                    clicked_ids=[str(a) for a in (d["article_id_fixed"][i] or [])],
                    times=list(d["impression_time_fixed"][i] or []),
                )


def get_reader(dataset: str, work_dir: Path, tier: str = "small"):
    """Build the reader for a dataset. The only place tiers map to paths."""
    work_dir = Path(work_dir)
    if dataset == "mind":
        return MindReader(work_dir / "mind")
    if dataset == "ebnerd":
        return EbnerdReader(work_dir / "ebnerd" / tier)
    raise KeyError(f"unknown dataset {dataset!r}; expected 'mind' or 'ebnerd'")


#: Split names differ between datasets; callers ask for a role, not a folder.
SPLIT_NAMES = {
    # "test" is the UNLABELLED leaderboard split (Q5 only). It exists at the
    # large tier alone and can never contribute to an offline metric -- see
    # findings F11/F14.
    "mind": {"train": "train", "heldout": "dev", "test": "large_test"},
    "ebnerd": {"train": "train", "heldout": "validation", "test": "test"},
}
