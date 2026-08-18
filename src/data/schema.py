"""The unified schema: what a dataset looks like once the dataset-specific
shape has been parsed away.

MIND and EB-NeRD disagree about almost everything -- format (TSV vs parquet),
id types (string vs int), where click history lives (inline vs a separate
file), and which fields exist at all. Everything downstream of this module is
written once against these three record types rather than twice against two
datasets.

Fields EB-NeRD has and MIND does not are Optional. That asymmetry is real and
permanent (findings F1, F10), so it is encoded in the types rather than
patched over with sentinel values.

See plan/1-Data-Pipeline.md for the schema table and the reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, Protocol


@dataclass(slots=True)
class Article:
    """One retrievable document.

    ``abstract`` carries MIND's abstract and EB-NeRD's subtitle -- they play
    the same role. ``body`` is EB-NeRD only: MIND-small ships no body text
    (F10), which is why the retrieval text is title + abstract on both.
    """

    article_id: str
    title: str
    abstract: str = ""
    body: str = ""
    category: str = ""
    subcategory: str = ""
    entities: list[str] = field(default_factory=list)
    published_time: datetime | None = None

    @property
    def retrieval_text(self) -> str:
        """The text that gets indexed. Q2.1: title + abstract, both datasets."""
        return f"{self.title} {self.abstract}".strip()


@dataclass(slots=True)
class Impression:
    """One moment a user was shown a slate of articles.

    ``clicked`` is a subset of ``candidates``. On EB-NeRD 99.5% of impressions
    have exactly one click (F7). MIND test impressions have none -- they are
    unlabelled (F14).
    """

    impression_id: str
    user_id: str
    time: datetime
    candidates: list[str]
    clicked: list[str]
    session_id: str | None = None

    @property
    def is_labelled(self) -> bool:
        return bool(self.clicked)


@dataclass(slots=True)
class History:
    """A user's prior clicks.

    ``times`` is EB-NeRD only. Without it the behaviour-window boundary cannot
    be verified from the data -- on MIND we rely on the authors' construction
    and say so (F1). This is the single most important asymmetry in the
    project, so it is a distinct Optional field rather than an empty list.
    """

    user_id: str
    clicked_ids: list[str]
    times: list[datetime] | None = None

    def before(self, cutoff: datetime) -> list[str]:
        """Clicks strictly before ``cutoff`` -- the leakage boundary.

        With timestamps this is exact. Without them (MIND) we cannot filter,
        so the caller gets everything and must document the assumption.
        """
        if self.times is None:
            return list(self.clicked_ids)
        return [aid for aid, t in zip(self.clicked_ids, self.times) if t < cutoff]

    @property
    def is_verifiable(self) -> bool:
        """Whether the leakage boundary can be *checked* for this user."""
        return self.times is not None


class DatasetReader(Protocol):
    """What every dataset reader provides.

    Iterators rather than materialised lists: EB-NeRD large has 12M
    impressions (F15), and the point of a --mem-gb budget is that nothing
    assumes the whole split fits in memory.
    """

    name: str

    def articles(self) -> Iterator[Article]: ...
    def impressions(self, split: str) -> Iterator[Impression]: ...
    def histories(self, split: str) -> Iterator[History]: ...
