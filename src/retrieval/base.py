"""The `Retriever` contract -- the interface one harness scores everything through.

Settled in plan/2-Lexical-BM25.md and reproduced here as executable code. Q4.5
requires a single evaluation harness over both retrievers, which is only
possible if both answer the same question the same way:

    "given this user's click history at time t, which K articles might they
     click?"  ->  [(article_id, score), ...] descending by score

Two details of the signature are deliberate and worth not undoing:

``at_time``
    Present even for retrievers that ignore it. It makes the temporal boundary
    visible at *every* call site, so an implementation that filters by publish
    time cannot silently forget to. Finding F16 made this load-bearing: on
    EB-NeRD, restricting to a 24h window moved token-overlap recall@50 from
    0.00 to 0.28.

``(id, score)`` pairs rather than bare ids
    Lets the harness do fusion, score analysis, and tie-break inspection
    without a second retrieval pass.

See architecture.md Part B for why the shared interface is the load-bearing
design decision of this component.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from ..data.schema import Article


@runtime_checkable
class Retriever(Protocol):
    """What every retriever provides. BM25, semantic, and every baseline."""

    #: Short identifier used as the row label in results tables.
    name: str

    def index(self, articles: list[Article]) -> None:
        """Build whatever structure this retriever searches.

        Called once per corpus. Must be idempotent -- the harness may re-index
        when sweeping a parameter that changes the index.
        """
        ...

    def retrieve(
        self,
        history_text: list[str],
        k: int,
        at_time: datetime | None = None,
    ) -> list[tuple[str, float]]:
        """Return at most ``k`` ``(article_id, score)`` pairs, best first.

        ``history_text`` is the retrieval text of the user's prior clicks,
        already truncated to the leakage boundary by the caller. An empty list
        means a cold-start user: returning ``[]`` is a legitimate answer, and
        the harness reports that slice separately.
        """
        ...
