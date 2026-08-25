"""Slices (Q4.3) -- where the findings live.

Q4.3 requires at least one; D4 picks cold-vs-warm users as primary and
head-vs-tail articles as second.

> [!warning] The threshold is a decision, not a given
> Q4.3 says "few clicks" and names no number. EB-NeRD's minimum history length
> is 5 (F9), so a fixed "< 5 clicks" rule selects *nobody* there, while MIND's
> distribution is entirely different (median 19 vs 92). A fixed constant would
> silently produce an empty slice on one dataset and a huge one on the other.
>
> So the threshold is **derived per dataset from the measured distribution**
> (bottom quartile by default) and **reported with every cold-start number**.
> A slice boundary chosen after seeing the results is not a finding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SliceDef:
    """A named subset of impressions, plus how its boundary was chosen.

    ``basis`` is carried so the results table can state *why* the threshold is
    what it is, rather than leaving a bare number to be taken on trust.
    """

    name: str
    #: Indices into the evaluated impression list.
    members: tuple[int, ...]
    basis: str

    def __len__(self) -> int:
        return len(self.members)


def cold_warm_slices(
    history_lengths: list[int], quantile: float = 0.25
) -> list[SliceDef]:
    """Split users into cold and warm by history length.

    The boundary is the ``quantile`` of the *observed* history-length
    distribution, so it adapts to each dataset instead of assuming one
    constant fits both. Ties at the threshold go to cold, which keeps the
    slice non-empty when the distribution is heavily discretised (EB-NeRD's
    minimum is exactly 5, so a strict < would empty the slice).
    """
    if not history_lengths:
        return []
    arr = np.asarray(history_lengths)
    threshold = float(np.quantile(arr, quantile))
    cold = tuple(i for i, h in enumerate(history_lengths) if h <= threshold)
    warm = tuple(i for i, h in enumerate(history_lengths) if h > threshold)
    basis = (
        f"history length <= {threshold:g} "
        f"(q{quantile:g} of observed; median {np.median(arr):g}, "
        f"min {arr.min()}, max {arr.max()})"
    )
    out = [SliceDef("cold", cold, basis)]
    if warm:
        out.append(SliceDef("warm", warm, f"history length > {threshold:g}"))
    return out


def head_tail_slices(
    clicked_per_impression: list[set[str]],
    train_popularity: dict[str, float],
    quantile: float = 0.5,
) -> list[SliceDef]:
    """Split impressions by whether the clicked article is popular in train.

    Tests whether the system is a recommender or a popularity list (D4): a
    retriever that only ever surfaces head articles will look fine overall and
    collapse on the tail slice.

    Impressions whose clicked article never appeared in train count as tail --
    an article with zero training clicks is as far into the tail as it gets.
    """
    if not clicked_per_impression or not train_popularity:
        return []
    pops = np.asarray(list(train_popularity.values()))
    threshold = float(np.quantile(pops, quantile))

    head: list[int] = []
    tail: list[int] = []
    for i, clicked in enumerate(clicked_per_impression):
        if not clicked:
            continue
        best = max((train_popularity.get(a, 0.0) for a in clicked), default=0.0)
        (head if best > threshold else tail).append(i)

    basis = f"max train popularity of clicked article > {threshold:.3g} (q{quantile:g})"
    out = []
    if head:
        out.append(SliceDef("head", tuple(head), basis))
    if tail:
        out.append(SliceDef("tail", tuple(tail), f"clicked article at or below q{quantile:g}"))
    return out
