"""Bootstrap confidence intervals (Q4.4).

Percentile bootstrap, B = 1000, seeded. Chosen in D5 over the normal
approximation (wrong for bounded, skewed metrics like recall) and over BCa
(more code, unnecessary at these sample sizes).

Two rules this module exists to enforce:

**Resample impressions, not predictions.** Predictions within one impression
are correlated; resampling them independently understates the interval, and
the symptom -- suspiciously narrow CIs -- is on the pitfall list. Every metric
must therefore arrive here as an array of *per-impression* values.

**Seeded.** A confidence interval that changes between runs cannot be quoted
in a design note.

> What the CI does not do: it quantifies sampling noise only. A leaking
> pipeline produces beautifully tight intervals around a wrong number. The
> bootstrap cannot detect bias -- that is what the leakage test is for.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

DEFAULT_B = 1000
DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 0


def bootstrap_ci(
    per_impression: Sequence[float],
    b: int = DEFAULT_B,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    """Return ``(mean, ci_low, ci_high)`` for a mean over per-impression values.

    Vectorised: the whole B x n resample matrix is built at once rather than
    looping in Python. At B=1000 and n=100K that is ~800 MB of int64 indices,
    so very large samples are chunked by the caller if needed -- in practice
    the evaluated slice is thousands, not hundreds of thousands.
    """
    values = np.asarray(per_impression, dtype=np.float64)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    if n == 1:
        # One observation carries no information about its own variability.
        # Returning a zero-width interval would overstate confidence.
        v = float(values[0])
        return (v, float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(b, n))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(values.mean()), float(lo), float(hi))


def point_only(
    units: Sequence, statistic: Callable[[Sequence], float]
) -> tuple[float, float, float]:
    """A statistic reported **without** a CI, because none is defensible.

    Coverage is the case this exists for, and the reason is worth stating in
    full because D5 asked for a CI on every number and this is the documented
    exception.

    Coverage counts **distinct** articles across the whole result set, so it
    is monotonically increasing in the number of impressions evaluated. That
    breaks every percentile-bootstrap scheme:

    * *Resample n with replacement* -- ~37% of draws are duplicates, so each
      resample sees fewer unique articles than the original. Measured: point
      0.9783 against an interval of [0.9035, 0.9235]. **The point estimate
      falls outside its own CI.**
    * *Subsample m < n without replacement* -- same bias, smaller. Measured
      across m/n in {0.5, 0.8, 0.9, 0.95, 0.99}: the interval fails to bracket
      the point at **every** ratio, approaching it only as m -> n.
    * *m = n without replacement* -- that is the original sample. Zero
      variance, an interval of zero width, which claims perfect certainty.

    There is no honest percentile interval here at fixed n. Reporting the
    point estimate and saying why beats manufacturing a plausible-looking
    interval that is biased by construction -- and a CI that excludes its own
    estimate would not survive a viva.

    What a coverage number *is* comparable across: two retrievers evaluated on
    the **same** impression sample, which is how the harness reports it.
    """
    if not units:
        return (float("nan"), float("nan"), float("nan"))
    return (float(statistic(units)), float("nan"), float("nan"))


def format_ci(mean: float, lo: float, hi: float, n: int | None = None) -> str:
    """Render as the IRE house style demands: never a bare number.

    >>> format_ci(0.34, 0.31, 0.37, 73152)
    '0.3400 [0.3100, 0.3700], n = 73,152'
    """
    if mean != mean:  # NaN
        return "n/a"
    body = f"{mean:.4f}"
    if lo == lo and hi == hi:
        body += f" [{lo:.4f}, {hi:.4f}]"
    if n is not None:
        body += f", n = {n:,}"
    return body


def paired_difference_ci(
    a: Sequence[float],
    b: Sequence[float],
    b_val: int = DEFAULT_B,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float, bool]:
    """CI on the DIFFERENCE between two retrievers, scored on the same impressions.

    Returns ``(mean_diff, ci_low, ci_high, significant)``.

    **Why this exists rather than comparing two separate intervals.**
    "Do the CIs overlap?" is a conservative approximation, not the test. Two
    intervals can overlap substantially while the difference is still
    significant, because the retrievers are evaluated on the *same*
    impressions -- so the per-impression noise is shared and cancels when you
    subtract. Comparing marginal intervals throws that pairing away.

    The elements of ``a`` and ``b`` must correspond to the same impressions in
    the same order, which is what ``measure()`` guarantees when two retrievers
    are scored over one split.

    > [!note] A null result here can mean two different things
    > Either the effect is genuinely absent, or the experiment could not have
    > seen it. Report how many impressions the two configurations actually
    > *differ* on: the dedup ablation differed on **4 of 800**, so no
    > statistical treatment could have resolved it. That is an underpowered
    > experiment, not evidence of no effect.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    n = min(len(x), len(y))
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), False)
    d = x[:n] - y[:n]

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(b_val, n))
    draws = d[idx].mean(axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(d.mean()), float(lo), float(hi), bool(lo > 0 or hi < 0))
