"""The evaluation harness (Q4) -- one function, every retriever.

D1: ``evaluate(retriever, split, config) -> list[ResultRow]``. BM25, semantic,
popularity, recency, random and any fusion all go through this. Q4.5 is
satisfied by construction rather than by discipline: there is nowhere else to
compute a metric.

The harness makes **one retrieval pass per impression** and derives both
regimes from it (D2):

    corpus regime   the top-K list as returned          -> recall@K
    slate regime    those scores restricted to the      -> AUC, MRR, nDCG
                    impression's own candidate list

That matters for cost as much as correctness -- retrieval dominates the
runtime, so computing eight metrics costs barely more than computing one.

Output is a tidy table (one row per dataset x retriever x slice x metric x
feature-set) so the design-note tables are a groupby rather than a
copy-and-paste job.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Sequence

from ..data.schema import Article, History, Impression
from .bootstrap import bootstrap_ci, format_ci, point_only
from .metrics import (
    auc,
    coverage,
    intra_list_diversity,
    mrr,
    ndcg_at_k,
    novelty,
    rank_slate,
    recall_at_k,
)
from .slices import SliceDef, cold_warm_slices, head_tail_slices

log = logging.getLogger("harness")

#: Q2.4/Q3.4 name these exactly.
KS = (50, 100, 200)


@dataclass
class ResultRow:
    """One number, with everything needed to interpret it.

    Never a bare value: the CI and n travel with it, per IRE convention, and
    ``regime`` states which candidate set it was computed over so corpus and
    slate numbers can never be silently compared.
    """

    dataset: str
    retriever: str
    slice: str
    metric: str
    regime: str
    value: float
    ci_low: float
    ci_high: float
    n_impressions: int
    features: str = "serving_only"
    slice_basis: str = ""

    def __str__(self) -> str:
        return (
            f"  {self.retriever:28s} {self.slice:6s} {self.metric:12s} "
            f"{format_ci(self.value, self.ci_low, self.ci_high, self.n_impressions)}"
        )


@dataclass
class PerImpression:
    """Raw per-impression measurements, kept so slices are a re-index.

    Computing metrics once and slicing afterwards is what makes N slices cost
    nothing extra. It also guarantees the slice numbers and the overall number
    come from identical measurements -- a re-run per slice could silently
    diverge.
    """

    recall: dict[int, list[float]] = field(default_factory=dict)
    auc: list[float] = field(default_factory=list)
    #: Indices of impressions where AUC was undefined (all-pos or all-neg
    #: slate). Tracked rather than filled with 0.5 -- see metrics.auc.
    auc_undefined: int = 0
    mrr: list[float] = field(default_factory=list)
    ndcg5: list[float] = field(default_factory=list)
    ndcg10: list[float] = field(default_factory=list)
    diversity: list[float] = field(default_factory=list)
    novelty: list[float] = field(default_factory=list)
    retrieved: list[list[str]] = field(default_factory=list)
    history_length: list[int] = field(default_factory=list)
    clicked: list[set[str]] = field(default_factory=list)
    #: True when the retriever fell back to a non-personalised answer. The
    #: cheapest Q9 with/without pair available (Phase 2 D4).
    used_fallback: list[bool] = field(default_factory=list)
    seconds: float = 0.0


def measure(
    retriever,
    impressions: Sequence[Impression],
    histories: dict[str, History],
    articles: dict[str, Article],
    ks: tuple[int, ...] = KS,
) -> PerImpression:
    """One retrieval pass; every metric derived from it.

    Skips impressions that are unlabelled (nothing to score against) or whose
    user has no usable history (the retriever would be answering a different
    question). Both counts are logged rather than silently absorbed.
    """
    out = PerImpression(recall={k: [] for k in ks})
    max_k = max(ks)
    category = {a.article_id: (a.category or "?") for a in articles.values()}
    skipped_unlabelled = skipped_nohistory = 0
    started = time.perf_counter()

    for imp in impressions:
        if not imp.is_labelled:
            skipped_unlabelled += 1
            continue
        hist = histories.get(imp.user_id)
        if hist is None:
            skipped_nohistory += 1
            continue

        # The leakage boundary, applied at the call site. Exact on EB-NeRD;
        # on MIND `before` returns everything because there are no timestamps
        # (F1) -- the harness reports that rather than implying otherwise.
        past = hist.before(imp.time)
        texts = [articles[a].retrieval_text for a in past if a in articles]

        scored = retriever.retrieve(texts, max_k, imp.time) if texts else []
        retrieved = [aid for aid, _ in scored]
        truth = set(imp.clicked)

        # --- corpus regime -------------------------------------------------
        for k in ks:
            out.recall[k].append(recall_at_k(retrieved, truth, k))

        # --- slate regime --------------------------------------------------
        # Restrict the corpus-wide scores onto this impression's own slate.
        # Candidates the retriever never scored keep their slate order rather
        # than being shuffled arbitrarily (see metrics.rank_slate).
        slate = imp.candidates or retrieved[:max_k]
        scores = dict(scored)
        ranked = rank_slate(scores, slate)
        slate_truth = truth & set(slate)

        a = auc(ranked, slate_truth)
        if a is None:
            out.auc_undefined += 1
        else:
            out.auc.append(a)
        out.mrr.append(mrr(ranked, slate_truth))
        out.ndcg5.append(ndcg_at_k(ranked, slate_truth, 5))
        out.ndcg10.append(ndcg_at_k(ranked, slate_truth, 10))

        # --- beyond-accuracy ------------------------------------------------
        div = intra_list_diversity(retrieved[:max_k], category)
        if div is not None:
            out.diversity.append(div)

        out.retrieved.append(retrieved[:max_k])
        out.history_length.append(len(past))
        out.clicked.append(truth)
        out.used_fallback.append(not texts)

    out.seconds = time.perf_counter() - started
    if skipped_unlabelled or skipped_nohistory:
        log.info(
            "  skipped %d unlabelled, %d without history",
            skipped_unlabelled,
            skipped_nohistory,
        )
    if out.auc_undefined:
        log.info(
            "  AUC undefined on %d impressions (all-positive or all-negative slate) "
            "-- dropped, not imputed",
            out.auc_undefined,
        )
    return out


def _rows_for(
    per: PerImpression,
    dataset: str,
    retriever_name: str,
    slice_def: SliceDef | None,
    train_popularity: dict[str, float],
    corpus_size: int,
    seed: int,
    features: str = "serving_only",
) -> list[ResultRow]:
    """Turn measurements (optionally re-indexed to a slice) into result rows."""
    idx = list(slice_def.members) if slice_def else None
    name = slice_def.name if slice_def else "all"
    basis = slice_def.basis if slice_def else "all evaluated impressions"

    def take(vals: list) -> list:
        if idx is None:
            return vals
        return [vals[i] for i in idx if i < len(vals)]

    rows: list[ResultRow] = []

    def add(metric: str, regime: str, values: list[float]) -> None:
        if not values:
            return
        mean, lo, hi = bootstrap_ci(values, seed=seed)
        rows.append(
            ResultRow(dataset, retriever_name, name, metric, regime, mean, lo, hi,
                      len(values), features, basis)
        )

    for k, vals in per.recall.items():
        add(f"recall@{k}", "corpus", take(vals))
    # AUC is filtered separately (undefined slates were dropped), so a slice
    # index into it would be misaligned. Only report AUC unsliced.
    if slice_def is None:
        add("auc", "slate", per.auc)
    add("mrr", "slate", take(per.mrr))
    add("ndcg@5", "slate", take(per.ndcg5))
    add("ndcg@10", "slate", take(per.ndcg10))
    add("diversity", "corpus", take(per.diversity))

    retrieved = take(per.retrieved)
    if retrieved and train_popularity:
        nov = [n for items in retrieved if (n := novelty(items, train_popularity)) is not None]
        add("novelty", "corpus", nov)

    # Coverage is global, so it is bootstrapped by resampling impressions and
    # recomputing (D5), not by averaging per-impression values.
    if retrieved and corpus_size:
        # No CI: coverage is a distinct-count, monotone in n, and every
        # percentile-bootstrap scheme is biased low for it. See point_only().
        point, lo, hi = point_only(retrieved, lambda units: coverage(units, corpus_size))
        rows.append(
            ResultRow(dataset, retriever_name, name, "coverage", "corpus", point, lo, hi,
                      len(retrieved), features, basis)
        )
    return rows


def evaluate(
    retriever,
    impressions: Sequence[Impression],
    histories: dict[str, History],
    articles: dict[str, Article],
    dataset: str,
    train_popularity: dict[str, float] | None = None,
    ks: tuple[int, ...] = KS,
    seed: int = 0,
    with_slices: bool = True,
) -> list[ResultRow]:
    """Score one retriever end to end. The only place a metric is computed.

    Returns the tidy table described in plan/4-Evaluation-Harness.md: one row
    per slice x metric, each carrying its CI, n, regime and slice basis.
    """
    train_popularity = train_popularity or {}
    per = measure(retriever, impressions, histories, articles, ks=ks)
    corpus_size = len(articles)

    rows = _rows_for(per, dataset, retriever.name, None, train_popularity, corpus_size, seed)

    if with_slices and per.history_length:
        for sd in cold_warm_slices(per.history_length):
            if len(sd):
                rows.extend(
                    _rows_for(per, dataset, retriever.name, sd, train_popularity, corpus_size, seed)
                )
        for sd in head_tail_slices(per.clicked, train_popularity):
            if len(sd):
                rows.extend(
                    _rows_for(per, dataset, retriever.name, sd, train_popularity, corpus_size, seed)
                )

    # Q9: the with/without pair, obtained from the fallback flag rather than
    # from a second run (D6). Only meaningful if the retriever ever fell back.
    n_fallback = sum(per.used_fallback)
    if n_fallback and n_fallback < len(per.used_fallback):
        keep = tuple(i for i, fb in enumerate(per.used_fallback) if not fb)
        sd = SliceDef("all", keep, f"excluding {n_fallback} fallback results")
        rows.extend(
            _rows_for(per, dataset, retriever.name, sd, train_popularity, corpus_size, seed,
                      features="no_fallback")
        )

    log.info("  %s: %d rows in %.1fs", retriever.name, len(rows), per.seconds)
    return rows


def to_dicts(rows: Sequence[ResultRow]) -> list[dict]:
    return [asdict(r) for r in rows]


def format_table(rows: Sequence[ResultRow], metrics: Sequence[str] | None = None) -> str:
    """Render as a markdown table -- paste-ready for the design note."""
    wanted = set(metrics) if metrics else None
    sel = [r for r in rows if (wanted is None or r.metric in wanted) and r.slice == "all"
           and r.features == "serving_only"]
    if not sel:
        return "(no rows)"
    names = sorted({r.retriever for r in sel})
    cols = list(dict.fromkeys(r.metric for r in sel))
    out = ["| retriever | " + " | ".join(cols) + " |",
           "|---" * (len(cols) + 1) + "|"]
    for n in names:
        cells = []
        for c in cols:
            m = next((r for r in sel if r.retriever == n and r.metric == c), None)
            if m is None:
                cells.append("—")
            elif m.ci_low == m.ci_low:  # not NaN -- has a real interval
                cells.append(f"{m.value:.4f} [{m.ci_low:.4f}, {m.ci_high:.4f}]")
            else:
                # Point-only metric (coverage). Marked so the table cannot be
                # misread as having lost its interval by accident.
                cells.append(f"{m.value:.4f} (no CI)")
        out.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(out)
