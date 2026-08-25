"""Metric correctness (Q4.1, Q4.2, Q4.4).

Every reported number flows through these functions, so they are checked
against hand-computable cases rather than against themselves. Where a metric
has a known closed form on a small example, that value is written out in the
test rather than computed by the code under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.eval.bootstrap import bootstrap_ci, format_ci, point_only
from src.eval.metrics import (
    auc,
    coverage,
    intra_list_diversity,
    mrr,
    ndcg_at_k,
    novelty,
    rank_slate,
    recall_at_k,
)
from src.eval.slices import cold_warm_slices, head_tail_slices


# --------------------------------------------------------------------------
# Corpus regime
# --------------------------------------------------------------------------


def test_recall_counts_fraction_of_clicks_found() -> None:
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, {"a"}, 4) == 1.0
    assert recall_at_k(retrieved, {"z"}, 4) == 0.0
    assert recall_at_k(retrieved, {"a", "z"}, 4) == 0.5


def test_recall_is_monotonic_in_k() -> None:
    """recall@200 < recall@50 is impossible by construction.

    The pitfall table lists this as a bug signature; asserting it here means a
    K-handling regression fails a test instead of producing a plausible table.
    """
    retrieved = [f"a{i}" for i in range(200)]
    clicked = {"a10", "a80", "a150"}
    r50 = recall_at_k(retrieved, clicked, 50)
    r100 = recall_at_k(retrieved, clicked, 100)
    r200 = recall_at_k(retrieved, clicked, 200)
    assert r50 <= r100 <= r200


# --------------------------------------------------------------------------
# Slate regime
# --------------------------------------------------------------------------


def test_auc_perfect_and_worst() -> None:
    ranked = ["hit", "miss1", "miss2"]
    assert auc(ranked, {"hit"}) == 1.0
    assert auc(["miss1", "miss2", "hit"], {"hit"}) == 0.0


def test_auc_midpoint() -> None:
    """One positive in the middle of four items: 2 of 3 negatives below it."""
    assert auc(["m1", "hit", "m2", "m3"], {"hit"}) == pytest.approx(2 / 3)


def test_auc_undefined_returns_none() -> None:
    """All-positive or all-negative slates have no AUC.

    Returning None rather than 0.5 keeps a fabricated value out of the mean --
    the harness drops these and reports the count.
    """
    assert auc(["a", "b"], {"a", "b"}) is None
    assert auc(["a", "b"], set()) is None


def test_mrr_is_reciprocal_of_first_hit() -> None:
    assert mrr(["a", "b", "c"], {"a"}) == 1.0
    assert mrr(["a", "b", "c"], {"b"}) == 0.5
    assert mrr(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert mrr(["a", "b", "c"], {"z"}) == 0.0


def test_ndcg_binary_relevance_closed_form() -> None:
    """Hit at position 2 of a 3-item slate: DCG = 1/log2(3), IDCG = 1/log2(2)."""
    got = ndcg_at_k(["m", "hit", "m2"], {"hit"}, 10)
    assert got == pytest.approx((1 / math.log2(3)) / (1 / math.log2(2)))


def test_ndcg_perfect_ranking_is_one() -> None:
    assert ndcg_at_k(["h1", "h2", "m"], {"h1", "h2"}, 10) == pytest.approx(1.0)


def test_ndcg_respects_cutoff() -> None:
    """A hit outside the cutoff contributes nothing."""
    assert ndcg_at_k(["m"] * 9 + ["hit"], {"hit"}, 5) == 0.0
    assert ndcg_at_k(["m"] * 9 + ["hit"], {"hit"}, 10) > 0.0


def test_rank_slate_orders_by_score_then_original_position() -> None:
    """Unscored candidates fall to the bottom in their original order.

    An arbitrary tie-break over unscored items is what made the popularity
    baseline behave as a slate-order ranker without saying so (F21).
    """
    slate = ["a", "b", "c", "d"]
    ranked = rank_slate({"c": 5.0, "a": 1.0}, slate)
    assert ranked[:2] == ["c", "a"]
    assert ranked[2:] == ["b", "d"], "unscored items lost their original order"


# --------------------------------------------------------------------------
# Beyond-accuracy
# --------------------------------------------------------------------------


def test_diversity_extremes() -> None:
    cats = {"a": "sport", "b": "sport", "c": "news"}
    assert intra_list_diversity(["a", "b"], cats) == 0.0        # all same
    assert intra_list_diversity(["a", "c"], cats) == 1.0        # all different
    assert intra_list_diversity(["a", "b", "c"], cats) == pytest.approx(2 / 3)


def test_diversity_needs_a_pair() -> None:
    assert intra_list_diversity(["a"], {"a": "sport"}) is None


def test_novelty_rewards_rare_items() -> None:
    pop = {"common": 0.5, "rare": 0.01}
    assert novelty(["rare"], pop) > novelty(["common"], pop)
    assert novelty(["common"], pop) == pytest.approx(-math.log2(0.5))


def test_novelty_unseen_article_is_finite() -> None:
    """An unseen article must not produce inf and poison the mean."""
    got = novelty(["never_seen"], {"a": 0.25})
    assert got is not None and math.isfinite(got)


def test_coverage_fraction_of_catalogue() -> None:
    assert coverage([["a", "b"], ["b", "c"]], corpus_size=4) == 0.75
    assert coverage([], corpus_size=4) == 0.0


# --------------------------------------------------------------------------
# Bootstrap (Q4.4)
# --------------------------------------------------------------------------


def test_bootstrap_ci_brackets_the_mean() -> None:
    rng = np.random.default_rng(1)
    values = rng.random(500).tolist()
    mean, lo, hi = bootstrap_ci(values)
    assert lo < mean < hi
    assert mean == pytest.approx(float(np.mean(values)))


def test_bootstrap_is_seeded_and_reproducible() -> None:
    """A CI that moves between runs cannot be quoted in a design note."""
    values = [0.0, 1.0] * 100
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)


def test_bootstrap_interval_narrows_with_n() -> None:
    """The sanity property: more data, tighter interval."""
    rng = np.random.default_rng(2)
    small = rng.random(50).tolist()
    large = rng.random(5000).tolist()
    _, lo_s, hi_s = bootstrap_ci(small)
    _, lo_l, hi_l = bootstrap_ci(large)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_bootstrap_constant_sample_has_zero_width() -> None:
    mean, lo, hi = bootstrap_ci([0.4] * 100)
    assert mean == pytest.approx(0.4)
    assert lo == pytest.approx(0.4) and hi == pytest.approx(0.4)


def test_bootstrap_single_observation_reports_no_interval() -> None:
    """One observation says nothing about its own variability."""
    mean, lo, hi = bootstrap_ci([0.5])
    assert mean == 0.5
    assert math.isnan(lo) and math.isnan(hi)


def test_coverage_is_reported_without_a_ci() -> None:
    """Coverage gets a point estimate and explicit NaN bounds.

    Not an oversight: coverage is a distinct-count and monotone in sample
    size, so every percentile-bootstrap scheme is biased low. Measured with
    replacement: point 0.9783 against CI [0.9035, 0.9235] -- the estimate
    outside its own interval. Subsampling without replacement fails to
    bracket at every ratio tried (0.5 through 0.99).

    Emitting NaN bounds makes the absence visible in the results table rather
    than silently shipping a manufactured interval.
    """
    units = [["a"], ["b"], ["c"], ["d"]]
    point, lo, hi = point_only(units, lambda u: coverage(u, 4))
    assert point == 1.0
    assert math.isnan(lo) and math.isnan(hi)


def test_format_ci_omits_bounds_when_absent() -> None:
    """A point-only metric must not render as though it had an interval."""
    rendered = format_ci(0.42, float("nan"), float("nan"), 400)
    assert "[" not in rendered
    assert rendered == "0.4200, n = 400"


def test_format_ci_is_the_house_style() -> None:
    assert format_ci(0.34, 0.31, 0.37, 73152) == "0.3400 [0.3100, 0.3700], n = 73,152"


# --------------------------------------------------------------------------
# Slices (Q4.3)
# --------------------------------------------------------------------------


def test_cold_warm_threshold_adapts_to_the_distribution() -> None:
    """A fixed '< 5 clicks' rule empties the slice on EB-NeRD (F9).

    Both fixtures below must yield a non-empty cold slice despite having
    completely different supports -- that is the whole point of deriving the
    threshold from the data.
    """
    ebnerd_like = [5, 6, 20, 90, 400, 2000]
    mind_like = [1, 2, 3, 19, 40, 500]
    for lengths in (ebnerd_like, mind_like):
        slices = {s.name: s for s in cold_warm_slices(lengths)}
        assert len(slices["cold"]) > 0, "cold slice must not be empty"
        assert len(slices["warm"]) > 0
        assert len(slices["cold"]) + len(slices["warm"]) == len(lengths)


def test_cold_slice_holds_the_shortest_histories() -> None:
    """Cold is the bottom quartile, so on 4 items it is the single shortest.

    q0.25 of [1, 2, 100, 200] is 1.75, so only the length-1 user is cold. The
    length-2 user being warm is the threshold working as specified, not an
    off-by-one -- with a quartile boundary the cold slice is ~25% of users by
    construction.
    """
    slices = {s.name: s for s in cold_warm_slices([1, 2, 100, 200])}
    assert set(slices["cold"].members) == {0}
    assert set(slices["warm"].members) == {1, 2, 3}

    # With 8 users the quartile admits two, confirming it scales with n
    # rather than being a fixed count.
    eight = {s.name: s for s in cold_warm_slices([1, 2, 3, 4, 100, 200, 300, 400])}
    assert len(eight["cold"]) == 2


def test_slice_basis_is_recorded() -> None:
    """A threshold quoted without its derivation is taken on trust."""
    cold = cold_warm_slices([1, 5, 50, 500])[0]
    assert "history length" in cold.basis and "q0.25" in cold.basis


def test_head_tail_splits_on_train_popularity() -> None:
    pop = {"hot": 0.9, "mid": 0.5, "cold": 0.05}
    clicked = [{"hot"}, {"cold"}, {"never_in_train"}]
    slices = {s.name: s for s in head_tail_slices(clicked, pop)}
    assert 0 in slices["head"].members
    assert 1 in slices["tail"].members
    assert 2 in slices["tail"].members, "unseen article belongs in the tail"


# --------------------------------------------------------------------------
# Fusion (Q3.5)
# --------------------------------------------------------------------------


def test_rrf_promotes_documents_both_retrievers_like() -> None:
    """A doc ranked mid-table by both should beat one ranked top by only one.

    That is the entire premise of fusion: agreement is evidence. If this
    property does not hold, the fusion is just an expensive alias for its
    strongest component.
    """
    from src.retrieval.fusion import RRFusion

    class Fake:
        def __init__(self, name, order):
            self.name = name
            self.order = order

        def index(self, articles):
            pass

        def retrieve(self, history_text, k, at_time=None):
            return [(a, 1.0 / (i + 1)) for i, a in enumerate(self.order[:k])]

    # "both" is 3rd and 3rd; "onlyA" is 1st then absent.
    a = Fake("a", ["onlyA", "x", "both", "y", "z"])
    b = Fake("b", ["onlyB", "p", "both", "q", "r"])

    fused = RRFusion([a, b])
    ranked = [aid for aid, _ in fused.retrieve(["q"], 5)]
    assert ranked[0] == "both", f"agreement not rewarded: {ranked}"


def test_rrf_is_deterministic() -> None:
    """Two runs must produce identical files, so ties break on id."""
    from src.retrieval.fusion import RRFusion

    class Fake:
        name = "f"

        def index(self, articles):
            pass

        def retrieve(self, history_text, k, at_time=None):
            return [("a", 1.0), ("b", 1.0)]

    fused = RRFusion([Fake(), Fake()])
    assert fused.retrieve(["q"], 2) == fused.retrieve(["q"], 2)


def test_popularity_prior_blends_without_erasing_the_retriever() -> None:
    """alpha=0 must be the bare retriever; alpha=1 must be pure popularity."""
    from src.retrieval.fusion import PopularityPrior

    class Fake:
        name = "inner"

        def index(self, articles):
            pass

        def retrieve(self, history_text, k, at_time=None):
            return [("r1", 5.0), ("r2", 4.0)]

    pop = {"p1": 0.9, "p2": 0.5}

    only_retriever = PopularityPrior(Fake(), pop, alpha=0.0)
    assert only_retriever.retrieve(["q"], 1)[0][0] == "r1"

    only_popular = PopularityPrior(Fake(), pop, alpha=1.0)
    assert only_popular.retrieve(["q"], 1)[0][0] == "p1"
