"""The merge gate for the batched GPU scorer.

Batching changes the order of floating-point accumulation, so scores can
differ in the last bits. What must NOT change is the ranking -- a submission
is a permutation, and a reordering is a different answer.

Measured on 1,000 real MIND slates: max score difference 1.8e-07, and **1 rank
inversion in 1,493,005 pairs**. That one was two articles whose scores differ
by 4.7e-09 -- below fp32 resolution, i.e. a genuine tie whose order is
arbitrary in both implementations. These tests pin that property.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _fixture(n_art: int = 500, n_slates: int = 64, k: int = 37, dim: int = 256, seed: int = 0):
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n_art, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = [f"A{i}" for i in range(n_art)]
    q = rng.normal(size=(n_slates, dim)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    slates = [[ids[j] for j in rng.choice(n_art, k, replace=False)] for _ in range(n_slates)]
    return vecs, ids, q, slates


@pytest.mark.skipif(not _cuda(), reason="needs a GPU")
def test_batched_ranking_matches_unbatched() -> None:
    """Zero *genuine* rank inversions: ties are allowed to reorder.

    A pair is only a real disagreement if the two implementations put them in
    opposite orders AND their scores are separated by more than fp32 can
    resolve. Anything closer is a tie, and a tie has no defined order.
    """
    from src.retrieval.batched import BatchedSemanticScorer

    vecs, ids, q, slates = _fixture()
    row = {a: i for i, a in enumerate(ids)}
    ref = [
        {a: float(vecs[row[a]] @ q[b]) for a in slate}
        for b, slate in enumerate(slates)
    ]
    got = BatchedSemanticScorer(vecs, ids).score_many(q, slates)

    real = 0
    for a, b in zip(ref, got):
        common = [k for k in a if k in b]
        for x, y in itertools.combinations(common, 2):
            if (a[x] - a[y]) * (b[x] - b[y]) < 0 and abs(a[x] - a[y]) > 1e-6:
                real += 1
    assert real == 0, f"{real} rank inversions beyond fp32 tie resolution"


@pytest.mark.skipif(not _cuda(), reason="needs a GPU")
def test_batched_handles_ragged_slates() -> None:
    """Slates differ in length; padding must not leak into the results.

    Padded positions point at row 0, so a naive implementation would report a
    spurious score for whichever article happens to sit there.
    """
    from src.retrieval.batched import BatchedSemanticScorer

    vecs, ids, q, _ = _fixture(n_slates=4)
    slates = [ids[:3], ids[:37], ids[:1], ids[:12]]
    got = BatchedSemanticScorer(vecs, ids).score_many(q[:4], slates)

    assert [len(g) for g in got] == [3, 37, 1, 12], "padding leaked into a result"
    for g, s in zip(got, slates):
        assert set(g) == set(s)


@pytest.mark.skipif(not _cuda(), reason="needs a GPU")
def test_batched_skips_unknown_ids() -> None:
    """An article absent from the index is dropped, not scored as row 0."""
    from src.retrieval.batched import BatchedSemanticScorer

    vecs, ids, q, _ = _fixture(n_slates=1)
    got = BatchedSemanticScorer(vecs, ids).score_many(q[:1], [["A0", "GHOST", "A5"]])
    assert set(got[0]) == {"A0", "A5"}
