"""Semantic retrieval (Q3): pooling, normalisation, and encode invariants.

The encoder itself needs a GPU and a model download, so those tests are
skipped unless one is available. The pooling logic is pure numpy and always
runs -- it is where the design decisions live.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.encode import l2_normalise
from src.retrieval.semantic import build_user_vector, log_decay_weights


def test_log_decay_is_mild_not_collapsing() -> None:
    """The whole point of log over exponential decay (D-LEX-QUERY).

    Exponential decay at any useful lambda leaves the 20th click at ~0.002 of
    the first -- i.e. the query is really the last two or three clicks. Log
    decay keeps it around 0.28, so older clicks still contribute.
    """
    w = log_decay_weights(20)
    ratio = w[19] / w[0]
    assert 0.2 < ratio < 0.4, f"20th click weighted {ratio:.3f} of the first"
    assert all(w[i] > w[i + 1] for i in range(len(w) - 1)), "not monotonically decaying"


def test_coherent_history_uses_the_mean() -> None:
    """One interest -> the centroid represents it, so use it."""
    base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vecs = l2_normalise(np.array([base + np.random.default_rng(i).normal(0, 0.05, 3)
                                  for i in range(6)], dtype=np.float32))
    _, strategy = build_user_vector(vecs, tau=0.35)
    assert strategy == "mean"


def test_incoherent_history_falls_back() -> None:
    """Two disjoint interests -> the centroid matches neither, so do not use it.

    This is the failure the conditional pooling exists to avoid: a user who
    reads football and recipes gets a vector sitting between the clusters.
    """
    a = np.tile(np.array([1.0, 0.0, 0.0]), (4, 1))
    b = np.tile(np.array([0.0, 1.0, 0.0]), (4, 1))
    # Alternating so neither half is coherent either.
    vecs = l2_normalise(np.vstack([a, b])[[0, 4, 1, 5, 2, 6, 3, 7]].astype(np.float32))
    _, strategy = build_user_vector(vecs, tau=0.9)
    assert strategy in ("recent_half", "max_pool"), strategy


def test_user_vector_is_always_unit_norm() -> None:
    """Inner-product search needs unit vectors or it ranks by magnitude."""
    rng = np.random.default_rng(0)
    for n in (1, 2, 5, 20):
        vecs = l2_normalise(rng.normal(size=(n, 16)).astype(np.float32))
        vec, _ = build_user_vector(vecs)
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)


def test_single_click_history_is_that_click() -> None:
    v = l2_normalise(np.array([[0.6, 0.8]], dtype=np.float32))
    vec, strategy = build_user_vector(v)
    assert strategy == "single"
    assert np.allclose(vec, v[0])


def test_l2_normalise_handles_a_zero_row() -> None:
    """A zero vector must not become NaN and poison every later dot product."""
    got = l2_normalise(np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32))
    assert not np.isnan(got).any()
    assert np.isclose(np.linalg.norm(got[1]), 1.0)


# --------------------------------------------------------------------------
# Encoder invariants -- these need the model, so they are opt-in.
# --------------------------------------------------------------------------


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="needs a GPU and the model")
def test_length_sorted_batching_does_not_change_results() -> None:
    """The 1.80x speed-up must be free, not approximate.

    Sorting by length changes only which texts share a padded batch; each is
    still encoded independently and padding is masked out of the mean pool.
    Verified against one-at-a-time encoding: agreement to fp16 rounding.
    """
    from src.retrieval.encode import encode_texts

    texts = [f"artikel {i} " + "ord " * (i % 30) for i in range(64)]
    batched, _ = encode_texts(texts, model_key="minilm", batch_size=32)
    one_at_a_time = np.vstack(
        [encode_texts([t], model_key="minilm", batch_size=1)[0] for t in texts[:8]]
    )
    assert np.abs(batched[:8] - one_at_a_time).max() < 1e-2, "batching changed the vectors"
    assert np.allclose(np.linalg.norm(batched, axis=1), 1.0, atol=1e-4)


def test_ef_search_scales_with_corpus_size() -> None:
    """A fixed efSearch silently degrades as the corpus grows (F52).

    HNSW explores `efSearch` candidates regardless of how many vectors exist,
    so the *fraction* of the corpus inspected shrinks with scale. Measured at
    1M vectors, the ef=128 that was lossless at 125K collapses to 0.68 recall
    -- a third of the answer lost, with no error and no warning.

    Recall is cheap to buy back: ef=1024 restores 0.9588 while still being 68x
    faster than exact search. So the default is derived from the corpus rather
    than pinned.
    """
    from src.retrieval.semantic import default_ef_search

    small = default_ef_search(20_738)
    large = default_ef_search(1_000_000)
    assert small < large, "ef must grow with the corpus"
    assert large >= 1024, f"1M corpus needs ef>=1024 for ~0.96 recall, got {large}"

    # Monotone: a bigger corpus never gets a smaller budget.
    sizes = [10_000, 50_000, 200_000, 500_000, 2_000_000]
    efs = [default_ef_search(n) for n in sizes]
    assert efs == sorted(efs), efs


def test_explicit_ef_search_is_respected() -> None:
    """An explicit value must win, so the sweep can pin what it measures."""
    from src.retrieval.semantic import SemanticRetriever

    assert SemanticRetriever(ef_search=64).ef_search == 64
    assert SemanticRetriever().ef_search is None  # derived at index time
