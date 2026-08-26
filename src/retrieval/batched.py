"""Batched GPU scoring for the submission path (branch: polars-gpu).

**What this is for.** `score_subset()` scores one slate per call: gather ~37
article vectors, one dot product, return. Measured at 623 us/slate. The
arithmetic is trivial; almost all of that is per-call overhead, which is why
batch=1 on a GPU is *slower* than CPU (85 slates/s) while batch=4,096 reaches
1.8M slates/s -- a ~1,200x span driven entirely by amortising the launch.

So this batches: pad B slates into one (B, K) index tensor, gather (B, K, d),
and do a single einsum against (B, d) query vectors.

> [!warning] This must produce the same RANKING as the unbatched path
> Padding and batching change the order of floating-point accumulation, so
> scores can differ in the last bits. The merge gate for this branch is a
> paired comparison showing zero rank inversions on real slates -- not that
> the code runs, and not that the scores are bitwise equal.

Nothing here changes what is retrieved or how it is scored; it changes only
how many slates are in flight. See `decisions.md` Part 4c for the costing.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

#: Measured knee (decisions.md Part 4c): 0.55 us/slate at 4,096, and no
#: further gain at 32,768. Larger batches only grow the padded tensor.
DEFAULT_BATCH = 4096


class BatchedSemanticScorer:
    """Score many slates at once against a fixed article matrix.

    Holds the article vectors on the GPU for the life of the run; only the
    per-batch index and query tensors move, which keeps transfer negligible
    (measured at ~1% of the unbatched loop).
    """

    def __init__(self, vectors: np.ndarray, ids: list[str], device: str | None = None) -> None:
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._ids = ids
        self._row = {aid: i for i, aid in enumerate(ids)}
        self._vecs = torch.from_numpy(np.ascontiguousarray(vectors)).to(self.device)
        self.dim = self._vecs.shape[1]
        log.info("batched scorer: %s vectors on %s (%.0f MB)",
                 f"{len(ids):,}", self.device, self._vecs.element_size() * self._vecs.nelement() / 1e6)

    def score_many(
        self, queries: np.ndarray, slates: list[list[str]]
    ) -> list[dict[str, float]]:
        """One dot product for the whole batch.

        ``queries`` is (B, d), already L2-normalised and aligned to ``slates``.
        Returns one ``{article_id: score}`` per slate, matching the shape
        ``score_subset()`` returns so the caller is unchanged.

        Slates have different lengths, so they are padded to the longest in
        the batch and the padding is masked out afterwards -- padded entries
        point at row 0 and are dropped by index, never by score, so a genuine
        row-0 hit is not lost.
        """
        import torch

        if not slates:
            return []
        B = len(slates)
        K = max(len(s) for s in slates)

        idx = np.zeros((B, K), dtype=np.int64)
        valid = np.zeros((B, K), dtype=bool)
        keep: list[list[str]] = []
        for b, slate in enumerate(slates):
            ids_here = []
            for k, aid in enumerate(slate):
                r = self._row.get(aid)
                if r is not None:
                    idx[b, k] = r
                    valid[b, k] = True
                    ids_here.append((k, aid))
            keep.append(ids_here)

        with torch.inference_mode():
            q = torch.from_numpy(np.ascontiguousarray(queries)).to(self.device)
            i = torch.from_numpy(idx).to(self.device)
            sub = self._vecs[i]                              # (B, K, d)
            scores = torch.einsum("bkd,bd->bk", sub, q)      # (B, K)
            out_scores = scores.cpu().numpy()

        results: list[dict[str, float]] = []
        for b, ids_here in enumerate(keep):
            results.append({aid: float(out_scores[b, k]) for k, aid in ids_here})
        return results
