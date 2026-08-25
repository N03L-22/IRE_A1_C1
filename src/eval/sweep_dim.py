"""Embedding dimension vs retrieval quality, inside the 24h window (O8).

**The question is efficiency, not accuracy.** Dimension is the one semantic
knob with a direct resource cost: memory, index size and search time all scale
linearly in it. So the useful result is not "does 384 beat 128" but **"how much
quality do we give up per byte saved"** -- and if the answer is "none
measurable", the smaller vector is strictly better.

Method: **truncate the cached 384-d MiniLM vectors** to 256 and 128, then
re-normalise. This isolates dimension exactly -- same model, same training,
same text, only the width changes. Re-encoding with a different model would
confound dimension with training objective, which is the mistake F37 exists to
warn about (XLM-R's 768-d lost to MiniLM's 384-d by 348x on the Danish probe,
because the objective mattered and the size did not).

> [!warning] MiniLM is not Matryoshka-trained
> Models trained with Matryoshka representation learning are designed so that
> a prefix of the vector remains a valid embedding. MiniLM is not, so
> truncation has no guarantee of degrading gracefully. **That is the point of
> measuring it** -- a graceful curve is a real finding, and a cliff is equally
> informative.

Run inside the **24h recency window**, because F16/F41 showed full-corpus
semantic retrieval on EB-NeRD scores under 0.008 for everything. Comparing
numbers that are all nearly zero resolves nothing; inside the window the
retrievers actually work and a difference would be visible.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from ..data.readers import SPLIT_NAMES, get_reader
from ..eval.bootstrap import paired_difference_ci
from ..eval.harness import evaluate, measure
from ..eval.run import _train_popularity
from ..retrieval.encode import l2_normalise
from ..retrieval.semantic import SemanticRetriever
from ..skeleton import WindowedRetriever, temporal_split

log = logging.getLogger("sweep_dim")

DIMS = (384, 256, 128, 64)
WINDOW_HOURS = 24.0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("bm25s", "transformers", "src.retrieval.encode",
                  "src.retrieval.semantic", "harness"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    reader = get_reader("ebnerd", Path("data/work"), "small")
    articles = {a.article_id: a for a in reader.articles()}
    imps = []
    for i in reader.impressions(SPLIT_NAMES["ebnerd"]["train"]):
        imps.append(i)
        if len(imps) >= 8000:
            break
    hist = {h.user_id: h for h in reader.histories(SPLIT_NAMES["ebnerd"]["train"])}
    train, val = temporal_split(imps)
    val = val[:4000]
    pop = _train_popularity(train)
    alist = list(articles.values())
    log.info("val=%d corpus=%d window=%gh\n", len(val), len(alist), WINDOW_HOURS)

    # Encode once at full width; every smaller dimension is a prefix of it.
    full = SemanticRetriever(model_key="minilm", last_n=20)
    full.index(alist)
    base_vecs = full._vecs
    ids = full._ids
    log.info("encoded once: %s\n", base_vecs.shape)

    rows, per_impression = [], {}
    for d in DIMS:
        vecs = l2_normalise(base_vecs[:, :d].astype(np.float32).copy())
        r = SemanticRetriever(vectors=vecs, vector_ids=ids, last_n=20)
        r.name = f"minilm-{d}d"
        w = WindowedRetriever(r, window_hours=WINDOW_HOURS)

        t0 = time.perf_counter()
        w.index(alist)
        index_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        m = measure(w, val, hist, articles)
        query_s = time.perf_counter() - t0
        per_impression[d] = m

        res = evaluate(w, val, hist, articles, "ebnerd",
                       train_popularity=pop, with_slices=False)
        rec = {x.metric: (x.value, x.ci_low, x.ci_high) for x in res if x.slice == "all"}
        mb = vecs.nbytes / 1e6

        rows.append({
            "dim": d, "vectors_mb": round(mb, 1),
            "index_seconds": round(index_s, 1), "query_seconds": round(query_s, 1),
            **{k: rec[k][0] for k in rec},
            "ci": {k: [rec[k][1], rec[k][2]] for k in rec},
        })
        log.info("  %3dd  %6.1f MB  r@50=%.4f [%.4f,%.4f]  ndcg@10=%.4f  "
                 "index %4.1fs  query %5.1fs",
                 d, mb, *rec["recall@50"], rec["ndcg@10"][0], index_s, query_s)

    # Paired against full width: the only test that can say "no worse".
    log.info("\npaired vs 384d (same impressions, so noise cancels):")
    base = per_impression[384]
    for d in DIMS[1:]:
        m = per_impression[d]
        for metric, A, B in (("recall@50", m.recall[50], base.recall[50]),
                             ("ndcg@10", m.ndcg10, base.ndcg10)):
            diff, lo, hi, sig = paired_difference_ci(A, B)
            n_diff = int((np.array(A[:len(B)]) - np.array(B[:len(A)]) != 0).sum())
            log.info("  %3dd %-10s %+.4f [%+.4f, %+.4f]  %-16s differ on %d/%d",
                     d, metric, diff, lo, hi,
                     "SIGNIFICANT" if sig else "not significant",
                     n_diff, min(len(A), len(B)))

    Path("results").mkdir(exist_ok=True)
    Path("results/dim_sweep_ebnerd.json").write_text(json.dumps(rows, indent=2))
    log.info("\nwrote results/dim_sweep_ebnerd.json")
    log.info("DIMDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
