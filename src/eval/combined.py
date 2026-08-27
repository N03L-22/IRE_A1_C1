"""Do F73/F74's fixes compound? Individually significant != jointly better.

shipped:  tau=0.35, log decay, max_pool fallback
fixed:    tau=0.20, flat decay, mean fallback (no max_pool)
Also test on EB-NeRD -- the fixes were found on MIND and may not transfer.
"""
import logging, numpy as np
logging.basicConfig(level=logging.INFO, format="%(message)s")
for n in ("transformers","src.retrieval.encode","src.retrieval.semantic","bm25s","harness"):
    logging.getLogger(n).setLevel(logging.ERROR)
from pathlib import Path
import src.retrieval.semantic as sem
from src.data.readers import get_reader, SPLIT_NAMES
from src.retrieval.encode import encode_cached
from src.eval.harness import evaluate, measure
from src.eval.run import _train_popularity
from src.eval.bootstrap import paired_difference_ci
from src.skeleton import temporal_split
log = logging.getLogger("comb")

_orig = sem.build_user_vector
def no_maxpool(vectors, tau=0.35, decay="log", lam=0.3):
    """F74: drop the max_pool rung -- keep the mean instead."""
    v, strat = _orig(vectors, tau=tau, decay=decay, lam=lam)
    if strat != "max_pool":
        return v, strat
    w = sem.decay_weights(len(vectors), decay, lam)[::-1]
    m = (vectors * w[:, None]).sum(axis=0) / w.sum()
    return (m / max(np.linalg.norm(m), 1e-12)).astype(np.float32), "mean_fallback"

for ds in ("mind", "ebnerd"):
    r = get_reader(ds, Path("data/work"), "small")
    arts = {a.article_id: a for a in r.articles()}
    alist = list(arts.values()); ids = [a.article_id for a in alist]
    V, _ = encode_cached([arts[i].retrieval_text for i in ids], ids, model_key="minilm")
    hist = {h.user_id: h for h in r.histories(SPLIT_NAMES[ds]["train"])}
    imps = []
    for i in r.impressions(SPLIT_NAMES[ds]["train"]):
        imps.append(i)
        if len(imps) >= 20000: break
    train, val = temporal_split(imps); val = val[:6000]
    pop = _train_popularity(train)
    log.info(f"\n=== {ds.upper()}  val={len(val)} corpus={len(alist)} ===")

    per = {}
    def go(name, patch, **kw):
        sem.build_user_vector = patch
        try:
            rr = sem.SemanticRetriever(vectors=V, vector_ids=ids, last_n=20, **kw)
            rr.name = name; rr.index(alist)
            per[name] = measure(rr, val, hist, arts)
            rows = evaluate(rr, val, hist, arts, ds, train_popularity=pop, with_slices=False)
            d = {x.metric: x.value for x in rows if x.slice == "all"}
            log.info(f"  {name:26} r@100={d['recall@100']:.4f}  r@200={d['recall@200']:.4f}  ndcg@10={d['ndcg@10']:.4f}")
        finally:
            sem.build_user_vector = _orig

    go("shipped", _orig, tau=0.35, decay="log")
    go("F73 only (tau .2 + flat)", _orig, tau=0.20, decay="flat")
    go("F74 only (no max_pool)", no_maxpool, tau=0.35, decay="log")
    go("BOTH", no_maxpool, tau=0.20, decay="flat")

    base = per["shipped"]
    log.info("  paired vs shipped (recall@100):")
    for k, m in per.items():
        if k == "shipped": continue
        diff, lo, hi, sig = paired_difference_ci(m.recall[100], base.recall[100])
        log.info(f"    {k:26} {diff:+.4f} [{lo:+.4f},{hi:+.4f}] {'SIGNIFICANT' if sig else 'ns'}")
log.info("COMBDONE")
