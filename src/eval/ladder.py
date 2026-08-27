"""Graded-N ladder vs the current n -> n/2 -> max_pool ladder.

The user's proposal: on incoherence, try N=15, then 10, then 5, rather than
jumping straight to half. Question: does the finer ladder rescue more users
into a "mean" strategy, and does that change retrieval?
"""
import logging, numpy as np
logging.basicConfig(level=logging.ERROR)
for n in ("transformers","src.retrieval.encode","src.retrieval.semantic","bm25s"):
    logging.getLogger(n).setLevel(logging.ERROR)
from pathlib import Path
from src.data.readers import get_reader, SPLIT_NAMES
from src.retrieval.semantic import decay_weights, DEFAULT_COHERENCE_TAU as TAU
from src.retrieval.encode import encode_cached

def vec_for(vs, k, decay="log", lam=0.3):
    v = vs[-k:] if k < len(vs) else vs
    w = decay_weights(len(v), decay, lam)[::-1]
    out = (v * w[:, None]).sum(axis=0) / w.sum()
    out = out / max(np.linalg.norm(out), 1e-12)
    return out, float(np.mean(v @ out))

for ds in ("mind", "ebnerd"):
    r = get_reader(ds, Path("data/work"), "small")
    arts = {a.article_id: a for a in r.articles()}
    ids = [a.article_id for a in arts.values()]
    vecs, _ = encode_cached([arts[i].retrieval_text for i in ids], ids, model_key="minilm")
    row = {a: i for i, a in enumerate(ids)}
    hist = list(r.histories(SPLIT_NAMES[ds]["train"]))[:4000]

    cur = {"mean":0,"recent_half":0,"max_pool":0}
    grad = {}
    n_multi = 0
    for h in hist:
        rows = [row[a] for a in h.clicked_ids if a in row]
        if len(rows) < 2: continue
        n_multi += 1
        vs = vecs[rows][-20:]
        # current ladder
        _, c = vec_for(vs, len(vs))
        if c >= TAU: cur["mean"] += 1
        else:
            half = max(2, len(vs)//2)
            _, c2 = vec_for(vs, half)
            cur["recent_half" if c2 >= TAU else "max_pool"] += 1
        # graded ladder: 20 -> 15 -> 10 -> 5 -> 3
        landed = "max_pool"
        for k in (20, 15, 10, 5, 3):
            if k > len(vs): continue
            _, ck = vec_for(vs, k)
            if ck >= TAU: landed = f"N={k}"; break
        grad[landed] = grad.get(landed, 0) + 1

    print(f"\n=== {ds.upper()}  (n={n_multi} users with >=2 clicks) ===")
    print("  current ladder:", {k: f"{v} ({v/n_multi*100:.1f}%)" for k,v in cur.items()})
    print("  graded ladder :", {k: f"{v} ({v/n_multi*100:.1f}%)" for k,v in sorted(grad.items())})
    rescued = n_multi - grad.get("max_pool", 0)
    cur_ok = cur["mean"] + cur["recent_half"]
    print(f"  -> reach a real centroid: current {cur_ok/n_multi*100:.1f}%  graded {rescued/n_multi*100:.1f}%")
print("\nLADDERDONE")
