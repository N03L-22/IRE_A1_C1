"""HNSW recall/latency at 10x scale: where is the knee? (F50 follow-up)

F50 found that at 1M vectors ef=128 gives 160x the speed of exact search but
loses 17% of the answer. That is a *default*, not a limit -- efSearch trades
latency back for recall at query time, and M trades index memory for graph
quality at build time.

The question this answers: **how much latency do we have to give back to get
recall above 0.95, and is it still worth doing versus exact search?**

Clustered vectors (50 centroids + noise), which is the structure real
embeddings have -- F49 established that uniform-random vectors are a worst
case that makes HNSW look far worse than it is.
"""
import json, time
from pathlib import Path

import faiss
import numpy as np

faiss.omp_set_num_threads(28)
N, D, NQ, K = 1_000_000, 384, 200, 200

rng = np.random.default_rng(0)
cent = rng.normal(size=(50, D)).astype(np.float32)
x = (cent[rng.integers(0, 50, N)] + rng.normal(scale=0.35, size=(N, D))).astype(np.float32)
faiss.normalize_L2(x)
q = x[rng.choice(N, NQ, replace=False)].copy()
print(f"n={N:,} d={D}  {x.nbytes/1e9:.2f} GB of vectors\n", flush=True)

flat = faiss.IndexFlatIP(D); flat.add(x)
t0 = time.perf_counter(); _, E = flat.search(q, K); exact_ms = (time.perf_counter()-t0)*1000
print(f"exact: {exact_ms:.0f} ms for {NQ} queries = {exact_ms/NQ:.1f} ms/query\n", flush=True)

rows = []
for M in (16, 32, 64):
    t0 = time.perf_counter()
    idx = faiss.IndexHNSWFlat(D, M, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = 200
    idx.add(x)
    build = time.perf_counter() - t0
    mem = idx.hnsw.neighbors.size() * 4 / 1e9  # graph links, int32
    for ef in (64, 128, 256, 512, 1024):
        idx.hnsw.efSearch = ef
        t0 = time.perf_counter(); _, I = idx.search(q, K); ms = (time.perf_counter()-t0)*1000
        rec = float(np.mean([len(set(I[i]) & set(E[i]))/K for i in range(NQ)]))
        rows.append({"M": M, "efSearch": ef, "recall": rec, "ms_total": round(ms,1),
                     "ms_per_query": round(ms/NQ,2), "speedup": round(exact_ms/ms,1),
                     "build_s": round(build,1), "graph_gb": round(mem,2)})
        print(f"  M={M:2d} ef={ef:4d}  recall {rec:.4f}  {ms/NQ:6.2f} ms/q  "
              f"{exact_ms/ms:6.1f}x  build {build:5.1f}s  graph {mem:.2f} GB", flush=True)
    print(flush=True)

Path("results").mkdir(exist_ok=True)
Path("results/ann_sweep_1m.json").write_text(json.dumps(
    {"n": N, "dim": D, "exact_ms_per_query": round(exact_ms/NQ,2), "rows": rows}, indent=2))
print("ANNSWEEPDONE", flush=True)
