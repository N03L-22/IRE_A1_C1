"""last_n INSIDE the 24h window -- the D4 question asked in the shipped regime.

F23 swept last_n over the full corpus and found it worth ~0.03, inside the CI.
But F22/F41 showed the window and K interact, and the full-corpus regime is one
where semantic retrieval scores under 0.008 for everything -- so that sweep
compared numbers that were all nearly zero.

This asks the same question where the retrievers actually work. Paired against
the shipped default so the comparison is the tight test, not the overlap
heuristic (F46).
"""
import logging, json, numpy as np
logging.basicConfig(level=logging.INFO, format="%(message)s")
for n in ("bm25s","transformers","src.retrieval.encode","src.retrieval.semantic","harness"):
    logging.getLogger(n).setLevel(logging.ERROR)
from pathlib import Path
from src.data.readers import get_reader, SPLIT_NAMES
from src.retrieval.bm25 import BM25Retriever
from src.skeleton import WindowedRetriever, temporal_split
from src.eval.harness import evaluate, measure
from src.eval.bootstrap import paired_difference_ci
from src.eval.run import _train_popularity

LAST_N = (5, 10, 15, 20, 30, 50)
DEFAULT = 15

r = get_reader("ebnerd", Path("data/work"), "small")
arts = {a.article_id: a for a in r.articles()}
imps = []
for i in r.impressions(SPLIT_NAMES["ebnerd"]["train"]):
    imps.append(i)
    if len(imps) >= 8000: break
hist = {h.user_id: h for h in r.histories(SPLIT_NAMES["ebnerd"]["train"])}
train, val = temporal_split(imps); val = val[:4000]
pop = _train_popularity(train); alist = list(arts.values())
print(f"val={len(val)} corpus={len(alist)} window=24h\n", flush=True)

rows, per = [], {}
for n in LAST_N:
    w = WindowedRetriever(BM25Retriever(k1=1.6, b=1.0, last_n=n), window_hours=24.0)
    w.index(alist)
    per[n] = measure(w, val, hist, arts)
    res = evaluate(w, val, hist, arts, "ebnerd", train_popularity=pop, with_slices=False)
    rec = {x.metric: (x.value, x.ci_low, x.ci_high) for x in res if x.slice == "all"}
    rows.append({"last_n": n, **{k: rec[k][0] for k in rec},
                 "ci": {k: [rec[k][1], rec[k][2]] for k in rec}})
    print(f"  last_n={n:2d}  r@50={rec['recall@50'][0]:.4f} "
          f"[{rec['recall@50'][1]:.4f},{rec['recall@50'][2]:.4f}]  "
          f"ndcg@10={rec['ndcg@10'][0]:.4f}", flush=True)

print(f"\npaired vs last_n={DEFAULT} (the shipped default):", flush=True)
base = per[DEFAULT]
for n in LAST_N:
    if n == DEFAULT: continue
    for metric, A, B in (("recall@50", per[n].recall[50], base.recall[50]),
                         ("ndcg@10", per[n].ndcg10, base.ndcg10)):
        d, lo, hi, sig = paired_difference_ci(A, B)
        nd = int((np.array(A[:len(B)]) - np.array(B[:len(A)]) != 0).sum())
        print(f"  {n:2d} {metric:10s} {d:+.4f} [{lo:+.4f},{hi:+.4f}]  "
              f"{'SIGNIFICANT' if sig else 'not significant':16s} differ on {nd}/{min(len(A),len(B))}", flush=True)

Path("results").mkdir(exist_ok=True)
Path("results/lastn_sweep_ebnerd.json").write_text(json.dumps(rows, indent=2))
print("\nLNDONE", flush=True)
