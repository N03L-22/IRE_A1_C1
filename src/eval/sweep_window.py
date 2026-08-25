"""Recency-window sweep (open question O2/O6).

F23 measured the window as worth 33x while k1/b/last_n were worth ~0.01, so
this is the one knob where a sweep is likely to move a number. F22 also found
window and K interact -- 72h scored WORSE than 24h at K=50, because a wider
pool admits more stale-but-similar distractors -- so both axes are varied.

EB-NeRD only: MIND has no publish time (F20).
"""

import logging, json, time
logging.basicConfig(level=logging.ERROR)
for n in ("bm25s","transformers","src.retrieval.encode","src.retrieval.semantic","harness"):
    logging.getLogger(n).setLevel(logging.ERROR)
from pathlib import Path
from src.data.readers import get_reader, SPLIT_NAMES
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.semantic import HistoryIdRetriever
from src.skeleton import WindowedRetriever, temporal_split
from src.eval.harness import evaluate
from src.eval.run import _train_popularity

r = get_reader("ebnerd", Path("data/work"), "small")
arts = {a.article_id: a for a in r.articles()}
imps = []
for i in r.impressions(SPLIT_NAMES["ebnerd"]["train"]):
    imps.append(i)
    if len(imps) >= 8000: break
hist = {h.user_id: h for h in r.histories(SPLIT_NAMES["ebnerd"]["train"])}
train, val = temporal_split(imps); val = val[:4000]
pop = _train_popularity(train)
alist = list(arts.values())
print(f"val={len(val)} corpus={len(alist)}", flush=True)

rows = []
for hrs in (6, 12, 24, 48, 72):
    for kind, mk in (("bm25", lambda: BM25Retriever(k1=1.6, b=1.0, last_n=15)),
                     ("semantic", lambda: HistoryIdRetriever(model_key="minilm", last_n=20))):
        w = WindowedRetriever(mk(), window_hours=hrs)
        w.index(alist)
        res = evaluate(w, val, hist, arts, "ebnerd", train_popularity=pop, with_slices=False)
        rec = {x.metric: (x.value, x.ci_low, x.ci_high) for x in res if x.slice == "all"}
        rows.append({"window_hours": hrs, "retriever": kind,
                     **{m: rec[m][0] for m in ("recall@50","recall@100","recall@200","auc","ndcg@10") if m in rec},
                     "ci50": rec.get("recall@50",(0,0,0))[1:]})
        print(f"  {hrs:3d}h {kind:9s} r@50={rec['recall@50'][0]:.4f} "
              f"[{rec['recall@50'][1]:.4f},{rec['recall@50'][2]:.4f}]  "
              f"r@200={rec['recall@200'][0]:.4f}  ndcg@10={rec['ndcg@10'][0]:.4f}", flush=True)

Path("results").mkdir(exist_ok=True)
json.dump(rows, open("results/window_sweep_ebnerd.json","w"), indent=2)
print("SWDONE", flush=True)
