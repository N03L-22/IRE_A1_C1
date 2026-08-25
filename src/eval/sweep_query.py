"""Query construction: decay x last_n x retriever, both datasets (F58).

D-LEX-QUERY chose log decay and last_n by REASONING, never measurement. F51
swept last_n inside the window for BM25 only. This sweeps the remaining
question -- does the decay shape matter, and does "strictly the last few
clicks" beat a longer weighted history -- across lexical, semantic AND fusion,
because F39 showed the two datasets disagree about which retriever wins.

Paired against each dataset's shipped default so the comparison is the tight
test (F46), not the overlap heuristic.
"""
import json, logging, numpy as np, time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
for _n in ("bm25s","transformers","src.retrieval.encode","src.retrieval.semantic","harness"):
    logging.getLogger(_n).setLevel(logging.ERROR)

from src.data.readers import SPLIT_NAMES, get_reader
from src.eval.bootstrap import paired_difference_ci
from src.eval.harness import evaluate, measure
from src.eval.run import _train_popularity
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.fusion import RRFusion
from src.retrieval.semantic import HistoryIdRetriever
from src.skeleton import temporal_split

BM25P = {"mind": dict(k1=1.6, b=0.75), "ebnerd": dict(k1=1.6, b=1.0)}
DEFAULT_N = {"mind": 5, "ebnerd": 15}
LAST_N = (3, 5, 10, 20)
DECAYS = ("log", "exp", "flat")

out, per = [], {}
for ds in ("mind", "ebnerd"):
    r = get_reader(ds, Path("data/work"), "small")
    arts = {a.article_id: a for a in r.articles()}
    imps = []
    for i in r.impressions(SPLIT_NAMES[ds]["train"]):
        imps.append(i)
        if len(imps) >= 8000: break
    hist = {h.user_id: h for h in r.histories(SPLIT_NAMES[ds]["train"])}
    train, val = temporal_split(imps); val = val[:4000]
    pop = _train_popularity(train); alist = list(arts.values())
    print(f"\n=== {ds.upper()}  val={len(val)} corpus={len(alist)} ===", flush=True)

    def score(tag, rt):
        rt.index(alist)
        m = measure(rt, val, hist, arts)
        per[(ds, tag)] = m
        res = evaluate(rt, val, hist, arts, ds, train_popularity=pop, with_slices=False)
        rec = {x.metric: (x.value, x.ci_low, x.ci_high) for x in res if x.slice == "all"}
        out.append({"dataset": ds, "config": tag, **{k: rec[k][0] for k in rec},
                    "ci": {k: [rec[k][1], rec[k][2]] for k in rec}})
        print(f"  {tag:34s} r@50={rec['recall@50'][0]:.4f} "
              f"[{rec['recall@50'][1]:.4f},{rec['recall@50'][2]:.4f}] "
              f"ndcg@10={rec['ndcg@10'][0]:.4f}", flush=True)

    # lexical: last_n only (BM25 has no decay -- terms are a bag)
    for n in LAST_N:
        score(f"bm25 n={n}", BM25Retriever(**BM25P[ds], last_n=n))
    # semantic and fusion: decay x last_n
    for dk in DECAYS:
        for n in LAST_N:
            score(f"sem {dk} n={n}",
                  HistoryIdRetriever(model_key="minilm", last_n=n, decay=dk))
    for dk in DECAYS:
        for n in (3, 5, 20):
            score(f"fusion {dk} n={n}", RRFusion(
                [BM25Retriever(**BM25P[ds], last_n=n),
                 HistoryIdRetriever(model_key="minilm", last_n=n, decay=dk)],
                name=f"rrf-{dk}-{n}"))

    base = per[(ds, f"fusion log n={DEFAULT_N[ds]}")] if (ds, f"fusion log n={DEFAULT_N[ds]}") in per else None
    if base is None:
        base = per[(ds, f"fusion log n=5")]
    print(f"\n  paired vs fusion log n=5 ({ds}):", flush=True)
    for (d2, tag), m in per.items():
        if d2 != ds or tag == "fusion log n=5": continue
        diff, lo, hi, sig = paired_difference_ci(m.recall[50], base.recall[50])
        if sig:
            print(f"    ** {tag:32s} {diff:+.4f} [{lo:+.4f},{hi:+.4f}] SIGNIFICANT", flush=True)

Path("results").mkdir(exist_ok=True)
Path("results/query_sweep.json").write_text(json.dumps(out, indent=2))
print("\nQSDONE", flush=True)
