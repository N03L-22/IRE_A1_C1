"""Multi-query vs single-vector, on MIND where 46% of users are incoherent (F40)."""
import logging, json, time, numpy as np
logging.basicConfig(level=logging.INFO, format="%(message)s")
for n in ("bm25s","transformers","src.retrieval.encode","src.retrieval.semantic","harness"):
    logging.getLogger(n).setLevel(logging.ERROR)
from pathlib import Path
from src.data.readers import get_reader, SPLIT_NAMES
from src.retrieval.semantic import HistoryIdRetriever, MultiQueryRetriever
from src.skeleton import temporal_split
from src.eval.harness import evaluate, measure
from src.eval.bootstrap import paired_difference_ci
from src.eval.run import _train_popularity

out=[]
for ds in ("mind","ebnerd"):
    r=get_reader(ds, Path("data/work"),"small")
    arts={a.article_id:a for a in r.articles()}
    imps=[]
    for i in r.impressions(SPLIT_NAMES[ds]["train"]):
        imps.append(i)
        if len(imps)>=8000: break
    hist={h.user_id:h for h in r.histories(SPLIT_NAMES[ds]["train"])}
    train,val=temporal_split(imps); val=val[:4000]
    pop=_train_popularity(train); alist=list(arts.values())
    print(f"\n=== {ds.upper()}  val={len(val)} ===", flush=True)
    per={}
    for lab, mk in (("single", lambda: HistoryIdRetriever(model_key="minilm", last_n=20)),
                    ("multi-c3", lambda: MultiQueryRetriever(model_key="minilm", last_n=20, n_clusters=3))):
        rt=mk(); t0=time.perf_counter(); rt.index(alist); idx=time.perf_counter()-t0
        t0=time.perf_counter(); m=measure(rt,val,hist,arts); q=time.perf_counter()-t0
        per[lab]=m
        res=evaluate(rt,val,hist,arts,ds,train_popularity=pop,with_slices=False)
        rec={x.metric:(x.value,x.ci_low,x.ci_high) for x in res if x.slice=="all"}
        out.append({"dataset":ds,"variant":lab,"query_seconds":round(q,1),
                    **{k:rec[k][0] for k in rec},"ci":{k:[rec[k][1],rec[k][2]] for k in rec},
                    "clusters":getattr(rt,"cluster_counts",None)})
        print(f"  {lab:9s} r@50={rec['recall@50'][0]:.4f} [{rec['recall@50'][1]:.4f},{rec['recall@50'][2]:.4f}]  "
              f"ndcg@10={rec['ndcg@10'][0]:.4f}  query {q:5.1f}s  clusters={getattr(rt,'cluster_counts',{})}", flush=True)
    for metric,A,B in (("recall@50",per["multi-c3"].recall[50],per["single"].recall[50]),
                       ("ndcg@10",per["multi-c3"].ndcg10,per["single"].ndcg10)):
        d,lo,hi,sig=paired_difference_ci(A,B)
        nd=int((np.array(A[:len(B)])-np.array(B[:len(A)])!=0).sum())
        print(f"    paired {metric:10s} {d:+.4f} [{lo:+.4f},{hi:+.4f}]  "
              f"{'SIGNIFICANT' if sig else 'not significant'}  differ on {nd}/{min(len(A),len(B))}", flush=True)
Path("results").mkdir(exist_ok=True)
Path("results/multiquery.json").write_text(json.dumps(out,indent=2))
print("\nMQDONE", flush=True)
