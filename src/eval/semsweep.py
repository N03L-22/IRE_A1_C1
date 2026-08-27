"""The semantic knobs never swept: tau, decay, lam -- plus category text.

F-log audit showed tau/decay/lam appear in NO results file as a varied
parameter. This closes that gap on MIND, where semantic underperforms.
"""
import logging, json, numpy as np, time
logging.basicConfig(level=logging.INFO, format="%(message)s")
for n in ("transformers","src.retrieval.encode","src.retrieval.semantic","bm25s","harness"):
    logging.getLogger(n).setLevel(logging.ERROR)
from pathlib import Path
from src.data.readers import get_reader, SPLIT_NAMES
from src.retrieval.semantic import SemanticRetriever
from src.retrieval.encode import encode_cached
from src.eval.harness import evaluate, measure
from src.eval.run import _train_popularity
from src.eval.bootstrap import paired_difference_ci
from src.skeleton import temporal_split

ds="mind"
r=get_reader(ds, Path("data/work"), "small")
arts={a.article_id:a for a in r.articles()}
alist=list(arts.values()); ids=[a.article_id for a in alist]
hist={h.user_id:h for h in r.histories(SPLIT_NAMES[ds]["train"])}
imps=[]
for i in r.impressions(SPLIT_NAMES[ds]["train"]):
    imps.append(i)
    if len(imps)>=12000: break
train,val=temporal_split(imps); val=val[:4000]
pop=_train_popularity(train)
log=logging.getLogger("sweep")
log.info(f"val={len(val)} corpus={len(alist)}\n")

# two text variants
plain=[arts[i].retrieval_text for i in ids]
withcat=[f"{arts[i].category} {arts[i].subcategory} {arts[i].retrieval_text}".strip() for i in ids]
V_plain,_=encode_cached(plain, ids, model_key="minilm")
V_cat,_=encode_cached(withcat, [f"cat_{i}" for i in ids], model_key="minilm")
log.info("encoded both variants\n")

per={}
def run(name, vecs, **kw):
    rr=SemanticRetriever(vectors=vecs, vector_ids=ids, **kw); rr.name=name
    rr.index(alist)
    m=measure(rr, val, hist, arts); per[name]=m
    rows=evaluate(rr, val, hist, arts, ds, train_popularity=pop, with_slices=False)
    d={x.metric:(x.value,x.ci_low,x.ci_high) for x in rows if x.slice=="all"}
    log.info(f"  {name:38} r@100={d['recall@100'][0]:.4f} [{d['recall@100'][1]:.4f},{d['recall@100'][2]:.4f}]  ndcg@10={d['ndcg@10'][0]:.4f}")
    return d

log.info("=== tau sweep (never swept before) ===")
for tau in (0.20, 0.35, 0.50, 0.65, 0.80):
    run(f"tau={tau}", V_plain, tau=tau, last_n=20)

log.info("\n=== decay / lam (never swept) ===")
for dec,lam in (("log",0.3),("exp",0.1),("exp",0.3),("exp",0.6),("flat",0.0)):
    run(f"decay={dec},lam={lam}", V_plain, last_n=20, decay=dec, lam=lam)

log.info("\n=== category in the embedded text (Q1.4 names category) ===")
run("plain title+abstract", V_plain, last_n=20)
run("category + title+abstract", V_cat, last_n=20)

log.info("\n=== paired tests vs the shipped default (tau=0.35, log, n=20) ===")
base=per["tau=0.35"]
for k,m in per.items():
    if k=="tau=0.35": continue
    diff,lo,hi,sig=paired_difference_ci(m.recall[100], base.recall[100])
    if sig or abs(diff)>0.002:
        log.info(f"  {k:38} {diff:+.4f} [{lo:+.4f},{hi:+.4f}] {'SIGNIFICANT' if sig else ''}")
log.info("SWEEPDONE")
