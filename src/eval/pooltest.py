"""Should max_pool be replaced by mean? Test the fallback in isolation.

The tau sweep says looser tau (more mean, less max_pool) scores better, which
implies max_pool is a liability. But that is inference. Test it directly:
take ONLY the users the ladder sends to max_pool, and score them both ways.
"""
import logging, numpy as np
logging.basicConfig(level=logging.ERROR)
for n in ("transformers","src.retrieval.encode","src.retrieval.semantic","bm25s","harness"):
    logging.getLogger(n).setLevel(logging.ERROR)
from pathlib import Path
from src.data.readers import get_reader, SPLIT_NAMES
from src.retrieval.encode import encode_cached
from src.retrieval.semantic import decay_weights, DEFAULT_COHERENCE_TAU as TAU

ds="mind"
r=get_reader(ds, Path("data/work"), "small")
arts={a.article_id:a for a in r.articles()}
ids=list(arts); vecs,_=encode_cached([arts[i].retrieval_text for i in ids], ids, model_key="minilm")
row={a:i for i,a in enumerate(ids)}
hist={h.user_id:h for h in r.histories(SPLIT_NAMES[ds]["train"])}
imps=[]
for i in r.impressions(SPLIT_NAMES[ds]["train"]):
    imps.append(i)
    if len(imps)>=20000: break

def wmean(V):
    w=decay_weights(len(V),"log",0.3)[::-1]
    v=(V*w[:,None]).sum(axis=0)/w.sum()
    return v/max(np.linalg.norm(v),1e-12)

# find impressions whose user lands in max_pool under the current ladder
cases=[]
for imp in imps:
    h=hist.get(imp.user_id)
    if not h or not imp.clicked: continue
    rr=[row[a] for a in h.clicked_ids if a in row][-20:]
    if len(rr)<3: continue
    V=vecs[rr]; m=wmean(V)
    if float(np.mean(V@m))>=TAU: continue
    half=max(2,len(V)//2); Vh=V[-half:]; mh=wmean(Vh)
    if float(np.mean(Vh@mh))>=TAU: continue
    cases.append((imp, V, m))
print(f"impressions whose user hits max_pool: {len(cases)}\n")

def rank_of_click(q, imp):
    cand=[a for a in imp.candidates if a in row]
    if not cand: return None
    s=vecs[[row[a] for a in cand]] @ q
    order=[cand[i] for i in np.argsort(-s)]
    for pos,a in enumerate(order,1):
        if a in imp.clicked: return pos, len(order)
    return None

res={"max_pool":[], "mean":[]}
for imp,V,m in cases:
    p=V.max(axis=0); p=p/max(np.linalg.norm(p),1e-12)
    for name,q in (("max_pool",p),("mean",m)):
        got=rank_of_click(q,imp)
        if got: res[name].append(1.0/got[0])
n=min(len(res["max_pool"]),len(res["mean"]))
if n:
    a=np.array(res["max_pool"][:n]); b=np.array(res["mean"][:n])
    print(f"on those {n} impressions (MRR within slate):")
    print(f"  max_pool (current) {a.mean():.4f}")
    print(f"  mean     (proposed){b.mean():.4f}")
    d=b-a
    rng=np.random.default_rng(0); idx=rng.integers(0,n,size=(1000,n))
    draws=d[idx].mean(axis=1); lo,hi=np.percentile(draws,[2.5,97.5])
    print(f"  paired diff {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] {'SIGNIFICANT' if lo>0 or hi<0 else 'not significant'}")
print("POOLDONE")
