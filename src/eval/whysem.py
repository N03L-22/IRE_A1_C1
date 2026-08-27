"""Why is semantic not beating lexical on MIND? Diagnose, don't guess.

Three candidate causes:
  A. the query vector is diffuse (many topics averaged)
  B. the corpus is dominated by near-duplicates competing in-slate
  C. category signal is discarded (not in retrieval_text)
"""
import logging, numpy as np, collections
logging.basicConfig(level=logging.ERROR)
for n in ("transformers","src.retrieval.encode","src.retrieval.semantic","bm25s"):
    logging.getLogger(n).setLevel(logging.ERROR)
from pathlib import Path
from src.data.readers import get_reader, SPLIT_NAMES
from src.retrieval.encode import encode_cached

ds="mind"
r=get_reader(ds, Path("data/work"), "small")
arts={a.article_id:a for a in r.articles()}
ids=list(arts)
vecs,_=encode_cached([arts[i].retrieval_text for i in ids], ids, model_key="minilm")
row={a:i for i,a in enumerate(ids)}
print(f"corpus {len(ids):,} articles\n")

# C. how much category signal exists, and is it recoverable from the text?
cats=collections.Counter(arts[i].category for i in ids)
print("=== C. category ===")
print(f"  {len(cats)} categories, top: {cats.most_common(5)}")
# within-category vs across-category mean cosine
by={}
for i in ids:
    by.setdefault(arts[i].category, []).append(row[i])
rng=np.random.default_rng(0)
big=[c for c,v in by.items() if len(v)>=200][:6]
wi, ac = [], []
for c in big:
    idx=rng.choice(by[c], 200, replace=False)
    V=vecs[idx]; s=V@V.T; wi.append((s.sum()-len(idx))/(len(idx)**2-len(idx)))
for a in range(len(big)):
    for b in range(a+1,len(big)):
        A=vecs[rng.choice(by[big[a]],150,replace=False)]
        B=vecs[rng.choice(by[big[b]],150,replace=False)]
        ac.append(float((A@B.T).mean()))
print(f"  within-category  cosine {np.mean(wi):.4f}")
print(f"  across-category  cosine {np.mean(ac):.4f}")
print(f"  -> separation {np.mean(wi)-np.mean(ac):+.4f}  ({'category IS already in the embedding' if np.mean(wi)-np.mean(ac)>0.05 else 'category is NOT captured -- adding it could help'})")

# A. query diffuseness: how spread is a user's history in embedding space?
hist=list(r.histories(SPLIT_NAMES[ds]["train"]))[:3000]
spreads=[]
for h in hist:
    rr=[row[a] for a in h.clicked_ids if a in row][-20:]
    if len(rr)<3: continue
    V=vecs[rr]; m=V.mean(axis=0); m/=max(np.linalg.norm(m),1e-12)
    spreads.append(float(np.mean(V@m)))
print(f"\n=== A. query diffuseness (n={len(spreads)}) ===")
print(f"  mean coherence of a history to its own centroid: {np.mean(spreads):.4f}")
print(f"  quartiles: {np.percentile(spreads,[25,50,75]).round(4)}")

# B. near-duplicate competition INSIDE slates
imps=[]
for i in r.impressions(SPLIT_NAMES[ds]["train"]):
    imps.append(i)
    if len(imps)>=3000: break
dupe_slates=0; tot=0
for imp in imps:
    c=[row[a] for a in imp.candidates if a in row]
    if len(c)<3: continue
    tot+=1
    V=vecs[c]; s=V@V.T; np.fill_diagonal(s,0)
    if s.max()>0.95: dupe_slates+=1
print(f"\n=== B. near-dupes inside slates (n={tot}) ===")
print(f"  slates containing a pair with cosine>0.95: {dupe_slates} ({dupe_slates/tot*100:.1f}%)")
print("WHYDONE")
