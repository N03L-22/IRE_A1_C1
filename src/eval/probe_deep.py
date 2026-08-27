"""Why does XLM-R collapse? Isolate the cause instead of assuming it.

Four hypotheses for the +0.0018 margin, tested separately:
  H1 anisotropy   -- the cone is real; centering/whitening should rescue it
  H2 pooling      -- mean-pool over subwords is wrong for this model
  H3 dimension    -- truncating helps or hurts (the user's question)
  H4 measurement  -- cosine on a collapsed space; rank correlation may survive
"""
import logging, numpy as np
logging.basicConfig(level=logging.ERROR)
for n in ("transformers","src.retrieval.encode"): logging.getLogger(n).setLevel(logging.ERROR)
from src.retrieval.encode import encode_texts, l2_normalise

related = [
    ("Brøndby vandt kampen mod FCK", "FC København tabte til Brøndby i går"),
    ("Regeringen hæver skatten", "Ny skattestigning vedtaget i Folketinget"),
    ("Storm rammer Jylland i nat", "Kraftig blæst ventes over Vestjylland"),
    ("Prisen på benzin stiger igen", "Benzinpriserne er steget markant"),
    ("Ny corona-variant opdaget", "Sundhedsstyrelsen advarer om ny variant"),
]
unrelated = [
    ("Brøndby vandt kampen mod FCK", "Ny opskrift på æblekage med kanel"),
    ("Regeringen hæver skatten", "Håndboldlandsholdet træner i Herning"),
    ("Storm rammer Jylland i nat", "Aktiekursen på Novo Nordisk falder"),
    ("Prisen på benzin stiger igen", "Kongehuset offentliggør nye billeder"),
    ("Ny corona-variant opdaget", "Fodboldklubben skifter cheftræner"),
]
texts = [t for p in related + unrelated for t in p]

def margin(v):
    sims = [float(v[2*i] @ v[2*i+1]) for i in range(len(v)//2)]
    rel, unrel = sims[:5], sims[5:]
    r, u = np.mean(rel), np.mean(unrel)
    return r, u, r-u, min(rel) > max(unrel)

def show(tag, v):
    r,u,m,sep = margin(v)
    print(f"  {tag:34} rel {r:.4f}  unrel {u:.4f}  margin {m:+.4f}  {'SEPARATES' if sep else 'overlaps'}")

for key in ("xlmr-base", "minilm"):
    vecs, st = encode_texts(texts, model_key=key, batch_size=16)
    print(f"\n=== {st.model} ({st.dim}-d) ===")
    show("baseline (mean-pool + L2)", vecs)

    # H1: anisotropy -- remove the common direction, then re-normalise
    centered = l2_normalise(vecs - vecs.mean(axis=0, keepdims=True))
    show("H1 centered (remove mean dir)", centered)

    # H1b: whitening -- decorrelate and equalise variance
    X = vecs - vecs.mean(axis=0, keepdims=True)
    cov = np.cov(X.T) + 1e-6*np.eye(X.shape[1])
    w, V = np.linalg.eigh(cov)
    W = V @ np.diag(1.0/np.sqrt(np.maximum(w,1e-12))) @ V.T
    show("H1b whitened", l2_normalise(X @ W))

    # H3: truncation, on raw and centered
    for d in (512, 256, 128, 64):
        if d >= st.dim: continue
        show(f"H3 truncated {d}d", l2_normalise(vecs[:, :d].copy()))
        show(f"H3 truncated {d}d + centered", l2_normalise(centered[:, :d].copy()))

    # H4: does the RANKING survive even if absolute cosines collapse?
    sims = [float(vecs[2*i] @ vecs[2*i+1]) for i in range(len(vecs)//2)]
    order = np.argsort(sims)[::-1]
    n_rel_in_top5 = sum(1 for i in order[:5] if i < 5)
    print(f"  H4 ranking: {n_rel_in_top5}/5 related pairs in the top 5 by cosine")
print("\nPROBEDONE")
