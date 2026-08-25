"""The remaining named ablations, in one pass (O5, O2, U7, U4).

Each closes a question the plan docs promise an answer to:

    O5  query-term dedup on/off        -- 2-Lexical-BM25.md D4
    O2  HNSW vs exact on REAL vectors  -- F31 used random vectors, which are
                                          near-orthogonal and the worst case
                                          for a proximity graph
    U7  provided EB-NeRD embeddings    -- a correctness check as much as a
                                          baseline: they are click-trained and
                                          SHOULD beat generic MiniLM. If ours
                                          wins, suspect a bug in our pipeline
    U4  TF-IDF                         -- named in the brief's lexical axis,
                                          isolates what BM25's knobs buy

Run:  python -m src.eval.ablations
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from ..data.readers import SPLIT_NAMES, get_reader
from ..eval.harness import evaluate
from ..eval.run import _train_popularity
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.semantic import SemanticRetriever
from ..skeleton import temporal_split

log = logging.getLogger("ablations")

ARTIFACTS = Path("data/work/ebnerd/artifacts")


class TfidfRetriever:
    """TF-IDF + cosine, the baseline the brief names alongside BM25 (U4).

    Isolates what BM25's two knobs actually buy: TF-IDF has neither term
    saturation (``k1``) nor length normalisation (``b``), so the gap between
    them is exactly the value of those two mechanisms on this data.
    """

    name = "tfidf"

    def __init__(self, last_n: int = 15) -> None:
        self.last_n = last_n
        self._ids: list[str] = []
        self._matrix = None
        self._vec = None

    def index(self, articles) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        from ..retrieval.tokenise import tokenise

        self._ids = [a.article_id for a in articles]
        # Same tokeniser as BM25, so the comparison isolates the *weighting*
        # scheme rather than accidentally measuring two different tokenisers.
        self._vec = TfidfVectorizer(analyzer=tokenise)
        self._matrix = self._vec.fit_transform(a.retrieval_text for a in articles)

    def retrieve(self, history_text, k, at_time=None):
        from ..retrieval.tokenise import build_query

        terms = build_query(history_text, last_n=self.last_n, dedup=True)
        if not terms:
            return []
        q = self._vec.transform([" ".join(terms)])
        scores = (self._matrix @ q.T).toarray().ravel()
        k = min(k, len(self._ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._ids[int(i)], float(scores[int(i)])) for i in top if scores[int(i)] > 0]


def load_provided(kind: str, article_ids: list[str]):
    """EB-NeRD's provided article vectors, joined onto our corpus (U7).

    F13 verified these cover all 125,541 large-tier articles with zero missing
    ids, so the small tier is a strict subset and the join is exact.
    """
    import pyarrow.parquet as pq

    path = {
        "word2vec": ARTIFACTS / "word2vec" / "document_vector.parquet",
        "bert": ARTIFACTS / "bert_multilingual" / "bert_base_multilingual_cased.parquet",
    }[kind]
    tbl = pq.read_table(path).to_pydict()
    cols = list(tbl)
    id_col = cols[0]
    vec_col = cols[1]
    lookup = {str(a): v for a, v in zip(tbl[id_col], tbl[vec_col])}

    wanted, vecs = [], []
    for aid in article_ids:
        v = lookup.get(str(aid))
        if v is not None:
            wanted.append(aid)
            vecs.append(np.asarray(v, dtype=np.float32))
    if not vecs:
        raise RuntimeError(f"no {kind} vectors matched the corpus")
    log.info("  %s: matched %d/%d articles, dim %d",
             kind, len(wanted), len(article_ids), len(vecs[0]))
    return np.vstack(vecs), wanted


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("bm25s", "transformers", "src.retrieval.encode",
                  "src.retrieval.semantic", "harness"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    reader = get_reader("ebnerd", Path("data/work"), "small")
    articles = {a.article_id: a for a in reader.articles()}
    imps = []
    for i in reader.impressions(SPLIT_NAMES["ebnerd"]["train"]):
        imps.append(i)
        if len(imps) >= 8000:
            break
    hist = {h.user_id: h for h in reader.histories(SPLIT_NAMES["ebnerd"]["train"])}
    train, val = temporal_split(imps)
    val = val[:4000]
    pop = _train_popularity(train)
    alist = list(articles.values())
    ids = [a.article_id for a in alist]
    log.info("val=%d corpus=%d\n", len(val), len(alist))

    out = []

    def score(name, retriever, note=""):
        t0 = time.perf_counter()
        retriever.index(alist)
        idx_s = time.perf_counter() - t0
        rows = evaluate(retriever, val, hist, articles, "ebnerd",
                        train_popularity=pop, with_slices=False)
        rec = {r.metric: (r.value, r.ci_low, r.ci_high) for r in rows if r.slice == "all"}
        out.append({"ablation": name, "retriever": retriever.name, "note": note,
                    "index_seconds": round(idx_s, 1),
                    **{m: rec[m][0] for m in rec}})
        log.info("  %-34s r@50=%.4f [%.4f,%.4f]  r@200=%.4f  ndcg@10=%.4f  (%.0fs)",
                 name, *rec["recall@50"], rec["recall@200"][0], rec["ndcg@10"][0], idx_s)

    log.info("=== O5: query-term dedup ===")
    score("dedup=True (default)", BM25Retriever(k1=1.6, b=1.0, last_n=15, dedup=True))
    score("dedup=False", BM25Retriever(k1=1.6, b=1.0, last_n=15, dedup=False))

    log.info("\n=== U4: TF-IDF vs BM25 ===")
    score("tfidf", TfidfRetriever(last_n=15))

    log.info("\n=== U7: provided vectors vs our MiniLM ===")
    score("ours: minilm 384d", SemanticRetriever(model_key="minilm", last_n=20))
    for kind in ("bert", "word2vec"):
        try:
            vecs, vids = load_provided(kind, ids)
            r = SemanticRetriever(vectors=vecs, vector_ids=vids, last_n=20)
            r.name = f"provided-{kind}({vecs.shape[1]}d)"
            score(f"provided: {kind} {vecs.shape[1]}d", r,
                  "click-trained; SHOULD beat generic MiniLM")
        except Exception as e:  # noqa: BLE001
            log.warning("  provided %s failed: %s", kind, str(e)[:120])

    log.info("\n=== O2: HNSW vs exact, REAL vectors ===")
    exact = SemanticRetriever(model_key="minilm", last_n=20, index_kind="brute")
    score("exact (brute force)", exact)
    for ef in (64, 128, 256):
        h = SemanticRetriever(model_key="minilm", last_n=20,
                              index_kind="hnsw", ef_search=ef)
        score(f"hnsw efSearch={ef}", h, "F31 used RANDOM vectors; these are real")

    Path("results").mkdir(exist_ok=True)
    Path("results/ablations_ebnerd.json").write_text(json.dumps(out, indent=2))
    log.info("\nwrote results/ablations_ebnerd.json")
    log.info("ABDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
