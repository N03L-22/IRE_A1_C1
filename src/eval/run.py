"""Score every retriever through the one harness (Q4.5).

    python -m src.eval.run --dataset ebnerd
    python -m src.eval.run --dataset mind --limit 4000 --out results/eval_mind.json

Retrievers are evaluated in parallel across processes -- each is independent,
and the retrieval pass dominates the runtime, so this is close to linear in
--n-jobs.

Everything reported here carries a bootstrap CI and an n. Numbers without both
are not reportable under the IRE conventions, which is why the harness has no
mode that omits them.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

from ..data.readers import SPLIT_NAMES, get_reader
from ..resources import add_arguments, from_args
from ..retrieval.bm25 import BM25Retriever
from ..skeleton import (
    PopularityRetriever,
    RecencyRetriever,
    WindowedRetriever,
    temporal_split,
)
from .harness import ResultRow, evaluate, format_table, to_dicts

log = logging.getLogger("eval")


class RandomRetriever:
    """The chance floor. Makes recall@200 interpretable (Phase 2 D6).

    Seeded per instance so the number is reproducible; without it the floor
    would move between runs and could not be quoted.
    """

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._ids: list[str] = []

    def index(self, articles) -> None:
        self._ids = [a.article_id for a in articles]

    def retrieve(self, history_text, k, at_time=None):
        import random

        rng = random.Random(self.seed)
        picks = rng.sample(self._ids, min(k, len(self._ids)))
        return [(aid, 1.0 / (i + 1)) for i, aid in enumerate(picks)]


def _load(dataset: str, tier: str, work_dir: Path, limit: int):
    reader = get_reader(dataset, work_dir, tier)
    split = SPLIT_NAMES[dataset]["train"]
    articles = {a.article_id: a for a in reader.articles()}
    impressions = []
    for imp in reader.impressions(split):
        impressions.append(imp)
        if len(impressions) >= limit * 2:
            break
    histories = {h.user_id: h for h in reader.histories(split)}
    train, val = temporal_split(impressions)
    return articles, train, val[:limit], histories


def _train_popularity(train) -> dict[str, float]:
    """Click distribution over the train split ONLY.

    Novelty and the head/tail slice both depend on this, and both would be
    leakage if it were computed over the evaluated period (D3).
    """
    counts: Counter[str] = Counter()
    for imp in train:
        counts.update(imp.clicked)
    total = max(1, sum(counts.values()))
    return {aid: c / total for aid, c in counts.items()}


#: Populated once per worker.
_DATA: tuple = ()


def _init(dataset: str, tier: str, work_dir: str, limit: int) -> None:
    global _DATA
    logging.getLogger("bm25s").setLevel(logging.ERROR)
    _DATA = _load(dataset, tier, Path(work_dir), limit)


def _score_one(spec: tuple[str, dict]) -> list[dict]:
    """Build one retriever from a spec and score it. Picklable by name."""
    kind, kwargs = spec
    articles, train, val, histories = _DATA
    dataset = kwargs.pop("_dataset")

    if kind == "recency":
        r = RecencyRetriever()
    elif kind == "popularity":
        r = PopularityRetriever()
    elif kind == "random":
        r = RandomRetriever()
    elif kind == "bm25":
        r = BM25Retriever(**kwargs)
    elif kind == "bm25+window":
        window = kwargs.pop("window_hours")
        r = WindowedRetriever(BM25Retriever(**kwargs), window_hours=window)
    else:
        raise ValueError(f"unknown retriever kind: {kind}")

    if isinstance(r, PopularityRetriever):
        r.index_from_clicks(train)  # train only -- never the evaluated split
    else:
        r.index(list(articles.values()))

    rows = evaluate(r, val, histories, articles, dataset,
                    train_popularity=_train_popularity(train))
    return to_dicts(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", choices=["mind", "ebnerd"], default="ebnerd")
    p.add_argument("--tier", default="small")
    p.add_argument("--work-dir", type=Path, default=Path("data/work"))
    p.add_argument("--limit", type=int, default=2000, help="impressions to evaluate")
    p.add_argument("--window-hours", type=float, default=24.0)
    p.add_argument("--out", type=Path)
    add_arguments(p)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("bm25s").setLevel(logging.ERROR)
    budget = from_args(args)
    log.info("%s", budget)

    # Parameters chosen on val in the Phase 2 sweep (F23), recorded before
    # test was touched.
    best = {"ebnerd": dict(k1=1.6, b=1.0, last_n=15),
            "mind": dict(k1=1.6, b=0.75, last_n=5)}[args.dataset]

    specs: list[tuple[str, dict]] = [
        ("random", {"_dataset": args.dataset}),
        ("popularity", {"_dataset": args.dataset}),
        ("bm25", {"_dataset": args.dataset, **best}),
    ]
    # Recency and the windowed retriever need publish times, which MIND does
    # not have (F20). Omitting them loudly beats reporting an unwindowed row
    # under a windowed label.
    if args.dataset == "ebnerd":
        specs.insert(1, ("recency", {"_dataset": args.dataset}))
        specs.append(("bm25+window",
                      {"_dataset": args.dataset, **best, "window_hours": args.window_hours}))
    else:
        log.info("note: mind has no publish times -- recency retrievers unavailable (F20)")

    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=min(budget.n_jobs, len(specs)),
        initializer=_init,
        initargs=(args.dataset, args.tier, str(args.work_dir), args.limit),
    ) as pool:
        chunks = list(pool.map(_score_one, specs))
    elapsed = time.perf_counter() - t0

    dicts = [d for chunk in chunks for d in chunk]
    rows = [ResultRow(**d) for d in dicts]
    log.info("\nscored %d retrievers -> %d rows in %.1fs\n", len(specs), len(rows), elapsed)

    log.info("=== %s %s -- accuracy (slate regime) ===", args.dataset, args.tier)
    log.info("%s", format_table(rows, ["auc", "mrr", "ndcg@5", "ndcg@10"]))
    log.info("\n=== %s %s -- candidate generation (corpus regime) ===", args.dataset, args.tier)
    log.info("%s", format_table(rows, ["recall@50", "recall@100", "recall@200"]))
    log.info("\n=== %s %s -- beyond accuracy ===", args.dataset, args.tier)
    log.info("%s", format_table(rows, ["diversity", "novelty", "coverage"]))

    log.info("\n=== slices ===")
    for r in rows:
        if r.slice != "all" and r.metric in ("recall@100", "ndcg@10"):
            log.info("%s", r)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "dataset": args.dataset, "tier": args.tier, "limit": args.limit,
            "budget": budget.as_dict(), "bm25_params": best,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "seconds": round(elapsed, 1), "rows": dicts,
        }, indent=2))
        log.info("\nwrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
