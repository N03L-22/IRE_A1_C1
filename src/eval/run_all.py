"""Score every retriever through the one harness -- the Q3.5 comparison.

    python -m src.eval.run_all --dataset ebnerd --limit 20000

This is what Q3.5 asks for: lexical and semantic, on the same impressions,
through the same harness, with confidence intervals, sliced. Plus the
baselines that make those numbers mean anything -- random for the floor,
popularity because it beat BM25 outright on MIND (F25), and recency because
on EB-NeRD it beats everything (F16).

> [!important] Sample size, after F34
> The MIND leaderboard scored AUC 0.5568 where our offline harness said
> 0.4981 [0.4776, 0.5190] -- an interval that excluded the true value, from
> n = 800. The offline sample was simply too small. **The default here is
> 20,000 impressions, not 800**, and any comparison used to choose a retriever
> should use at least that.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from ..data.readers import SPLIT_NAMES, get_reader
from ..resources import add_arguments, from_args
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.fusion import PopularityPrior, RRFusion
from ..retrieval.semantic import HistoryIdRetriever
from ..skeleton import (
    PopularityRetriever,
    RecencyRetriever,
    WindowedRetriever,
    temporal_split,
)
from .harness import ResultRow, evaluate, format_table, to_dicts
from .run import RandomRetriever, _train_popularity

log = logging.getLogger("eval_all")

#: Chosen on val in the Phase 2 sweep (F23), before test was touched.
BM25_PARAMS = {
    "ebnerd": dict(k1=1.6, b=1.0, last_n=15),
    "mind": dict(k1=1.6, b=0.75, last_n=5),
}


def build_retrievers(dataset: str, train, articles, window_hours: float, encoder: str):
    """Every retriever the comparison needs, in one place.

    Recency-based rows are EB-NeRD-only: MIND has no publish time (F20), so a
    recency filter cannot be built there at all. Omitting them loudly beats
    reporting an unwindowed row under a windowed label.
    """
    popularity = _train_popularity(train)
    bm25_params = BM25_PARAMS[dataset]

    lexical = BM25Retriever(**bm25_params)
    semantic = HistoryIdRetriever(model_key=encoder, last_n=20)

    retrievers = [
        RandomRetriever(),
        PopularityRetriever(),
        lexical,
        semantic,
        # Q3.5's natural follow-up: they fail differently, so fuse them.
        RRFusion([BM25Retriever(**bm25_params), HistoryIdRetriever(model_key=encoder, last_n=20)],
                 name="rrf(bm25+semantic)"),
        # Popularity beat BM25 outright on MIND (F25); blending it in is the
        # cheapest measured improvement available.
        PopularityPrior(BM25Retriever(**bm25_params), popularity, alpha=0.3),
    ]
    if dataset == "ebnerd":
        retrievers.insert(2, RecencyRetriever())
        retrievers.append(
            WindowedRetriever(BM25Retriever(**bm25_params), window_hours=window_hours)
        )
        retrievers.append(
            WindowedRetriever(HistoryIdRetriever(model_key=encoder, last_n=20),
                              window_hours=window_hours)
        )
    return retrievers, popularity


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", choices=["mind", "ebnerd"], default="ebnerd")
    p.add_argument("--tier", default="small")
    p.add_argument("--work-dir", type=Path, default=Path("data/work"))
    p.add_argument("--limit", type=int, default=20_000,
                   help="impressions to evaluate (default 20000; see F34)")
    p.add_argument("--window-hours", type=float, default=24.0)
    p.add_argument("--encoder", default="minilm", choices=["minilm", "xlmr-base", "xlmr-large"])
    p.add_argument("--out", type=Path)
    add_arguments(p)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("bm25s").setLevel(logging.ERROR)
    budget = from_args(args)
    log.info("%s", budget)

    reader = get_reader(args.dataset, args.work_dir, args.tier)
    split = SPLIT_NAMES[args.dataset]["train"]

    t0 = time.perf_counter()
    articles = {a.article_id: a for a in reader.articles()}
    log.info("articles     %7d  (%.1fs)", len(articles), time.perf_counter() - t0)

    impressions = []
    for imp in reader.impressions(split):
        impressions.append(imp)
        if len(impressions) >= args.limit * 2:
            break
    histories = {h.user_id: h for h in reader.histories(split)}
    train, val = temporal_split(impressions)
    val = val[: args.limit]
    log.info("evaluating   %7d impressions (train %d)", len(val), len(train))

    retrievers, popularity = build_retrievers(
        args.dataset, train, articles, args.window_hours, args.encoder
    )
    article_list = list(articles.values())

    rows: list[ResultRow] = []
    for r in retrievers:
        t0 = time.perf_counter()
        if isinstance(r, PopularityRetriever):
            r.index_from_clicks(train)  # train only -- never the evaluated split
        else:
            r.index(article_list)
        log.info("%-42s indexed %6.1fs", r.name, time.perf_counter() - t0)
        rows.extend(
            evaluate(r, val, histories, articles, args.dataset, train_popularity=popularity)
        )
        # The conditional pooling only earns its complexity if it branches.
        counts = getattr(r, "strategy_counts", None)
        if counts:
            log.info("   pooling strategies: %s", dict(counts))

    log.info("\n=== %s %s -- candidate generation (corpus regime) ===", args.dataset, args.tier)
    log.info("%s", format_table(rows, ["recall@50", "recall@100", "recall@200"]))
    log.info("\n=== %s %s -- ranking (slate regime) ===", args.dataset, args.tier)
    log.info("%s", format_table(rows, ["auc", "mrr", "ndcg@5", "ndcg@10"]))
    log.info("\n=== %s %s -- beyond accuracy ===", args.dataset, args.tier)
    log.info("%s", format_table(rows, ["diversity", "novelty", "coverage"]))

    log.info("\n=== slices (recall@100, ndcg@10) ===")
    for r in rows:
        if r.slice != "all" and r.metric in ("recall@100", "ndcg@10"):
            log.info("%s", r)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "dataset": args.dataset, "tier": args.tier, "limit": args.limit,
            "encoder": args.encoder, "bm25_params": BM25_PARAMS[args.dataset],
            "budget": budget.as_dict(),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "rows": to_dicts(rows),
        }, indent=2))
        log.info("\nwrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
