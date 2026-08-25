"""Run BM25 against the baselines, or sweep its parameters (Q2).

Two modes:

    python -m src.retrieval.run_bm25 --dataset ebnerd
        BM25 (full corpus and windowed) against recency and popularity.

    python -m src.retrieval.run_bm25 --dataset ebnerd --sweep
        k1 x b x last_n x window, in parallel across --n-jobs processes.

The sweep runs on **val only**. Choosing parameters on val and reporting on
test is what keeps the test split honest; the chosen values are printed so
they can be recorded before test is touched (plan/2-Lexical-BM25.md D5).

These are exploratory numbers -- no confidence intervals, so nothing here is
reportable. The Phase 4 harness attaches CIs and owns the deliverable numbers.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

from ..data.readers import SPLIT_NAMES, get_reader
from ..data.schema import Article, History, Impression
from ..resources import add_arguments, from_args
from ..skeleton import (
    PopularityRetriever,
    RecencyRetriever,
    WindowedRetriever,
    evaluate,
    temporal_split,
)
from .bm25 import BM25Retriever

log = logging.getLogger("run_bm25")

#: The K values Q2.4 names. Not configurable -- the brief fixes them.
KS = (50, 100, 200)


def load(
    dataset: str, tier: str, work_dir: Path, limit: int
) -> tuple[dict[str, Article], list[Impression], list[Impression], dict[str, History]]:
    """Read the working tier and carve a val split from the tail of train."""
    reader = get_reader(dataset, work_dir, tier)
    split = SPLIT_NAMES[dataset]["train"]

    t0 = time.perf_counter()
    articles = {a.article_id: a for a in reader.articles()}
    log.info("articles     %6d  (%.1fs)", len(articles), time.perf_counter() - t0)

    impressions: list[Impression] = []
    for imp in reader.impressions(split):
        impressions.append(imp)
        if len(impressions) >= limit * 2:
            break

    histories = {h.user_id: h for h in reader.histories(split)}
    train, val = temporal_split(impressions)
    log.info(
        "impressions  %6d  train=%d val=%d  histories=%d",
        len(impressions),
        len(train),
        len(val),
        len(histories),
    )
    return articles, train, val[:limit], histories


def _sweep_one(
    params: tuple[float, float, int, float | None],
) -> dict[str, object]:
    """One sweep cell. Module-level so ProcessPoolExecutor can pickle it.

    Re-reads the data in each worker rather than inheriting it: the corpus is
    ~20-65K articles, cheap to reload, and passing it through pickle for every
    cell would cost more than the reload does.
    """
    k1, b, last_n, window = params
    articles, train, val, histories = _WORKER_DATA
    base = BM25Retriever(k1=k1, b=b, last_n=last_n)
    retriever = base if window is None else WindowedRetriever(base, window_hours=window)
    retriever.index(list(articles.values()))
    results = evaluate(retriever, val, histories, articles, ks=KS)
    return {
        "k1": k1,
        "b": b,
        "last_n": last_n,
        "window_hours": window,
        **{f"recall@{r.k}": round(r.recall, 4) for r in results},
        "n": results[0].n_impressions if results else 0,
        "seconds": round(results[0].seconds, 1) if results else 0.0,
    }


#: Populated once per worker by the initialiser below.
_WORKER_DATA: tuple = ()


def _init_worker(dataset: str, tier: str, work_dir: str, limit: int) -> None:
    global _WORKER_DATA
    logging.getLogger("bm25s").setLevel(logging.ERROR)
    _WORKER_DATA = load(dataset, tier, Path(work_dir), limit)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", choices=["mind", "ebnerd"], default="ebnerd")
    p.add_argument("--tier", default="small")
    p.add_argument("--work-dir", type=Path, default=Path("data/work"))
    p.add_argument("--limit", type=int, default=1500, help="impressions to evaluate")
    p.add_argument("--last-n", type=int, default=15, help="click titles per query (D4)")
    p.add_argument(
        "--window-hours",
        type=float,
        default=24.0,
        help="recency window for the windowed row; EB-NeRD only (F16/F20)",
    )
    p.add_argument("--sweep", action="store_true", help="run the k1/b/last_n/window sweep")
    p.add_argument("--out", type=Path, help="write results as JSON")
    add_arguments(p)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("bm25s").setLevel(logging.ERROR)
    budget = from_args(args)
    log.info("%s", budget)

    # MIND has no publish times (F20), so the recency filter cannot be built
    # there. Refusing loudly beats silently reporting an unwindowed row as a
    # windowed one.
    windowed_available = args.dataset == "ebnerd"
    if not windowed_available:
        log.info("note: %s has no publish times -- recency window unavailable (F20)", args.dataset)

    if args.sweep:
        grid = list(
            itertools.product(
                (0.9, 1.2, 1.6),
                (0.3, 0.75, 1.0),
                (5, 15, 50),
                (24.0, None) if windowed_available else (None,),
            )
        )
        log.info("sweeping %d cells across %d workers", len(grid), budget.n_jobs)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(
            max_workers=budget.n_jobs,
            initializer=_init_worker,
            initargs=(args.dataset, args.tier, str(args.work_dir), args.limit),
        ) as pool:
            rows = list(pool.map(_sweep_one, grid))
        log.info("sweep finished in %.1fs\n", time.perf_counter() - t0)

        rows.sort(key=lambda r: -float(r["recall@100"]))
        log.info("%-5s %-5s %-6s %-8s %-9s %-9s %-9s", "k1", "b", "last_n", "window", *[f"r@{k}" for k in KS])
        for r in rows:
            log.info(
                "%-5g %-5g %-6d %-8s %-9.4f %-9.4f %-9.4f",
                r["k1"], r["b"], r["last_n"],
                f"{r['window_hours']:g}h" if r["window_hours"] else "none",
                r["recall@50"], r["recall@100"], r["recall@200"],
            )
        best = rows[0]
        log.info(
            "\nbest by recall@100 (val only): k1=%g b=%g last_n=%d window=%s",
            best["k1"], best["b"], best["last_n"],
            f"{best['window_hours']:g}h" if best["window_hours"] else "none",
        )
        payload: object = rows
    else:
        articles, train, val, histories = load(
            args.dataset, args.tier, args.work_dir, args.limit
        )
        alist = list(articles.values())

        retrievers: list = [RecencyRetriever(), PopularityRetriever(), BM25Retriever(last_n=args.last_n)]
        if windowed_available:
            retrievers.append(
                WindowedRetriever(BM25Retriever(last_n=args.last_n), window_hours=args.window_hours)
            )

        rows = []
        for r in retrievers:
            t0 = time.perf_counter()
            if isinstance(r, PopularityRetriever):
                r.index_from_clicks(train)  # train only -- never val
            else:
                r.index(alist)
            log.info("%-34s indexed %6.1fs", r.name, time.perf_counter() - t0)
            for res in evaluate(r, val, histories, articles, ks=KS):
                rows.append(
                    {"retriever": res.retriever, "k": res.k, "recall": round(res.recall, 4),
                     "n": res.n_impressions}
                )

        log.info("\n=== recall@K  %s %s (exploratory -- NO CIs, not reportable) ===",
                 args.dataset, args.tier)
        for row in rows:
            log.info("  %-34s recall@%-4d %.4f   n=%d",
                     row["retriever"], row["k"], row["recall"], row["n"])
        payload = rows

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"dataset": args.dataset, "tier": args.tier, "limit": args.limit,
             "budget": budget.as_dict(), "generated_at": datetime.now().isoformat(timespec="seconds"),
             "results": payload}, indent=2))
        log.info("\nwrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
