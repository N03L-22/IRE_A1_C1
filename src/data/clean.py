"""Step 3 of the Q1 pipeline: both datasets -> one unified, split store.

The highest-leverage module in the project. Get the unified schema right and
every downstream component is written once for both datasets; get it wrong and
everything gets written twice.

What it does, in order:

    read (per-dataset readers)
      -> temporal split          (src/data/split.py -- the leakage boundary)
      -> truncate history        (per impression, < t where checkable)
      -> write parquet           (data/store/<dataset>/)
      -> run manifest            (counts, budget, boundaries, provenance)

Deliberately NOT done here: any generative or non-deterministic cleanup. Q1.5
requires one command to rebuild everything from raw files, and an LLM in the
data path breaks that while quietly rewriting the corpus that BM25 is scored
against. See architecture.md decision 9.

Post-click fields are excluded upstream in readers.py -- they are never read at
all, which is stronger than reading and remembering to drop them (F6).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..resources import Budget, add_arguments, from_args
from .readers import SPLIT_NAMES, get_reader
from .schema import Article, History, Impression
from .split import TRAIN, TEST, VAL, temporal_split, truncate_history

log = logging.getLogger(__name__)

DATASETS = ("mind", "ebnerd")

ARTICLE_SCHEMA = pa.schema(
    [
        ("article_id", pa.string()),
        ("title", pa.string()),
        ("abstract", pa.string()),
        ("body", pa.string()),
        ("category", pa.string()),
        ("subcategory", pa.string()),
        ("entities", pa.list_(pa.string())),
        ("published_time", pa.timestamp("us")),
        ("retrieval_text", pa.string()),
    ]
)

IMPRESSION_SCHEMA = pa.schema(
    [
        ("impression_id", pa.string()),
        ("user_id", pa.string()),
        ("time", pa.timestamp("us")),
        ("candidates", pa.list_(pa.string())),
        ("clicked", pa.list_(pa.string())),
        ("session_id", pa.string()),
        ("split", pa.string()),
        # History truncated to < time. The leakage boundary, materialised.
        ("history", pa.list_(pa.string())),
        # False when the boundary could not be checked (MIND, F1).
        ("history_verifiable", pa.bool_()),
    ]
)


def _write_articles(articles: list[Article], out: Path) -> int:
    table = pa.Table.from_pydict(
        {
            "article_id": [a.article_id for a in articles],
            "title": [a.title for a in articles],
            "abstract": [a.abstract for a in articles],
            "body": [a.body for a in articles],
            "category": [a.category for a in articles],
            "subcategory": [a.subcategory for a in articles],
            "entities": [a.entities for a in articles],
            "published_time": [a.published_time for a in articles],
            "retrieval_text": [a.retrieval_text for a in articles],
        },
        schema=ARTICLE_SCHEMA,
    )
    pq.write_table(table, out, compression="zstd")
    return table.num_rows


def _write_impressions(
    rows: list[tuple[Impression, list[str]]], split: str, verifiable: bool, out: Path
) -> int:
    table = pa.Table.from_pydict(
        {
            "impression_id": [i.impression_id for i, _ in rows],
            "user_id": [i.user_id for i, _ in rows],
            "time": [i.time for i, _ in rows],
            "candidates": [i.candidates for i, _ in rows],
            "clicked": [i.clicked for i, _ in rows],
            "session_id": [i.session_id for i, _ in rows],
            "split": [split] * len(rows),
            "history": [h for _, h in rows],
            "history_verifiable": [verifiable] * len(rows),
        },
        schema=IMPRESSION_SCHEMA,
    )
    pq.write_table(table, out, compression="zstd")
    return table.num_rows


def build(
    dataset: str,
    work_dir: Path,
    store_dir: Path,
    *,
    tier: str = "small",
    val_fraction: float = 0.1,
    budget: Budget | None = None,
    limit: int | None = None,
) -> dict:
    """Build the unified store for one dataset. Returns its manifest."""
    started = time.perf_counter()
    reader = get_reader(dataset, work_dir, tier)
    names = SPLIT_NAMES[dataset]
    out_dir = store_dir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- articles -------------------------------------------------------
    # MIND: the union of train and dev news.tsv, de-duplicated (D3/F2). Taking
    # the union rather than per-split corpora avoids a recall ceiling that has
    # nothing to do with the retriever -- exactly the artefact F17 caught.
    t0 = time.perf_counter()
    articles = list(reader.articles())
    n_articles = _write_articles(articles, out_dir / "articles.parquet")
    n_with_time = sum(1 for a in articles if a.published_time)
    log.info(
        "  articles      %7d  (%.1fs)  with publish_time: %d (%.0f%%)",
        n_articles,
        time.perf_counter() - t0,
        n_with_time,
        100 * n_with_time / max(1, n_articles),
    )

    # --- impressions + split -------------------------------------------
    t0 = time.perf_counter()
    train_period = list(reader.impressions(names["train"]))
    heldout_period = list(reader.impressions(names["heldout"]))
    if limit:
        train_period = train_period[:limit]
        heldout_period = heldout_period[: max(1, limit // 4)]

    splits, split_report = temporal_split(
        train_period, heldout_period, dataset=dataset, val_fraction=val_fraction
    )
    log.info("  %s", str(split_report).replace("\n", "\n  "))

    # --- history truncation --------------------------------------------
    # Histories are keyed per split by the dataset. MIND emits one record per
    # impression (history is inline), so later records overwrite earlier ones
    # for the same user -- acceptable because MIND history is untimestamped and
    # cannot be filtered anyway.
    manifest_splits = {}
    verifiable_overall = False

    for split_name, impressions in splits.items():
        if not impressions:
            manifest_splits[split_name] = {"impressions": 0}
            continue
        source = names["heldout"] if split_name == TEST else names["train"]
        histories = {h.user_id: h for h in reader.histories(source)}
        pairs_iter, trunc = truncate_history(impressions, histories, dataset=dataset)
        rows = list(pairs_iter)
        verifiable_overall = verifiable_overall or trunc.verifiable

        n = _write_impressions(
            rows, split_name, trunc.verifiable, out_dir / f"impressions_{split_name}.parquet"
        )
        log.info("  %s", trunc)
        manifest_splits[split_name] = {
            "impressions": n,
            "users": len(histories),
            "verifiable": trunc.verifiable,
            "clicks_before": trunc.clicks_before,
            "clicks_after": trunc.clicks_after,
            "dropped_fraction": round(trunc.dropped_fraction, 6),
            "span": [str(x) for x in (split_report.boundaries.get(split_name) or ())],
        }

    manifest = {
        "dataset": dataset,
        "tier": tier,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "articles": n_articles,
        "articles_with_publish_time": n_with_time,
        "val_fraction_requested": val_fraction,
        "splits": manifest_splits,
        "proportions": {k: round(v, 4) for k, v in split_report.proportions.items()},
        "history_boundary_verifiable": verifiable_overall,
        "budget": budget.as_dict() if budget else None,
        "seconds": round(time.perf_counter() - started, 2),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", action="append", choices=DATASETS, help="repeatable; default both")
    p.add_argument("--tier", default="small")
    p.add_argument("--work-dir", type=Path, default=Path("data/work"))
    p.add_argument("--store-dir", type=Path, default=Path("data/store"))
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--limit", type=int, help="cap impressions (development only)")
    add_arguments(p)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    budget = from_args(args)
    log.info("%s\n", budget)

    for dataset in args.dataset or list(DATASETS):
        log.info("=== %s (%s) ===", dataset, args.tier)
        m = build(
            dataset,
            args.work_dir,
            args.store_dir,
            tier=args.tier,
            val_fraction=args.val_fraction,
            budget=budget,
            limit=args.limit,
        )
        log.info("  -> %s/%s  (%.1fs)\n", args.store_dir, dataset, m["seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
