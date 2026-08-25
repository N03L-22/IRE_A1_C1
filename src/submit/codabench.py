"""Generate Codabench prediction files (Q5).

Both leaderboards take the same format -- one line per impression::

    impression_id [rank1,rank2,...,rankN]

The rank list is a permutation of 1..N aligned to the candidate list **in the
order it was given**, rank 1 = most likely click. It is a mark-up of the slate
handed to you, not a re-ordering of it: emitting your own preferred order is
scored as a different answer entirely.

Scale is the design constraint (brief v2): MIND large test is 2,370,727
impressions and EB-NeRD's is 13,536,710, neither of which fits in RAM
alongside an index. So this streams -- one batch at a time, the only per-row
work being the file write.

    python -m src.submit.codabench --dataset mind --retriever bm25
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import zipfile
from datetime import datetime
from pathlib import Path

from ..data.readers import SPLIT_NAMES, get_reader
from ..data.schema import Article
from ..resources import add_arguments, from_args
from ..retrieval.bm25 import BM25Retriever

log = logging.getLogger("submit")

#: Params chosen on val in the Phase 2 sweep (F23), before test was touched.
BEST = {
    "ebnerd": dict(k1=1.6, b=1.0, last_n=15),
    "mind": dict(k1=1.6, b=0.75, last_n=5),
}


def rank_candidates(
    candidates: list[str], scores: dict[str, float]
) -> list[int]:
    """Rank a slate, returning one rank per candidate in ORIGINAL order.

    Unscored candidates tie at -inf and are broken by original position, so
    the output is deterministic. That determinism matters: F21 found that an
    arbitrary tie-break over a mostly-unscored slate silently turns a
    popularity baseline into a slate-order baseline.
    """
    order = sorted(
        range(len(candidates)),
        key=lambda i: (-scores.get(candidates[i], float("-inf")), i),
    )
    ranks = [0] * len(candidates)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def build_predictions(
    reader,
    retriever,
    articles: dict[str, Article],
    histories: dict[str, "object"],
    split: str,
    out_path: Path,
    max_k: int = 500,
    log_every: int = 100_000,
) -> dict:
    """Stream the test split, writing one prediction line per impression.

    ``max_k`` over-fetches relative to slate size so that most candidates
    receive a real score; slates average 11-40 items, so 500 is generous
    without being expensive.
    """
    written = 0
    unscored_slates = 0
    seen: set[str] = set()
    started = time.perf_counter()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for imp in reader.impressions(split):
            # Deduplicate: EB-NeRD's test set repeats ~200K impression ids,
            # and the format is one line per id.
            if imp.impression_id in seen:
                continue
            seen.add(imp.impression_id)

            hist = histories.get(imp.user_id)
            texts: list[str] = []
            if hist is not None:
                past = hist.before(imp.time)
                texts = [articles[a].retrieval_text for a in past if a in articles]

            scores: dict[str, float] = {}
            if texts:
                # Score the slate directly rather than retrieving top-K over
                # the whole corpus and discarding 99% of it: ~16x faster, and
                # the submission format only ever records the ordering.
                scores = retriever.score_subset(texts, imp.candidates)
            else:
                unscored_slates += 1

            ranks = rank_candidates(imp.candidates, scores)
            f.write(f"{imp.impression_id} [{','.join(map(str, ranks))}]\n")
            written += 1
            if written % log_every == 0:
                rate = written / (time.perf_counter() - started)
                log.info("  %s lines written (%.0f/s)", f"{written:,}", rate)

    elapsed = time.perf_counter() - started
    return {
        "lines": written,
        "cold_slates": unscored_slates,
        "seconds": round(elapsed, 1),
        "bytes": out_path.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    p.add_argument("--tier", default="large")
    p.add_argument("--work-dir", type=Path, default=Path("data/work"))
    p.add_argument("--out-dir", type=Path, default=Path("submissions"))
    p.add_argument("--max-k", type=int, default=500)
    add_arguments(p)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("bm25s").setLevel(logging.ERROR)
    budget = from_args(args)
    log.info("%s", budget)

    reader = get_reader(args.dataset, args.work_dir, args.tier)
    test_split = SPLIT_NAMES[args.dataset].get("test")
    if test_split is None:
        log.error("no test split defined for %s", args.dataset)
        return 2

    # The corpus must be the TEST split's own news.tsv, not train+dev.
    # MIND-large test ships 120,961 articles against small-train's 51,282
    # (F14), and an article absent from the index can never be scored -- it
    # would fall to the bottom of every slate it appears in. This is the same
    # recall-ceiling trap as F17/D-CORPUS, in the submission path.
    t0 = time.perf_counter()
    try:
        articles = {a.article_id: a for a in reader.articles(splits=(test_split,))}
    except TypeError:  # readers that do not take a splits argument
        articles = {a.article_id: a for a in reader.articles()}
    log.info("articles     %8d  (%.1fs)", len(articles), time.perf_counter() - t0)

    t0 = time.perf_counter()
    histories = {h.user_id: h for h in reader.histories(test_split)}
    log.info("histories    %8d  (%.1fs)", len(histories), time.perf_counter() - t0)

    params = BEST[args.dataset]
    retriever = BM25Retriever(**params)
    t0 = time.perf_counter()
    retriever.index(list(articles.values()))
    log.info("indexed %s in %.1fs", retriever.name, time.perf_counter() - t0)

    txt = args.out_dir / f"{args.dataset}_prediction.txt"
    stats = build_predictions(
        reader, retriever, articles, histories, test_split, txt, max_k=args.max_k
    )
    log.info(
        "\nwrote %s: %s lines, %.1f MB, %.1fs (%s cold slates)",
        txt, f"{stats['lines']:,}", stats["bytes"] / 1e6, stats["seconds"],
        f"{stats['cold_slates']:,}",
    )

    zip_path = args.out_dir / f"{args.dataset}_prediction.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt, arcname=txt.name)
    log.info("zipped -> %s (%.1f MB)", zip_path, zip_path.stat().st_size / 1e6)

    meta = args.out_dir / f"{args.dataset}_prediction.meta.json"
    meta.write_text(json.dumps({
        "dataset": args.dataset, "tier": args.tier,
        "retriever": retriever.name, "params": params,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "budget": budget.as_dict(), **stats,
    }, indent=2))
    log.info("metadata -> %s", meta)

    log.info(
        "\nupload %s to %s",
        zip_path.name,
        "https://www.codabench.org/competitions/13967/" if args.dataset == "mind"
        else "https://www.codabench.org/competitions/2469/",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
