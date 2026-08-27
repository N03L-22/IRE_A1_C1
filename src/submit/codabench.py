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
import gc
import json
import logging
import os
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import psutil
import pyarrow.parquet as pq

from ..data.readers import SPLIT_NAMES, get_reader
from ..data.schema import Article
from ..resources import add_arguments, from_args
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.fusion import RRFusion
from ..retrieval.semantic import HistoryIdRetriever

log = logging.getLogger("submit")

#: The filename each scorer opens inside the uploaded archive. **The two
#: competitions differ by one letter**, and getting it wrong costs a
#: submission from the daily quota:
#:
#:   MIND     evaluate.py does open(os.path.join(submit_dir, "prediction.txt"))
#:            -> FileNotFoundError on anything else. Confirmed by a rejected
#:            upload (F35).
#:   EB-NeRD  ebrec.utils._python.write_submission_file defaults to
#:            Path("predictions.txt") and zips with arcname=path.name.
#:
#: Verified against both upstream sources on 2026-08-26.
SUBMISSION_MEMBER = {
    "mind": "prediction.txt",
    "ebnerd": "predictions.txt",
}

#: Measured peak RSS of one worker on EB-NeRD's test split: index + articles
#: (1.8 GB) plus CompactHistories over 807,677 users (~7.4 GB). Was 15.1 GB
#: before the int-array change. Used to clamp --n-jobs before a run starts.
WORKER_GB = 9.5
#: With ColumnarTexts (F67) an EB-NeRD worker is 5.32 GB rather than ~9.2 GB.
#: MEASURED, not derived: a first estimate of 2.9 GB (articles + the 0.93 GB of
#: click arrays) was wrong by 2.8x, because peak RSS is set by peak *allocation*
#: during the load, not by the size of what survives it. Rounded up to 5.6 for
#: headroom -- a worker that does not fit does not fail loudly, it swaps, and a
#: swapping run looks exactly like a hang (F38).
WORKER_GB_COLUMNAR = 5.6

#: Marginal cost of one more worker when the shared state is inherited through
#: fork (F70). Not another full copy -- just the child's own scratch: the
#: per-row-group buffers and the scored-slate dicts it builds and drops. Kept
#: deliberately generous; the failure mode for underestimating is a thrash.
WORKER_INCREMENT_GB = 1.5

#: Left free for the single-threaded merge, which builds a set over 13.3M
#: impression ids after the workers exit.
MERGE_HEADROOM_GB = 3.0

def submission_stem(dataset: str, retriever: str, params: dict, out_dir: Path) -> str:
    """`{dataset}_{retriever}_{paramhash}_i{n}` -- unique per run.

    Four components, each earning its place:

    ``dataset``     two competitions, two formats.
    ``retriever``   bm25 vs fusion is the comparison we want to keep.
    ``paramhash``   6 hex chars over the sorted params, so a window or decay
                    sweep does not collapse into one filename.
    ``i{n}``        auto-incremented. Identical params re-run is normal --
                    after a bug fix, or on a different machine -- and those
                    are still distinct artefacts with distinct leaderboard
                    rows. Without this the second run would silently replace
                    the first.

    The full params live in the .meta.json beside each file; the hash is only
    for uniqueness, not for reading back.
    """
    import hashlib

    digest = hashlib.sha256(
        "|".join(f"{k}={params[k]}" for k in sorted(params)).encode()
    ).hexdigest()[:6]
    base = f"{dataset}_{retriever}_{digest}"
    n = 1
    while (out_dir / f"{base}_i{n}_prediction.zip").exists():
        n += 1
    return f"{base}_i{n}"


def build_retriever(kind: str, dataset: str):
    """The retriever a submission is generated with.

    Defaults to fusion: on MIND it was the best retriever on **both**
    leaderboard metrics (AUC 0.5095 vs BM25's 0.5057, nDCG@10 0.2914 vs
    0.2853, F39). The margin is small and the CIs overlap, so this is a
    measured preference, not a demonstrated improvement.

    On EB-NeRD fusion showed **no gain** over its components -- the two
    retrievers agree there, and RRF needs disagreement. It is still offered so
    both datasets can be generated the same way, but `bm25` remains a
    defensible choice for EB-NeRD and is what the first submission used.
    """
    params = BEST[dataset]
    if kind == "bm25":
        return BM25Retriever(**params)
    if kind == "semantic":
        return HistoryIdRetriever(model_key="minilm", last_n=20, **SEMANTIC)
    if kind == "fusion":
        return RRFusion(
            [BM25Retriever(**params),
             HistoryIdRetriever(model_key="minilm", last_n=20, **SEMANTIC)],
            name="rrf(bm25+semantic)",
        )
    raise ValueError(f"unknown retriever {kind!r}")


#: Semantic query-vector params (F73/F75). The shipped defaults -- tau=0.35,
#: log decay -- were never swept: an audit found neither appears in any results
#: file as a *varied* quantity. Sweeping them found tau monotonically worse as
#: it tightens (0.20 -> r@100 0.0146, 0.80 -> 0.0039) and flat decay beating
#: log, i.e. recency weighting on the semantic side was *costing* recall.
#:
#: Recency decay was imported here by analogy with the lexical side (D3) and
#: never tested here. On MIND, which has no publish times at all, down-weighting
#: older clicks discards history that carries signal.
#:
#: > [!warning] The offline effect is +0.0026 [-0.0011, +0.0070] -- NOT
#: > significant at n=2,000 (F75). This ships only because the offline harness
#: > has disagreed with the leaderboard three times out of three (F34/F42/F58),
#: > so "not significant offline" is not evidence of no effect. The leaderboard
#: > is the measurement; this is the experiment.
SEMANTIC = dict(tau=0.20, decay="flat")

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


# ---------------------------------------------------------------------------
# Parallel path -- one worker per parquet row group.
#
# Each impression is scored independently, so the only thing preventing
# parallelism is that the output must be one file. Row groups solve that:
# each worker writes its own shard, and the shards are concatenated in row
# group order at the end. Ordering within the file does not matter to either
# leaderboard (each line carries its impression id), but keeping it stable
# makes two runs diffable, which is worth the trivial cost.
#
# The index is rebuilt per worker rather than pickled across: bm25s indices
# are large and pickling one per task would cost more than the ~6 s rebuild.
# ---------------------------------------------------------------------------

def _warn_if_swapping(threshold_gb: float = 2.0) -> None:
    """Say something before the machine starts thrashing, not after.

    A run that swaps does not fail -- it slows by two orders of magnitude and
    looks like a hang, which is exactly how the first EB-NeRD attempt
    presented. A log line at the moment memory runs out turns a mystery into
    a diagnosis.
    """
    vm = psutil.virtual_memory()
    avail = vm.available / (1024 ** 3)
    if avail < threshold_gb:
        swap = psutil.swap_memory()
        log.warning(
            "LOW MEMORY: %.1f GB available, %.1f GB swap in use -- "
            "reduce --n-jobs; each worker holds the index plus all histories",
            avail, swap.used / (1024 ** 3),
        )


_WORKER: dict = {}


class CompactHistories:
    """Worker-local click histories, stored as int arrays instead of strings.

    **Why this exists.** Measured on EB-NeRD's test split, a worker peaked at
    **15.08 GB**, of which the articles and the BM25 index were only 1.82 GB.
    The remaining ~13.3 GB was 807,677 users x 116,825,984 click ids held as
    Python ``str`` objects -- roughly 57 bytes of interpreter overhead apiece
    for a value the parquet file stores as int32. One worker therefore did not
    leave room for a second, and the parallel path could not be used at all.

    Storing the same ids as ``array('i')`` indices into a shared article table
    costs 4 contiguous bytes each: **~0.47 GB instead of ~13.3 GB**, so a
    worker fits in ~2.5 GB and six of them fit comfortably in 20 GB.

    **Why it is worker-local and not a schema change.** ``History.clicked_ids``
    is also read by ``src/eval/harness.py`` and, critically, by
    ``tests/test_no_leakage.py`` -- ``History.before()`` *is* the leakage
    boundary that Q9 requires a test for. Changing its element type would put
    a memory optimisation on Q9's correctness surface for no benefit, since
    only this path is memory-bound.

    Here the ids never escape: they are consumed immediately as a lookup key
    on the way to article text, and nothing downstream ever sees them. That
    makes the representation a private detail of one loop.

    > [!warning] This reimplements the truncation, so it is tested against the
    > > original
    > ``texts_before()`` duplicates the ``< cutoff`` logic of
    > ``History.before()``. Duplicated logic on the leakage boundary is exactly
    > the kind of thing that drifts silently, so
    > ``test_submit.py::test_compact_histories_match_history_before`` asserts
    > the two agree on the same inputs. Without that test this optimisation
    > would not be worth making.
    """

    __slots__ = ("_ids", "_times", "_texts")

    def __init__(self, histories, texts_by_id: dict[str, str]) -> None:
        from array import array

        # One dense index per article, shared by every user.
        order = list(texts_by_id)
        index = {aid: i for i, aid in enumerate(order)}
        self._texts: list[str] = [texts_by_id[a] for a in order]

        self._ids: dict[str, "array"] = {}
        self._times: dict[str, list] = {}
        for h in histories:
            pairs = [
                (index[a], t)
                for a, t in zip(h.clicked_ids, h.times or [None] * len(h.clicked_ids))
                if a in index
            ]
            if not pairs:
                continue
            self._ids[h.user_id] = array("i", [p[0] for p in pairs])
            # Timestamps are kept only when the dataset has them. On MIND they
            # are None (F1) and truncation is not verifiable anyway.
            if h.times is not None:
                self._times[h.user_id] = [p[1] for p in pairs]

    def __len__(self) -> int:
        return len(self._ids)

    def texts_before(self, user_id: str, cutoff) -> list[str]:
        """Retrieval text of this user's clicks strictly before ``cutoff``.

        Mirrors ``History.before()`` followed by the id-to-text lookup the
        caller would otherwise do, but without materialising the id list.
        """
        ids = self._ids.get(user_id)
        if ids is None:
            return []
        times = self._times.get(user_id)
        if times is None:
            # No timestamps: return everything, exactly as History.before()
            # does when times is None.
            return [self._texts[i] for i in ids]
        return [self._texts[i] for i, t in zip(ids, times) if t < cutoff]


#: Set by build_predictions_parallel before the pool starts, so each worker
#: constructs the same retriever the parent was asked for.
_WORKER_KIND = "bm25"


def _init_worker_shared() -> None:
    """Adopt state the parent already built, inherited through fork() (F70).

    ``_init_worker`` has every worker build its own copy of data that is
    identical and never written: the BM25 index, 125,541 article objects, and
    all 807,677 histories. Three workers meant three copies -- measured at
    19.3 GB each, which is what put 48 GB of RSS on a 31 GB machine and
    thrashed.

    On Linux ``fork`` is copy-on-write, so anything built *before* the pool
    starts is shared until written to. Scoring only reads, so it stays shared:
    measured, three children reading 0.96 GB of parent numpy arrays added
    **0.00 GB** rather than 2.88 GB.

    This works because F64/F67 moved histories to contiguous numpy arrays.
    Python objects would defeat it -- refcounting writes to every object header
    a child touches, dirtying the page and copying it. Arrays keep their data
    in one buffer that refcounting never touches, so the 116.8M clicks stay in
    exactly one physical copy.
    """
    global _WORKER
    if _WORKER is None:  # pragma: no cover - the parent always sets it
        raise RuntimeError("_PREFORK not populated before the pool started")


def _init_worker(dataset: str, tier: str, work_dir: str, test_split: str,
                 retriever_kind: str = "bm25") -> None:
    global _WORKER, _WORKER_KIND
    _WORKER_KIND = retriever_kind
    logging.getLogger("bm25s").setLevel(logging.ERROR)
    reader = get_reader(dataset, Path(work_dir), tier)
    try:
        articles = {a.article_id: a for a in reader.articles(splits=(test_split,))}
    except TypeError:
        articles = {a.article_id: a for a in reader.articles()}
    retriever = build_retriever(_WORKER_KIND, dataset)
    retriever.index(list(articles.values()))

    texts_by_id = {aid: a.retrieval_text for aid, a in articles.items()}
    # F67: load histories columnar where the dataset allows it. Same
    # texts_before() interface, pinned to CompactHistories by
    # test_columnar_texts_matches_compact_histories -- so this is a speed and
    # memory change, never a behaviour change.
    #
    # EB-NeRD only: its ids are numeric and it has click timestamps. MIND's
    # ids are strings ("N12345") with no timestamps (F1), so it keeps the
    # original path rather than having a second encoding guessed for it.
    histories = None
    if dataset == "ebnerd":
        try:
            from ..data.columnar import ColumnarTexts
            hist_path = reader._split_dir(test_split) / "history.parquet"
            histories = ColumnarTexts(hist_path, texts_by_id)
            log.info("worker: columnar histories (%d users)", len(histories))
        except Exception as e:  # noqa: BLE001
            # Never fail the run over an optimisation; fall back to the path
            # that produced every submitted file.
            log.warning("columnar histories unavailable (%s); using CompactHistories", e)
            histories = None
    if histories is None:
        histories = CompactHistories(reader.histories(test_split), texts_by_id)

    _WORKER = {
        "reader": reader,
        "retriever": retriever,
        "histories": histories,
        "split": test_split,
    }


def _predict_row_group(task: tuple[int, str]) -> tuple[int, str, int, int]:
    """Score one row group, write a shard, return its path and counts."""
    rg_idx, shard_path = task
    reader = _WORKER["reader"]
    histories: CompactHistories = _WORKER["histories"]
    retriever = _WORKER["retriever"]

    written = cold = 0
    with open(shard_path, "w") as f:
        for imp in reader.impressions_row_group(_WORKER["split"], rg_idx):
            # One call replaces truncate-then-look-up-text. The leakage
            # boundary is still applied -- see CompactHistories.texts_before.
            texts = histories.texts_before(imp.user_id, imp.time)

            scores = retriever.score_subset(texts, imp.candidates) if texts else {}
            if not texts:
                cold += 1
            ranks = rank_candidates(imp.candidates, scores)
            f.write(f"{imp.impression_id} [{','.join(map(str, ranks))}]\n")
            written += 1
    return rg_idx, shard_path, written, cold


def build_predictions_parallel(
    dataset: str,
    tier: str,
    work_dir: Path,
    test_split: str,
    out_path: Path,
    n_jobs: int,
    allow_swap: bool = False,
    retriever_kind: str = "bm25",
) -> dict:
    """Row-group-parallel prediction. Falls back to serial if unsupported.

    > [!warning] Swap is fine for scoring and fatal for the merge
    > The scoring phase streams a row group at a time, so paging degrades
    > gracefully. The **merge does not**: it builds a set over 13.3M impression
    > ids and hits it randomly, and random access against swap is roughly five
    > orders of magnitude slower than RAM. A first run wrote all 51 shards at
    > 6,426 lines/s, then produced *zero* output for 20+ minutes once the merge
    > started paging.
    >
    > So ``--allow-swap`` relaxes the worker clamp, but the merge still runs
    > only after every worker has exited and its memory is returned.
    """
    import tempfile

    behaviours = get_reader(dataset, work_dir, tier)._split_dir(test_split) / "behaviors.parquet"
    n_groups = pq.ParquetFile(behaviours).metadata.num_row_groups

    # Preflight. Each worker holds its own index plus all histories -- measured
    # at ~9.2 GB on EB-NeRD's test split after the CompactHistories change
    # (15.1 GB before it). Starting more workers than fit does not fail
    # loudly; it swaps, and a swapping run looks exactly like a hang. Clamp
    # here rather than discovering it 40 minutes in.
    import multiprocessing as _mp

    # Decided before the preflight, because it changes how much memory a
    # worker costs (F70): with fork the shared state exists once, without it
    # every worker rebuilds its own copy.
    can_fork = hasattr(os, "fork") and _mp.get_start_method(allow_none=True) in (None, "fork")

    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    # EB-NeRD workers use ColumnarTexts (F67), so they are ~2.9 GB rather than
    # ~9.5 GB and many more fit. Measured, not assumed: F64 put the click
    # arrays at 0.93 GB against ~7.4 GB of Python objects.
    worker_gb = WORKER_GB_COLUMNAR if dataset == "ebnerd" else WORKER_GB
    # Under fork-sharing (F70) the big structures exist once and the workers
    # inherit them read-only, so an extra worker costs its own scratch space
    # rather than another full copy. Budget the shared set once, then a small
    # per-worker increment.
    if can_fork:
        fits = max(1, int((available_gb - MERGE_HEADROOM_GB - worker_gb) // WORKER_INCREMENT_GB) + 1)
    else:
        fits = max(1, int((available_gb - MERGE_HEADROOM_GB) // worker_gb))
    if fits < n_jobs:
        if allow_swap:
            log.warning(
                "%.1f GB available: %d workers x %.1f GB will use swap. "
                "Proceeding because --allow-swap was passed.",
                available_gb, n_jobs, worker_gb,
            )
        else:
            log.warning(
                "%.1f GB available: %d workers x %.1f GB would swap. Using %d. "
                "Pass --allow-swap to override.",
                available_gb, n_jobs, worker_gb, fits,
            )
            n_jobs = fits
    log.info(
        "parallel: %d row groups across %d workers (%.1f GB free, ~%.1f GB/worker)",
        n_groups, n_jobs, available_gb, WORKER_GB,
    )

    started = time.perf_counter()
    tmpdir = Path(tempfile.mkdtemp(prefix="preds_", dir=out_path.parent))
    tasks = [(i, str(tmpdir / f"shard_{i:05d}.txt")) for i in range(n_groups)]

    written = cold = 0
    shards: dict[int, str] = {}
    done = 0

    # The pool is created and shut down inside this block so that every worker
    # has exited before concatenation begins. Leaving concatenation inside the
    # `with` kept two workers resident holding 13.6 GB and 10.9 GB while the
    # single-threaded merge tried to allocate a 13.3M-element set -- which put
    # 24 GB into swap and hung a run whose shards were already complete.
    # Build the shared, read-only state ONCE here, then let fork() share it
    # (F70). Falls back to per-worker construction on any platform without a
    # real fork, where each child would have to rebuild anyway.
    if can_fork:
        _init_worker(dataset, tier, str(work_dir), test_split, retriever_kind)
        log.info("shared state built once in the parent; workers inherit it via fork")
        ctx = _mp.get_context("fork")
        pool = ProcessPoolExecutor(
            max_workers=min(n_jobs, n_groups),
            initializer=_init_worker_shared,
            mp_context=ctx,
        )
    else:
        pool = ProcessPoolExecutor(
            max_workers=min(n_jobs, n_groups),
            initializer=_init_worker,
            initargs=(dataset, tier, str(work_dir), test_split, retriever_kind),
        )
    try:
        for rg_idx, shard, n, c in pool.map(_predict_row_group, tasks):
            shards[rg_idx] = shard
            written += n
            cold += c
            done += 1
            if done % 10 == 0 or done == n_groups:
                rate = written / (time.perf_counter() - started)
                log.info(
                    "  %d/%d groups, %s lines (%.0f/s)",
                    done, n_groups, f"{written:,}", rate,
                )
                _warn_if_swapping()
    finally:
        pool.shutdown(wait=True)
        gc.collect()
        log.info("  workers shut down; %.1f GB available for the merge",
                 psutil.virtual_memory().available / (1024 ** 3))

    # Concatenate in row-group order, deduplicating on impression id. EB-NeRD
    # repeats ~200K ids across the file (13,536,710 rows -> 13,336,711
    # unique), and the format is one line per id, so a straight concatenation
    # would emit duplicates.
    #
    # > [!warning] This step once drove the machine into 24 GB of swap
    # > The first EB-NeRD run wrote all 51 shards correctly and then hung. Two
    # > causes, both fixed here:
    # >
    # > 1. **The worker pool was still alive.** Concatenation ran inside the
    # >    `with ProcessPoolExecutor(...)` block, so two workers holding 13.6 GB
    # >    and 10.9 GB were kept resident for a phase that does not need them.
    # >    The pool is now shut down *before* this point.
    # > 2. **`set[str]` of 13.3M ids costs ~1.5 GB** in Python string objects.
    # >    Impression ids are integers, so `set[int]` is used instead -- roughly
    # >    a third of the memory and faster to hash.
    #
    # A non-integer id falls back to hashing the string, so this stays correct
    # for any dataset whose ids are not numeric.
    seen_int: set[int] = set()
    seen_str: set[str] = set()
    kept = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as out:
        for i in range(n_groups):
            with open(shards[i]) as sh:
                for line in sh:
                    iid = line.split(" ", 1)[0]
                    try:
                        key = int(iid)
                    except ValueError:
                        if iid in seen_str:
                            continue
                        seen_str.add(iid)
                    else:
                        if key in seen_int:
                            continue
                        seen_int.add(key)
                    out.write(line)
                    kept += 1
            Path(shards[i]).unlink()
    tmpdir.rmdir()

    elapsed = time.perf_counter() - started
    return {
        "lines": kept,
        "rows_read": written,
        "duplicates_dropped": written - kept,
        "cold_slates": cold,
        "seconds": round(elapsed, 1),
        "bytes": out_path.stat().st_size,
        "row_groups": n_groups,
        "workers": min(n_jobs, n_groups),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    p.add_argument("--tier", default="large")
    p.add_argument("--work-dir", type=Path, default=Path("data/work"))
    p.add_argument("--out-dir", type=Path, default=Path("submissions"))
    p.add_argument("--max-k", type=int, default=500)
    p.add_argument(
        "--window-hours", type=float, default=0.0,
        help="recency window in hours; 0 disables it. EB-NeRD only -- MIND has "
             "no publish time (F20). Part of the run identity, so a window "
             "sweep produces separately-named submissions.",
    )
    p.add_argument(
        "--retriever", choices=["bm25", "semantic", "fusion"], default="fusion",
        help="which retriever generates the predictions (default: fusion, "
             "the best MIND retriever on both leaderboard metrics -- F39)",
    )
    p.add_argument(
        "--allow-swap",
        action="store_true",
        help="run the requested worker count even if it exceeds free RAM. "
             "Fine for the scoring phase (sequential); the merge still waits "
             "for workers to exit first, because random access against swap "
             "does not degrade -- it stalls.",
    )
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

    # Only the SERIAL path uses these. build_predictions_parallel builds its
    # own per-worker histories in _init_worker, so loading them here is dead
    # weight on the parallel path -- and expensive dead weight: measured at
    # ~14.5 GB of Python objects for EB-NeRD's 807,677 users / 116.8M clicks
    # (F60: 158.6 s to build).
    #
    # Worse, the parent holds it for the whole run, so the preflight sees far
    # less free memory than the machine really has and clamps the worker count
    # accordingly. On a 31 GB box that turned "3 workers" into "1 worker, and
    # 36 GB requested against 31 GB of RAM" (F68).
    will_parallelise = hasattr(reader, "impressions_row_group") and budget.n_jobs > 1
    histories = {}
    if not will_parallelise:
        t0 = time.perf_counter()
        histories = {h.user_id: h for h in reader.histories(test_split)}
        log.info("histories    %8d  (%.1fs)", len(histories), time.perf_counter() - t0)
    else:
        log.info("histories    (skipped -- workers build their own)")

    params = BEST[args.dataset]
    run_params = {
        **params,
        "retriever": args.retriever,
        "window_hours": args.window_hours,
        "max_k": args.max_k,
    }
    # The semantic query-vector params are part of the run identity too.
    # Without them a tau/decay change produces the SAME stem as the run it
    # differs from -- silently overwriting a submitted artefact and making two
    # different configurations indistinguishable on the leaderboard.
    if args.retriever in ("semantic", "fusion"):
        run_params.update(SEMANTIC)
    retriever = build_retriever(args.retriever, args.dataset)
    t0 = time.perf_counter()
    retriever.index(list(articles.values()))
    log.info("indexed %s in %.1fs", retriever.name, time.perf_counter() - t0)

    # Name by dataset, retriever, params AND iteration, so no run can ever
    # overwrite another. Comparing two submissions on one leaderboard is a
    # reportable result (F39) and needs both artefacts to survive.
    #
    # The param hash alone is not enough: re-running an identical
    # configuration is a normal thing to do (a fixed bug, a different
    # machine), and those are genuinely different artefacts with different
    # leaderboard rows. The iteration counter is what separates them.
    stem = submission_stem(args.dataset, args.retriever, run_params, args.out_dir)
    log.info("submission stem: %s", stem)
    txt = args.out_dir / f"{stem}_prediction.txt"
    if will_parallelise:
        # Parquet-backed readers expose row groups, the natural unit of
        # parallelism. MIND is TSV and has none, so it takes the serial path.
        stats = build_predictions_parallel(
            args.dataset, args.tier, args.work_dir, test_split, txt, budget.n_jobs,
            allow_swap=args.allow_swap, retriever_kind=args.retriever,
        )
    else:
        stats = build_predictions(
            reader, retriever, articles, histories, test_split, txt, max_k=args.max_k
        )
    log.info(
        "\nwrote %s: %s lines, %.1f MB, %.1fs (%s cold slates)",
        txt, f"{stats['lines']:,}", stats["bytes"] / 1e6, stats["seconds"],
        f"{stats['cold_slates']:,}",
    )

    # The archive member MUST be named exactly "prediction.txt": the MIND
    # scorer does open(os.path.join(submit_dir, "prediction.txt")) and dies
    # with FileNotFoundError on anything else. A first submission failed this
    # way -- the file was correct, only its name inside the zip was not.
    # The local .txt keeps its dataset prefix so the two datasets' outputs do
    # not overwrite each other; only the arcname is fixed.
    zip_path = args.out_dir / f"{stem}_prediction.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt, arcname=SUBMISSION_MEMBER[args.dataset])
    log.info("zipped -> %s (%.1f MB)", zip_path, zip_path.stat().st_size / 1e6)

    meta = args.out_dir / f"{stem}_prediction.meta.json"
    meta.write_text(json.dumps({
        "dataset": args.dataset, "tier": args.tier,
        "retriever": retriever.name, "retriever_kind": args.retriever,
        "params": params, "run_params": run_params, "stem": stem,
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
