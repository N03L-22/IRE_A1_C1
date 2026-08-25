"""Step 1 of the Q1 pipeline: raw zips -> a working directory of real files.

Extraction is separated from parsing for one reason: parquet needs random
access, so EB-NeRD's files cannot be streamed out of the zip. Rather than have
each reader decide whether to stream or extract, everything lands on disk once
and the readers only ever see paths.

Extraction is idempotent and skipped when the output is newer than the archive,
which is what lets ``make data`` be re-run cheaply.

Nothing here writes into ``data/raw/`` -- the archives are inputs and stay
untouched. See plan/1-Data-Pipeline.md step 1.
"""

from __future__ import annotations

import argparse
import logging
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

#: Junk that macOS leaves in zips. Never wanted, and ``__MACOSX`` shadows real
#: filenames if extracted alongside them.
JUNK_PREFIXES = ("__MACOSX/",)
JUNK_NAMES = (".DS_Store", "__placeholder__")


@dataclass(frozen=True)
class Archive:
    """One raw archive and where its contents belong."""

    name: str
    #: Path relative to data/raw/
    zip_path: str
    #: Directory under data/work/ that receives the contents
    dest: str
    #: A file that must exist afterwards, as a completeness check
    sentinel: str


#: The small tier -- the working and headline tier. Large tiers are downloaded
#: but deliberately idle; see plan/Pipeline.md.
SMALL_TIER: tuple[Archive, ...] = (
    Archive(
        name="mind-train",
        zip_path="mind/MINDsmall_train.zip",
        dest="mind/train",
        sentinel="behaviors.tsv",
    ),
    Archive(
        name="mind-dev",
        zip_path="mind/MINDsmall_dev.zip",
        dest="mind/dev",
        sentinel="behaviors.tsv",
    ),
    Archive(
        name="ebnerd-small",
        zip_path="ebnerd/ebnerd_small.zip",
        dest="ebnerd/small",
        sentinel="articles.parquet",
    ),
)

#: EB-NeRD demo -- the smoke test. Every code change runs against this first.
DEMO_TIER: tuple[Archive, ...] = (
    Archive(
        name="ebnerd-demo",
        zip_path="ebnerd/ebnerd_demo.zip",
        dest="ebnerd/demo",
        sentinel="articles.parquet",
    ),
)

#: Provided article embeddings -- the Phase 3 baseline. Both cover all 125,541
#: EB-NeRD articles, so the small tier is a strict subset (finding F13).
ARTIFACTS: tuple[Archive, ...] = (
    Archive(
        name="word2vec",
        zip_path="ebnerd/Ekstra_Bladet_word2vec.zip",
        dest="ebnerd/artifacts/word2vec",
        sentinel="document_vector.parquet",
    ),
    Archive(
        name="bert-multilingual",
        zip_path="ebnerd/google_bert_base_multilingual_cased.zip",
        dest="ebnerd/artifacts/bert_multilingual",
        sentinel="bert_base_multilingual_cased.parquet",
    ),
)

#: The large tier -- Codabench submission only (brief v2 makes it mandatory
#: for Q5). Never the source of a headline metric: MIND-large test and
#: EB-NeRD's testset are UNLABELLED, so no offline number can be computed
#: from them (F14). Small stays the measurement tier.
LARGE_TIER: tuple[Archive, ...] = (
    Archive(
        name="mind-large-test",
        zip_path="mind/MINDlarge_test.zip",
        dest="mind/large_test",
        sentinel="behaviors.tsv",
    ),
    Archive(
        name="ebnerd-large",
        zip_path="ebnerd/ebnerd_large.zip",
        dest="ebnerd/large",
        sentinel="articles.parquet",
    ),
    # NOTE: ebnerd_testset.zip is a SEPARATE download and is not present.
    # ebnerd_large.zip ships train/ and validation/ only -- verified, no test
    # member (F11). Without it the EB-NeRD leaderboard submission is blocked.
)

TIERS: dict[str, tuple[Archive, ...]] = {
    "small": SMALL_TIER,
    "demo": DEMO_TIER,
    "large": LARGE_TIER,
    "artifacts": ARTIFACTS,
}


def _is_junk(member: str) -> bool:
    if member.startswith(JUNK_PREFIXES):
        return True
    return Path(member).name in JUNK_NAMES


def _strip_top_level(members: list[str]) -> bool:
    """Should we drop a shared top-level directory from every path?

    MIND zips wrap everything in ``MINDsmall_train/``; EB-NeRD zips do not.
    Normalising here means the readers see the same layout for both, instead of
    each one knowing its archive's quirk.
    """
    tops = {m.split("/", 1)[0] for m in members if "/" in m}
    files_at_root = any("/" not in m for m in members)
    return len(tops) == 1 and not files_at_root


def extract_one(archive: Archive, raw_dir: Path, work_dir: Path, *, force: bool = False) -> Path:
    """Extract one archive. Returns the destination directory.

    Skipped when the sentinel exists and is newer than the zip -- so a re-run
    costs a stat() rather than a re-extraction.
    """
    zip_path = raw_dir / archive.zip_path
    dest = work_dir / archive.dest
    sentinel = dest / archive.sentinel

    if not zip_path.exists():
        raise FileNotFoundError(
            f"{archive.name}: missing archive {zip_path}. "
            "Downloads are documented in the tracking note."
        )

    if not force and sentinel.exists() and sentinel.stat().st_mtime >= zip_path.stat().st_mtime:
        log.info("%-18s skip (up to date)", archive.name)
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if not _is_junk(m)]
        strip = _strip_top_level(members)

        n_files = 0
        n_bytes = 0
        for member in members:
            if member.endswith("/"):
                continue
            rel = member.split("/", 1)[1] if strip else member
            if not rel:
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as out:
                # Chunked so a 1.4 GB member does not become 1.4 GB of RSS.
                while chunk := src.read(1 << 20):
                    out.write(chunk)
                    n_bytes += len(chunk)
            n_files += 1

    if not sentinel.exists():
        raise RuntimeError(
            f"{archive.name}: extracted {n_files} files but sentinel "
            f"{archive.sentinel} is missing -- the archive layout changed."
        )

    elapsed = time.perf_counter() - started
    log.info(
        "%-18s %2d files  %6.1f MB  %5.1fs  -> %s",
        archive.name,
        n_files,
        n_bytes / 1e6,
        elapsed,
        dest.relative_to(work_dir.parent) if work_dir.parent in dest.parents else dest,
    )
    return dest


def extract_tier(
    tier: str, raw_dir: Path, work_dir: Path, *, force: bool = False
) -> dict[str, Path]:
    """Extract every archive in a named tier."""
    if tier not in TIERS:
        raise KeyError(f"unknown tier {tier!r}; expected one of {sorted(TIERS)}")
    return {a.name: extract_one(a, raw_dir, work_dir, force=force) for a in TIERS[tier]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tier",
        action="append",
        choices=sorted(TIERS),
        help="tier(s) to extract; repeatable. Default: small",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--work-dir", type=Path, default=Path("data/work"))
    parser.add_argument("--force", action="store_true", help="re-extract even if up to date")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for tier in args.tier or ["small"]:
        log.info("--- %s ---", tier)
        extract_tier(tier, args.raw_dir, args.work_dir, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
