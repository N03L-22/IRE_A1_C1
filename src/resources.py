"""Resource budget: how many cores and how much memory a stage may use.

The machine this runs on is shared with other work, so no stage may help itself
to everything it finds. Every entry point takes ``--n-jobs`` and ``--mem-gb``,
and the *resolved* values are written into the run manifest beside the metrics
so any number can be traced back to the budget that produced it.

Defaults assume an idle machine. They are ceilings, not reservations: if the
machine is busy the budget scales down (or refuses) rather than swapping itself
to death. See plan/Pipeline.md and architecture.md decision 7b.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict
from typing import Any

import psutil

#: Ceiling on an *idle* machine, leaving headroom for the OS and editor.
DEFAULT_N_JOBS = 26
DEFAULT_MEM_GB = 26.0
#: Encoder batch size. This is the VRAM dial and is independent of --mem-gb.
DEFAULT_BATCH_SIZE = 64

#: Refuse to start below this much free memory -- under it we would only swap.
MIN_VIABLE_MEM_GB = 2.0


@dataclass(frozen=True)
class Budget:
    """A resolved resource budget. Immutable, and safe to log verbatim."""

    n_jobs: int
    mem_gb: float
    batch_size: int
    #: What was actually free when we resolved, for the manifest.
    available_mem_gb: float
    total_cores: int
    load_average: float
    #: True when we had to come down from what was requested.
    scaled_down: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        note = " (scaled down)" if self.scaled_down else ""
        return (
            f"budget: {self.n_jobs} jobs, {self.mem_gb:.1f} GB, "
            f"batch {self.batch_size}{note} "
            f"[machine: {self.total_cores} cores, load {self.load_average:.1f}, "
            f"{self.available_mem_gb:.1f} GB free]"
        )


class InsufficientResources(RuntimeError):
    """Raised when the machine cannot honour even a minimal budget."""


def resolve(
    n_jobs: int = DEFAULT_N_JOBS,
    mem_gb: float = DEFAULT_MEM_GB,
    batch_size: int = DEFAULT_BATCH_SIZE,
    *,
    check_availability: bool = True,
) -> Budget:
    """Turn a requested budget into one the machine can actually honour.

    Requesting 26 GB when 8 GB is free does not give you 26 GB -- it gives you
    thrashing. So the request is clamped to what is genuinely available, and if
    even a minimal budget is impossible we raise rather than start a run that
    will die three hours in.
    """
    total_cores = os.cpu_count() or 1
    load_1min = os.getloadavg()[0]
    vm = psutil.virtual_memory()
    available_gb = vm.available / (1024**3)

    requested = (n_jobs, mem_gb)

    # Never more cores than exist, and never fewer than one.
    n_jobs = max(1, min(n_jobs, total_cores))

    if check_availability:
        # Leave the cores that are already busy alone. Load average is a decent
        # proxy for "runnable tasks", so free_cores is what is genuinely idle.
        free_cores = max(1, int(total_cores - load_1min))
        n_jobs = min(n_jobs, free_cores)

        # Keep a little headroom -- allocating every free byte is how the OOM
        # killer gets involved.
        usable_gb = max(0.0, available_gb * 0.9)
        mem_gb = min(mem_gb, usable_gb)

        if mem_gb < MIN_VIABLE_MEM_GB:
            raise InsufficientResources(
                f"only {available_gb:.1f} GB available "
                f"(need at least {MIN_VIABLE_MEM_GB} GB). "
                f"Machine load is {load_1min:.1f} across {total_cores} cores. "
                "Wait for the other job to finish, or pass a smaller --mem-gb."
            )

    return Budget(
        n_jobs=n_jobs,
        mem_gb=round(mem_gb, 2),
        batch_size=batch_size,
        available_mem_gb=round(available_gb, 2),
        total_cores=total_cores,
        load_average=round(load_1min, 2),
        scaled_down=(n_jobs, mem_gb) != requested,
    )


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the standard budget flags. Every entry point calls this."""
    group = parser.add_argument_group("resource budget")
    group.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help=f"worker processes/threads (default: {DEFAULT_N_JOBS})",
    )
    group.add_argument(
        "--mem-gb",
        type=float,
        default=DEFAULT_MEM_GB,
        help=f"memory ceiling in GB (default: {DEFAULT_MEM_GB})",
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"encoder batch size, the VRAM dial (default: {DEFAULT_BATCH_SIZE})",
    )
    group.add_argument(
        "--ignore-availability",
        action="store_true",
        help="use the requested budget without checking what is free",
    )
    return parser


def from_args(args: argparse.Namespace) -> Budget:
    """Build a Budget from parsed arguments."""
    return resolve(
        n_jobs=args.n_jobs,
        mem_gb=args.mem_gb,
        batch_size=args.batch_size,
        check_availability=not args.ignore_availability,
    )
