"""Env-gated phase profiler (issue #39 speed work).

Set PPGRID_PROFILE=1 to print one line per phase to stderr:

    [prof] <label> wall=<s> user=<s> sys=<s> faults=<M> rss=<peak GB>

The line breaks wall into user (CPU in the process) and sys (kernel:
page faults, mmap/munmap, cgroup charge) so a fault storm is visible as
a sys >> user ratio. faults is the delta of resource.ru_minflt over the
phase. All accounting is process-global (ru_* includes threads), which
is what the phase breakdown needs.

Zero cost when PPGRID_PROFILE is unset: the context manager is a no-op
pair of attribute reads, and the module imports only stdlib at import.
"""

from __future__ import annotations

import os
import resource
import sys
import time
from typing import Self

__all__ = ["enabled", "phase"]

# Read once at import: PPGRID_PROFILE is a process-start knob.
_ENABLED: bool = os.environ.get("PPGRID_PROFILE") == "1"


class _Phase:
    """Context manager: timed phase with user/sys/fault/RSS accounting."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.enabled = _ENABLED
        self.t0 = 0.0
        self.u0 = 0.0
        self.s0 = 0.0
        self.f0 = 0

    def __enter__(self) -> Self:
        if self.enabled:
            ru = resource.getrusage(resource.RUSAGE_SELF)
            self.t0 = time.perf_counter()
            self.u0 = ru.ru_utime
            self.s0 = ru.ru_stime
            self.f0 = ru.ru_minflt
        return self

    def __exit__(self, *exc: object) -> bool:
        if not self.enabled:
            return False
        ru = resource.getrusage(resource.RUSAGE_SELF)
        wall = time.perf_counter() - self.t0
        user = ru.ru_utime - self.u0
        sysc = ru.ru_stime - self.s0
        faults = ru.ru_minflt - self.f0
        rss = ru.ru_maxrss / 1024.0 / 1024.0  # KB -> GB on Linux
        print(  # ruff: ignore[print]
            f"[prof] {self.label} wall={wall:.2f}s user={user:.2f}s sys={sysc:.2f}s "
            f"faults={faults / 1e6:.1f}M rss_peak={rss:.2f}GB",
            file=sys.stderr,
        )
        return False


def phase(label: str) -> _Phase:
    """Start a profiler phase (no-op unless PPGRID_PROFILE=1)."""
    return _Phase(label)


def enabled() -> bool:
    """Report whether PPGRID_PROFILE=1 was set at import."""
    return _ENABLED
