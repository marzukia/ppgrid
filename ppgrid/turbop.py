"""Turbo pre-check estimator and RAM regime sizing (issue #30).

Implements sections 2.2, 3.5, and 3.6 (all) of the approved turbo design
(/home/monky/reports/ppgrid-turbo-design.md):

- ``est_peak``: worst-case RAM estimate from the 2.2 phase formulas, the
  MAX of phases A through D (phases do not run at once). Phase A dominates
  at ~24.5 B/cell (4-array box_count transient); phase C is ~19.7 B/cell
  (pyramid + val/sup).
- Budget model: sizing budget ``Bt = 0.85 * C - 4 GB`` against pre-check
  budget ``0.85 * C`` (reconciliation in :func:`sizing_budget_bytes`).
- Regime selection A/B/C per 3.6.2 with in-flight sizing: descent bands,
  box_count chunks, workers, ZSTD context pool, write tile cache. Regime C
  needs an 8 GB page-cache floor and >= 2 in-flight units; else per-box
  fallback with turbo warp/write. The 3.6.2 worked table is
  :data:`SIZING_TABLE`, the one source of truth for the derived row
  numbers (pending A/B validation, 3.6.5).
- ``RssBackstop``: mid-run RSS hook on the shared path, a single [warn]
  when resource.getrusage RSS exceeds budget x 0.95 (performance only).

Stdlib only: no numpy/pandas/rasterio at import, so the pre-check can run
before the first heavy allocation (and before heavy imports) once wired.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import resource
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ANCHOR_WALL_S",
    "B0_BYTES",
    "SIZING_TABLE",
    "PrecheckDecision",
    "RssBackstop",
    "SizingRow",
    "TurboPlan",
    "box_chunk_bytes",
    "descent_band_bytes",
    "detect_cap_gb",
    "est_peak",
    "per_box_wall_s",
    "phase_peak_bytes",
    "plan",
    "precheck",
    "precheck_budget_bytes",
    "resolve_cap",
    "sizing_budget_bytes",
    "table_row",
    "tile_cache_tiles",
    "wall_model_s",
]

log = logging.getLogger("ppgrid.turbop")

# ---------------------------------------------------------------------------
# Price list (design 3.6.2). Every value is derived, pending A/B validation
# (3.6.5). GB is decimal, 1 GB = 1e9 bytes, matching the design arithmetic.
# ---------------------------------------------------------------------------
GB: float = 1.0e9

B0_BYTES: float = 4.0 * GB  # non-turbo base: OS + runtime + ingest transient
AUTO_HEADROOM_GB: float = 8.0  # auto-cap headroom (3.6.1), fixed, not a percentage
CAP_FLOOR_GB: float = 8.0  # below this, Bt is near zero; sub-floor caps run per-box
PAGE_CACHE_FLOOR_BYTES: float = 8.0 * GB  # regime C floor vs writeback throttle (2.1)

BUDGET_FACTOR: float = 0.85  # planning factor (3.1)
PHASE_A_B_PER_CELL: float = 24.5  # 2.2 phase A: s0+c0 8 + near 1 + box_count 4-array transient
PYRAMID_SC_B_PER_CELL: float = 8.0 * 4.0 / 3.0  # 10.67: s+c pyramid, geometric sum
PHASE_B_B_PER_CELL: float = PYRAMID_SC_B_PER_CELL + 1.0  # pyramid build: s+c pyramid + near
PHASE_C_B_PER_CELL: float = PYRAMID_SC_B_PER_CELL + 8.0 + 1.0  # descent: pyramid + val/sup + near (~19.7)
PHASE_D_B_PER_CELL: float = 8.0 + 1.0 + 2.0  # write: val/sup + near + int16 DN
POINTS_B_PER_PTS: float = 40.0  # 3 f64 memmap bands (24 B, file-backed) + ix/iy f64 transient (16 B)

M_IN_B_PER_CELL: float = 9.0  # s0+c0 f32 + near bool
M_VS_B_PER_CELL: float = 8.0  # val+sup f32
M_TILE_BYTES: float = 1.0e6  # 0.5 MB raw + 0.44 MB zstd per 512^2 int16 tile
M_Z_BYTES: float = 2.0e6  # one libzstd 1.5.7 level-3 streaming context

M_BOX_B_PER_CELL: float = 12.0  # p, w, q f32 temps per box_count chunk cell
BOX_CHUNK_ROWS: int = 4096  # box_count_banded band height
DESCENT_BAND_B_PER_CELL: float = 40.0  # a/local/pv/ps slices over band rows
DESCENT_PARENT_B_PER_CELL: float = 8.0  # descent parent-row term
DESCENT_BAND_ROWS: int = 1024  # _descent_banded band height
DESCENT_PARENT_ROWS: int = 512  # parent rows read per band

TILE_CACHE_MIN: int = 8
TILE_CACHE_MAX: int = 256
MAX_WORKERS: int = 8  # thread cap, min(8, os.process_cpu_count()) (3.1)
PER_BOX_WORKERS: int = 4  # per-box pool, matches pipeline.DEFAULT_WORKERS

RSS_BACKSTOP_FRAC: float = 0.95  # mid-run RSS warn threshold vs budget (3.5)

# Wall model (3.6.2 table note, calibrated to the A4 target).
ANCHOR_WALL_S: float = 479.6  # full-AU non-turbo anchor (DESIGN box, 4 workers)
SERIAL_FLOOR_S: float = 30.0  # ingest + calibrate + grid + serial flush
PARALLEL_EQUIV_S: float = 740.0  # 1-thread parallel equivalent (30 + 740/4 = 215 s target)
PER_BOX_WARP_SAVE_S: float = 65.2  # reproj 127.2 s -> 62 s (warp 2.88x, flush serial)
PER_BOX_ZSTD_SAVE_S: float = 21.9  # zstd 29.2 s @1t -> 7.3 s @4t
PER_BOX_SKIP_WORKFILE_S: float = 89.3  # skip the work-CRS intermediate (S3.4)

# Per-box peak: au50m measured 3.69 GB RSS (A.2) + derived turbo write buffers.
PER_BOX_BASE_RSS_BYTES: float = 3.69e9
PER_BOX_WRITE_BUFFER_BYTES: float = 1.0e9

# Host files for auto-detect (monkeypatched in tests).
CGROUP_MEMORY_MAX: str = "/sys/fs/cgroup/memory.max"  # cgroup v2
CGROUP_V1_LIMIT: str = "/sys/fs/cgroup/memory/memory.limit_in_bytes"  # cgroup v1 fallback
CGROUP_ROOT: str = "/sys/fs/cgroup"  # cgroup v2 mount root (test seam)
PROC_SELF_CGROUP: str = "/proc/self/cgroup"  # process cgroup paths (v2 unified line)
MEMINFO_PATH: str = "/proc/meminfo"

_PHASE_B_PER_CELL: dict[str, float] = {
    "A": PHASE_A_B_PER_CELL,
    "B": PHASE_B_B_PER_CELL,
    "C": PHASE_C_B_PER_CELL,
    "D": PHASE_D_B_PER_CELL,
}


# ---------------------------------------------------------------------------
# Budget model
# ---------------------------------------------------------------------------


def sizing_budget_bytes(cap_gb: float) -> float:
    """Compute the sizing budget Bt = 0.85 * C - 4 GB (design 3.6.2).

    Reconciliation with 3.5: the plain 3.5 pre-check budget is the
    ``ram_gb x 0.85`` form; the 3.6 addendum supersedes it. Sizing works
    against Bt, which splits off B0 = 4 GB; the pre-check still compares
    the regime est peak against ``0.85 * C`` (see
    :func:`precheck_budget_bytes`). B0 is counted inside the pre-check
    budget, so a regime sized to Bt always fits ``0.85 * C``.

    Args:
        cap_gb: RAM cap C in GB (after :func:`resolve_cap`).

    Returns:
        Bt in bytes.

    """
    return (BUDGET_FACTOR * cap_gb - B0_BYTES / GB) * GB


def precheck_budget_bytes(cap_gb: float) -> float:
    """Compute the pre-check budget 0.85 * C (design 3.5, 3.6.2).

    Args:
        cap_gb: RAM cap C in GB.

    Returns:
        Pre-check budget in bytes, B0 included.

    """
    return BUDGET_FACTOR * cap_gb * GB


# ---------------------------------------------------------------------------
# 2.2 phase estimates
# ---------------------------------------------------------------------------


def phase_peak_bytes(phase: str, cells: int, n_pts: int) -> float:
    """Estimate one 2.2 phase's peak RAM: cells x per-cell price + points.

    Args:
        phase: Phase letter, one of A (bin + box_count), B (pyramid
            build), C (descent to val/sup), D (slice + write).
        cells: Total padded cells.
        n_pts: Point count (40 B/pt: 3 f64 memmap bands + ix/iy transient).

    Returns:
        Phase peak in bytes (excludes B0, the sizing base floor).

    Raises:
        ValueError: If phase is not one of A/B/C/D.

    """
    try:
        per_cell = _PHASE_B_PER_CELL[phase]
    except KeyError:
        msg = f"unknown phase {phase!r} (expected A/B/C/D)"
        raise ValueError(msg) from None
    return cells * per_cell + n_pts * POINTS_B_PER_PTS


def est_peak(cells: int, n_pts: int) -> float:
    """Estimate worst-case RAM (bytes) from the 2.2 phase formulas.

    The MAX of phases A through D, not the sum: the phases do not run at
    once. Phase A (bin + box_count) dominates at ~24.5 B/cell with the
    4-array box_count transient; phase C (pyramid + val/sup) is ~19.7
    B/cell. Excludes B0 (see :func:`sizing_budget_bytes`).

    Args:
        cells: Total padded cells.
        n_pts: Point count.

    Returns:
        Worst-case phase peak in bytes.

    """
    return max(phase_peak_bytes(p, cells, n_pts) for p in _PHASE_B_PER_CELL)


# ---------------------------------------------------------------------------
# Per-regime in-flight sizing (3.6.2)
# ---------------------------------------------------------------------------


def box_chunk_bytes(wc: int, radius: int) -> float:
    """Compute M_box: bytes of one box_count chunk (4096 + 2r rows).

    The p/w/q f32 temps are 12 B/cell over (BOX_CHUNK_ROWS + 2*radius)
    rows (pullpush.box_count_banded).

    Args:
        wc: Padded columns.
        radius: Cap radius in cells.

    Returns:
        One chunk's transient in bytes.

    """
    return M_BOX_B_PER_CELL * wc * (BOX_CHUNK_ROWS + 2 * radius)


def descent_band_bytes(wc: int) -> float:
    """Compute M_band: bytes of one descent band.

    40 B/cell over 1024 band rows (a/local/pv/ps slices) + 8 B/cell over
    512 parent rows (pullpush._descent_banded).

    Args:
        wc: Padded columns.

    Returns:
        One band's transient in bytes.

    """
    return (DESCENT_BAND_B_PER_CELL * DESCENT_BAND_ROWS + DESCENT_PARENT_B_PER_CELL * DESCENT_PARENT_ROWS) * wc


def tile_cache_tiles(sizing_budget: float) -> int:
    """Compute the write tile cache T = clamp(floor(0.1 * Bt / M_tile), 8, 256).

    Args:
        sizing_budget: Bt in bytes.

    Returns:
        Tile cache depth in tiles (flush stays serial raster-scan, 3.3).

    """
    t = int((0.1 * sizing_budget) // M_TILE_BYTES)
    return max(TILE_CACHE_MIN, min(TILE_CACHE_MAX, t))


def wall_model_s(workers: int) -> float:
    """Estimate shared-path wall: 30 s serial floor + 740 s / workers (3.6.2).

    Args:
        workers: Thread count for the parallel phases.

    Returns:
        Central wall estimate in seconds.

    """
    return SERIAL_FLOOR_S + PARALLEL_EQUIV_S / workers


def per_box_wall_s() -> float:
    """Estimate per-box wall: anchor minus the turbo output gains (3.6.2 note).

    Returns:
        Central wall estimate in seconds (479.6 - 65.2 - 21.9 - 89.3).

    """
    return ANCHOR_WALL_S - PER_BOX_WARP_SAVE_S - PER_BOX_ZSTD_SAVE_S - PER_BOX_SKIP_WORKFILE_S


# ---------------------------------------------------------------------------
# The 3.6.2 worked table (full-AU anchor geometry). One source of truth for
# the derived row numbers; A/B validation (3.6.5) edits this table only.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SizingRow:
    """One row of the 3.6.2 worked table at the full-AU anchor geometry."""

    cap_gb: float
    regime: str  # "A" | "B" | "C" | "per_box"
    box_chunks: int  # in-flight box_count chunks (0 = full field or per-box)
    descent_bands: int  # in-flight descent bands
    workers: int
    est_peak_range_gb: tuple[float, float]  # planned est peak, low/high
    wall_range_s: tuple[float, float]  # hand-widened wall range (3.6.2 note)


SIZING_TABLE: tuple[SizingRow, ...] = (
    # 16 GB: regime C fails (0 in-flight after the 8 GB floor) -> per-box
    # with turbo warp/write. Peak: au50m measured 3.69 GB + write buffers.
    SizingRow(16.0, "per_box", 0, 0, 4, (4.0, 6.0), (290.0, 380.0)),
    # 32 GB: regime B, val/sup memmap. Est peak bounded by Bt + B0 = 0.85*C.
    SizingRow(32.0, "B", 3, 4, 3, (27.2, 27.2), (240.0, 330.0)),
    # 64 GB: regime A, full in-RAM. Est peak = 24.5*N + B0 (formula base;
    # doc value 42.7 rounded, exact 42.71; the 2.2 RSS sum without B0 is
    # 38-40 GB, anchored to measured HWM).
    SizingRow(64.0, "A", 0, 0, 8, (42.7, 42.8), (115.0, 160.0)),
    # 128 GB: same plan as 64 GB; headroom for bigger grids, not speed here.
    SizingRow(128.0, "A", 0, 0, 8, (42.7, 42.8), (115.0, 160.0)),
)


def table_row(cap_gb: float) -> SizingRow | None:
    """Look up a 3.6.2 worked table row by cap.

    Args:
        cap_gb: RAM cap in GB.

    Returns:
        The row, or None for caps not in the table.

    """
    for row in SIZING_TABLE:
        if abs(row.cap_gb - cap_gb) < 1e-9:
            return row
    return None


@dataclass(frozen=True)
class TurboPlan:
    """A RAM-scaled turbo plan for one geometry at one cap (design 3.6.2)."""

    cap_gb: float
    n_cells: int
    regime: str  # "A" | "B" | "C" | "per_box"
    box_chunks: int  # in-flight box_count chunks (A/per_box: 0)
    descent_bands: int  # in-flight descent bands (A/per_box: 0)
    workers: int
    tile_cache_tiles: int  # write tile cache depth (per_box: 0)
    val_sup_memmap: bool  # output val/sup file-backed (True) vs in RAM (False)
    zstd_ctx_bytes: int  # ZSTD context pool: workers x M_z
    est_peak_bytes: float  # regime est peak (3.6.2 per-regime formula)
    budget_bytes: float  # pre-check budget 0.85*C (B0 included)
    sizing_bytes: float  # sizing budget Bt = 0.85*C - B0
    wall_s: float  # central wall estimate
    wall_range_s: tuple[float, float]  # table row range, else generic +/-15%


def plan(
    cap_gb: float,
    n_cells: int,
    wc: int,
    radius: int,
    cpu: int | None = None,
) -> TurboPlan:
    """Select the regime and size in-flight parameters from the cap (3.6.2).

    Selection: A if ``N <= (0.85*C - 4)/24.5`` (full in-RAM), else B if
    ``Bt >= N*M_in + max(M_box, M_band)`` (input in RAM, banded), else C
    (all memmap, 8 GB page-cache floor, viable with >= 2 in-flight
    units), else per-box fallback with turbo warp/write.

    Args:
        cap_gb: RAM cap C in GB (after :func:`resolve_cap`).
        n_cells: Total padded cells N.
        wc: Padded columns Wc.
        radius: Cap radius in cells r.
        cpu: Thread count; defaults to os.process_cpu_count().

    Returns:
        TurboPlan sized to fit the 0.85*C budget.

    Raises:
        ValueError: If n_cells or wc is non-positive, or radius is negative.

    """
    if n_cells <= 0 or wc <= 0 or radius < 0:
        msg = f"need n_cells > 0, wc > 0, radius >= 0; got {n_cells}, {wc}, {radius}"
        raise ValueError(msg)
    if cpu is None:
        cpu = os.process_cpu_count() or 1
    sizing = sizing_budget_bytes(cap_gb)
    budget = precheck_budget_bytes(cap_gb)
    m_box = box_chunk_bytes(wc, radius)
    m_band = descent_band_bytes(wc)
    n_in = M_IN_B_PER_CELL * n_cells
    row = table_row(cap_gb)

    if n_cells * PHASE_A_B_PER_CELL <= sizing:
        # Regime A: whole field in RAM.
        workers = min(MAX_WORKERS, cpu)
        est = n_cells * PHASE_A_B_PER_CELL + B0_BYTES
        wall = wall_model_s(workers)
        return _finish_plan(
            row,
            cap_gb,
            n_cells,
            "A",
            0,
            0,
            workers,
            tile_cache_tiles(sizing),
            val_sup_memmap=False,
            est=est,
            budget=budget,
            sizing=sizing,
            wall=wall,
        )

    if sizing >= n_in + max(m_box, m_band):
        # Regime B: input in RAM, banded in-flight sized from the leftover.
        t = tile_cache_tiles(sizing)
        left = sizing - n_in - t * M_TILE_BYTES
        c = max(0, int(left // m_box))
        b = max(0, int(left // m_band))
        workers = min(MAX_WORKERS, cpu, c, b)
        inflight = max(c * m_box, b * m_band)
        val_ram = sizing >= n_cells * (M_IN_B_PER_CELL + M_VS_B_PER_CELL) + t * M_TILE_BYTES + inflight
        # Component-bound est peak (3.6.2). The in-flight sizing keeps the
        # planned peak at or below Bt + B0 = 0.85*C, the bound the
        # pre-check compares against.
        comp = n_in + t * M_TILE_BYTES + c * m_box + b * m_band + B0_BYTES
        est = min(comp, sizing + B0_BYTES)
        wall = wall_model_s(workers)
        return _finish_plan(
            row,
            cap_gb,
            n_cells,
            "B",
            c,
            b,
            workers,
            t,
            val_sup_memmap=not val_ram,
            est=est,
            budget=budget,
            sizing=sizing,
            wall=wall,
        )

    # Regime C candidate: all memmap, 8 GB page-cache floor.
    left = sizing - PAGE_CACHE_FLOOR_BYTES
    c = max(0, int(left // m_box))
    b = max(0, int(left // m_band))
    if max(c, b) >= 2:
        workers = min(MAX_WORKERS, cpu, c, b)
        # The tile cache lives inside the page-cache floor here.
        est = max(c * m_box, b * m_band) + PAGE_CACHE_FLOOR_BYTES + B0_BYTES
        wall = wall_model_s(workers)
        return _finish_plan(
            row,
            cap_gb,
            n_cells,
            "C",
            c,
            b,
            workers,
            tile_cache_tiles(sizing),
            val_sup_memmap=True,
            est=est,
            budget=budget,
            sizing=sizing,
            wall=wall,
        )

    # Per-box fallback with turbo warp/write (S3.3, S3.4).
    est = PER_BOX_BASE_RSS_BYTES + PER_BOX_WRITE_BUFFER_BYTES
    wall = per_box_wall_s()
    return _finish_plan(
        row,
        cap_gb,
        n_cells,
        "per_box",
        0,
        0,
        PER_BOX_WORKERS,
        0,
        val_sup_memmap=True,
        est=est,
        budget=budget,
        sizing=sizing,
        wall=wall,
    )


def _finish_plan(
    row: SizingRow | None,
    cap_gb: float,
    n_cells: int,
    regime: str,
    c: int,
    b: int,
    workers: int,
    t: int,
    *,
    val_sup_memmap: bool,
    est: float,
    budget: float,
    sizing: float,
    wall: float,
) -> TurboPlan:
    """Assemble a TurboPlan; wall range from the table row, else +/-15%."""
    wall_range = row.wall_range_s if row is not None else (0.85 * wall, 1.15 * wall)
    return TurboPlan(
        cap_gb=cap_gb,
        n_cells=n_cells,
        regime=regime,
        box_chunks=c,
        descent_bands=b,
        workers=workers,
        tile_cache_tiles=t,
        val_sup_memmap=val_sup_memmap,
        zstd_ctx_bytes=workers * int(M_Z_BYTES),
        est_peak_bytes=est,
        budget_bytes=budget,
        sizing_bytes=sizing,
        wall_s=wall,
        wall_range_s=wall_range,
    )


# ---------------------------------------------------------------------------
# Cap resolution (auto-detect + presets, 3.6.1)
# ---------------------------------------------------------------------------


def _read_cgroup_max_gb() -> float:
    """Read the effective cgroup memory limit in GB for this process.

    Walks the process's own cgroup up to the root and returns the smallest
    finite memory.max found: a slice limit above the service cgroup (e.g.
    user-1000.slice) binds even when the service cgroup itself is 'max'.
    Falls back to the fixed v2/v1 paths when /proc/self/cgroup is
    unavailable (old kernels, some containers). 'max', missing, or
    unreadable everywhere -> inf.

    """
    limit = math.inf
    try:
        rel = ""
        for line in Path(PROC_SELF_CGROUP).read_text(encoding="utf-8").splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[0] == "0":  # cgroup v2 unified hierarchy
                rel = parts[2].strip("/")
                break
        root = Path(CGROUP_ROOT)
        cur = root / rel if rel else root
        while True:
            try:
                raw = (cur / "memory.max").read_text(encoding="utf-8").strip()
            except OSError:
                raw = "max"
            if raw not in ("max", ""):
                with contextlib.suppress(ValueError):
                    limit = min(limit, int(raw) / GB)
            if cur in (root, cur.parent):
                break
            cur = cur.parent
    except OSError:
        pass
    for path_str in (CGROUP_MEMORY_MAX, CGROUP_V1_LIMIT):
        try:
            raw = Path(path_str).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if raw == "max":
            continue
        try:
            limit = min(limit, int(raw) / GB)
        except ValueError:
            continue
    return limit


def _read_physical_gb() -> float:
    """Read physical RAM in GB from /proc/meminfo MemTotal (kB).

    Raises:
        ValueError: If MemTotal is missing.

    """
    with Path(MEMINFO_PATH).open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                return float(line.split()[1]) * 1024.0 / GB
    msg = "MemTotal not found in /proc/meminfo"
    raise ValueError(msg)


def detect_cap_gb(physical_gb: float | None = None, cgroup_gb: float | None = None) -> tuple[float, str]:
    """Compute the auto cap: min(physical, cgroup memory.max) - 8 GB, floor 8 GB.

    Supersedes the 3.1 auto form (min x 0.85): one formula for both
    spellings (--ram-gb alias and --max-ram auto), so the alias cannot
    carry the old default silently. The 8 GB headroom is fixed, not a
    percentage (co-tenant anon, OS/kernel, python/numpy/GDAL runtime,
    CSV + points ingest transient, allocator + page-cache floor); it must
    not double-count the 0.85 planning factor.

    Args:
        physical_gb: Physical RAM in GB; read from /proc/meminfo if None.
        cgroup_gb: cgroup memory.max in GB; read if None (inf = unlimited).

    Returns:
        (cap_gb, source) tuple; source is printable for deploy visibility.

    """
    if physical_gb is None:
        physical_gb = _read_physical_gb()
    if cgroup_gb is None:
        cgroup_gb = _read_cgroup_max_gb()
    cap = max(min(physical_gb, cgroup_gb) - AUTO_HEADROOM_GB, CAP_FLOOR_GB)
    cgroup_str = "unlimited" if math.isinf(cgroup_gb) else f"{cgroup_gb:.1f} GB"
    source = f"auto min(physical {physical_gb:.1f} GB, cgroup {cgroup_str}) - {AUTO_HEADROOM_GB:g} GB headroom"
    return cap, source


def resolve_cap(
    preset_gb: float | None,
    physical_gb: float | None = None,
    cgroup_gb: float | None = None,
) -> tuple[float, str]:
    """Resolve the RAM cap C from an explicit preset or auto-detect (3.6.1).

    Precedence: --max-ram > --turbo preset > auto. Clamp rule: a preset
    >= physical RAM means whole box, so C = physical - 8 GB (keep OS
    headroom); below physical the preset stands as given. Floor
    C = max(C, 8).

    Args:
        preset_gb: Explicit --max-ram or --turbo preset in GB; None = auto.
        physical_gb: Physical RAM in GB, for the clamp and auto; read if None.
        cgroup_gb: cgroup memory.max in GB, for the auto path; read if None.

    Returns:
        (cap_gb, source) tuple; source is printable for deploy visibility.

    """
    if preset_gb is None:
        return detect_cap_gb(physical_gb, cgroup_gb)
    cap = preset_gb
    if physical_gb is not None and preset_gb >= physical_gb:
        cap = physical_gb - AUTO_HEADROOM_GB
        source = (
            f"preset {preset_gb:g} GB >= physical {physical_gb:g} GB; clamped to physical - {AUTO_HEADROOM_GB:g} GB"
        )
    else:
        source = f"preset {preset_gb:g} GB"
    return max(cap, CAP_FLOOR_GB), source


# ---------------------------------------------------------------------------
# Pre-check (3.5, regime-aware per 3.6.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrecheckDecision:
    """Outcome of the regime-aware pre-check (3.5, superseded by 3.6.2)."""

    path: str  # "shared" | "per_box"
    regime: str  # selected regime: "A" | "B" | "C" | "per_box"
    est_peak_gb: float  # est peak used in the gate/message (B0 per 3.6.2)
    budget_gb: float  # pre-check budget 0.85 * C
    exit_code: int  # 0, or 3 when strict and the shared path is infeasible
    message: str | None  # [warn]/error line, None when clean
    summary: str  # cap + budgets + plan used, printable for deploy visibility


def precheck(plan: TurboPlan, *, strict: bool = False, cap_source: str = "auto") -> PrecheckDecision:
    """Gate the selected regime's est peak against the 0.85*C budget.

    A feasible regime fits the budget by construction (3.6.2), so the
    gate's job is the regime selection and the per-box fallback. A
    selected regime below A (full in-RAM) logs a [warn] naming the
    regime-A field estimate, per A9. Non-strict always continues (exit
    0); strict exits 3 when only per-box fits (A9 pin: 16 GB full-AU).

    Args:
        plan: TurboPlan from :func:`plan`.
        strict: True for --turbo-strict (hard-fail instead of fallback).
        cap_source: Human-readable cap source from :func:`resolve_cap`.

    Returns:
        PrecheckDecision with exit_code 0 or 3 and the [warn]/error line.

    """
    budget_gb = plan.budget_bytes / GB
    summary = (
        f"--turbo: cap {plan.cap_gb:g} GB ({cap_source}); budget 0.85*C = {budget_gb:.1f} GB; "
        f"sizing Bt = {plan.sizing_bytes / GB:.1f} GB; regime {plan.regime} "
        f"(box_chunks={plan.box_chunks}, descent_bands={plan.descent_bands}, workers={plan.workers}, "
        f"tile_cache={plan.tile_cache_tiles}, zstd_ctx={plan.zstd_ctx_bytes / 1e6:.0f} MB)"
    )
    shared_ok = plan.regime in ("A", "B", "C") and plan.est_peak_bytes <= plan.budget_bytes * (1.0 + 1e-12) + 4096.0
    if shared_ok:
        est_gb = plan.est_peak_bytes / GB
        if plan.regime == "A":
            return PrecheckDecision("shared", "A", est_gb, budget_gb, 0, None, summary)
        est_a_gb = plan.n_cells * PHASE_A_B_PER_CELL / GB
        message = (
            f"[warn] --turbo: est peak {est_a_gb:.1f} GB (regime A, full in-RAM) "
            f"> budget {budget_gb:.1f} GB; using regime {plan.regime} shared"
        )
        return PrecheckDecision("shared", plan.regime, est_gb, budget_gb, 0, message, summary)

    # Per-box fallback: the natural shared estimate (regime A, field-only,
    # 2.2 base) is what overran; the per-box plan itself peaks ~4-6 GB.
    est_a_gb = plan.n_cells * PHASE_A_B_PER_CELL / GB
    if strict:
        message = f"error: --turbo est peak {est_a_gb:.1f} GB > budget {budget_gb:.1f} GB; shared path infeasible"
        return PrecheckDecision("per_box", "per_box", est_a_gb, budget_gb, 3, message, summary)
    message = (
        f"[warn] --turbo: est peak {est_a_gb:.1f} GB > budget {budget_gb:.1f} GB; using per-box (turbo warp/write)"
    )
    return PrecheckDecision("per_box", "per_box", est_a_gb, budget_gb, 0, message, summary)


# ---------------------------------------------------------------------------
# Runtime RSS backstop (3.5, extended by 3.6.4)
# ---------------------------------------------------------------------------


class RssBackstop:
    """Mid-run RSS hook for the shared path: a single [warn] at budget x 0.95.

    Call :meth:`check` from phase loops. The first time resource.getrusage
    RSS exceeds budget x 0.95, log one [warn] (performance only, the run
    completes; the pre-check is the real gate). The 3.6.4 pre-emptive
    step-down at 0.90 is a follow-up issue, not wired here.
    """

    def __init__(self, budget_gb: float) -> None:
        """Create a backstop that warns once when RSS crosses budget x 0.95.

        Args:
            budget_gb: Pre-check budget in GB (0.85 * C).

        """
        self._limit_bytes = budget_gb * RSS_BACKSTOP_FRAC * GB
        self._warned = False

    @property
    def limit_bytes(self) -> float:
        """RSS limit in bytes (budget x 0.95)."""
        return self._limit_bytes

    def check(self) -> bool:
        """Poll once; True exactly once, the first time the limit is crossed."""
        if self._warned:
            return False
        rss_bytes = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024.0
        if rss_bytes <= self._limit_bytes:
            return False
        self._warned = True
        log.warning(
            "[warn] turbo: rss %.2f GB > budget x %.2f = %.2f GB; performance only, run completes",
            rss_bytes / GB,
            RSS_BACKSTOP_FRAC,
            self._limit_bytes / GB,
        )
        return True
