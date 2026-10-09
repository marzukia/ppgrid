"""ppgrid — pull-push scattered-data interpolation pipeline + CLI.

Turns scattered geolocated point values into a continent-scale, gapless,
capped raster surface, in minutes, on a single machine, with no GPU.
"""

from __future__ import annotations

import argparse
import difflib
import errno
import io
import json
import logging
import math
import os
import shutil
import struct
import subprocess  # ruff: ignore[suspicious-subprocess-import] — git rev-parse for the --json summary
import sys
import threading
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from itertools import starmap
from pathlib import Path
from typing import Any, NoReturn, Self

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from pyproj.exceptions import CRSError
from rasterio._io import MemoryDataset
from rasterio.crs import CRS
from rasterio.errors import CRSError as RasterioCRSError
from rasterio.errors import NotGeoreferencedWarning, RasterioIOError
from rasterio.transform import Affine, from_bounds, from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

from . import __version__, _prof, turbop, zstdmt
from ._tempfiles import _open_fresh_memmap
from .calibrate import (
    M_PER_KM,
    PERCENTILE_MAX,
    PercentileTransform,
    calibrate_fill_cap,
    choose_transform,
    make_transform,
    transforms,
    validate_calibration,
)
from .pullpush import (
    _descent_banded,
    _nohuge,
    _pull_push_descent,
    bin_points,
    bin_points_banded,
    box_count,
    box_count_banded,
    box_count_mt,
    downsample_sum,
    pull_push,
)

# rasterio's MemoryDataset.__init__ (used by reproject for ndarray buffers)
# always warns NotGeoreferencedWarning on the initial transform read of the
# MEM::: dataset, then sets the real transform immediately after. It normally
# hides the warning in a nested warnings.catch_warnings, but CPython warning
# filters are process-global, not thread-safe: with concurrent warps (S3.3)
# another thread can restore a stale filter snapshot and leak the benign
# warning. It carries no information here, so ignore the category.
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# Module logger (pattern shared with turbop.py). The CLI configures the
# `ppgrid` logger (level + compact stderr format) in main(); library users
# who never call main() see at most WARNING+ via the lastResort handler.
log = logging.getLogger("ppgrid.pipeline")

_LOG_FORMAT = "%(levelname).1s %(name)s: %(message)s"
_LOG_LEVELS: tuple[str, ...] = ("error", "warning", "info", "debug")

_EPILOG = """\
examples:
  # Basic run: 10 km cells, auto transform and fill cap, default 4 workers
  ppgrid data/melb_houses.csv -o out/melb_10km --res 10000 --value-col price

  # Custom transform, CRS and DN scale: log10 values, WGS84 output, DN = 10*percentile
  ppgrid data/all_equakes.csv -o out/eq --transform log10 --out-crs 4326 --scale 10 --cap-km 20 --value-col mag

  # Turbo / low-RAM: 32 GB budget, strict (exit 3 if infeasible).
  # au_gcc_sparse.csv is the production dataset (not in the repo); any sparse
  # CSV with lng/lat/value columns works.
  ppgrid data/au_gcc_sparse.csv -o out/au --turbo 32 --turbo-strict --workers 8

logging:
  Progress goes to stderr (stdout stays clean). Default level is info (one line
  per phase + a final summary). --verbose (or --log-level debug, or
  LOGGING=verbose) adds a per-phase wall-time
  breakdown and volumetric detail (rows read, NaN/inf dropped, cells per level,
  output bytes raw vs compressed, resolved plan).

exit codes:
  0  ok (also: `ppgrid help` and `--plan`)
  1  pipeline / I/O error (bad output path, missing input file, disk full)
  2  validation error (bad flag value, missing input column, out dir not
     empty without --force)
  3  --turbo-strict: shared path infeasible under the RAM budget

Run `ppgrid help` to print this text at any time."""


class _WallPhase:
    """Context manager: accumulate wall seconds for `label` into `times`.

    Feeds the DEBUG per-phase breakdown in Pipeline.run() (time.perf_counter
    only; no I/O, and the logged lines are level-gated, so the default INFO
    run pays nothing but two counter reads per phase).
    """

    __slots__ = ("label", "t0", "times")

    def __init__(self, label: str, times: dict[str, float]) -> None:
        self.label = label
        self.times = times
        self.t0 = 0.0

    def __enter__(self) -> Self:
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.times[self.label] = self.times.get(self.label, 0.0) + (time.perf_counter() - self.t0)
        return False


WORK_CRS: int = 6933  # Wagner VII — global equal-area, metres are true
SRC_CRS: int = 4326  # WGS 84 lon/lat (default input)
OUT_CRS: int = 3857  # Web Mercator (default output)
NODATA: int = -32768
INT16_MAX: int = 32767

# Value band: DN = percentile * scale, percentile in [0, PERCENTILE_MAX]
# (PERCENTILE_MAX is defined in calibrate.py so both modules share it).
DEFAULT_SCALE: float = 100.0  # DN per percentile point -> DN 0..10000 at default

# Support band encode/decode pair — the two MUST stay in sync.
# Encode: DN is log2 of the support in km (floored at SUPPORT_KM_FLOOR) times
# SUPPORT_LOG2_SCALE, clipped to +/-SUPPORT_DN_MAX.
# Decode: support_km is 2 to the power DN / SUPPORT_LOG2_SCALE.
SUPPORT_LOG2_SCALE: int = 8
SUPPORT_DN_MAX: int = 32000
SUPPORT_KM_FLOOR: float = 1e-3  # keeps log2 finite for sub-mm support

# Fill cap (km) used when no calibrated cap is available (skip-calibration,
# or a calibration file without cap_km). Distinct from FILL_CAP_DEFAULT_KM
# (calibrate.py): that one is the fallback the blocked-CV fill-cap search
# returns when no bin clears the skill bar.
FILL_FALLBACK_CAP_KM: float = 64.0

# GeoTIFF tile size. Must stay 512: _reproject_band flushes TILE_PX blocks
# in raster-scan order to reproduce the on-disk layout of the 512px baseline
# (byte-identity of the reprojected output).
TILE_PX: int = 512

# CLI/API defaults — single source for the Pipeline constructor and argparse
# so the two can't drift.
DEFAULT_RES: float = 500.0
DEFAULT_WORKERS: int = 4
DEFAULT_SATURATION: float = 1.0
DEFAULT_BLOCK: int = 2048
CALIB_MAX_POINTS_DEFAULT: int = 2_000_000

# Point memmap bands (written by Pipeline.grid, read by _block_points):
# band PTS_X = projected x (m), PTS_Y = projected y (m), PTS_TV = transformed value.
PTS_X: int = 0
PTS_Y: int = 1
PTS_TV: int = 2
PTS_NBANDS: int = 3

# Shared full-field path: bin all points once, build the pyramid once, run
# the descent once over the whole padded grid, and let every block slice
# the result. Used only while the padded grid is small enough that one full
# descent beats per-box descents. Measured on hydrogen (12.88 GB cgroup):
# at ~2e9 cells the banded memmap descent is ~20x slower than the per-box
# path (page-cache writeback throttling), so large grids fall back to
# per-box. 2.5e8 covers the 1e8-2.5e8 band where the shared memmap path
# still wins (wide1m: 32s vs 46s per-box).
_SHARED_MAX_CELLS = int(2.5e8)

# Below this many cells the full field fits in RAM comfortably; larger grids
# write the finest descent level to memmap in row bands.
_SHARED_MEMMAP_CELLS = int(1e8)

# Classic TIFF address space (S3.4 parallel write): the assembled file is
# built on a classic-TIFF reference head; a total size beyond 0xFFFFFFFF
# needs the BIGTIFF format, so the band falls back to the serial write.
_TIFF_CLASSIC_MAX_BYTES: int = 0xFFFFFFFF

# Temp files created in the output dir by a run; all removed on success and
# on the failure path (issue #15). The list is generated from the write
# sites, not hand-maintained (audit #83): _descent_banded (pullpush) writes
# _val_lvl{k}.npy / _sup_lvl{k}.npy for every intermediate level
# k = 1 .. levels-1, and the old literal kept only lvl1, so any run with
# levels >= 3 leaked _val_lvl2.npy and up.
# 64 covers cells up to 2**64 (levels = ceil(log2(cap_cells)); the shared
# cell cap is 2.5e8 non-turbo, budget-derived in turbo).
_RUN_LEVELS_MAX: int = 64


def _run_temp_names(n_levels: int = _RUN_LEVELS_MAX) -> tuple[str, ...]:
    """Names of every per-run temp file for a grid of up to n_levels."""
    names = [
        "_points.npy",
        "_s0.npy",
        "_c0.npy",
        "_near.npy",
        "_val_full.npy",
        "_sup_full.npy",
        "_val_dn.npy",
        "_sup_dn.npy",
    ]
    names.extend(f"_{band}_lvl{k}.npy" for k in range(1, n_levels) for band in ("val", "sup"))
    # Write-phase staging files (issue #40): the final rasters are written
    # to these names and os.replace()'d over value.tif / support_km.tif
    # only on success, so a mid-write crash never truncates the finals.
    names.extend(("_value_tmp.tif", "_support_tmp.tif", "value.tif.tmp", "support_km.tif.tmp"))
    return tuple(names)


_RUN_TEMP_FILES = _run_temp_names()

# Ingest parallelism (S3.5): minimum data rows per CSV chunk for the process
# pool. Below this the pool overhead exceeds the parse time, so ingest stays
# on the single serial parse.
_INGEST_MIN_CHUNK_ROWS = 100_000


@dataclass
class _WorkerConfig:
    """Configuration + resolved state for the worker threads.

    The pipeline fields are set by Pipeline.grid(); the runtime fields
    (pts/tf/pct) are resolved once per run by Pipeline._write_rasters; the
    shared fields hold the full-field results when the shared path is
    active. The object is stored in _CTX as-is (no per-field copy), so a
    new field needs exactly one edit here.
    """

    pts_path: str
    transform_state: dict[str, Any]
    pct_quantiles: list[float]
    starts: list[int]
    nbx: int
    nby: int
    bsize: int
    halo: int
    res: float
    levels: int
    cap_km: float
    x0: float
    y0: float
    nx: int
    ny: int
    nx_padded: int
    ny_padded: int
    sat: float
    scale: float
    pct_step: float | None
    # Resolved at run start by Pipeline._write_rasters.
    pts: np.ndarray | None = None
    tf: Any | None = None
    pct: PercentileTransform | None = None
    # Shared full-field results (set by Pipeline._prepare_shared).
    shared: bool = False
    val_full: np.ndarray | None = None
    sup_full: np.ndarray | None = None
    near_full: np.ndarray | None = None


# Worker state, shared by all worker threads in this process (one
# ThreadPoolExecutor, one object — not local to each process). Holds the
# _WorkerConfig as-is under "cfg"; workers read attributes off it.
_CTX: dict[str, Any] = {}


def _die(msg: str) -> NoReturn:
    """Print a clean CLI error (no traceback) and exit 1.

    Args:
        msg: Error message, printed as 'error: <msg>'.

    Raises:
        SystemExit: Always, with status code 1.

    """
    print(f"error: {msg}", file=sys.stderr)  # ruff: ignore[print]
    raise SystemExit(1)


def _check_default_transform(
    dst_transform: Any,
    dst_width: int,
    dst_height: int,
    work_crs: int,
    out_crs: int,
) -> None:
    """Validate a calculate_default_transform result (issue #74, P0).

    When the work-CRS grid extent crosses the output projection's domain
    (e.g. the Wagner VII pole at y = +/-7,342,230 m with a Web Mercator
    target), GDAL returns a NaN transform and width = height = -2147483648
    (INT32_MIN); every writer then dies at rasterio.open() with the cryptic
    "Attempt to create -2147483648x-2147483648 dataset is illegal".
    Name the cause instead.

    Args:
        dst_transform: Affine returned by calculate_default_transform.
        dst_width: Width returned by calculate_default_transform.
        dst_height: Height returned by calculate_default_transform.
        work_crs: Work CRS EPSG code (for the message).
        out_crs: Output CRS EPSG code (for the message).

    Raises:
        ValueError: If the transform is non-finite or the raster size is
            non-positive. Mapped to exit 2 by _map_pipeline_errors, the
            user-data-derived failure code.

    """
    if dst_width > 0 and dst_height > 0 and all(math.isfinite(v) for v in dst_transform):
        return
    msg = (
        f"work-CRS grid extent (EPSG:{work_crs}) exceeds the output projection's domain "
        f"(EPSG:{out_crs}): the derived output transform is not finite "
        f"(raster size {dst_width}x{dst_height}). "
        "Retry with --out-crs 4326, a coarser --res, or a clipped input extent."
    )
    raise ValueError(msg)


def _warn_stale_outputs(out: Path, published: set[str] | None = None) -> None:
    """Warn when a failed run leaves an earlier run's rasters behind (issues #40, #77).

    On a non-zero exit, if value.tif / support_km.tif exist in the output dir
    and were not published by this run, they are from an earlier run. A
    post-publish failure must not report the just-written rasters as untouched.
    The overwrite guard exits on its own message, so it does not call this.

    Args:
        out: Output directory to check.
        published: Final-output names published by this run (None = none).

    """
    if not out.is_dir():
        return
    published = published or set()
    existing = [name for name in ("value.tif", "support_km.tif") if (out / name).is_file() and name not in published]
    if not existing:
        return
    print(  # ruff: ignore[print]
        f"warning: this run failed before writing outputs; {', '.join(existing)} in {out} "
        "are from an earlier run and were left untouched",
        file=sys.stderr,
    )


def _staging_fd(path: Path) -> int:
    """Create `path` as a fresh 0600 regular file and return an open write fd.

    Symlinks are replaced, never followed (issues #15, #77): the path is
    unlinked, then recreated with O_CREAT | O_EXCL | O_NOFOLLOW — a symlink
    (or any file) planted in that window makes os.open fail (ELOOP / EEXIST)
    instead of being followed, and one retry resolves it.

    Args:
        path: Staging path in the output dir.

    Returns:
        Open fd on the fresh regular file.

    Raises:
        OSError: If the path still cannot be created as a regular file
            after one retry.

    """
    path.unlink(missing_ok=True)
    try:
        return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except OSError as e:
        if e.errno not in (errno.EEXIST, errno.ELOOP):
            raise
        # Planted in the unlink->open window: replace it once, try again.
        path.unlink(missing_ok=True)
        return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)


def _open_tiff_staging(path: Path, **profile: Any) -> rasterio.DatasetBase:
    """Open `path` for GTiff write without ever following a symlink (issues #15, #77).

    The file is created fresh via _staging_fd (O_EXCL | O_NOFOLLOW, 0600),
    then the path is handed to rasterio.open and re-checked: a symlink
    planted in the create->open window is caught (close, replace, retry
    once) instead of having its target overwritten. Mirrors the
    _open_fresh_memmap guard (issue #45).

    Args:
        path: Staging path in the output dir.
        **profile: rasterio.open write profile (driver, dtype, width, ...).

    Returns:
        Open rasterio dataset; use under `with`.

    Raises:
        OSError: If the path is still a symlink after one retry.

    """
    for _attempt in (1, 2):
        fd = _staging_fd(path)
        os.close(fd)
        ds = rasterio.open(path, "w", **profile)
        if not path.is_symlink():
            return ds
        ds.close()
        path.unlink(missing_ok=True)
    msg = f"staging path is a symlink after retry: {path}"
    raise OSError(msg)


def _pos_float(text: str) -> float:
    """Argparse type: a finite, strictly positive float (issue #13).

    Rejects nan/inf/negative/zero/non-numeric at parse time so the
    float64 pipeline math and the int16 DN range can never see them
    (silent all-0 rasters, inverted SAT windows, deep OverflowErrors).

    Args:
        text: Raw CLI argument string.

    Returns:
        The parsed value, finite and > 0.

    Raises:
        argparse.ArgumentTypeError: If text is not a finite positive number.

    """
    try:
        v = float(text)
    except ValueError as exc:
        msg = f"invalid number: {text!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if not math.isfinite(v) or v <= 0:
        msg = f"must be a finite positive number: {text!r}"
        raise argparse.ArgumentTypeError(msg) from None
    return v


def _cap_km_arg(text: str) -> float | str:
    """Argparse type: 'auto' or a finite positive fill cap in km (issue #13).

    Args:
        text: Raw CLI argument string.

    Returns:
        'auto' unchanged, else the parsed finite positive cap.

    """
    if text == "auto":
        return "auto"
    return _pos_float(text)


def _crs_arg(text: str) -> int:
    """Argparse type: an EPSG code, bare int or 'EPSG:<code>' string (#58).

    Args:
        text: Raw CLI argument string (e.g. '4326' or 'EPSG:4326').

    Returns:
        The parsed EPSG code.

    Raises:
        argparse.ArgumentTypeError: If text is not a bare int or EPSG-prefixed int.

    """
    t = text.strip()
    if t.upper().startswith("EPSG:"):
        t = t[len("EPSG:") :]
    try:
        return int(t)
    except ValueError as exc:
        msg = f"invalid EPSG code: {text!r} (use e.g. 4326 or 'EPSG:4326')"
        raise argparse.ArgumentTypeError(msg) from exc


def _workers_arg(text: str) -> int:
    """Argparse type: a worker count, or 'auto' = os.cpu_count() (#59).

    Args:
        text: Raw CLI argument string.

    Returns:
        The parsed count ('auto' resolves to the logical CPU count).

    Raises:
        argparse.ArgumentTypeError: If text is not an int or 'auto'.

    """
    if text.lower() == "auto":
        return os.cpu_count() or 1
    try:
        return int(text)
    except ValueError as exc:
        msg = f"invalid worker count: {text!r} (use a positive integer or 'auto')"
        raise argparse.ArgumentTypeError(msg) from exc


def _neighbour_block_ids(bx: int, by: int, nbx: int, nby: int) -> list[int]:
    """Row-major ids of the 3x3 block neighbourhood, clamped to the grid."""
    return [
        jx * nby + jy
        for jx in range(max(0, bx - 1), min(nbx, bx + 2))
        for jy in range(max(0, by - 1), min(nby, by + 2))
    ]


def _block_bounds(bx: int, by: int, bsize: int, nx: int, ny: int) -> tuple[int, int, int, int]:
    """(i0, j0, i1, j1) cell bounds of output block (bx, by), clamped to the grid."""
    i0, j0 = bx * bsize, by * bsize
    return i0, j0, min(i0 + bsize, nx), min(j0 + bsize, ny)


def _block_window(bx: int, by: int, cfg: _WorkerConfig) -> Window:
    """Rasterio Window of block (bx, by) in the output raster (grid [x, y] -> raster [row, col])."""
    i0, j0, i1, j1 = _block_bounds(bx, by, cfg.bsize, cfg.nx, cfg.ny)
    return Window(i0, cfg.ny - j1, i1 - i0, j1 - j0)


def _snapped_box(
    i0: int,
    j0: int,
    i1: int,
    j1: int,
    halo: int,
    step: int,
    nx_padded: int,
    ny_padded: int,
) -> tuple[int, int, int, int]:
    """Snap the halo box around a block to pyramid-step multiples.

    Exact alignment: the tent filter taps land on grid cells, so per-box and
    shared-field descents agree everywhere outside the halo.
    """
    hi0 = max(0, ((i0 - halo) // step) * step)
    hj0 = max(0, ((j0 - halo) // step) * step)
    hi1 = min(nx_padded, -(-(i1 + halo) // step) * step)
    hj1 = min(ny_padded, -(-(j1 + halo) // step) * step)
    return hi0, hj0, hi1, hj1


def _block_points(cfg: _WorkerConfig, bx: int, by: int) -> np.ndarray:
    """Gather points from the 3x3 block neighbourhood.

    Returns views of the memmap; np.concatenate below copies once.

    Returns:
        Stacked point array of shape (PTS_NBANDS, n_points).

    """
    pts = cfg.pts
    starts = cfg.starts
    out: list[np.ndarray] = []
    for bid in _neighbour_block_ids(bx, by, cfg.nbx, cfg.nby):
        lo, hi = starts[bid], starts[bid + 1]
        if hi > lo:
            out.append(pts[:, lo:hi])
    return np.concatenate(out, axis=1) if out else np.empty((PTS_NBANDS, 0))


# Thread-local f32 scratch for _quantize's support chain (see docstring
# there). The int16 outputs are owned by the caller - returning scratch
# raced the worker pool's ex.map prefetch (worker overwrote the buffer
# before the main thread finished vd.write), corrupting blocks randomly.
_QUANT_TLS = threading.local()


def _quantize(
    cfg: _WorkerConfig,
    v_out: np.ndarray,
    r_out: np.ndarray,
    near_out: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform interpolated (value, support) to int16 percentile DN grids.

    Shared by the per-block and shared-field paths so the rounding is
    structurally identical. `r_out` is the support field in METRES (the raw
    sup slice); the /M_PER_KM division runs into the per-thread f32 scratch
    (issue #39 pass 3: the old out-of-place division was a fresh 16 MB f32
    array per block).

    Returns:
        Tuple of (value, support) int16 arrays with NODATA where not near data.
        The arrays are owned by the caller (fresh each call) - the f32 work
        buffer is per-thread scratch (issue #39 pass 3: returning scratch
        raced the worker pool's prefetch and corrupted ~10% of blocks).

    """
    # Transform: interpolate space -> raw -> percentile
    tf = cfg.tf
    pct = cfg.pct
    s = _QUANT_TLS.__dict__.get("q")
    if s is None or s[0].shape != v_out.shape:
        s = (
            np.empty(v_out.shape, np.float32),
            np.empty(v_out.shape, np.float64),
        )
        _nohuge(s[0], s[1])
        _QUANT_TLS.q = s
    w, x64 = s
    v_in = tf.inv(v_out)
    # np.interp's C core casts a non-f64 x to a FRESH f64 array (32 MB per
    # block at full scale). Cast into the per-thread scratch instead: an
    # exact f32 -> f64 conversion, same bits, no allocation (issue #39 pass
    # 3). If v_in is already f64 (non-identity transform), reuse it as-is.
    if v_in.dtype != np.float64:
        np.copyto(x64, v_in)
        v_in = x64
    pv = pct.fwd(v_in)
    if cfg.pct_step is not None:
        # Round to the nearest step (e.g. 5 -> 90/95/100), clamp to 0-100.
        # In-place: the out-of-place chain allocated 3 fresh full-size f64
        # arrays per block; at full-AU scale that was the dominant fault
        # source of write.dnfill. Same IEEE ops in the same order, so the
        # bits are identical.
        np.divide(pv, cfg.pct_step, out=pv)
        np.round(pv, out=pv)
        np.multiply(pv, cfg.pct_step, out=pv)
        np.clip(pv, 0.0, PERCENTILE_MAX, out=pv)
    vq = np.empty(v_out.shape, np.int16)
    rq = np.empty(v_out.shape, np.int16)
    _nohuge(vq, rq)
    # Integer-valued intermediates (round first, exact int16 cast second), so
    # fill + copyto(where=) match the old where-then-astype bits (np.where has
    # no out=; copyto uses the same standard f64/f32 -> int16 cast).
    np.multiply(pv, cfg.scale, out=pv)
    np.round(pv, out=pv)
    vq.fill(NODATA)
    np.copyto(vq, pv, where=near_out, casting="unsafe")
    np.divide(r_out, M_PER_KM, out=w)
    np.maximum(w, SUPPORT_KM_FLOOR, out=w)
    np.log2(w, out=w)
    np.multiply(w, SUPPORT_LOG2_SCALE, out=w)
    np.round(w, out=w)
    np.clip(w, -SUPPORT_DN_MAX, SUPPORT_DN_MAX, out=w)
    rq.fill(NODATA)
    np.copyto(rq, w, where=near_out, casting="unsafe")
    return vq, rq


def _block_shared(cfg: _WorkerConfig, bx: int, by: int) -> tuple[np.ndarray, np.ndarray]:
    """Shared-field block core: slice the prebuilt full-grid field.

    Bit-identical to the per-block path: the full-grid descent differs from a
    per-box descent only within one pyramid step of each box edge (tent-filter
    edge rules), and the halo is a full step, so the block region is clean.
    The 3x3 point gather is exactly the set of points inside the box
    (enforced by Pipeline._prepare_shared).

    Returns:
        Tuple of (value, support) int16 DN grids.

    """
    bsize = cfg.bsize
    i0, j0, i1, j1 = _block_bounds(bx, by, bsize, cfg.nx, cfg.ny)
    sl = (slice(i0, i1), slice(j0, j1))
    return _quantize(cfg, cfg.val_full[sl], cfg.sup_full[sl], cfg.near_full[sl])


def _process_block(
    args: tuple[int, int],
) -> tuple[int, int, tuple[np.ndarray, np.ndarray] | None]:
    """Interpolate a single output block using pull-push.

    Returns:
        Tuple of block coords and the (value, support) DN grids, or None when
        the block neighbourhood has no points (written as nodata).

    """
    bx, by = args
    cfg = _CTX["cfg"]
    if cfg.shared:
        return bx, by, _block_shared(cfg, bx, by)
    i0, j0, i1, j1 = _block_bounds(bx, by, cfg.bsize, cfg.nx, cfg.ny)
    hi0, hj0, hi1, hj1 = _snapped_box(i0, j0, i1, j1, cfg.halo, 1 << cfg.levels, cfg.nx_padded, cfg.ny_padded)

    sel = _block_points(cfg, bx, by)
    if sel.shape[1] == 0:
        return bx, by, None

    ix = ((sel[PTS_X] - cfg.x0) // cfg.res).astype(np.int64)
    iy = ((sel[PTS_Y] - cfg.y0) // cfg.res).astype(np.int64)
    m = (ix >= hi0) & (ix < hi1) & (iy >= hj0) & (iy < hj1)
    if not m.any():
        return bx, by, None

    iloc = ix[m] - hi0
    jloc = iy[m] - hj0
    tv = sel[PTS_TV][m]

    s, c_grid = bin_points(iloc, jloc, tv, hi1 - hi0, hj1 - hj0)
    val, sup = pull_push(s, c_grid, cfg.res, cfg.levels, saturation=cfg.sat)

    # Mask from exact radius query
    cap_cells = round(cfg.cap_km * M_PER_KM / cfg.res)
    near = box_count(c_grid, cap_cells) > 0

    # Extract output window
    a0 = i0 - hi0
    b0 = j0 - hj0
    sl = (slice(a0, a0 + (i1 - i0)), slice(b0, b0 + (j1 - j0)))
    return bx, by, _quantize(cfg, val[sl], sup[sl], near[sl])


def _block_neighbourhood_nonempty(
    bx: int,
    by: int,
    starts: np.ndarray,
    nbx: int,
    nby: int,
) -> bool:
    """Return True if any block in the 3x3 neighbourhood holds at least one point."""
    return any(starts[bid + 1] > starts[bid] for bid in _neighbour_block_ids(bx, by, nbx, nby))


def _warp_dst_tile(
    src_path: str,
    i0: int,
    j0: int,
    band_h: int,
    warp_tile: int,
    dst_width: int,
    dst_transform: Any,
    dst_crs: str,
) -> tuple[int, np.ndarray]:
    """Nearest-neighbour warp of one coarse 2048px dst tile.

    Pure per-pixel function of the dst grid with no state shared between
    tiles, so the tile can be warped from any thread and the result is
    identical to the serial loop. The source is opened per tile: GDAL
    dataset handles are not thread-safe, and opening is cheap relative to
    the warp (S3.3).

    Returns:
        Tuple of (tile column origin i0, warped int16 tile buffer).

    """
    band_w = min(warp_tile, dst_width - i0)
    with rasterio.open(src_path) as src:
        w = Window(i0, j0, band_w, band_h)
        w_bounds = rasterio.windows.bounds(w, dst_transform)
        local_dst_transform = from_bounds(*w_bounds, band_w, band_h)
        buf = np.zeros((band_h, band_w), dtype=np.int16)
        _nohuge(buf)
        reproject(
            rasterio.band(src, 1),
            buf,
            src_transform=src.transform,
            dst_transform=local_dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            nodata=NODATA,
        )
    return i0, buf


def _warp_dst_tile_arr(
    src_arr: np.ndarray,
    src_transform: Any,
    src_crs: str,
    i0: int,
    j0: int,
    band_h: int,
    warp_tile: int,
    dst_width: int,
    dst_transform: Any,
    dst_crs: str,
    src_ds: Any | None = None,
) -> tuple[int, np.ndarray]:
    """Nearest-neighbour warp of one coarse 2048px dst tile.

    Sourced from an in-RAM work-CRS int16 array (S3.4a turbo write). Same
    per-tile math as _warp_dst_tile, but the source is the array itself
    (reproject wraps it in an internal MemoryDataset) instead of the
    work-CRS intermediate file. The array is read-only shared state;
    reproject builds a fresh dataset handle per call, so tiles are
    thread-safe.

    Args:
        src_arr: Work-CRS int16 array, shape (ny, nx), north-up raster order.
        src_transform: Geotransform of the work-CRS raster.
        src_crs: Work-CRS string.
        i0: Left column of the coarse tile in the dst raster.
        j0: Top row of the coarse tile in the dst raster.
        band_h: Height of the current warp row band (<= warp_tile).
        warp_tile: Width of the coarse tile.
        dst_width: Full dst width.
        dst_transform: Geotransform of the target raster.
        dst_crs: Target CRS.
        src_ds: Optional prewrapped MemoryDataset over src_arr (copy=False).
            Uses reproject's MultiBand tuple form, which consumes the
            dataset's GDAL handle directly; the ndarray form copies the
            whole band into a fresh MEM dataset on every call (3 GB per
            block at full-AU scale, issue #39 pass 3). Must stay alive for
            the duration of the warp.

    Returns:
        (i0, buf) - buf shape (band_h, band_w).

    """
    band_w = min(warp_tile, dst_width - i0)
    w = Window(i0, j0, band_w, band_h)
    w_bounds = rasterio.windows.bounds(w, dst_transform)
    local_dst_transform = from_bounds(*w_bounds, band_w, band_h)
    buf = np.zeros((band_h, band_w), dtype=np.int16)
    _nohuge(buf)
    src = src_arr if src_ds is None else (src_ds, 1, src_arr.dtype, src_arr.shape)
    reproject(
        src,
        buf,
        src_transform=src_transform,
        dst_transform=local_dst_transform,
        src_crs=src_crs,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        nodata=NODATA,
    )
    return i0, buf


def _reproject_core(
    make_warp: Callable[[int, int], Callable[[int], tuple[int, np.ndarray]]],
    dst_path: str,
    profile: dict[str, Any],
    dst_crs: str,
    dst_transform: Any,
    dst_width: int,
    dst_height: int,
    tags: dict[str, str],
    scales: tuple[float, ...] | None,
    offsets: tuple[float, ...] | None,
    n_threads: int = 1,
) -> None:
    """Warp one dst row band and flush it to the output GeoTIFF.

    Shared core for the file-source (_reproject_band) and array-source
    (_reproject_band_array) reprojections. make_warp(j0, band_h) returns
    the warp function for one 2048px dst row band: fn(i0) -> (i0, buf).
    Warp: the 2048px tiles of a row band are independent (pure per-pixel
    function of the dst grid), so n_threads > 1 overlaps them in a pool.
    Flush: TILE_PX blocks in serial raster-scan order, because the file
    bytes depend on the tile write order (GDAL GTiff appends tiles in
    write order).

    Args:
        make_warp: Builds the per-band warp function (source already bound).
        dst_path: Destination GeoTIFF path.
        profile: Base write profile (driver, compress, tiled, block size).
        dst_crs: Target CRS string.
        dst_transform: Geotransform of the target raster.
        dst_width: Target raster width.
        dst_height: Target raster height.
        tags: Dataset tags to copy onto the output (work-file parity).
        scales: Band scales to set, or None to leave unset.
        offsets: Band offsets to set, or None to leave unset.
        n_threads: Warp threads. 1 = fully serial (A8 path).

    """
    dst_profile = dict(
        profile,
        width=dst_width,
        height=dst_height,
        transform=dst_transform,
        crs=dst_crs,
    )
    with _open_tiff_staging(Path(dst_path), **dst_profile) as dst:
        dst.update_tags(**tags)
        if scales:
            dst.scales = scales
        if offsets:
            dst.offsets = offsets
        warp_tile = 2048
        for j0 in range(0, dst_height, warp_tile):
            band_h = min(warp_tile, dst_height - j0)
            warp_i0 = make_warp(j0, band_h)
            coarse: dict[int, np.ndarray] = {}
            if n_threads > 1:
                with ThreadPoolExecutor(max_workers=n_threads) as ex:
                    coarse.update(ex.map(warp_i0, range(0, dst_width, warp_tile)))
            else:
                for i0 in range(0, dst_width, warp_tile):
                    _, buf = warp_i0(i0)
                    coarse[i0] = buf
            # Flush TILE_PX blocks in raster-scan order (serial: the file
            # bytes depend on the tile write order).
            for j in range(j0, j0 + band_h, TILE_PX):
                w_h = min(TILE_PX, j0 + band_h - j)
                for i in range(0, dst_width, TILE_PX):
                    w_w = min(TILE_PX, dst_width - i)
                    i0 = (i // warp_tile) * warp_tile
                    sub = coarse[i0][j - j0 : j - j0 + w_h, i - i0 : i - i0 + w_w]
                    dst.write(sub, 1, window=Window(i, j, w_w, w_h))


def _footprint_has_task(
    w_bounds: tuple[float, float, float, float],
    dst_crs: Any,
    src_crs: Any,
    g_left: float,
    g_bottom: float,
    g_res_x: float,
    g_res_y: float,
    g_width: int,
    g_height: int,
    bsize: int,
    nby: int,
    task_bids: set[int],
) -> bool:
    """Return True when the source footprint of a dst window intersects any task block.

    The footprint is the dst window bounds transformed into source (work-CRS)
    georeferenced space via `transform_bounds` on a densified boundary, then
    converted to work-grid cell ranges rounded OUT (one extra cell of pad on
    every side). Raster row 0 is at the top edge (g_top), so the footprint's
    row range is derived from the top edge, then flipped to bottom-anchored
    row indices before the block test: task_bids encodes block rows from the
    bottom (self.tasks convention, by = (y - y.min()) // res // bsize; the
    raster write flips it via Window(i0, ny - j1)). The padded cell ranges
    are tested against the task blocks, so a false "has data" costs one warp,
    but a false "empty" is not possible: the task blocks are the superset of
    every non-NoData cell.
    """
    sb = rasterio.warp.transform_bounds(dst_crs, src_crs, *w_bounds, densify_pts=257)
    pad = max(g_res_x, g_res_y)
    xmin, ymin, xmax, ymax = sb[0] - pad, sb[1] - pad, sb[2] + pad, sb[3] + pad
    i0 = max(0, math.floor((xmin - g_left) / g_res_x))
    i1 = min(g_width, math.ceil((xmax - g_left) / g_res_x))
    g_top = g_bottom + g_height * g_res_y
    j0 = max(0, math.floor((g_top - ymax) / g_res_y))
    j1 = min(g_height, math.floor((g_top - ymin) / g_res_y) + 1)
    if i1 <= i0 or j1 <= j0:
        return False
    # j0/j1 are raster rows from the TOP (row 0 = g_top); task_bids encodes
    # block rows from the BOTTOM. Top row [j0, j1) is bottom row
    # [g_height - j1, g_height - j0); flip before deriving block rows, else
    # top/bottom tiles are falsely skipped whenever nby >= 4 (B1, PR #27).
    bj0 = g_height - j1
    bj1 = g_height - j0
    for bx in range(i0 // bsize, (i1 - 1) // bsize + 1):
        base = bx * nby
        for by in range(bj0 // bsize, (bj1 - 1) // bsize + 1):
            if base + by in task_bids:
                return True
    return False


def _reproject_band(
    src_path: str,
    dst_path: str,
    profile: dict[str, Any],
    dst_crs: str,
    dst_transform: Any,
    dst_width: int,
    dst_height: int,
    task_info: dict[str, Any] | None = None,
    n_threads: int = 1,
) -> None:
    """Nearest-neighbour reproject one band into a fresh output-CRS GeoTIFF.

    Vectorised. The nearest-neighbour warp is a per-pixel function of the dst
    grid, so the *warp* chunk size never changes the output values. The
    *write* chunk size, however, changes GDAL's on-disk block layout (and
    thus the file bytes). So: warp in coarse 2048px tiles (fewer, larger gdal
    calls -> ~15% faster) but flush TILE_PX blocks in raster-scan order,
    exactly as the 512px baseline does. The result is byte-identical to the
    baseline.

    n_threads > 1 (S3.3) warps the 2048px tiles of each row band in a thread
    pool: reproject releases the GIL (2.88x at 4 threads, design A.4) and the
    tiles are independent, so the pool changes wall time only, never bytes.
    The 512px flush loop stays serial, raster-scan: the file bytes depend on
    the tile write order. n_threads == 1 keeps the exact serial code path (A8);
    the outer src handle is still read by the main thread for profile, tags,
    scales and offsets.

    NoData-tile skip (C(ii), PR #27): when `task_info` is provided (work-grid
    block geometry + the populated-block superset from grid()), a coarse dst
    tile whose source footprint intersects no task block cannot contain data
    (empty blocks are written NoData, and data cells exist only inside task
    blocks). Its warp is skipped and a plain NoData buffer is written to the
    tile instead. The footprint is conservatively rounded out, so the skip
    is safe: at worst one tile that is all-NoData gets warped anyway. When
    no tile is skippable (or task_info is None) the output is identical to
    the unskipped path.
    """
    with rasterio.open(src_path) as src:
        skip = task_info is not None
        if skip:
            bsize = task_info["bsize"]
            nby = task_info["nby"]
            task_bids: set[int] = task_info["task_bids"]
            g_left, g_bottom, g_right, g_top = src.bounds
            g_res_x = (g_right - g_left) / src.width
            g_res_y = (g_top - g_bottom) / src.height
            src_crs = src.crs
        # Tile counters are shared with the warp pool (n_threads > 1); the
        # count feeds a log line only, but keep it exact under the GIL anyway.
        counters = {"tiles": 0, "skipped": 0}
        count_lock = threading.Lock()

        def make_warp(j0: int, band_h: int) -> Callable[[int], tuple[int, np.ndarray]]:
            warp = partial(
                _warp_dst_tile,
                src_path,
                j0=j0,
                band_h=band_h,
                warp_tile=2048,
                dst_width=dst_width,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
            )

            def warp_tile(i0: int) -> tuple[int, np.ndarray]:
                with count_lock:
                    counters["tiles"] += 1
                    if skip and not _footprint_has_task(
                        rasterio.windows.bounds(Window(i0, j0, min(2048, dst_width - i0), band_h), dst_transform),
                        dst_crs,
                        src_crs,
                        g_left,
                        g_bottom,
                        g_res_x,
                        g_res_y,
                        src.width,
                        src.height,
                        bsize,
                        nby,
                        task_bids,
                    ):
                        # All-NoData tile: no warp, plain NoData buffer.
                        counters["skipped"] += 1
                        return i0, np.full((band_h, min(2048, dst_width - i0)), NODATA, np.int16)
                return warp(i0)

            return warp_tile

        _reproject_core(
            make_warp,
            dst_path,
            profile,
            dst_crs,
            dst_transform,
            dst_width,
            dst_height,
            tags=dict(src.tags()),
            scales=tuple(src.scales) if getattr(src, "scales", None) else None,
            offsets=tuple(src.offsets) if getattr(src, "offsets", None) else None,
            n_threads=n_threads,
        )
        if skip and counters["skipped"] > 0:
            log.debug("[reproj] skipped %d/%d all-NoData tiles", counters["skipped"], counters["tiles"])


def _reproject_band_array(
    src_arr: np.ndarray,
    src_transform: Any,
    src_crs: str,
    dst_path: str,
    profile: dict[str, Any],
    dst_crs: str,
    dst_transform: Any,
    dst_width: int,
    dst_height: int,
    tags: dict[str, str],
    scales: tuple[float, ...] | None = None,
    offsets: tuple[float, ...] | None = None,
    n_threads: int = 1,
) -> None:
    """Reproject an in-RAM work-CRS int16 array.

    Target: a fresh output-CRS GeoTIFF (S3.4a turbo write, serial
    strategy). Byte-identical to writing the same array to the work-CRS
    intermediate file and running _reproject_band on it: same GDAL warp
    kernel per tile, same serial raster-scan flush order, same
    tags/scales/offsets.

    Args:
        src_arr: Work-CRS int16 array, shape (ny, nx), north-up raster order.
        src_transform: Geotransform of the work-CRS raster.
        src_crs: Work-CRS string.
        dst_path: Destination GeoTIFF path.
        profile: Base write profile (driver, compress, tiled, block size).
        dst_crs: Target CRS string.
        dst_transform: Geotransform of the target raster.
        dst_width: Target raster width.
        dst_height: Target raster height.
        tags: Dataset tags for the output (work-file parity).
        scales: Band scales to set, or None.
        offsets: Band offsets to set, or None.
        n_threads: Warp threads. 1 = fully serial (A8 path).

    """
    # Wrap the source array in a MEM dataset ONCE, no copy (issue #39
    # pass 3, ported to the serial array reproject - audit #78 P1-3):
    # reproject's ndarray source form copies the whole band into a fresh
    # MEM dataset per warp call. The tuple form reads the shared handle;
    # src_arr is not mutated during the warp.
    src_ds = MemoryDataset(src_arr, transform=src_transform, crs=src_crs, copy=False)

    def make_warp(j0: int, band_h: int) -> Callable[[int], tuple[int, np.ndarray]]:
        return partial(
            _warp_dst_tile_arr,
            src_arr,
            src_transform,
            src_crs,
            j0=j0,
            band_h=band_h,
            warp_tile=2048,
            dst_width=dst_width,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_ds=src_ds,
        )

    _reproject_core(
        make_warp,
        dst_path,
        profile,
        dst_crs,
        dst_transform,
        dst_width,
        dst_height,
        tags=tags,
        scales=scales,
        offsets=offsets,
        n_threads=n_threads,
    )


# TIFF base type -> bytes per element (Y-1). The parser and the head-
# overlap guard share it so a type can never be sized in one place and
# mis-sized in the other. 16/17/18 = the BigTIFF 64-bit types (LONG8,
# SLONG8, IFD8).
_TIFF_TYPE_SIZES = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 8,
    6: 8,
    7: 8,
    8: 1,
    9: 2,
    10: 4,
    11: 4,
    12: 8,
    16: 8,
    17: 8,
    18: 8,
}


def _tif_parse_ifd(data: bytes) -> dict[str, Any]:
    """Parse the first IFD of a single-band classic or BigTIFF tiled raster.

    Args:
        data: Full file bytes.

    Returns:
        Dict: e (struct endian), big (bool), off_fmt (offset width name),
        tags {tag: (typ, count, value-or-inline offset)}, t324/t325
        (tag 324/325 entries), n_tiles, first (offset of the first stored
        tile), sz324/sz325 (per-tag slot bytes: classic 4/4, BigTIFF
        324=8 (LONG8) / 325=4 (LONG)) and the matching fmt324/fmt325
        struct formats.

    Raises:
        ValueError: On unknown endian/magic, missing 324/325, an unknown
            TIFF type, a non-LONG/LONG8 tile array type, inline tile
            arrays, or a count mismatch.

    """
    if data[:2] == b"II":
        e = "<"
    elif data[:2] == b"MM":
        e = ">"
    else:
        msg = f"not a little/big-endian TIFF: {data[:2]!r}"
        raise ValueError(msg)
    magic = struct.unpack_from(e + "H", data, 2)[0]
    big = magic == 43
    if not big and magic != 42:
        msg = f"unexpected TIFF magic {magic}"
        raise ValueError(msg)
    if not big:
        ifd = struct.unpack_from(e + "I", data, 4)[0]
        cnt = struct.unpack_from(e + "H", data, ifd)[0]
        # 12-byte entries: tag(2) type(2) count(4) value-or-offset(4).
        ent_size, count_fmt, off_fmt, entry_off, val_at, inline = 12, "I", "I", 2, 8, 4
    else:
        # 16-byte BigTIFF header: first-IFD offset (8) at bytes 8-15; the
        # IFD count is followed by 6 pad bytes; 20-byte entries: tag(2)
        # type(2) count(8) value-or-offset(8, inline capacity 12).
        ifd = struct.unpack_from(e + "Q", data, 8)[0]
        cnt = struct.unpack_from(e + "H", data, ifd)[0]
        ent_size, count_fmt, off_fmt, entry_off, val_at, inline = 20, "Q", "Q", 8, 12, 12
    per_typ = _TIFF_TYPE_SIZES
    tags: dict[int, tuple[int, int, int]] = {}
    for i in range(cnt):
        off = ifd + entry_off + i * ent_size
        tag, typ = struct.unpack_from(e + "HH", data, off)
        count = struct.unpack_from(e + count_fmt, data, off + 4)[0]
        sz = per_typ.get(typ)
        if sz is None:
            msg = f"unsupported TIFF type {typ} for tag {tag}"
            raise ValueError(msg)
        total = sz * count
        if total <= inline:
            tags[tag] = (typ, count, off + val_at)
        else:
            voff = struct.unpack_from(e + off_fmt, data, off + val_at)[0]
            tags[tag] = (typ, count, voff)
    if 324 not in tags or 325 not in tags:
        msg = "missing TileOffsets/TileByteCounts"
        raise ValueError(msg)
    t324, t325 = tags[324], tags[325]
    if t324[1] != t325[1]:
        msg = "TileOffsets/TileByteCounts count mismatch"
        raise ValueError(msg)
    if t324[0] not in (4, 16) or t325[0] not in (4, 16):
        msg = f"unexpected tile array types (324: {t324[0]}, 325: {t325[0]})"
        raise ValueError(msg)
    n_tiles = t324[1]
    # Slot width is per tag (Y-1): BigTIFF stores 324 as LONG8 (8-byte
    # slots) but 325 as LONG (4-byte slots); classic stores both as LONG.
    sz324, sz325 = per_typ[t324[0]], per_typ[t325[0]]
    fmt324 = e + ("Q" if sz324 == 8 else "I")
    fmt325 = e + ("Q" if sz325 == 8 else "I")
    if t324[2] + sz324 * n_tiles > len(data) or t325[2] + sz325 * n_tiles > len(data):
        msg = "tile value arrays stored inline or beyond EOF"
        raise ValueError(msg)
    first = struct.unpack_from(fmt324, data, t324[2])[0]
    return {
        "e": e,
        "big": big,
        "off_fmt": off_fmt,
        "inline": inline,
        "tags": tags,
        "t324": t324,
        "t325": t325,
        "n_tiles": n_tiles,
        "first": first,
        "sz324": sz324,
        "sz325": sz325,
        "fmt324": fmt324,
        "fmt325": fmt325,
    }


def _turbo_write_parallel(
    arr: np.ndarray,
    src_transform: Any,
    src_crs: str,
    dst_path: str,
    profile: dict[str, Any],
    dst_crs: str,
    dst_transform: Any,
    dst_width: int,
    dst_height: int,
    tags: dict[str, str],
    scales: tuple[float, ...],
    offsets: tuple[float, ...] | None,
    n_threads: int,
) -> bool:
    """Assemble the output-CRS GeoTIFF from an in-RAM array.

    Parallel ZSTD strategy (S3.4a, S3.6.3). Only called when the A.7 oracle
    passed (i.e. zstdmt.compress_tiles reproduces the stock codec bytes
    exactly).

    1. Warp the full output raster from the work-CRS array (2048px tiles
       in a thread pool - same kernel as the serial path).
    2. Compress every TILE_PX^2 tile in parallel (predictor 2 + zstdmt).
    3. Write a reference raster with the identical profile holding ONE
       TILE_PX^2 zero tile; GDAL serialises the exact head (IFD + external
       tag arrays) from the raster dimensions alone, so the single tile
       avoids a dst_height*dst_width int16 reference array.
    4. Assemble: reference head with the TileOffsets/TileByteCounts arrays
       patched to the real frames, then the frames appended in raster-scan
       order.

    Returns:
        True if the file was written, False if the caller must fall back
        to the serial array reproject (layout guard: value arrays not all
        before the first tile, tile count mismatch, or classic head with a
        >= 4 GiB assembled file).

    """
    warp_tile = 2048
    # Wrap the source array in a MEM dataset ONCE, no copy (issue #39
    # pass 3, ported to the zstd writer - audit #78 P1-3): reproject's
    # ndarray source form copies the whole band into a fresh MEM dataset
    # per warp call (3.1 GB per 2048^2 block at full-AU; up to n_threads
    # copies in flight). The tuple form reads the shared handle; arr is
    # not mutated during the warp, so concurrent read-only access is safe.
    src_ds = MemoryDataset(arr, transform=src_transform, crs=src_crs, copy=False)
    tiles: list[bytes] = []
    band_tiles: list[np.ndarray] = []
    coarse: dict[int, np.ndarray] = {}
    for j0 in range(0, dst_height, warp_tile):
        band_h = min(warp_tile, dst_height - j0)
        warp_i0 = partial(
            _warp_dst_tile_arr,
            arr,
            src_transform,
            src_crs,
            j0=j0,
            band_h=band_h,
            warp_tile=warp_tile,
            dst_width=dst_width,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_ds=src_ds,
        )
        coarse = {}
        if n_threads > 1:
            with ThreadPoolExecutor(max_workers=n_threads) as ex:
                coarse.update(ex.map(warp_i0, range(0, dst_width, warp_tile)))
        else:
            for i0 in range(0, dst_width, warp_tile):
                _, buf = warp_i0(i0)
                coarse[i0] = buf
        band_tiles = []
        for j in range(j0, j0 + band_h, TILE_PX):
            h = min(TILE_PX, j0 + band_h - j)
            for i in range(0, dst_width, TILE_PX):
                w = min(TILE_PX, dst_width - i)
                i0 = (i // warp_tile) * warp_tile
                t = coarse[i0][j - j0 : j - j0 + h, i - i0 : i - i0 + w]
                if h < TILE_PX or w < TILE_PX:
                    # GDAL zero-pads partial edge tiles to the full block
                    # before the codec runs (audit #81 F-1): unpadded
                    # frames decompress to less than the declared tile
                    # size, breaking S3 bit-identity on non-512-multiple
                    # rasters. Pad to TILE_PX^2 first; predictor 2 then
                    # runs over the padded row exactly as libtiff does.
                    pad = np.zeros((TILE_PX, TILE_PX), dtype=np.int16)
                    pad[:h, :w] = t
                    t = pad
                band_tiles.append(t)
        # Predictor 2 must be applied before compression: the stock codec
        # (and the oracle) compress the predicted bytes, not the raw tiles.
        band_tiles = [zstdmt.predictor2(t) for t in band_tiles]
        tiles.extend(zstdmt.compress_tiles(band_tiles, n_threads=n_threads))
    del coarse, band_tiles, src_ds

    ref_path = Path(dst_path + ".refhead.tif")
    try:
        # One TILE_PX^2 reference tile, not a full-raster zero write: the
        # head (IFD + tag arrays) is sized from the raster dimensions at
        # creation, so a single tile write yields the byte-identical head a
        # full-band zero write would, without touching a
        # dst_height*dst_width int16 array (~3 GB at full-AU) (audit #84
        # P2-4; same property the stock parallel writer relies on).
        th = min(TILE_PX, dst_height)
        tw = min(TILE_PX, dst_width)
        zero = np.zeros((th, tw), dtype=np.int16)
        with _open_tiff_staging(
            ref_path,
            **dict(
                profile,
                width=dst_width,
                height=dst_height,
                transform=dst_transform,
                crs=dst_crs,
            ),
        ) as dst:
            dst.update_tags(**tags)
            if scales:
                dst.scales = scales
            if offsets:
                dst.offsets = offsets
            dst.write(zero, 1, window=Window(0, 0, tw, th))
        data = ref_path.read_bytes()
    finally:
        ref_path.unlink(missing_ok=True)
        del zero

    try:
        info = _tif_parse_ifd(data)
    except ValueError as e:
        print(f"[warn] turbo zstd head parse: {e}; serial write", file=sys.stderr)  # ruff: ignore[print]
        return False
    t324, t325, n_tiles = info["t324"], info["t325"], info["n_tiles"]
    if n_tiles != len(tiles):
        print(f"[warn] turbo zstd head: tile count {n_tiles} != {len(tiles)}; serial write", file=sys.stderr)  # ruff: ignore[print]
        return False
    # All external value arrays must end before the first stored tile, so
    # the head cut keeps every array and drops every reference tile.
    head_end = info["first"]
    for typ, count, voff in info["tags"].values():
        total = _TIFF_TYPE_SIZES[typ] * count
        if total > info["inline"] and voff + total > info["first"]:
            print("[warn] turbo zstd head: value array overlaps tiles; serial write", file=sys.stderr)  # ruff: ignore[print]
            return False
    if not info["big"] and head_end + sum(len(t) for t in tiles) > _TIFF_CLASSIC_MAX_BYTES:
        print("[warn] turbo zstd head: assembled file beyond classic 4 GiB; serial write", file=sys.stderr)  # ruff: ignore[print]
        return False

    head = bytearray(data[:head_end])
    # Patch stride is per tag (Y-1): BigTIFF 324 slots are 8 bytes, 325
    # slots 4; patching both at one width corrupts the 325 array region.
    sz324, sz325 = info["sz324"], info["sz325"]
    fmt324, fmt325 = info["fmt324"], info["fmt325"]
    pos = 0
    for i, t in enumerate(tiles):
        struct.pack_into(fmt324, head, t324[2] + i * sz324, head_end + pos)
        struct.pack_into(fmt325, head, t325[2] + i * sz325, len(t))
        pos += len(t)
    with os.fdopen(_staging_fd(Path(dst_path)), "wb") as f:
        f.write(bytes(head))
        f.writelines(tiles)
    return True


def _stock_band_frames(
    buf: np.ndarray,
    i0: int,
    j0: int,
    bw: int,
    bh: int,
    dst_transform: Any,
    out_crs: str,
    profile: dict[str, Any],
    scratch: Path,
) -> list[bytes]:
    """Compress one warp band's 512^2 tiles with the stock GTiff zstd codec.

    A per-band scratch dataset carries the identical codec profile (zstd,
    tiled, predictor 2, same tile windows, same nodata), so libtiff's
    streaming Ctx compresses each tile exactly as the serial full-raster
    write would: the frames are byte-equal to the serial path by
    construction, on any GDAL/libtiff stack (issue #39). GDAL writes tiles
    in the order requested, so the frames come back in raster-scan order.

    Args:
        buf: Warped band, shape (bh, bw), int16.
        i0: Band origin column in the dst raster (the transform keeps the
            scratch a valid GeoTIFF; frames are codec-only).
        j0: Band origin row in the dst raster.
        bw: Band size (px, columns).
        bh: Band size (px, rows).
        dst_transform: Geotransform of the full dst raster.
        out_crs: CRS string for the scratch.
        profile: The final raster's profile (width/height/transform/crs are
            overridden to the band's).
        scratch: Scratch path (reused per thread across bands).

    Returns:
        One zstd frame per tile, raster-scan order.

    """
    t = dst_transform * Affine(1, 0, i0, 0, 1, -j0)
    prof = dict(profile, width=bw, height=bh, transform=t, crs=out_crs)
    with _open_tiff_staging(scratch, **prof) as dst:
        for j in range(0, bh, TILE_PX):
            h = min(TILE_PX, bh - j)
            for i in range(0, bw, TILE_PX):
                w = min(TILE_PX, bw - i)
                dst.write(buf[j : j + h, i : i + w], 1, window=Window(i, j, w, h))
    data = scratch.read_bytes()
    info = _tif_parse_ifd(data)
    n = info["n_tiles"]
    offs = [struct.unpack_from(info["fmt324"], data, info["t324"][2] + k * info["sz324"])[0] for k in range(n)]
    szs = [struct.unpack_from(info["fmt325"], data, info["t325"][2] + k * info["sz325"])[0] for k in range(n)]
    return [data[o : o + s] for o, s in zip(offs, szs, strict=True)]


def _turbo_write_stock_parallel(
    arr: np.ndarray,
    src_transform: Any,
    src_crs: str,
    dst_path: str,
    profile: dict[str, Any],
    dst_crs: str,
    dst_transform: Any,
    dst_width: int,
    dst_height: int,
    tags: dict[str, str],
    scales: tuple[float, ...],
    offsets: tuple[float, ...] | None,
    n_threads: int,
    scratch_dir: str,
) -> bool:
    """Assemble the output-CRS GeoTIFF with parallel stock-codec compression.

    Fallback for stacks where the CPL zstd pfn drifts from libtiff's
    streaming codec (the A.7 oracle mismatches - e.g. the bundled GDAL
    3.12.4 / libtiff 6.2 stack, whose pfn emits a different frame header:
    single-segment flag + 8-byte FCS + a different windowLog, and which
    ignores every zstd option). Strategy (S3.4a, issue #39):

    1. Warp 2048^2 blocks in a thread pool (same kernel and windows as the
       serial path).
    2. Each block's 512^2 tiles are compressed with the stock codec via a
       per-thread scratch raster with the identical profile - frames are
       byte-equal to the serial flush by construction.
    3. The head comes from a zero-filled reference raster with the identical
       profile + tags + scales (layout matches by construction); the
       TileOffsets/TileByteCounts arrays are patched and the frames
       appended in raster-scan order.

    Returns:
        True if the file was written, False if the caller must fall back to
        the serial array reproject (layout guards).

    """
    warp_tile = 2048
    # 1. Reference raster: stock GDAL head + exact tag layout. The head
    # (IFD + tag arrays) is sized from the raster dimensions at creation,
    # so ONE 512^2 tile write yields the byte-identical head a full-band
    # zero write would (verified: BigTIFF IFD is complete on close, tag
    # layout and first-tile offset match a full NODATA write). The full
    # zero write touched a dst_height*dst_width int16 array (~3 GB at
    # full-AU) plus the whole file; under memory pressure that was the
    # write phase's dominant cost (issue #39 pass 3).
    ref_path = Path(str(dst_path) + ".refhead.tif")
    try:
        th = min(TILE_PX, dst_height)
        tw = min(TILE_PX, dst_width)
        zero = np.zeros((th, tw), dtype=np.int16)
        with _open_tiff_staging(
            ref_path,
            **dict(
                profile,
                width=dst_width,
                height=dst_height,
                transform=dst_transform,
                crs=dst_crs,
            ),
        ) as dst:
            dst.update_tags(**tags)
            if scales:
                dst.scales = scales
            if offsets:
                dst.offsets = offsets
            dst.write(zero, 1, window=Window(0, 0, tw, th))
        data = ref_path.read_bytes()
    finally:
        # No `del zero` here (audit #78 P1-1): if the reference head is not
        # created, `zero` is unbound and the del raises UnboundLocalError,
        # masking the root error. The 512^2 tile local dies at function
        # return anyway; the del buys nothing.
        ref_path.unlink(missing_ok=True)
    try:
        info = _tif_parse_ifd(data)
    except ValueError as e:
        print(f"[warn] turbo stock head parse: {e}; serial write", file=sys.stderr)  # ruff: ignore[print]
        return False
    t324, t325, n_tiles = info["t324"], info["t325"], info["n_tiles"]
    head_end = info["first"]
    for typ, count, voff in info["tags"].values():
        total = _TIFF_TYPE_SIZES[typ] * count
        if total > info["inline"] and voff + total > info["first"]:
            print("[warn] turbo stock head: value array overlaps tiles; serial write", file=sys.stderr)  # ruff: ignore[print]
            return False
    # Compressed frames are never longer than the raw band, so raw size is
    # a sound classic-4GiB upper bound.
    if not info["big"] and head_end + arr.nbytes > _TIFF_CLASSIC_MAX_BYTES:
        print("[warn] turbo stock head: assembled file may exceed classic 4 GiB; serial write", file=sys.stderr)  # ruff: ignore[print]
        return False

    # 2. Warp blocks in parallel; each block compresses through its thread's
    #    scratch raster. band_jobs is block raster-scan (j0 outer, i0 inner):
    #    flat index br*n_bcols + bc.
    n_brows = -(-dst_height // warp_tile)
    n_bcols = -(-dst_width // warp_tile)
    band_jobs = [(bc * warp_tile, br * warp_tile) for br in range(n_brows) for bc in range(n_bcols)]
    scratch_local = threading.local()
    # Wrap the source array in a MEM dataset ONCE, no copy (issue #39
    # pass 3): reproject's ndarray source form copies the whole band into
    # a fresh MEM dataset per warp call (3 GB per 2048^2 block at full-AU
    # scale; 8 such copies in flight OOM-killed the run). The tuple form
    # reads the shared handle; val_dn/sup_dn are not mutated during the
    # warp, so concurrent read-only access is safe.
    src_ds = MemoryDataset(arr, transform=src_transform, crs=src_crs, copy=False)
    prog = [0, 0.0, 0.0, 0]  # done, warp_s, frames_s, t0 (issue #39 pass 3 progress)
    prog[3] = time.monotonic()
    prog_lock = threading.Lock()

    def _block(i0: int, j0: int) -> list[bytes]:
        bw = min(warp_tile, dst_width - i0)
        bh = min(warp_tile, dst_height - j0)
        scratch: Path | None = getattr(scratch_local, "path", None)
        if scratch is None:
            # Full TID (no mask): two live threads can collide on the low
            # 16 bits of the kernel TID (observed on hydrogen, 2026-10-07);
            # a shared path made one thread truncate the other's scratch
            # mid-write (issue #39, pass 3).
            scratch = Path(scratch_dir) / f"_stock_scratch_{threading.get_ident()}.tif"
            scratch_local.path = scratch
        tw0 = time.monotonic()
        _, buf = _warp_dst_tile_arr(
            arr, src_transform, src_crs, i0, j0, bh, warp_tile, dst_width, dst_transform, dst_crs, src_ds=src_ds
        )
        tw1 = time.monotonic()
        frames = _stock_band_frames(buf, i0, j0, bw, bh, dst_transform, dst_crs, profile, scratch)
        tf1 = time.monotonic()
        with prog_lock:
            prog[0] += 1
            prog[1] += tw1 - tw0
            prog[2] += tf1 - tw1
            if prog[0] % 20 == 0 and _prof.enabled():
                print(  # ruff: ignore[print]
                    f"[reproj-prog] blocks={prog[0]}/{len(band_jobs)} "
                    f"warp={prog[1]:.1f}s frames={prog[2]:.1f}s "
                    f"elapsed={time.monotonic() - prog[3]:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
        return frames

    if n_threads > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            block_frames = list(ex.map(lambda jb: _block(*jb), band_jobs))
    else:
        block_frames = list(starmap(_block, band_jobs))

    # Reassemble into global 512 raster-scan (the serial/CPL order). A block's
    # frames are block-local raster-scan, so block-major concatenation would
    # interleave tile rows wrongly; expand each block row/col into its 512
    # tile rows/cols and index frames by (trow_local * tcols + tcol_local).
    bh_by_br = [min(warp_tile, dst_height - br * warp_tile) for br in range(n_brows)]
    bw_by_bc = [min(warp_tile, dst_width - bc * warp_tile) for bc in range(n_bcols)]
    trows_by_br = [-(-bh // TILE_PX) for bh in bh_by_br]
    tcols_by_bc = [-(-bw // TILE_PX) for bw in bw_by_bc]
    row_map = [(br, tl) for br in range(n_brows) for tl in range(trows_by_br[br])]
    col_map = [(bc, cl) for bc in range(n_bcols) for cl in range(tcols_by_bc[bc])]
    tiles: list[bytes] = []
    for br, tl in row_map:
        for bc, cl in col_map:
            tiles.append(block_frames[br * n_bcols + bc][tl * tcols_by_bc[bc] + cl])
    if len(tiles) != n_tiles:
        print(f"[warn] turbo stock: tile count {len(tiles)} != {n_tiles}; serial write", file=sys.stderr)  # ruff: ignore[print]
        return False

    # 3. Patch the head and assemble: head + frames in raster-scan order.
    head = bytearray(data[:head_end])
    sz324, sz325 = info["sz324"], info["sz325"]
    fmt324, fmt325 = info["fmt324"], info["fmt325"]
    pos = 0
    for i, t in enumerate(tiles):
        struct.pack_into(fmt324, head, t324[2] + i * sz324, head_end + pos)
        struct.pack_into(fmt325, head, t325[2] + i * sz325, len(t))
        pos += len(t)
    with os.fdopen(_staging_fd(Path(dst_path)), "wb") as f:
        f.write(bytes(head))
        f.writelines(tiles)
    return True


def _read_csv_chunk(
    task: tuple[str, int, int, bool, list[str], list[str]],
) -> dict[str, np.ndarray]:
    """Parse one CSV row chunk in a worker process (S3.5).

    The parent split the file into complete records at newline offsets and
    this worker seeks its byte range. The wanted columns parse with explicit
    float64 dtypes and the per-chunk dtype equality is asserted (review F3):
    the serial read pins the same dtypes (audit P0-2, 2026-10-09, decision
    (a)), so the assert is a structural invariant — a wanted column must
    reach the worker as float64, never an inferred int64.

    Args:
        task: Tuple of (path, byte_start, byte_end, has_header, wanted, all_names).

    Returns:
        Dict of column name -> float64 array in original row order. The
        caller indexes by name (audit #83: the chunk arrays used to be
        returned positionally, so the order of `wanted` was an unspoken
        value/lng/lat contract shared with the caller).

    Raises:
        RuntimeError: If a wanted column does not parse as float64 (schema pin).

    """
    path, start, end, has_header, wanted, all_names = task
    with Path(path).open("rb") as f:
        f.seek(start)
        raw = f.read(end - start)
    dtype = dict.fromkeys(wanted, np.float64)
    if has_header:
        df = pd.read_csv(io.BytesIO(raw), usecols=wanted, dtype=dtype)
    else:
        df = pd.read_csv(io.BytesIO(raw), header=None, names=all_names, usecols=wanted, dtype=dtype)
    for c in wanted:
        if df[c].dtype != np.dtype(np.float64):
            msg = f"ingest chunk dtype mismatch for column {c!r}: {df[c].dtype}"
            raise RuntimeError(msg)
    return {c: df[c].to_numpy(dtype=np.float64) for c in wanted}


def _csv_chunk_tasks(
    path: str,
    wanted: list[str],
    n_threads: int,
) -> tuple[list[tuple[str, int, int, bool, list[str], list[str]]], int] | None:
    """Plan the parallel CSV read (S3.5): split the file into row chunks.

    Reads the file once and finds record boundaries at newline offsets. The
    split is used only when every chunk is a set of complete CSV records:
    an even quote parity per chunk (a boundary inside a quoted field leaves
    an unbalanced quote) and a single-line header. Any doubt returns None
    and the caller keeps the single serial parse, so the parallel path can
    never change values, only wall time.

    Returns:
        Tuple of (chunk tasks, expected data rows) or None for the serial
        parse. Each task is (path, byte_start, byte_end, has_header,
        wanted, all_names); byte ranges are half-open and include the row
        terminator.

    """
    try:
        data = Path(path).read_bytes()
        # Empty or compressed input cannot be split by raw byte offsets: pandas
        # cannot sniff compression from a byte buffer, so the serial parse
        # (which reads via the path) is the only correct read (issue #80).
        if (
            not data
            or data[:2] == b"\x1f\x8b"  # gzip
            or data[:3] == b"BZh"  # bzip2
            or data[:4] == b"PK\x03\x04"  # zip
            or data[:5] == b"\xfd7zXZ"  # xz (5-byte magic; review M-1)
        ):
            return None
        nl = np.flatnonzero(np.frombuffer(data, dtype=np.uint8) == 10)
        n_nl = int(nl.size)
        # One physical line per row, the header is the first line; a file
        # without a final newline has one extra unterminated row.
        data_rows = n_nl - 1 if data[-1] == 10 else n_nl
        if n_nl == 0 or data_rows < 2 * _INGEST_MIN_CHUNK_ROWS:
            return None
        # n_threads >= 2 and the data_rows guard above make this >= 2.
        n_chunks = min(n_threads, data_rows // _INGEST_MIN_CHUNK_ROWS)

        # Header via pandas so the names match exactly what a serial parse
        # would see (BOM, quoting, coercion included).
        hdr = pd.read_csv(path, nrows=0)
        all_names = [str(c) for c in hdr.columns]
        if data[: int(nl[0])].count(b'"') % 2 != 0 or any(all_names.count(w) != 1 for w in wanted):
            return None  # embedded-newline header, or missing/duplicated column

        tasks: list[tuple[str, int, int, bool, list[str], list[str]]] = []
        lo = np.linspace(0, data_rows, n_chunks + 1).astype(np.int64)
        for k in range(n_chunks):
            r0, r1 = int(lo[k]), int(lo[k + 1])
            # Data row r (0-based) is physical line r + 1: the bytes after
            # newline nl[r], up to and including newline nl[r + 1].
            start = 0 if r0 == 0 else int(nl[r0]) + 1
            end = len(data) if r1 >= n_nl else int(nl[r1]) + 1
            if data[start:end].count(b'"') % 2 != 0:
                return None  # chunk boundary splits a quoted field
            tasks.append((path, start, end, r0 == 0, list(wanted), list(all_names)))
    except (OSError, ValueError, IndexError, MemoryError):
        # Planning is best effort: any surprise (odd encoding, ragged
        # header, memory pressure at plan time, ...) falls back to the
        # serial parse, which has the historical behaviour (issue #80).
        return None
    else:
        return tasks, data_rows


class Pipeline:
    """Pull-push interpolation pipeline."""

    def __init__(
        self,
        input_path: str,
        value_col: str,
        lng_col: str,
        lat_col: str,
        out_dir: str,
        res: float = DEFAULT_RES,
        cap_km: float | str = "auto",
        transform: str = "auto",
        saturation: float = DEFAULT_SATURATION,
        block_size: int = DEFAULT_BLOCK,
        workers: int = DEFAULT_WORKERS,
        calib_path: str | None = None,
        *,
        scale: float = DEFAULT_SCALE,
        compress: str = "ZSTD",
        percentile_step: float | None = None,
        calib_max_points: int = CALIB_MAX_POINTS_DEFAULT,
        src_crs: int = SRC_CRS,
        work_crs: int = WORK_CRS,
        out_crs: int = OUT_CRS,
        skip_calibration: bool = False,
        max_band_parallel: int | None = None,
        n_threads: int = 1,
        turbo: bool = False,
        turbo_cap_gb: float | None = None,
        turbo_strict: bool = False,
        seed: int = 0,
        dry_run: bool = False,
    ) -> None:
        """Initialise the interpolation pipeline.

        Raises:
            ValueError: If scale * PERCENTILE_MAX exceeds int16 max.
            ValueError: If percentile_step is outside (0, PERCENTILE_MAX].
            ValueError: If saturation is not > 0.
            ValueError: If n_threads is not >= 1.
            ValueError: If max_band_parallel is not >= 1.
            ValueError: If turbo_cap_gb is not > 0.
            ValueError: If seed is not >= 0.

        """
        self.input_path = input_path
        self.value_col = value_col
        self.lng_col = lng_col
        self.lat_col = lat_col
        self.out_dir = out_dir
        self.res = res
        self.cap_km = cap_km
        self.transform = transform
        self.saturation = saturation
        self.block_size = block_size
        self.workers = workers
        self.calib_path = calib_path
        self.scale = scale
        self.compress = compress
        self.percentile_step = percentile_step
        if percentile_step is not None and not (0 < percentile_step <= PERCENTILE_MAX):
            msg = f"percentile_step must be in (0, {PERCENTILE_MAX:g}]: {percentile_step}"
            raise ValueError(msg)
        if saturation <= 0:
            msg = f"saturation must be > 0: {saturation}"
            raise ValueError(msg)
        self.calib_max_points = calib_max_points
        self.src_crs = src_crs
        self.work_crs = work_crs
        self.out_crs = out_crs
        self.skip_calibration = skip_calibration
        if max_band_parallel is not None and max_band_parallel < 1:
            msg = f"max_band_parallel must be >= 1: {max_band_parallel}"
            raise ValueError(msg)
        self.max_band_parallel = max_band_parallel
        if n_threads < 1:
            msg = f"n_threads must be >= 1: {n_threads}"
            raise ValueError(msg)
        self.n_threads = n_threads
        if turbo_cap_gb is not None and turbo_cap_gb <= 0:
            msg = f"turbo_cap_gb must be > 0: {turbo_cap_gb}"
            raise ValueError(msg)
        self.turbo = turbo
        self.turbo_cap_gb = turbo_cap_gb
        self.turbo_strict = turbo_strict
        if seed < 0:
            msg = f"seed must be >= 0: {seed}"
            raise ValueError(msg)
        self.seed = seed
        # --plan dry run: ingest/calibrate/grid run in memory only; no files
        # (calibration.json, _points.npy) and no directories are created (#53).
        self.dry_run = dry_run
        # Set by _turbo_precheck() (run(), after grid()).
        self._turbo_plan: turbop.TurboPlan | None = None
        self._turbo_decision: turbop.PrecheckDecision | None = None
        self._turbo_budget_bytes: float | None = None
        # Set by run(): labelled wall seconds (ingest sub-phases + write
        # sub-phases) for the DEBUG perf breakdown.
        self._phase_wall: dict[str, float] = {}
        # Final-output names published (via os.replace) this run, set by the
        # write phase: post-publish failures (e.g. a --json write error)
        # then don't report the just-written rasters as stale "left
        # untouched" (issues #40, #77).
        self._published: set[str] = set()
        # Set by run() on success: machine-readable summary for --json (#55).
        self._summary: dict[str, Any] | None = None

        if self.scale * PERCENTILE_MAX > INT16_MAX:
            msg = (
                f"scale * {PERCENTILE_MAX:g} exceeds int16 max: {self.scale * PERCENTILE_MAX} > "
                f"{INT16_MAX}. Reduce scale."
            )
            raise ValueError(msg)

        # Ingest outputs
        self.v: np.ndarray
        self.x: np.ndarray
        self.y: np.ndarray
        self.n: int
        self.n_total: int  # rows read (pre-filter), for the volumetric log lines
        self.n_dropped: int  # all rows dropped (NaN/inf + out-of-domain + non-finite projection)
        self.n_dropped_nan_inf: int  # non-finite value/lon/lat rows
        self.n_dropped_out_of_domain: int  # geographic rows outside lat/lon range (issue #76)
        self.n_dropped_nonfinite_proj: int  # rows with non-finite projected x/y (issue #76)
        self._drop_examples: list[str]  # first offender(s), for the ingest drop summary

        # Calibration outputs
        self.tf: Any
        self.pct_q: Any
        self.cap_km_val: float
        self.tv: np.ndarray
        self.tname: str

        # Grid outputs
        self.x0: float
        self.y0: float
        self.nx: int
        self.ny: int
        self.levels: int
        self.step: int
        self.halo: int
        self.nx_padded: int
        self.ny_padded: int
        self.bsize: int
        self.nbx: int
        self.nby: int
        self.starts: np.ndarray
        self.tasks: list[tuple[int, int]]
        self.empty_blocks: list[tuple[int, int]]
        self.pts_path: Path
        self.cfg: _WorkerConfig
        self._cal: dict[str, Any] | None

    @property
    def published(self) -> set[str]:
        """Final-output names published (via os.replace) this run (issues #40, #77)."""
        return self._published

    def ingest(self) -> None:
        """Read input, filter, project to working CRS.

        CSV input with n_threads > 1 and enough rows is split into complete
        row records and parsed in a process pool (S3.5): read_csv holds the
        GIL, so threads cannot overlap the parse. Every chunk — and the
        serial fallback — parses the wanted columns with explicit float64
        dtypes (audit P0-2, 2026-10-09, decision (a): one C-parser float
        path for all n_threads, S3), the chunks assert per-chunk dtype
        equality, and the chunks are concatenated in original row order.
        Any split uncertainty (small file, blank lines, quoted newlines,
        header quirks) falls back to the single serial parse.

        Raises:
            ValueError: If no valid points remain after filtering.
            ImportError: If pyarrow is missing for Parquet input.

        """
        wanted = [self.value_col, self.lng_col, self.lat_col]
        if self.input_path.endswith((".parquet", ".pq")):
            try:
                import pyarrow as pa  # ruff: ignore[unused-import]
            except ImportError:
                msg = "pyarrow is required for Parquet files. Install with: pip install ppgrid[parquet]"
                raise ImportError(msg) from None
            df = pd.read_parquet(self.input_path, columns=wanted)
            v = df[self.value_col].to_numpy(dtype=np.float64)
            lon = df[self.lng_col].to_numpy(dtype=np.float64)
            lat = df[self.lat_col].to_numpy(dtype=np.float64)
        else:
            with _prof.phase("ingest.csv"):
                v, lon, lat = self._read_points(wanted)

        # Volumetric (DEBUG log): rows read vs valid points after filtering.
        self.n_total = int(v.size)
        good = np.isfinite(v) & np.isfinite(lon) & np.isfinite(lat)
        v, lon, lat = v[good], lon[good], lat[good]
        self.n_dropped_nan_inf = self.n_total - len(v)
        kept = np.flatnonzero(good)  # original 0-based data-row numbers of kept rows

        # Raw geographic domain (issue #76): pyproj is silent about
        # out-of-range lat/lon — lat 95 maps to inf (int64-cast inf crashes
        # grid() with "ix indices out of range"), lon 181 maps to a finite
        # x that balloons the grid to global span. Drop such rows and name
        # the first offender, same summary style as the NaN/inf line.
        self.n_dropped_out_of_domain = 0
        self.n_dropped_nonfinite_proj = 0
        self._drop_examples = []
        if CRS.from_epsg(self.src_crs).is_geographic:
            in_range = (lat >= -90.0) & (lat <= 90.0) & (lon >= -180.0) & (lon <= 180.0)
            if not in_range.all():
                i = int(np.nonzero(~in_range)[0][0])
                row = int(kept[i])
                if not (-90.0 <= lat[i] <= 90.0):
                    self._drop_examples.append(f"out-of-domain lat {lat[i]:.6f} outside [-90, 90] (row {row})")
                else:
                    self._drop_examples.append(f"out-of-domain lon {lon[i]:.6f} outside [-180, 180] (row {row})")
                self.n_dropped_out_of_domain = int(np.count_nonzero(~in_range))
                v, lon, lat, kept = v[in_range], lon[in_range], lat[in_range], kept[in_range]

        self.n = len(v)
        self.n_dropped = self.n_total - self.n

        if self.n == 0:
            msg = "No valid points found in input. Check columns and data."
            if self._drop_examples:
                msg += f" ({self._drop_examples[0]})"
            raise ValueError(msg)

        with _prof.phase("ingest.proj"):
            tr = Transformer.from_crs(self.src_crs, self.work_crs, always_xy=True)
            x, y = tr.transform(lon, lat)
            x = np.asarray(x)
            y = np.asarray(y)
            # Transformed coords must be finite (issue #76): singular or
            # out-of-domain inputs come back as +/-inf/NaN from pyproj
            # (silent) and reach grid() as int64 min/max after casting.
            good_xy = np.isfinite(x) & np.isfinite(y)
            if not good_xy.all():
                i = int(np.nonzero(~good_xy)[0][0])
                row = int(kept[i])
                self.n_dropped_nonfinite_proj = int(np.count_nonzero(~good_xy))
                self._drop_examples.append(
                    f"non-finite after projection to EPSG:{self.work_crs} "
                    f"(row {row}: lon {lon[i]:.6f}, lat {lat[i]:.6f})",
                )
                v, lon, lat, x, y = v[good_xy], lon[good_xy], lat[good_xy], x[good_xy], y[good_xy]
                self.n = len(v)
                self.n_dropped = self.n_total - self.n
                if self.n == 0:
                    msg = f"No valid points remain after projection to EPSG:{self.work_crs} ({self._drop_examples[0]})"
                    raise ValueError(msg)
            self.x = x
            self.y = y
            self.v = v

    def _drop_summary(self) -> str:
        """Ingest drop text for the plan/log lines (issue #76).

        Returns:
            'N NaN/inf dropped' when only non-finite rows were dropped
            (byte-identical to the pre-#76 output), else an extended
            breakdown naming the first offender.

        """
        if self.n_dropped_out_of_domain == 0 and self.n_dropped_nonfinite_proj == 0:
            return f"{self.n_dropped} NaN/inf dropped"
        detail = []
        if self.n_dropped_nan_inf:
            detail.append(f"{self.n_dropped_nan_inf} NaN/inf")
        if self.n_dropped_out_of_domain:
            detail.append(f"{self.n_dropped_out_of_domain} out-of-domain")
        if self.n_dropped_nonfinite_proj:
            detail.append(f"{self.n_dropped_nonfinite_proj} non-finite after projection")
        text = f"{self.n_dropped} dropped ({'; '.join(detail)})"
        if self._drop_examples:
            text += f"; first: {self._drop_examples[0]}"
        return text

    def _ingest_summary_info(self) -> dict[str, int]:
        """Ingest section of the --json run summary (issue #76).

        The out-of-domain / non-finite-projection keys are only present when
        non-zero, so an all-clean run keeps the historical three-key shape.

        Returns:
            Dict with rows_read, points, dropped_nan_inf (+ optional new keys).

        """
        info: dict[str, int] = {
            "rows_read": self.n_total,
            "points": self.n,
            "dropped_nan_inf": self.n_dropped_nan_inf,
        }
        if self.n_dropped_out_of_domain:
            info["dropped_out_of_domain"] = self.n_dropped_out_of_domain
        if self.n_dropped_nonfinite_proj:
            info["dropped_nonfinite_proj"] = self.n_dropped_nonfinite_proj
        return info

    def _read_points(self, wanted: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read (value, lon, lat) float64 arrays from the CSV input.

        Single serial parse by default. With n_threads > 1 and a file large
        enough, a process pool parses row chunks in original order (see
        ingest). A row-count mismatch after the parallel read (e.g. blank
        lines, which pandas skips but the newline scan counts) re-runs the
        serial parse.

        The serial read pins dtype=float64 on the wanted columns exactly as
        the chunk workers do (audit P0-2, 2026-10-09, decision (a)): both
        paths then take pandas' identical C-parser float path, so the values
        are bit-identical at every n_threads (S3) by construction.

        Returns:
            Tuple of (value, lon, lat) float64 arrays in original row order.

        """

        def serial() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            dtype = dict.fromkeys(wanted, np.float64)
            df = pd.read_csv(self.input_path, usecols=wanted, dtype=dtype)
            return (
                df[self.value_col].to_numpy(dtype=np.float64),
                df[self.lng_col].to_numpy(dtype=np.float64),
                df[self.lat_col].to_numpy(dtype=np.float64),
            )

        tasks = _csv_chunk_tasks(self.input_path, wanted, self.n_threads) if self.n_threads > 1 else None
        if tasks is None:
            return serial()
        task_list, expected = tasks
        with ProcessPoolExecutor(max_workers=len(task_list)) as ex:
            chunks = list(ex.map(_read_csv_chunk, task_list))
        # Index by column name, not position (audit #83): the chunk dicts
        # carry the same names the serial path uses.
        v = np.concatenate([c[self.value_col] for c in chunks])
        lon = np.concatenate([c[self.lng_col] for c in chunks])
        lat = np.concatenate([c[self.lat_col] for c in chunks])
        if v.size != expected:
            return serial()
        return v, lon, lat

    def calibrate(self) -> None:
        """Load or run calibration. Sets transform, percentile, cap.

        If an explicitly forced transform is invalid for the data (e.g. log10
        with negative values), a ValueError is raised. For the ``auto`` path,
        an invalid transform is rejected with a warning and identity is used.

        Raises:
            ValueError: If an explicit --transform cannot be applied to the data.
            ValueError: If calib_path is set but the file does not exist.

        """
        cal: dict[str, Any] | None = None
        cal_fresh = False
        cpath_obj = Path(self.calib_path) if self.calib_path else None
        if cpath_obj is not None and not cpath_obj.exists():
            # A named cal file is a request to reuse it. A missing path is a
            # user error (issue #21), not a chance to silently recalibrate
            # and create the file.
            msg = f"--calibration file not found: {cpath_obj} (run without --calibration to create one)"
            raise ValueError(msg)
        if cpath_obj is not None:
            with cpath_obj.open(encoding="utf-8") as f:
                cal = json.load(f)
            # User-supplied file: validate the schema before use so a bad
            # file is a friendly exit-2 naming the file, not a raw
            # traceback mid-run (issue #79).
            validate_calibration(cal, str(cpath_obj))
        elif not self.skip_calibration:
            if self.n > self.calib_max_points:
                sub = np.random.default_rng(self.seed).choice(self.n, self.calib_max_points, replace=False)
                cx, cy, cv = self.x[sub], self.y[sub], self.v[sub]
            else:
                cx, cy, cv = self.x, self.y, self.v

            tf, scores = choose_transform(cx, cy, cv)
            pct_fit = PercentileTransform().fit(cv)
            if self.cap_km != "auto":
                # Explicit cap: skip the blocked-CV fill-cap search (it is
                # discarded anyway, and it dominates run time / memory).
                cap, detail = float(self.cap_km), {}
            else:
                cap, detail = calibrate_fill_cap(cx, cy, tf.fwd(cv), seed=self.seed, saturation=self.saturation)

            cal = {
                "transform": tf.name,
                "n_calibration_points": len(cv),
                "transform_scores": {s[0].name: s[1] for s in scores},
                "transform_state": tf.state(),
                "percentile_quantiles": pct_fit.q.tolist(),
                "cap_km": cap,
                "cv": {
                    str(k): {"overall_skill": d["overall_skill"], "cap_km": d.get("cap_km")} for k, d in detail.items()
                },
            }
            # The file is written below, after the forced-transform override
            # is final (issue #23).
            cal_fresh = True
        else:
            cal = {
                "transform": "identity",
                "cap_km": float(self.cap_km) if self.cap_km != "auto" else FILL_FALLBACK_CAP_KM,
            }

        self.tname = (
            self.transform if self.transform != "auto" else (cal.get("transform", "identity") if cal else "identity")
        )
        if cal and "transform_state" in cal and cal["transform_state"].get("name") == self.tname:
            self.tf = make_transform(cal["transform_state"])
        else:
            self.tf = next(t for t in transforms() if t.name == self.tname).fit(self.v)

        # Same valid() guard choose_transform applies: a transform can be
        # invalid for this data (log10 on negatives -> NaN -> silent all-0
        # surface). An explicitly forced --transform is a hard error (the user
        # was deliberate); the auto path falls back to identity with a warning.
        if not self.tf.valid(self.v):
            if self.transform != "auto":
                msg = (
                    f"--transform {self.tname} not valid for this data "
                    f"(min={np.min(self.v):.6g}); use --transform auto or a different transform"
                )
                raise ValueError(msg)
            msg = (
                f"transform {self.tname!r} is invalid for this data "
                f"(min={np.min(self.v):.6g}); falling back to identity"
            )
            warnings.warn(msg, stacklevel=2)
            self.tname = "identity"
            self.tf = next(t for t in transforms() if t.name == "identity")

        # Keep the cal dict in sync with the fitted transform (a forced
        # transform can differ from the calibration's choice): the transform
        # name and its state must always agree, or a later auto run re-fits
        # from a stale name.
        if cal is not None:
            cal["transform"] = self.tname
            cal["transform_state"] = self.tf.state()

        # Write the cal file only after the transform above is final
        # (issue #23): the file must record the transform actually used, else
        # a later --transform auto run reloading it would silently change the
        # output. allow_nan=False: a NaN that slips through is a loud error,
        # not silent non-strict JSON (issue #14).
        if cal_fresh and cal is not None and not self.dry_run:
            Path(self.out_dir).mkdir(parents=True, exist_ok=True)
            cal_path = cpath_obj or Path(self.out_dir) / "calibration.json"
            with cal_path.open("w", encoding="utf-8") as f:
                json.dump(cal, f, indent=2, allow_nan=False)

        self.pct_q = (
            PercentileTransform(cal["percentile_quantiles"])
            if cal and "percentile_quantiles" in cal
            else PercentileTransform().fit(self.v)
        )

        self.cap_km_val = (
            float(self.cap_km)
            if self.cap_km != "auto"
            else float(cal.get("cap_km", FILL_FALLBACK_CAP_KM) if cal else FILL_FALLBACK_CAP_KM)
        )

        self.tv = self.tf.fwd(self.v)
        self._cal = cal

    def grid(self) -> None:
        """Compute grid geometry and spatial index.

        Raises:
            ValueError: If halo exceeds block size.

        """
        self.x0 = self.x.min()
        self.y0 = self.y.min()
        self.nx = int((self.x.max() - self.x0) // self.res) + 1
        self.ny = int((self.y.max() - self.y0) // self.res) + 1

        cap_cells = self.cap_km_val * M_PER_KM / self.res
        if not math.isfinite(cap_cells):
            msg = f"cap_km/res is not finite (cap_km={self.cap_km_val} km, res={self.res} m)"
            raise ValueError(msg)
        self.levels = max(1, math.ceil(math.log2(max(cap_cells, 2.0))))
        self.step = 1 << self.levels
        self.halo = max(math.ceil(cap_cells) + 2, self.step)
        self.nx_padded = -(-self.nx // self.step) * self.step
        self.ny_padded = -(-self.ny // self.step) * self.step
        self.bsize = max(self.block_size, self.step)
        if self.halo > self.bsize:
            msg = (
                f"halo ({self.halo}) exceeds block size ({self.bsize}). "
                f"Reduce cap_km or increase block size. "
                f"Currently: cap_km={self.cap_km_val}, res={self.res}, "
                f"block={self.block_size}"
            )
            raise ValueError(msg)
        self.nbx = math.ceil(self.nx / self.bsize)
        self.nby = math.ceil(self.ny / self.bsize)

        # Spatial index
        bx_ = np.clip(((self.x - self.x0) // self.res // self.bsize).astype(np.int64), 0, self.nbx - 1)
        by_ = np.clip(((self.y - self.y0) // self.res // self.bsize).astype(np.int64), 0, self.nby - 1)
        bid = bx_ * self.nby + by_
        order = np.argsort(bid, kind="stable")
        self.starts = np.searchsorted(bid[order], np.arange(self.nbx * self.nby + 1))

        # Memmap for workers (bands PTS_X/PTS_Y/PTS_TV — see constants above).
        # Fresh 0600 file, never follows a pre-planted symlink (issue #15).
        # Skipped in --plan dry-run mode (no files are created; the write
        # phase, which reads it, never runs).
        self.pts_path = Path(self.out_dir) / "_points.npy"
        if not self.dry_run:
            pts = _open_fresh_memmap(self.pts_path, np.float64, (PTS_NBANDS, self.n))
            pts[PTS_X] = self.x[order]
            pts[PTS_Y] = self.y[order]
            pts[PTS_TV] = self.tv[order]
            pts.flush()

        # Task list (one pass: blocks whose 3x3 neighbourhood holds points
        # are computed, the rest are written as nodata)
        tasks: list[tuple[int, int]] = []
        empty_blocks: list[tuple[int, int]] = []
        for i in range(self.nbx):
            for j in range(self.nby):
                if _block_neighbourhood_nonempty(i, j, self.starts, self.nbx, self.nby):
                    tasks.append((i, j))
                else:
                    empty_blocks.append((i, j))
        self.tasks = tasks
        self.empty_blocks = empty_blocks

        # Worker config
        self.cfg = _WorkerConfig(
            pts_path=str(self.pts_path),
            transform_state=self._cal.get("transform_state", {"name": self.tname})
            if self._cal
            else {"name": self.tname},
            pct_quantiles=self._cal.get("percentile_quantiles", self.pct_q.q.tolist())
            if self._cal
            else self.pct_q.q.tolist(),
            starts=self.starts.tolist(),
            nbx=self.nbx,
            nby=self.nby,
            bsize=self.bsize,
            halo=self.halo,
            res=self.res,
            levels=self.levels,
            cap_km=self.cap_km_val,
            x0=float(self.x0),
            y0=float(self.y0),
            nx=self.nx,
            ny=self.ny,
            nx_padded=self.nx_padded,
            ny_padded=self.ny_padded,
            sat=self.saturation,
            scale=self.scale,
            pct_step=self.percentile_step,
        )

    def run(self) -> tuple[str, str]:
        """Execute the full pipeline.

        Logs phase progress at INFO and a per-phase wall-time + volumetric
        breakdown at DEBUG (see _log_run_summary). Logging is side-channel
        only: it never touches the raster bytes.

        Returns:
            Tuple of value and support GeoTIFF file paths.

        """
        self._phase_wall = {}
        t_total = time.perf_counter()
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        with _prof.phase("ingest"):
            self.ingest()
        t_ingest = time.perf_counter() - t0
        t0 = time.perf_counter()
        with _prof.phase("calibrate"):
            self.calibrate()
        t_calibrate = time.perf_counter() - t0
        try:
            t0 = time.perf_counter()
            with _prof.phase("grid"):
                self.grid()
            t_grid = time.perf_counter() - t0
            self._turbo_precheck()
            t0 = time.perf_counter()
            vpath, spath = self._write_rasters()
            t_write = time.perf_counter() - t0
        finally:
            # Temp cleanup on every path: grid failure, mid-run worker error,
            # OOM, disk full — _points.npy + partial TIFF temps are removed
            # (issue #15).
            self._remove_run_temps()
        t_total = time.perf_counter() - t_total
        self._log_run_summary(t_ingest, t_calibrate, t_grid, t_write, t_total, vpath, spath)
        self._summary = self._collect_summary(vpath, spath, t_ingest, t_calibrate, t_grid, t_write, t_total)
        return vpath, spath

    def plan(self) -> None:
        """Resolve the full run plan without the write phase (--plan, #53).

        Runs ingest -> calibrate -> grid -> turbo precheck; with dry_run
        active no files or directories are created. A --turbo-strict
        infeasibility in the precheck exits 3 (SystemExit propagates).

        """
        self.ingest()
        self.calibrate()
        self.grid()
        self._turbo_precheck()

    def plan_text(self, args: argparse.Namespace) -> str:
        """Human-readable run plan for --plan (issue #53); printed on stdout.

        Args:
            args: Parsed CLI namespace (input/columns/out paths, --json).

        Returns:
            The multi-line plan text.

        """
        out = Path(args.out)
        pct_step = f"{self.percentile_step:g}" if self.percentile_step is not None else "off"
        mbp = f"{args.max_band_parallel}" if args.max_band_parallel is not None else "default (--workers)"
        json_extra = " + run_summary.json" if args.json else ""
        cells_line = (
            f"  grid cells:    {self.nx} x {self.ny} = {self.nx * self.ny:,} cells @ res={self.res:g} m "
            f"(padded {self.nx_padded} x {self.ny_padded}, step {self.step})"
        )
        blocks_line = (
            f"  blocks:        {self.nbx} x {self.nby} of bsize={self.bsize} "
            f"({len(self.tasks)} tasks, {len(self.empty_blocks)} empty)"
        )
        lines = [
            "ppgrid run plan (--plan: nothing written, exit 0)",
            f"  input:         {args.input}",
            f"  columns:       value={args.value_col!r} lng={args.lng_col!r} lat={args.lat_col!r}",
            f"  out dir:       {args.out} ({'exists' if out.is_dir() else 'does not exist yet'})",
            f"  ingest:        {self.n_total} rows -> {self.n} points ({self._drop_summary()})",
            f"  grid extent:   x0={self.x0:.3f} m, y0={self.y0:.3f} m (work CRS EPSG:{self.work_crs}, metres)",
            cells_line,
            f"  levels:        {self.levels} (halo {self.halo}, cap {self.cap_km_val:g} km)",
            blocks_line,
            f"  transform:     {self.tname}",
            f"  saturation:    {self.saturation:g}",
            f"  scale:         {self.scale:g} (DN = percentile * scale), percentile-step: {pct_step}",
            f"  seed:          {self.seed}",
            f"  workers:       {self.workers}, max-band-parallel: {mbp}",
            f"  compress:      {self.compress}",
            f"  outputs:       value.tif (EPSG:{self.out_crs}) + support_km.tif{json_extra}",
        ]
        if self.turbo and self._turbo_decision is not None and self._turbo_plan is not None:
            d, t = self._turbo_decision, self._turbo_plan
            lines.append(
                f"  turbo:         on — regime {d.regime}, path {d.path}, cap {t.cap_gb:g} GB, "
                f"budget {d.budget_gb:g} GB, est peak {d.est_peak_gb:g} GB, workers {t.workers}"
            )
        elif self.turbo:
            lines.append("  turbo:         on")
        else:
            lines.append("  turbo:         off")
        return "\n".join(lines)

    def run_summary(self) -> dict[str, Any]:
        """Machine-readable summary of the last successful run() (for --json, #55).

        Returns:
            Dict with timings, volumetrics, grid metadata, calibration,
            seed and output paths (CLI params + git commit are added by main).

        Raises:
            RuntimeError: If run() has not completed successfully.

        """
        if self._summary is None:
            msg = "run_summary() called before a successful run()"
            raise RuntimeError(msg)
        return self._summary

    def _collect_summary(
        self,
        vpath: str,
        spath: str,
        t_ingest: float,
        t_calibrate: float,
        t_grid: float,
        t_write: float,
        t_total: float,
    ) -> dict[str, Any]:
        """Assemble the --json run summary dict from post-run state (#55).

        All values are plain Python types (JSON-safe): numpy scalars are
        converted, the turbo dataclasses go through asdict().

        """
        pw = self._phase_wall
        disk_v = Path(vpath).stat().st_size
        disk_s = Path(spath).stat().st_size
        raw = self.nx * self.ny * 2  # one int16 band
        cells_per_level = [(self.nx_padded >> k) * (self.ny_padded >> k) for k in range(self.levels + 1)]
        turbo: dict[str, Any] | None = None
        if self._turbo_decision is not None:
            turbo = {
                "enabled": True,
                "decision": asdict(self._turbo_decision),
                "plan": asdict(self._turbo_plan) if self._turbo_plan is not None else None,
                "budget_bytes": self._turbo_budget_bytes,
            }

        def ms(seconds: float) -> float:
            return round(seconds * 1e3, 3)

        return {
            "ppgrid_version": __version__,
            "seed": self.seed,
            "transform": self.tname,
            "cap_km": float(self.cap_km_val),
            "grid": {
                "nx": self.nx,
                "ny": self.ny,
                "cells": self.nx * self.ny,
                "res_m": float(self.res),
                "x0_m": float(self.x0),
                "y0_m": float(self.y0),
                "nx_padded": self.nx_padded,
                "ny_padded": self.ny_padded,
                "levels": self.levels,
                "step": self.step,
                "halo": self.halo,
                "bsize": self.bsize,
                "nbx": self.nbx,
                "nby": self.nby,
                "n_tasks": len(self.tasks),
                "n_empty_blocks": len(self.empty_blocks),
                "src_crs": self.src_crs,
                "work_crs": self.work_crs,
                "out_crs": self.out_crs,
                "saturation": float(self.saturation),
                "scale": float(self.scale),
                "percentile_step": self.percentile_step,
                "block_size": self.block_size,
                "compress": self.compress,
            },
            "ingest": self._ingest_summary_info(),
            "calibration": self._cal,
            "timings_ms": {
                "ingest": ms(t_ingest),
                "calibrate": ms(t_calibrate),
                "grid": ms(t_grid),
                "write": ms(t_write),
                "total": ms(t_total),
                "subphases": {k: ms(v) for k, v in pw.items()},
            },
            "volumetrics": {
                "value": {
                    "raw_bytes": raw,
                    "compressed_bytes": disk_v,
                    "compression_ratio": (raw / disk_v) if disk_v else None,
                },
                "support": {
                    "raw_bytes": raw,
                    "compressed_bytes": disk_s,
                    "compression_ratio": (raw / disk_s) if disk_s else None,
                },
                "cells_per_level": cells_per_level,
            },
            "turbo": turbo,
            "outputs": {"value": vpath, "support": spath},
        }

    def _log_run_summary(
        self,
        t_ingest: float,
        t_calibrate: float,
        t_grid: float,
        t_write: float,
        t_total: float,
        vpath: str,
        spath: str,
    ) -> None:
        """Emit the run's log lines: phase progress (INFO) + detail (DEBUG).

        INFO (default): one line per phase, then a final summary
        (rows -> grid WxH cells, total wall time, output paths). DEBUG
        (--verbose / LOGGING=verbose): per-phase wall time in ms,
        volumetric counts (rows read, NaN/inf dropped, cells per level,
        output bytes raw vs compressed) and the resolved plan.

        Never raises: a logging failure must not fail a successful run.

        Args:
            t_ingest: ingest wall seconds.
            t_calibrate: calibrate wall seconds.
            t_grid: grid (spatial index) wall seconds.
            t_write: write (interpolate + reproject + raster I/O) wall seconds.
            t_total: total run wall seconds.
            vpath: Value GeoTIFF path.
            spath: Support GeoTIFF path.

        """
        pw = self._phase_wall
        try:
            reproj_s = pw.get("reproject_value", 0.0) + pw.get("reproject_support", 0.0)
            interp_s = pw.get("interpolate", 0.0) + pw.get("descent", 0.0)
            log.info(
                "ingest: %d rows -> %d points (%s) in %.2fs",
                self.n_total,
                self.n,
                self._drop_summary(),
                t_ingest,
            )
            log.info("calibrate: transform=%s cap_km=%.4g in %.2fs", self.tname, self.cap_km_val, t_calibrate)
            log.info("grid: %dx%d cells (%d) in %.2fs", self.nx, self.ny, self.nx * self.ny, t_grid)
            log.info("interpolate: done in %.2fs", interp_s)
            log.info("write: reproject %.2fs, done in %.2fs", reproj_s, t_write)
            disk_v = Path(vpath).stat().st_size
            disk_s = Path(spath).stat().st_size
            log.info(
                "done: %d rows -> %dx%d grid (%d cells) in %.2fs -> %s, %s",
                self.n_total,
                self.nx,
                self.ny,
                self.nx * self.ny,
                t_total,
                vpath,
                spath,
            )
            if log.isEnabledFor(logging.DEBUG):
                raw = self.nx * self.ny * 2  # one int16 band
                cells_per_level = [(self.nx_padded >> k) * (self.ny_padded >> k) for k in range(self.levels + 1)]
                total_disk = disk_v + disk_s
                log.debug(
                    "perf: ingest=%.0fms calibrate=%.0fms grid=%.0fms descent=%.0fms interpolate=%.0fms "
                    "reproject_value=%.0fms reproject_support=%.0fms write=%.0fms total=%.0fms",
                    t_ingest * 1e3,
                    t_calibrate * 1e3,
                    t_grid * 1e3,
                    pw.get("descent", 0.0) * 1e3,
                    pw.get("interpolate", 0.0) * 1e3,
                    pw.get("reproject_value", 0.0) * 1e3,
                    pw.get("reproject_support", 0.0) * 1e3,
                    t_write * 1e3,
                    t_total * 1e3,
                )
                log.debug(
                    "volumetrics: rows_read=%d points_ingested=%d dropped_nan_inf=%d "
                    "dropped_out_of_domain=%d dropped_nonfinite_proj=%d cells=%d levels=%d "
                    "cells_per_level=%s value=%dB (raw %dB) support=%dB (raw %dB) compression_ratio=%.2f",
                    self.n_total,
                    self.n,
                    self.n_dropped_nan_inf,
                    self.n_dropped_out_of_domain,
                    self.n_dropped_nonfinite_proj,
                    self.nx * self.ny,
                    self.levels,
                    cells_per_level,
                    disk_v,
                    raw,
                    disk_s,
                    raw,
                    (raw * 2) / total_disk if total_disk else 0.0,
                )
                if self._turbo_decision is not None:
                    turbo_desc = (
                        f"on (regime {self._turbo_decision.regime}, path {self._turbo_decision.path}, "
                        f"cap {self._turbo_plan.cap_gb:g} GB)"
                        if self._turbo_plan is not None
                        else "on"
                    )
                else:
                    turbo_desc = "off"
                log.debug(
                    "plan: workers=%d n_threads=%d res=%.4g m block=%d levels=%d cap_km=%.4g transform=%s turbo=%s",
                    self.workers,
                    self.n_threads,
                    self.res,
                    self.bsize,
                    self.levels,
                    self.cap_km_val,
                    self.tname,
                    turbo_desc,
                )
        except Exception:  # ruff: ignore[blind-except, try-except-pass] — logging must never fail a successful run
            pass

    def _remove_run_temps(self) -> None:
        """Remove every run temp file in out_dir (idempotent, best effort)."""
        for fname in _RUN_TEMP_FILES:
            (Path(self.out_dir) / fname).unlink(missing_ok=True)

    def _resolve_turbo_cap(self) -> tuple[float, str]:
        """Resolve the turbo RAM cap with the cgroup-aware preset clamp (M-2).

        Passes both physical RAM and the cgroup ceiling so a --turbo /
        --max-ram preset at/above the process-visible memory clamps to
        available - 8 GB instead of planning against RAM the process
        cannot see (a capped slice OOMs at its own limit).

        PPGRID_CAP_GB (debug/test only, not a CLI flag): forces the cap,
        skipping detection and the clamp. Used to exercise a regime on a
        capped slice without matching the slice to the box.

        """
        override = os.environ.get("PPGRID_CAP_GB")
        if override is not None:
            return max(float(override), turbop.CAP_FLOOR_GB), f"PPGRID_CAP_GB override {override:g} GB"
        return turbop.resolve_cap(
            self.turbo_cap_gb,
            physical_gb=turbop._read_physical_gb(),  # ruff: ignore[private-member-access]
            cgroup_gb=turbop._read_cgroup_max_gb(),  # ruff: ignore[private-member-access]
        )

    def _per_box_peak_bytes(self) -> int | None:
        """Worst-case per-box fallback peak, or None pre-grid (M-3).

        The per-box block loop runs up to `workers` boxes in flight, each
        holding a snapped (bsize + 2*halo) grid; on the turbo write the
        4 B/cell DN field is also alive when the output is reprojected.
        Used to announce a per-box OOM risk instead of promising exit-0
        correct output into an OOM.

        """
        if not all(hasattr(self, attr) for attr in ("bsize", "halo", "step", "nx", "ny", "tasks", "n")):
            return None
        worst_dim = -(-(self.bsize + 2 * self.halo) // self.step) * self.step
        extra = self.n * 32  # x/y/tv/v float64 points arrays, alive all run
        if self.out_crs != self.work_crs:
            extra += self.nx * self.ny * 4  # in-RAM DN field (turbo write)
        return turbop.per_box_peak_bytes(
            worst_dim * worst_dim,
            in_flight=min(self.workers, len(self.tasks)),
            extra_bytes=extra,
        )

    def _turbo_precheck(self) -> None:
        """Size the turbo plan against the RAM budget before heavy allocation (3.5).

        Runs after grid() (which knows n, cells, levels) and before
        _prepare_shared's first big allocation. Prints the budget value
        actually used. Shared infeasible: strict -> stderr error + exit 3;
        non-strict -> [warn] + continue on the per-box path (exit 0, correct
        output, turbo warp/write still apply). When the per-box est peak
        itself exceeds the budget the warning is explicit: best-effort,
        OOM risk, output may be partial (M-3).

        Raises:
            SystemExit: code 3 when --turbo-strict and the shared path does
                not fit under the budget.

        """
        if not self.turbo:
            self._turbo_plan = None
            self._turbo_decision = None
            self._turbo_budget_bytes = None
            return
        n_cells = self.nx_padded * self.ny_padded
        radius = round(self.cap_km_val * M_PER_KM / self.res)
        cap_gb, cap_source = self._resolve_turbo_cap()
        plan = turbop.plan(cap_gb, n_cells, self.nx_padded, radius)
        # The process's actual memory ceiling (audit #78 P1-2): the 8 GB
        # cap floor can plan a budget above it on a small cgroup slice;
        # precheck names the OOM risk instead of staying green.
        available_gb = min(
            turbop._read_physical_gb(),  # ruff: ignore[private-member-access]
            turbop._read_cgroup_max_gb(),  # ruff: ignore[private-member-access]
        )
        decision = turbop.precheck(
            plan,
            strict=self.turbo_strict,
            cap_source=cap_source,
            per_box_peak_bytes=self._per_box_peak_bytes(),
            available_gb=available_gb,
        )
        self._turbo_plan = plan
        self._turbo_decision = decision
        self._turbo_budget_bytes = plan.budget_bytes
        if self.dry_run:
            # --plan: the plan printout carries the decision; only a hard
            # failure (--turbo-strict infeasible) says its piece on stderr.
            if decision.exit_code == 3 and decision.message:
                print(decision.message, file=sys.stderr)  # ruff: ignore[print]
        else:
            # stdout stays clean (the --json/--help contract): the summary
            # goes through the run logger (stderr, INFO; -q suppresses it,
            # like every other progress line) (#77).
            log.info("%s", decision.summary)
            if decision.message:
                print(decision.message, file=sys.stderr)  # ruff: ignore[print]
        if decision.exit_code == 3:
            raise SystemExit(3)

    def _turbo_zstd_ok(self) -> bool:
        """A.7 calibration oracle gate for the parallel ZSTD write.

        Writes one deterministic 500x512 int16 probe (partial width, so
        the gate also covers GDAL's zero-padding of edge tiles - audit
        #81 F-1) with the stock codec settings, then asks zstdmt whether
        the turbo-compressed frame is byte-equal to the bytes GDAL
        stored. Mismatch (or unavailable) -> False: the caller writes
        with the stock-codec parallel path (_turbo_write_stock_parallel);
        the serial per-band write runs only if that writer's layout guards
        fail. Either way the output is byte-exact.
        """
        # Deterministic compressible content: both axes differ to small
        # values, so predictor 2 output is near-zero structured data.
        gx = np.arange(500, dtype=np.int16) * 13
        gy = np.arange(512, dtype=np.int16) * 7
        tile = (gx[None, :] + gy[:, None]).astype(np.int16)
        probe = Path(self.out_dir) / "_zstd_oracle.tif"
        try:
            # zstdmt.available() inside the try (audit #81 F-3): a gate
            # failure must mean "stock parallel write", never a crash.
            if not zstdmt.available():
                print("[warn] turbo zstd oracle: CPL zstd unavailable; stock parallel write", file=sys.stderr)  # ruff: ignore[print]
                return False
            with rasterio.open(
                probe,
                "w",
                driver="GTiff",
                dtype="int16",
                width=500,
                height=512,
                count=1,
                nodata=NODATA,
                crs=f"EPSG:{self.work_crs}",
                transform=from_origin(0, 512 * self.res, self.res, self.res),
                compress=self.compress,
                tiled=True,
                blockxsize=512,
                blockysize=512,
                predictor=2,
                BIGTIFF="IF_SAFER",
            ) as dst:
                dst.write(tile, 1)
            verdict = zstdmt.oracle_check(probe, tile=0)
            if verdict.ok:
                return True
            print(f"[warn] {verdict.detail}", file=sys.stderr)  # ruff: ignore[print]
        except Exception as e:  # ruff: ignore[blind-except] - any probe failure means the stock parallel writer runs
            print(f"[warn] turbo zstd oracle: {e}; stock parallel write", file=sys.stderr)  # ruff: ignore[print]
        finally:
            probe.unlink(missing_ok=True)
        return False

    def _is_regime_a(self) -> bool:
        """Return True when the pre-check selected regime A (full in-RAM)."""
        d = self._turbo_decision
        return d is not None and d.regime == "A"

    def _prepare_shared(self) -> bool:
        """Prebuild the full-grid pull-push field shared by all blocks (bit-exact).

        Bins every point once into the padded grid, builds the pyramid once,
        and runs the descent once over the whole grid; each block then slices
        the result. Bit-identical to the per-block path: the full-grid descent
        differs from a per-box descent only within one pyramid step of each
        box edge (tent-filter edge rules propagate 2**levels - 2 rows inward),
        and the halo margin is a full step, so block regions are clean. The 3x3
        point gather is exactly the set of points inside each box (checked
        below), so the binned grids match the per-box grids exactly. The near
        window (cap cells) is always inside the halo, so the global box count
        matches the per-box one in block regions.

        Returns False (per-block path) when the geometry or size check fails.

        """
        if self.turbo:
            # S3.3: the shared/per-box decision comes from the pre-check
            # (3.5), not the hard 2.5e8 cap; the thread count comes from
            # the turbo plan (in-flight units are budgeted, see turbop.plan).
            if self._turbo_decision is None or self._turbo_decision.path != "shared":
                return False
            n_threads = self._turbo_plan.workers if self._turbo_plan is not None else 1
        else:
            if self.nx_padded * self.ny_padded > _SHARED_MAX_CELLS:
                return False
            # PR #27 (low-RAM path): the shared descent runs parallel
            # row-banded. --max-band-parallel caps the per-level pool; without
            # the flag the read-side worker count is used (main's baseline
            # for this path was serial, n_threads = 1). Bit-identical at any
            # thread count (S3), and _descent_banded's RAM gate drops to
            # single-thread when available memory is tight.
            n_threads = self.max_band_parallel if self.max_band_parallel is not None else self.workers
        step = self.step
        bsize = self.bsize
        halo = self.halo
        for bx in range(self.nbx):
            for by in range(self.nby):
                i0, j0, i1, j1 = _block_bounds(bx, by, bsize, self.nx, self.ny)
                hi0, hj0, hi1, hj1 = _snapped_box(i0, j0, i1, j1, halo, step, self.nx_padded, self.ny_padded)
                # Box must stay within the cells _block_points gathers (3x3).
                if hi0 < max(0, (bx - 1) * bsize) or hi1 > min(self.nx_padded, (bx + 2) * bsize):
                    return False
                if hj0 < max(0, (by - 1) * bsize) or hj1 > min(self.ny_padded, (by + 2) * bsize):
                    return False

        pts = np.load(self.pts_path, mmap_mode="r")
        with _prof.phase("shared.ix"):
            ix = ((pts[PTS_X][:] - self.x0) // self.res).astype(np.int64)
            iy = ((pts[PTS_Y][:] - self.y0) // self.res).astype(np.int64)
        cap_cells = round(self.cap_km_val * M_PER_KM / self.res)
        out_dir = Path(self.out_dir)
        force_memmap = os.environ.get("PPGRID_FORCE_MEMMAP") == "1"
        # Regime A = "full in-RAM, all workers" (SIZING table): keep the
        # shared grids in anonymous RAM even past the 1e8 memmap threshold
        # (issue #39: on a box under memory pressure the tmpfs memmap pages
        # self-evict and the run pays a 46 us/fault re-fault storm).
        use_memmap = (
            self.nx_padded * self.ny_padded > _SHARED_MEMMAP_CELLS and not (self.turbo and self._is_regime_a())
        ) or force_memmap
        if use_memmap:
            # Keep the finest grids off the RAM budget: memmap + row banded
            # bin/near (bit-identical to the in-RAM pass, see pullpush).
            shape = (self.nx_padded, self.ny_padded)
            s0 = _open_fresh_memmap(out_dir / "_s0.npy", np.float32, shape)
            c0 = _open_fresh_memmap(out_dir / "_c0.npy", np.float32, shape)
            _nohuge(s0, c0)
            with _prof.phase("shared.bin"):
                bin_points_banded(s0, c0, ix, iy, pts[PTS_TV][:], self.ny_padded)
            near_full = _open_fresh_memmap(out_dir / "_near.npy", bool, shape)
            _nohuge(near_full)
            with _prof.phase("shared.boxcount"):
                box_count_banded(c0, cap_cells, near_full, n_threads=n_threads)
        else:
            if self.nx_padded * self.ny_padded > _SHARED_MEMMAP_CELLS:
                # Regime A scale (issue #39): the flat f64 bincount
                # intermediates of bin_points (two 8 B/cell arrays plus two
                # astype copies) are the single largest first-touch churn in
                # the run. The banded scatter is bit-identical (same per-cell
                # f64 accumulation order) and fills preallocated f32 grids.
                with _prof.phase("shared.bin"):
                    s0 = np.empty((self.nx_padded, self.ny_padded), np.float32)
                    c0 = np.empty((self.nx_padded, self.ny_padded), np.float32)
                    _nohuge(s0, c0)
                    # rows without points are never written by bin_points_banded
                    s0.fill(0)
                    c0.fill(0)
                    bin_points_banded(s0, c0, ix, iy, pts[PTS_TV][:], self.ny_padded)
            else:
                with _prof.phase("shared.bin"):
                    s0, c0 = bin_points(ix, iy, pts[PTS_TV][:], self.nx_padded, self.ny_padded)
            if n_threads > 1:
                with _prof.phase("shared.boxcount"):
                    near_full = box_count_mt(c0, cap_cells, n_threads=n_threads) > 0
            else:
                with _prof.phase("shared.boxcount"):
                    near_full = box_count(c0, cap_cells) > 0

        with _prof.phase("shared.pyramid"):
            sums: list[np.ndarray] = [s0]
            for _ in range(self.levels):
                sums.append(downsample_sum(sums[-1]))
            counts: list[np.ndarray] = [c0]
            for _ in range(self.levels):
                counts.append(downsample_sum(counts[-1]))

        if not use_memmap:
            out_val = out_sup = None
            if self.turbo and self._is_regime_a():
                # Regime A (issue #39 pass 3): back the output fields with
                # tmpfs instead of anon RAM. The dnfill pass re-reads every
                # field cell once; on this box external ~15 GB memory hogs
                # (qemu CI runners) push anon field pages into swap
                # (100 us+/fault - the 110 s sys in write.dnfill). tmpfs
                # pages reclaim as clean file cache; a re-fault is a tmpfs
                # read. Descent writes the final level into these memmaps.
                shape = (self.nx_padded, self.ny_padded)
                val_full = _open_fresh_memmap(out_dir / "_val_full.npy", np.float32, shape)
                sup_full = _open_fresh_memmap(out_dir / "_sup_full.npy", np.float32, shape)
                _nohuge(val_full, sup_full)
                out_val, out_sup = val_full, sup_full
            with _prof.phase("shared.descent"):
                val_full, sup_full = _pull_push_descent(
                    sums,
                    counts,
                    self.res,
                    self.levels,
                    saturation=self.cfg.sat,
                    free_levels=True,
                    n_threads=n_threads,
                    out_val=out_val,
                    out_sup=out_sup,
                )
            # Level 0 is dead after the descent: free the finest grids
            # (two f32 full arrays, ~12 GB at full-AU scale) before the
            # dnfill block loop (issue #39).
            sums[0] = None  # type: ignore[index]
            counts[0] = None  # type: ignore[index]
            s0 = c0 = None  # type: ignore[assignment]
            del ix, iy
        else:
            # Full descent to level 2 (small arrays), then band levels 2 -> 1
            # -> 0 through memmap to bound RAM on multi-billion-cell grids.
            # levels == 1 (cap_km*1000/res <= 2) has no level 2: stop/start the
            # descent at level 1 instead, else the banded k=1 step blends a
            # level-1 local with a level-2 upsample and broadcast-crashes.
            lvl2 = min(2, self.levels)
            val2, sup2 = _pull_push_descent(
                sums,
                counts,
                self.res,
                self.levels,
                saturation=self.cfg.sat,
                stop_level=lvl2,
                free_levels=True,
                n_threads=n_threads,
            )
            val_full = _open_fresh_memmap(Path(self.out_dir) / "_val_full.npy", np.float32, s0.shape)
            sup_full = _open_fresh_memmap(Path(self.out_dir) / "_sup_full.npy", np.float32, s0.shape)
            with _prof.phase("shared.descent_banded"):
                _descent_banded(
                    sums,
                    counts,
                    self.res,
                    self.cfg.sat,
                    start_level=lvl2,
                    val_in=val2,
                    sup_in=sup2,
                    out_val=val_full,
                    out_sup=sup_full,
                    level_dir=Path(self.out_dir),
                    n_threads=n_threads,
                )
            for k in range(min(3, self.levels + 1)):
                sums[k] = None  # type: ignore[assignment]
                counts[k] = None  # type: ignore[assignment]
            val2 = sup2 = None  # type: ignore[assignment]
            s0 = c0 = None  # type: ignore[assignment]

        self.cfg.val_full = val_full
        self.cfg.sup_full = sup_full
        self.cfg.near_full = near_full
        return True

    def _write_rasters(self) -> tuple[str, str]:
        """Write block outputs to GeoTIFFs (post-ingest/calibrate/grid).

        Returns:
            Tuple of value and support GeoTIFF file paths.

        """
        # Raster profile
        xform = from_origin(self.x0, self.y0 + self.ny * self.res, self.res, self.res)
        common = {
            "driver": "GTiff",
            "height": self.ny,
            "width": self.nx,
            "count": 1,
            "crs": f"EPSG:{self.work_crs}",
            "transform": xform,
            "compress": self.compress,
            "tiled": True,
            "blockxsize": TILE_PX,
            "blockysize": TILE_PX,
            "BIGTIFF": "IF_SAFER",
        }
        vprof = dict(common, dtype="int16", nodata=NODATA, predictor=2)
        sprof = dict(common, dtype="int16", nodata=NODATA, predictor=2)

        vpath = Path(self.out_dir) / "value.tif"
        spath = Path(self.out_dir) / "support_km.tif"

        vtags: dict[str, str] = {
            "transform": self.tname,
            "cap_km": str(self.cap_km_val),
            "res_m": str(self.res),
            "scale": str(self.scale),
            "units": "percentile",
            "decode": f"percentile = DN/{self.scale:g}",
        }
        if self.percentile_step is not None:
            vtags["percentile_step"] = str(self.percentile_step)
        stags = {"decode": f"support_km = 2**(DN/{SUPPORT_LOG2_SCALE})"}

        # Turbo write (S3.4a): skip the work-CRS intermediate file, keep the
        # quantised int16 DN field in RAM, write the output CRS directly.
        # Only applies when the output is reprojected (out_crs != work_crs):
        # when the two match, the block-write file IS the final file.
        if self.turbo and self.out_crs != self.work_crs:
            self._write_rasters_turbo(vprof, sprof, vpath, spath, vtags, stags, xform)
        else:
            self._write_rasters_serial(vprof, sprof, vpath, spath, vtags, stags)
        return str(vpath), str(spath)

    def _write_rasters_serial(
        self,
        vprof: dict[str, Any],
        sprof: dict[str, Any],
        vpath: Path,
        spath: Path,
        vtags: dict[str, str],
        stags: dict[str, str],
    ) -> None:
        """Write block outputs to work-CRS GeoTIFFs.

        Serial path (non-turbo, A8: byte-identical reference) and the turbo
        fallback when the in-RAM DN field does not fit the planning budget.
        When the output CRS differs, the work-CRS rasters are renamed to
        intermediates and reprojected through _reproject_band.
        """
        # Staging names (issue #40): the finals (value.tif / support_km.tif)
        # are only ever published via os.replace() on success, so a mid-write
        # crash leaves .tmp / _tmp files instead of truncating them.
        ftmp_v = Path(self.out_dir) / (vpath.name + ".tmp")
        ftmp_s = Path(self.out_dir) / (spath.name + ".tmp")
        if self.out_crs != self.work_crs:
            wtmp_v = Path(self.out_dir) / "_value_tmp.tif"
            wtmp_s = Path(self.out_dir) / "_support_tmp.tif"
        else:
            wtmp_v, wtmp_s = ftmp_v, ftmp_s
        partial = False
        try:
            # Staging opens create fresh 0600 regular files and never follow
            # a pre-planted symlink (issues #15, #77); the ftmp finals are
            # created the same way by _reproject_core below.
            with (
                _open_tiff_staging(wtmp_v, **vprof) as vd,
                _open_tiff_staging(wtmp_s, **sprof) as sd,
            ):
                vd.update_tags(**vtags)
                vd.scales = (1.0 / self.scale,)
                sd.update_tags(**stags)

                # Write empty blocks as nodata
                for bx, by in self.empty_blocks:
                    w = _block_window(bx, by, self.cfg)
                    blank = np.full((w.height, w.width), NODATA, np.int16)
                    vd.write(blank, 1, window=w)
                    sd.write(blank, 1, window=w)

                # Process blocks with parallel workers
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Setting the shape on a NumPy array")
                    # One shared config object for all worker threads (no
                    # per-field dict copy). Rebuilt fresh each run: stale state
                    # from a previous run in this process is dropped.
                    _CTX.clear()
                    cfg = self.cfg
                    cfg.pts = np.load(cfg.pts_path, mmap_mode="r")
                    cfg.tf = make_transform(cfg.transform_state)
                    cfg.pct = PercentileTransform(cfg.pct_quantiles)
                    _CTX["cfg"] = cfg
                    with _WallPhase("descent", self._phase_wall):
                        if self._prepare_shared():
                            cfg.shared = True
                    with (
                        ThreadPoolExecutor(max_workers=self.workers) as ex,
                        _prof.phase("write.serial"),
                        _WallPhase("interpolate", self._phase_wall),
                    ):
                        for bx, by, out in ex.map(_process_block, self.tasks):
                            w = _block_window(bx, by, self.cfg)

                            if out is None:
                                blank = np.full((w.height, w.width), NODATA, np.int16)
                                vd.write(blank, 1, window=w)
                                sd.write(blank, 1, window=w)
                                continue

                            vq, rq = out
                            vd.write(vq.T[::-1, :], 1, window=w)
                            sd.write(rq.T[::-1, :], 1, window=w)
            if self.out_crs != self.work_crs:
                # Both bands share the same work-CRS source grid (identical
                # bounds/size), so the output grid is identical too: compute
                # the default transform once instead of per band.
                with rasterio.open(wtmp_v) as grid_src:
                    dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
                        grid_src.crs,
                        f"EPSG:{self.out_crs}",
                        grid_src.width,
                        grid_src.height,
                        *grid_src.bounds,
                    )
                # Out-of-domain grid extent -> NaN/INT32_MIN transform (issue #74).
                _check_default_transform(dst_transform, dst_width, dst_height, self.work_crs, self.out_crs)

                out_crs = f"EPSG:{self.out_crs}"
                # NoData-tile skip (C(ii), PR #27): the task blocks are the
                # superset of all non-NoData cells in the work rasters, so a
                # dst tile whose source footprint touches no task block can
                # be written as plain NoData without warping.
                task_info = None
                if self.tasks:
                    task_info = {
                        "bsize": self.bsize,
                        "nby": self.nby,
                        "task_bids": {bx * self.nby + by for bx, by in self.tasks},
                    }
                with _prof.phase("write.reproj_value"), _WallPhase("reproject_value", self._phase_wall):
                    _reproject_band(
                        str(wtmp_v),
                        str(ftmp_v),
                        vprof,
                        out_crs,
                        dst_transform,
                        dst_width,
                        dst_height,
                        task_info=task_info,
                        n_threads=self.n_threads,
                    )
                with _prof.phase("write.reproj_support"), _WallPhase("reproject_support", self._phase_wall):
                    _reproject_band(
                        str(wtmp_s),
                        str(ftmp_s),
                        sprof,
                        out_crs,
                        dst_transform,
                        dst_width,
                        dst_height,
                        task_info=task_info,
                        n_threads=self.n_threads,
                    )
            # Publish: the finals only ever change via atomic rename.
            ftmp_v.replace(vpath)
            self._published.add(vpath.name)
            ftmp_s.replace(spath)
            self._published.add(spath.name)
            if wtmp_v is not ftmp_v:
                wtmp_v.unlink(missing_ok=True)
                wtmp_s.unlink(missing_ok=True)
        except BaseException:
            # Mid-run failure (worker error, OOM, disk full): the finals were
            # not reached (os.replace is last). Warn so nobody consumes an
            # earlier run's files as this run's output (issues #15, #40).
            partial = True
            raise
        finally:
            self._remove_run_temps()
            if partial:
                print(  # ruff: ignore[print]
                    f"warning: run failed during write; {vpath}, {spath} were not fully written "
                    "(any existing files are from an earlier run, left untouched)",
                    file=sys.stderr,
                )

    def _write_rasters_turbo(
        self,
        vprof: dict[str, Any],
        sprof: dict[str, Any],
        vpath: Path,
        spath: Path,
        vtags: dict[str, str],
        stags: dict[str, str],
        xform: Any,
    ) -> None:
        """Write block outputs to the output-CRS GeoTIFFs from in-RAM arrays.

        S3.4a turbo write: skip the work-CRS intermediate file, keep the
        quantised int16 DN field in RAM, write the output CRS directly.

        Budget gate: the DN field is 4 B/cell (two int16 bands). Above half
        of the 0.85*C pre-check budget the serial write (work file + MT
        reproject) is the safe path (10 m full-AU DN field: ~626 GB).

        Serial strategy (final fallback, byte-identical reference): warp the
        output raster tile-by-tile from the in-RAM arrays through the same
        GDAL warp kernel and flush in serial raster-scan order, byte-identical
        to _reproject_band on the work-CRS file.
        Parallel strategies (issue #39): when the A.7 zstd oracle passes, warp
        + compress every 512^2 tile in parallel through the CPL zstd pfn and
        assemble the file from the stock GDAL head with the
        TileOffsets/TileByteCounts arrays patched (_turbo_write_parallel).
        When the oracle mismatches (the pfn drifts from libtiff's streaming
        codec, as on the bundled GDAL 3.12.4 stack), the stock codec runs
        per 2048^2 warp block through a per-thread scratch raster with the
        identical profile - frames byte-equal to the serial flush by
        construction on any stack (_turbo_write_stock_parallel).
        """
        budget = self._turbo_budget_bytes
        if budget is not None and self.nx * self.ny * 4 > budget // 2:
            self._write_rasters_serial(vprof, sprof, vpath, spath, vtags, stags)
            return

        # Staging names (issue #40): the finals are only ever published via
        # os.replace() on success, so a mid-write crash leaves .tmp files
        # instead of truncating value.tif / support_km.tif.
        ftmp_v = Path(self.out_dir) / (vpath.name + ".tmp")
        ftmp_s = Path(self.out_dir) / (spath.name + ".tmp")
        out_dir = Path(self.out_dir)

        partial = False
        try:
            with _prof.phase("write.dnfill"), _WallPhase("dnfill", self._phase_wall):
                # File-backed (tmpfs) DN fields, not anon RAM (issue #39
                # pass 3): the warp pass re-reads every 2048^2 block once,
                # and external ~15 GB memory hogs swap anon pages out of the
                # 3 GB fields (30 s/warp block observed vs 0.35 s resident).
                # tmpfs pages reclaim as clean file cache; a re-fault is a
                # tmpfs read, never swap I/O.
                val_dn = _open_fresh_memmap(Path(self.out_dir) / "_val_dn.npy", np.int16, (self.ny, self.nx))
                sup_dn = _open_fresh_memmap(Path(self.out_dir) / "_sup_dn.npy", np.int16, (self.ny, self.nx))
                _nohuge(val_dn, sup_dn)
                val_dn.fill(NODATA)
                sup_dn.fill(NODATA)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Setting the shape on a NumPy array")
                    # One shared config object for all worker threads (no
                    # per-field dict copy). Rebuilt fresh each run: stale state
                    # from a previous run in this process is dropped.
                    _CTX.clear()
                    cfg = self.cfg
                    cfg.pts = np.load(cfg.pts_path, mmap_mode="r")
                    cfg.tf = make_transform(cfg.transform_state)
                    cfg.pct = PercentileTransform(cfg.pct_quantiles)
                    _CTX["cfg"] = cfg
                    with _WallPhase("descent", self._phase_wall):
                        if self._prepare_shared():
                            cfg.shared = True
                    with (
                        _WallPhase("interpolate", self._phase_wall),
                        ThreadPoolExecutor(max_workers=self.workers) as ex,
                    ):
                        for bx, by, out in ex.map(_process_block, self.tasks):
                            w = _block_window(bx, by, self.cfg)
                            if out is None:
                                continue  # already NODATA
                            rows = slice(w.row_off, w.row_off + w.height)
                            cols = slice(w.col_off, w.col_off + w.width)
                            vq, rq = out
                            val_dn[rows, cols] = vq.T[::-1, :]
                            sup_dn[rows, cols] = rq.T[::-1, :]
                    # The shared field is dead after the block loop: free it
                    # before the reproject/write (val/sup f32 + near bool,
                    # ~13.7 GB at full-AU scale) (issue #39).
                    cfg.val_full = cfg.sup_full = cfg.near_full = None
            # Same position as before the restructure: the DN fields are
            # still open memmaps (the inode outlives the unlink), but the
            # tmpfs space is freed while the write phase runs.
            self._remove_run_temps()

            # Both bands share the same work-CRS source grid (identical
            # bounds/size), so the output grid is identical too: compute the
            # default transform once instead of per band.
            work_crs = f"EPSG:{self.work_crs}"
            out_crs = f"EPSG:{self.out_crs}"
            dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
                work_crs,
                out_crs,
                self.nx,
                self.ny,
                self.x0,
                self.y0,
                self.x0 + self.nx * self.res,
                self.y0 + self.ny * self.res,
            )
            # Out-of-domain grid extent -> NaN/INT32_MIN transform (issue #74).
            _check_default_transform(dst_transform, dst_width, dst_height, self.work_crs, self.out_crs)
            # Budgeted MT thread count from the turbo plan drives the warp pool;
            # the n_threads kwarg stays the explicit non-turbo knob.
            warp_threads = self._turbo_plan.workers if self._turbo_plan is not None else self.n_threads
            with _prof.phase("write.oracle"):
                cpl = self._turbo_zstd_ok()
            # The writers below create ftmp_v / ftmp_s fresh (O_EXCL |
            # O_NOFOLLOW) and never follow a pre-planted symlink (issues #15, #77).
            for band, arr, prof, path, tags, scales in (
                ("value", val_dn, vprof, ftmp_v, vtags, (1.0 / self.scale,)),
                ("support", sup_dn, sprof, ftmp_s, stags, (1.0,)),
            ):
                with _prof.phase(f"write.reproj_{band}"), _WallPhase(f"reproject_{band}", self._phase_wall):
                    wrote = False
                    if cpl:
                        try:
                            wrote = _turbo_write_parallel(
                                arr,
                                xform,
                                work_crs,
                                str(path),
                                prof,
                                out_crs,
                                dst_transform,
                                dst_width,
                                dst_height,
                                tags=tags,
                                scales=scales,
                                offsets=(0.0,),
                                n_threads=warp_threads,
                            )
                        except Exception as e:  # ruff: ignore[blind-except] - layout surprise: fall back
                            print(f"[warn] turbo zstd parallel: {e}; stock parallel write", file=sys.stderr)  # ruff: ignore[print]
                    if not wrote:
                        try:
                            wrote = _turbo_write_stock_parallel(
                                arr,
                                xform,
                                work_crs,
                                str(path),
                                prof,
                                out_crs,
                                dst_transform,
                                dst_width,
                                dst_height,
                                tags=tags,
                                scales=scales,
                                offsets=(0.0,),
                                n_threads=warp_threads,
                                scratch_dir=str(out_dir),
                            )
                        except Exception as e:  # ruff: ignore[blind-except] - any surprise: serial
                            print(f"[warn] turbo stock parallel: {e}; serial write", file=sys.stderr)  # ruff: ignore[print]
                    if not wrote:
                        # Layout guard tripped (see the parallel writers): the
                        # serial array reproject is byte-identical by
                        # construction.
                        _reproject_band_array(
                            arr,
                            xform,
                            work_crs,
                            str(path),
                            prof,
                            out_crs,
                            dst_transform,
                            dst_width,
                            dst_height,
                            tags=tags,
                            scales=scales,
                            offsets=(0.0,),
                            n_threads=warp_threads,
                        )
            # Publish: the finals only ever change via atomic rename.
            ftmp_v.replace(vpath)
            self._published.add(vpath.name)
            ftmp_s.replace(spath)
            self._published.add(spath.name)
        except BaseException:
            # Mid-run failure (worker error, OOM, disk full): the finals were
            # not reached (os.replace is last). Warn so nobody consumes an
            # earlier run's files as this run's output (issues #15, #40).
            partial = True
            raise
        finally:
            self._remove_run_temps()
            for p in out_dir.glob("_stock_scratch_*.tif"):
                p.unlink(missing_ok=True)
            if partial:
                print(  # ruff: ignore[print]
                    f"warning: run failed during write; {vpath}, {spath} were not fully written "
                    "(any existing files are from an earlier run, left untouched)",
                    file=sys.stderr,
                )


def run(
    input_path: str,
    value_col: str,
    lng_col: str,
    lat_col: str,
    out_dir: str,
    **kwargs: Any,
) -> tuple[str, str]:
    """Full pipeline: load, calibrate, interpolate, write.

    Returns:
        Tuple of value and support GeoTIFF file paths.

    """
    p = Pipeline(
        input_path,
        value_col,
        lng_col,
        lat_col,
        out_dir,
        **kwargs,
    )
    return p.run()


def _input_columns(path: str) -> list[str] | None:
    """Cheaply read the input file's column names for the #57 preflight.

    Args:
        path: CSV or Parquet input path.

    Returns:
        Column name list, or None when they cannot be read cheaply (missing
        file, unreadable format, parquet without pyarrow) — the pipeline run
        then reports the real error.

    """
    try:
        if path.endswith((".parquet", ".pq")):
            import pyarrow.parquet as pq

            with pq.ParquetFile(path) as f:
                return [str(n) for n in f.schema.names]
        df = pd.read_csv(path, nrows=0)
        return [str(c) for c in df.columns]
    except Exception:  # ruff: ignore[blind-except] — preflight is best-effort; the run reports the real error
        return None


def _check_columns(path: str, value_col: str, lng_col: str, lat_col: str) -> None:
    """Exit 2 with the real columns + closest matches on a missing column (#57).

    Args:
        path: Input file path.
        value_col: --value-col name.
        lng_col: --lng-col name.
        lat_col: --lat-col name.

    Raises:
        SystemExit: code 2 when a requested column is absent from the input.

    """
    cols = _input_columns(path)
    if cols is None:
        return
    for flag, name in (("--value-col", value_col), ("--lng-col", lng_col), ("--lat-col", lat_col)):
        if name in cols:
            continue
        shown = ", ".join(cols[:24]) + (f", ... ({len(cols)} total)" if len(cols) > 24 else "")
        lines = [
            f"error: {flag} {name!r} not found in {path}",
            f"  columns found: {shown}",
        ]
        close = difflib.get_close_matches(name, cols, n=3)
        if close:
            lines.append(f"  did you mean: {', '.join(close)}?")
        print("\n".join(lines), file=sys.stderr)  # ruff: ignore[print]
        raise SystemExit(2)


def _git_commit() -> str | None:
    """HEAD commit of the checkout this package runs from (None outside git).

    Runs a short `git rev-parse HEAD` in the package directory so an installed
    (non-git) copy yields None instead of a hang or a wrong hash.

    Returns:
        Commit hash string, or None.

    """
    git = shutil.which("git")
    if git is None:
        return None
    try:
        # argv is [which(git), "rev-parse", "HEAD"] — fully controlled, not untrusted input
        r = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [git, "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def _make_pipeline(args: argparse.Namespace, *, dry_run: bool) -> Pipeline:
    """Build the Pipeline from parsed CLI args (shared by --plan and run).

    Resolves the turbo flag/cap pair in one place so the two paths cannot
    drift.

    Args:
        args: Parsed CLI namespace.
        dry_run: True for --plan (no files created).

    Returns:
        Configured Pipeline, not yet run.

    """
    turbo = args.turbo is not None or args.max_ram is not None
    turbo_cap_gb: float | None
    if args.max_ram is not None:
        turbo_cap_gb = args.max_ram
    elif args.turbo not in (None, "auto"):
        turbo_cap_gb = float(args.turbo)
    else:
        turbo_cap_gb = None
    return Pipeline(
        args.input,
        args.value_col,
        args.lng_col,
        args.lat_col,
        args.out,
        res=args.res,
        cap_km=args.cap_km,
        transform=args.transform,
        saturation=args.saturation,
        block_size=args.block,
        workers=args.workers,
        calib_path=args.calibration,
        scale=args.scale,
        percentile_step=args.percentile_step,
        compress=args.compress,
        calib_max_points=args.calib_max_points,
        src_crs=args.src_crs,
        work_crs=args.work_crs,
        out_crs=args.out_crs,
        skip_calibration=args.skip_calibration,
        max_band_parallel=args.max_band_parallel,
        turbo=turbo,
        turbo_cap_gb=turbo_cap_gb,
        turbo_strict=args.turbo_strict,
        seed=args.seed,
        dry_run=dry_run,
    )


def _map_pipeline_errors(fn: Callable[[], Any], input_path: str) -> Any:
    """Run a pipeline phase, mapping exceptions to the CLI exit-code contract.

    Shared by the --plan phases and the full run so both report identically:
    exit 1 for pipeline/IO errors, exit 2 for validation errors (bad
    columns, no valid points), exit 3 passes through (--turbo-strict).

    Args:
        fn: Zero-arg callable running the phase (or the full run).
        input_path: Input path for the FileNotFoundError message.

    Returns:
        The callable's return value.

    Raises:
        SystemExit: 1 (pipeline/IO) or 2 (validation, unknown EPSG).

    """
    try:
        return fn()
    except (CRSError, RasterioCRSError) as e:
        # Unknown EPSG: pyproj's CRSError (--src-crs/--work-crs) and
        # rasterio's CRSError (--out-crs, a ValueError subclass) both land
        # here — one exit code for all three CRS flags: 2, "bad flag value"
        # per the README exit-code table (#77). The message names the code.
        print(f"error: {e}", file=sys.stderr)  # ruff: ignore[print] — CLI error output
        raise SystemExit(2) from None
    except ImportError as e:
        # e.g. parquet input without the optional pyarrow extra.
        _die(str(e))
    except FileNotFoundError as e:
        _die(f"input file not found: {e.filename or input_path}")
    except pd.errors.ParserError as e:
        _die(f"malformed input file: {e}")
    except RasterioIOError as e:
        # A GDAL I/O failure mid-run (corrupt/short raster, driver error).
        # Subclass of OSError; named here so the exit-1 group reads as the
        # full list of pipeline/IO errors (audit #83).
        _die(str(e))
    except (IndexError, OverflowError, ZeroDivisionError, RuntimeError) as e:
        # Invariant-style failures (a bad shape, a div-by-zero in a derived
        # constant, a worker signalling failure via RuntimeError): pipeline
        # bugs or a broken run, not user input errors, so exit 1 like the
        # other pipeline failures rather than exit 2 (audit #83).
        _die(f"{type(e).__name__}: {e} (input: {input_path})")
    except (KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)  # ruff: ignore[print] — CLI error output
        raise SystemExit(2) from None
    except OSError as e:
        # NotADirectoryError, disk full, etc.
        _die(str(e))


def _sanitize_json(obj: Any) -> Any:
    """Recursively replace non-finite floats with None (strict JSON, #55).

    An external --calibration file may carry NaN/Infinity literals in
    fields that are not pre-validated; json.dump(allow_nan=False) would
    raise, so they become null in the summary.

    Args:
        obj: Summary structure (dict/list/scalar).

    Returns:
        A copy containing only finite floats.

    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    return obj


def _write_json_summary(p: Pipeline, args: argparse.Namespace, out: Path) -> None:
    """Write <out>/run_summary.json for --json (issue #55); stdout stays clean.

    Atomic like the raster finals (fresh 0600 tmp + os.replace): a
    mid-write crash leaves no partial summary to block the next run, and a
    write failure (e.g. run_summary.json pre-existing as a directory) is a
    named error — exit 1 — never a raw traceback (#77). The rasters were
    already published, so no stale-outputs warning fires either (the
    run-phase handler filters on Pipeline._published).

    Args:
        p: Pipeline after a successful run() (provides run_summary()).
        args: Parsed CLI namespace (recorded under 'cli').
        out: Output directory.

    """
    summary = p.run_summary()
    summary["cli"] = dict(vars(args))
    summary["git_commit"] = _git_commit()
    path = out / "run_summary.json"
    tmp = out / "run_summary.json.tmp"
    try:
        with os.fdopen(_staging_fd(tmp), "wb") as f:
            f.write((json.dumps(_sanitize_json(summary), indent=2, allow_nan=False) + "\n").encode("utf-8"))
        tmp.replace(path)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        _die(f"cannot write run summary: {e}")
    log.info("run summary: %s", path)


def _resolve_log_level(*, verbose: bool, log_level: str | None, quiet: bool = False) -> str:
    """Resolve the ppgrid log level: explicit flag > quiet > verbose > env > info.

    Precedence: `--log-level` (most specific) > `--quiet` (suppress below
    WARNING, #52) > `--verbose` (shorthand for debug) > `LOGGING` > `info`.
    Env values accept the four level names or the alias `verbose` (case-
    insensitive, = debug); an unrecognized value warns and falls back to
    info (a mistyped env var must not break the run).

    Args:
        verbose: --verbose flag.
        log_level: --log-level flag value, or None when the flag is absent.
        quiet: --quiet flag (suppress stderr below WARNING).

    Returns:
        One of 'error', 'warning', 'info', 'debug'.

    """
    if log_level is not None:
        return log_level
    if quiet:
        return "warning"
    if verbose:
        return "debug"
    raw = os.environ.get("LOGGING")
    if raw is None:
        return "info"
    val = raw.strip().lower()
    if val == "verbose":
        val = "debug"
    if val in _LOG_LEVELS:
        return val
    print(f"warning: unrecognized LOGGING={raw!r}; using 'info'", file=sys.stderr)  # ruff: ignore[print]
    return "info"


def _configure_logging(level_name: str) -> None:
    """Configure the `ppgrid` logger for one CLI run (stderr, compact format).

    Only the `ppgrid` logger is touched — never the root logger — so library
    users' logging setup is undisturbed and stdout stays clean for any
    machine-readable output. Idempotent: repeated main() calls in one
    process (tests) update the level instead of stacking handlers.

    Args:
        level_name: Resolved level name from _resolve_log_level.

    """
    levels = {"error": logging.ERROR, "warning": logging.WARNING, "info": logging.INFO, "debug": logging.DEBUG}
    root = logging.getLogger("ppgrid")
    root.setLevel(levels[level_name])
    for handler in root.handlers:
        if getattr(handler, "ppgrid_handler", False):
            # Rebind to the current stderr: tests (capsys) and TTY swaps
            # replace sys.stderr between runs; holding the first run's
            # stream object sends later records into a closed file
            # (logging-error spam, #77).
            if handler.stream is not sys.stderr:  # type: ignore[attr-defined]
                # Assign directly: setStream() flushes the old stream first,
                # and that raises on a closed file (capsys test teardown).
                handler.stream = sys.stderr  # type: ignore[attr-defined]
            return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.ppgrid_handler = True  # type: ignore[attr-defined]
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser (tests exercise the flags here)."""
    parser = argparse.ArgumentParser(
        description="Pull-push scattered-data interpolation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    parser.add_argument("--version", action="version", version=f"ppgrid {__version__}")
    parser.add_argument("input", nargs="?", help="CSV or Parquet input path")
    parser.add_argument("-o", "--out", default="out/", help="Output directory (default: ./out)")
    parser.add_argument("--value-col", default="value", help="Value column name")
    parser.add_argument("--lng-col", default="longitude", help="Longitude column name")
    parser.add_argument("--lat-col", default="latitude", help="Latitude column name")
    parser.add_argument("--res", type=_pos_float, default=DEFAULT_RES, help="Cell size in metres")
    parser.add_argument("--cap-km", type=_cap_km_arg, default="auto", help="Fill cap km, or 'auto'")
    parser.add_argument(
        "--transform",
        default="auto",
        # Pinned to the calibrate.transforms() factory names (audit #83):
        # adding a transform there adds its name here, nothing else to update.
        choices=["auto", *(t.name for t in transforms())],
    )
    parser.add_argument(
        "--saturation",
        type=_pos_float,
        default=DEFAULT_SATURATION,
        help="Counts for a cell to fully self-trust",
    )
    parser.add_argument("--block", type=int, default=DEFAULT_BLOCK, help="Block size in cells")
    parser.add_argument(
        "--workers",
        type=_workers_arg,
        default=DEFAULT_WORKERS,
        help="Number of workers (positive integer, or 'auto' = os.cpu_count()); default 4",
    )
    parser.add_argument("--scale", type=_pos_float, default=DEFAULT_SCALE, help="DN = percentile * scale")
    parser.add_argument(
        "--percentile-step",
        type=_pos_float,
        default=None,
        help="Round output percentiles to the nearest step (e.g. 5 -> 90/95/100). Default: no rounding",
    )
    parser.add_argument("--compress", default="ZSTD")
    parser.add_argument("--calib-max-points", type=int, default=CALIB_MAX_POINTS_DEFAULT)
    parser.add_argument("--calibration", default=None, help="Calibration JSON path")
    parser.add_argument(
        "--src-crs",
        type=_crs_arg,
        default=SRC_CRS,
        help=f"Input CRS as a bare int or 'EPSG:<code>' string (default {SRC_CRS})",
    )
    parser.add_argument(
        "--work-crs",
        type=_crs_arg,
        default=WORK_CRS,
        help=f"Working CRS for interpolation, bare int or 'EPSG:<code>' (default {WORK_CRS})",
    )
    parser.add_argument(
        "--out-crs",
        type=_crs_arg,
        default=OUT_CRS,
        help=f"Output CRS, bare int or 'EPSG:<code>' (default {OUT_CRS})",
    )
    parser.add_argument("--skip-calibration", action="store_true", help="Skip calibration, use defaults")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the calibration subsampling and blocked-CV bootstrap (default 0; "
        "same seed + same input -> reproducible outputs)",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Resolve the full run plan, print it, and exit 0 before running; creates no files or directories",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs in the out dir (default: exit 2 when value.tif / "
        "support_km.tif / calibration.json / run_summary.json are already present)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write a machine-readable run summary to <out>/run_summary.json "
        "(phase timings, volumetrics, grid, calibration, CLI params; stdout stays clean)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress stderr below WARNING (beats --verbose and LOGGING; an explicit --log-level wins)",
    )
    parser.add_argument(
        "--max-band-parallel",
        type=int,
        default=None,
        help="Cap parallel row-band workers for the low-RAM shared descent (default: --workers).",
    )
    parser.add_argument(
        "--turbo",
        nargs="?",
        const="auto",
        default=None,
        metavar="GB",
        help="Turbo mode: parallel phases + RAM-budgeted shared cap (S3.3). "
        "Optional RAM cap preset 16|32|64|128; bare --turbo = auto-detect.",
    )
    parser.add_argument(
        "--max-ram",
        type=_pos_float,
        default=None,
        metavar="GB",
        help="Explicit RAM cap in GB (overrides the --turbo preset). Implies --turbo.",
    )
    parser.add_argument("--ram-gb", type=_pos_float, default=None, metavar="GB", help="Alias of --max-ram.")
    parser.add_argument(
        "--turbo-strict",
        action="store_true",
        help="Shared path infeasible under the budget: error + exit 3 instead of [warn] + per-box fallback (exit 0).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Debug logging: per-phase wall times + volumetric detail (shorthand for --log-level debug)",
    )
    parser.add_argument(
        "--log-level",
        choices=list(_LOG_LEVELS),
        default=None,
        help="Progress log level on stderr (default info; env LOGGING; 'verbose' = debug)",
    )
    return parser


def _main_preflight(parser: argparse.ArgumentParser, args: argparse.Namespace, out: Path) -> None:
    """Preflight phase of main(): input check, logging, validation, --plan.

    Runs the whole pre-run sequence: the missing-input error, the `help`
    pseudo-command, logging setup, all parser.error flag validations, the
    out-is-a-file check, and the --plan branch (which exits 0). main() wraps
    the call so every non-zero exit here can warn about stale outputs
    (issues #40, #77).

    Args:
        parser: The argument parser (for print_help / error).
        args: Parsed namespace (mutated: max_ram alias resolution).
        out: Output directory path.

    Raises:
        SystemExit: 0 (help, plan), 1 (out is a file), 2 (validation).

    """
    if args.input is None:
        print("error: missing input file (CSV or Parquet)", file=sys.stderr)  # ruff: ignore[print]
        print("hint: run 'ppgrid help' for usage and examples", file=sys.stderr)  # ruff: ignore[print]
        raise SystemExit(2)
    if args.input == "help" and not Path("help").exists():
        # A literal file named "help" wins over the pseudo-command (use --help).
        parser.print_help()
        raise SystemExit(0)

    _configure_logging(_resolve_log_level(verbose=args.verbose, log_level=args.log_level, quiet=args.quiet))

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.percentile_step is not None and not (0 < args.percentile_step <= PERCENTILE_MAX):
        parser.error(f"--percentile-step must be in (0, {PERCENTILE_MAX:g}]")
    if args.block < 1:
        parser.error("--block must be at least 1")
    if args.max_band_parallel is not None and args.max_band_parallel < 1:
        parser.error("--max-band-parallel must be at least 1")
    if args.calib_max_points < 1:
        parser.error("--calib-max-points must be at least 1")
    if args.seed < 0:
        parser.error("--seed must be >= 0")
    for flag, val in (("--src-crs", args.src_crs), ("--work-crs", args.work_crs), ("--out-crs", args.out_crs)):
        if val <= 0:
            parser.error(f"{flag} must be a positive EPSG code")
    if args.turbo is not None and args.turbo not in ("auto", "16", "32", "64", "128"):
        parser.error(f"--turbo preset must be auto|16|32|64|128: {args.turbo}")
    if args.max_ram is None:
        args.max_ram = args.ram_gb
    if args.ram_gb is not None and args.max_ram != args.ram_gb:
        parser.error("--max-ram and --ram-gb are aliases and must match")
    if args.turbo_strict and args.turbo is None and args.max_ram is None:
        parser.error("--turbo-strict requires --turbo or --max-ram")

    if out.exists() and not out.is_dir():
        _die(f"output path is a file, not a directory: {args.out}")

    if args.plan:
        # #53: resolve the full plan (ingest -> calibrate -> grid -> turbo
        # precheck) in memory, print it, and exit 0. dry_run keeps calibrate
        # and grid from writing files, so nothing is created.
        # #57: column preflight first (same error as the run path).
        _check_columns(args.input, args.value_col, args.lng_col, args.lat_col)
        p = _map_pipeline_errors(lambda: _make_pipeline(args, dry_run=True), args.input)
        _map_pipeline_errors(p.plan, args.input)
        print(p.plan_text(args))  # ruff: ignore[print]
        raise SystemExit(0)


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the pipeline.

    Args:
        argv: Argument list (defaults to sys.argv[1:]; tests pass their own).

    Raises:
        SystemExit: exit 0 for `ppgrid help` (and `--plan`), exit 1 for
        pipeline/IO errors (`_die`), exit 2 for missing input, validation
        errors, missing input columns (#57), or a non-empty out dir without
        --force (#54), exit 3 for `--turbo-strict` shared-path infeasibility.
        Any non-zero exit over an earlier run's rasters also prints the
        stale-outputs warning (issues #40, #77).

    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    out = Path(args.out)

    try:
        # Preflight (input check, logging, flag validation, --plan branch):
        # any non-zero exit over an earlier run's rasters warns instead of
        # staying silent (issues #40, #77).
        _main_preflight(parser, args, out)
    except SystemExit as e:
        if e.code not in (0, None):
            _warn_stale_outputs(out)
        raise
    except BaseException:
        _warn_stale_outputs(out)
        raise

    # #54: overwrite guard — a non-empty out dir means an earlier run's
    # outputs; require --force to clobber them.
    if not args.force:
        guarded = ("value.tif", "support_km.tif", "calibration.json", "run_summary.json")
        existing = [name for name in guarded if (out / name).is_file()]
        if existing:
            print(f"error: output directory {args.out} already contains: {', '.join(existing)}", file=sys.stderr)  # ruff: ignore[print]
            print("hint: pass --force to overwrite them", file=sys.stderr)  # ruff: ignore[print]
            raise SystemExit(2)

    try:
        # #57: friendly missing-column error before any pipeline work; inside
        # the stale-outputs handler so a failed run over an earlier run's
        # outputs still warns (issue #40).
        _check_columns(args.input, args.value_col, args.lng_col, args.lat_col)
        out.mkdir(parents=True, exist_ok=True)
    except SystemExit:
        _warn_stale_outputs(out)
        raise
    except OSError as e:
        _die(f"cannot create output directory {args.out}: {e}")
    p: Pipeline | None = None
    try:
        # Construction (init validation, e.g. --scale range) gets the same
        # exit-code mapping as the run phase (review M-1, #56 contract).
        p = _map_pipeline_errors(lambda: _make_pipeline(args, dry_run=False), args.input)
        _map_pipeline_errors(p.run, args.input)
        if args.json:
            _write_json_summary(p, args, out)
    except SystemExit as e:
        # Any non-zero exit with an earlier run's rasters in place: warn
        # instead of staying silent (issue #40). Rasters this run already
        # published (a post-publish failure, e.g. a --json write error)
        # are not stale (#77).
        if e.code not in (0, None):
            _warn_stale_outputs(out, p.published if p is not None else None)
        raise
    except BaseException:
        _warn_stale_outputs(out, p.published if p is not None else None)
        raise


if __name__ == "__main__":
    main()
