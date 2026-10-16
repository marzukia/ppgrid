"""ppgrid — pull-push scattered-data interpolation pipeline + CLI.

Turns scattered geolocated point values into a continent-scale, gapless,
capped raster surface, in minutes, on a single machine, with no GPU.
"""

from __future__ import annotations

import argparse
import difflib
import errno
import hashlib
import importlib
import io
import json
import logging
import math
import os
import platform
import shutil
import struct
import subprocess  # ruff: ignore[suspicious-subprocess-import] — git rev-parse for the --json summary
import sys
import tempfile
import threading
import time
import tomllib
import traceback
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, Self

import numpy as np

from . import __version__, _cinterp, _prof, turbop, zstdmt
from ._tempfiles import _open_fresh_memmap
from .calibrate import (
    M_PER_KM,
    PERCENTILE_MAX,
    PercentileTransform,
    calibrate_fill_cap,
    choose_transform,
    make_transform,
    transform_named,
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

# --- lazy native imports (PEP 562) ------------------------------------------
# pandas (~160 ms to import), rasterio (GDAL C extension) and pyproj (PROJ)
# are only needed once a run actually parses input, projects points, or does
# raster I/O. `import ppgrid` and `ppgrid --help` must not pay for them, so
# the names below resolve on first attribute access and are then cached in
# the module namespace. The public surface is unchanged: `from ppgrid.pipeline
# import reproject` and `monkeypatch.setattr(ppgrid.pipeline, "reproject",
# ...)` keep working; a patched attribute lives in the module dict, which
# shadows __getattr__ (it only fires on a lookup miss).
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Affine": ("rasterio.transform", "Affine"),
    "CRS": ("rasterio.crs", "CRS"),
    "CRSError": ("pyproj.exceptions", "CRSError"),
    "MemoryDataset": ("rasterio._io", "MemoryDataset"),
    "NotGeoreferencedWarning": ("rasterio.errors", "NotGeoreferencedWarning"),
    "RasterioCRSError": ("rasterio.errors", "CRSError"),
    "RasterioIOError": ("rasterio.errors", "RasterioIOError"),
    "Resampling": ("rasterio.warp", "Resampling"),
    "Transformer": ("pyproj", "Transformer"),
    "Window": ("rasterio.windows", "Window"),
    "from_bounds": ("rasterio.transform", "from_bounds"),
    "from_origin": ("rasterio.transform", "from_origin"),
    "pd": ("pandas", ""),
    "reproject": ("rasterio.warp", "reproject"),
    "rasterio": ("rasterio", ""),
}

if TYPE_CHECKING:  # static bindings only; _load_native() provides the runtime values
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

    from . import ui as _ui


def _load_native() -> None:
    """Import pandas, rasterio and pyproj, and publish all lazy names in the module dict.

    CPython's LOAD_GLOBAL does not consult a module __getattr__, so function
    bodies can only see these names once they are in the module dict. This
    runs the one-time import (sys.modules-cached afterwards) and populates
    the dict. Idempotent; names already present (e.g. monkeypatched) are
    left alone. Call it at every entry point that can run the functions
    below directly (CLI, Pipeline methods, reproject helpers).

    """
    if not all(name in globals() for name in _LAZY_IMPORTS):
        for name, (module_name, attr) in _LAZY_IMPORTS.items():
            if name in globals():
                continue
            module = importlib.import_module(module_name)
            globals()[name] = module if not attr else getattr(module, attr)
    # _CLI_MAPPED_EXCEPTIONS references lazy names, so it is built here, not
    # at import time (module-level, but unevaluable before the native stack
    # is published).
    if not globals().get("_CLI_MAPPED_EXCEPTIONS"):
        globals()["_CLI_MAPPED_EXCEPTIONS"] = (
            CRSError,
            RasterioCRSError,
            ImportError,
            FileNotFoundError,
            pd.errors.ParserError,
            RasterioIOError,
            IndexError,
            OverflowError,
            ZeroDivisionError,
            RuntimeError,
            KeyError,
            ValueError,
            OSError,
            SystemExit,
        )


def __getattr__(name: str) -> Any:
    """Resolve a lazily imported rasterio/pyproj name (see _LAZY_IMPORTS).

    Also serves ``_ui`` (the ui module -> rich): importing it eagerly at
    module load would pull the terminal-render stack into ``import
    ppgrid.pipeline`` (issue #109).

    Raises:
        AttributeError: If name is not a lazy import.

    """
    if name in _LAZY_IMPORTS:
        _load_native()
        val = globals().get(name)
        if val is not None:
            return val
    if name == "_ui":
        from . import ui as _ui_mod

        return _ui_mod
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg) from None


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_IMPORTS) | {"_ui"})


# rasterio's MemoryDataset.__init__ (used by reproject for ndarray buffers)
# always warns NotGeoreferencedWarning on the initial transform read of the
# MEM::: dataset, then sets the real transform immediately after. It normally
# hides the warning in a nested warnings.catch_warnings, but CPython warning
# filters are process-global, not thread-safe: with concurrent warps (S3.3)
# another thread can restore a stale filter snapshot and leak the benign
# warning. It carries no information here, so ignore the category. The filter
# is installed lazily in _ensure_warp_ready (first warp, before the pool
# starts) because the category lives in the heavy rasterio package.
_WARP_READY: dict[str, bool] = {}


def _ensure_warp_ready() -> None:
    """Load the native stack and install the NotGeoreferencedWarning filter.

    Call at the top of every function that warps, before its thread pool
    starts: _load_native makes the module globals visible to the warp
    functions, and the process-global filter (installed once per process)
    then covers every warp thread.

    """
    _load_native()
    if _WARP_READY.get("filter"):
        return
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    _WARP_READY["filter"] = True


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

  # Turbo: 32 GB RAM budget, strict (exit 3 if infeasible), 8 workers.
  ppgrid data/melb_houses.csv -o out/melb_turbo --turbo 32 --turbo-strict --workers 8 --value-col price

  # Verify: compare two run outputs (out dirs or GeoTIFFs); exit 0 identical,
  # 1 different, 2 usage error
  ppgrid compare out/melb_10km out/melb_10km

progress logging:
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
  130 interrupted (SIGINT): one `interrupted` line, no traceback; run
      temp files removed (retained partials named), earlier run's rasters
      warned as stale

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

# A.7 oracle probe geometry, single source for BOTH probe writers
# (Pipeline._turbo_zstd_ok and the doctor check _doc_zstdmt; audit #83:
# the literal width was duplicated in each, so a change could land in one
# and not the other). One deterministic 500 x TILE_PX int16 tile: the
# width is deliberately NOT a multiple of TILE_PX, so the tile spans a
# partial edge column and GDAL zero-pads it — the gate covers the
# partial-tile codec path (audit #81 F-1) in addition to the full-tile
# path. The non-multiple invariant is pinned in tests.
_ORACLE_PROBE_W: int = 500


def _oracle_probe_tile() -> np.ndarray:
    """Deterministic compressible A.7 oracle probe tile.

    Both axes differ to small values, so predictor-2 output is near-zero
    structured data (compressible, and the two codec paths compare
    meaningfully).

    Returns:
        The int16 array, _ORACLE_PROBE_W x TILE_PX.

    """
    gx = np.arange(_ORACLE_PROBE_W, dtype=np.int16) * 13
    gy = np.arange(TILE_PX, dtype=np.int16) * 7
    return (gx[None, :] + gy[:, None]).astype(np.int16)


# CLI/API defaults — single source for the Pipeline constructor and argparse
# so the two can't drift.
DEFAULT_RES: float = 500.0
DEFAULT_WORKERS: int = 4
DEFAULT_SATURATION: float = 1.0
DEFAULT_BLOCK: int = 2048
CALIB_MAX_POINTS_DEFAULT: int = 2_000_000

# --- config file (issue #63) ----------------------------------------------
# `ppgrid.toml` holds pipeline options. Precedence: CLI flag > config >
# built-in default. Argparse must tell "not passed" apart from "passed with
# the default value", so every config-able option carries the UNSET sentinel
# as its parser default; _apply_config fills it from the file and
# _resolve_sentinels falls back to the built-in default, so downstream code
# (preflight, _make_pipeline, summaries) only ever sees real values.


class _Unset:
    """Sentinel: a config-able CLI flag that was not passed (issue #63)."""

    __slots__ = ()
    _inst: Self | None = None

    def __new__(cls) -> Self:
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self) -> str:
        return "<unset>"


UNSET = _Unset()

# (config key, argparse dest, accepted TOML kind, example value for the
# `init` template when the built-in default is None). Order is the --help
# group order (input/columns, CRS/grid, interpolation, memory/turbo, output);
# the template and the effective-params dict use this order.
_CONFIG_OPTIONS: tuple[tuple[str, str, str, Any], ...] = (
    ("value-col", "value_col", "str", "price"),
    ("lng-col", "lng_col", "str", "longitude"),
    ("lat-col", "lat_col", "str", "latitude"),
    ("src-crs", "src_crs", "crs", 4326),
    ("work-crs", "work_crs", "crs", 6933),
    ("out-crs", "out_crs", "crs", 3857),
    ("res", "res", "num", 10000),
    ("cap-km", "cap_km", "str-or-num", 20),
    ("block", "block", "int", 2048),
    ("transform", "transform", "str", "log10"),
    ("saturation", "saturation", "num", 1.0),
    ("scale", "scale", "num", 10.0),
    ("percentile-step", "percentile_step", "num", 5),
    ("skip-calibration", "skip_calibration", "bool", True),
    ("seed", "seed", "int", 1234),
    ("calib-max-points", "calib_max_points", "int", 1_000_000),
    ("calibration", "calibration", "str", "calibration.json"),
    ("workers", "workers", "str-or-num", "auto"),
    ("max-band-parallel", "max_band_parallel", "int", 8),
    ("turbo", "turbo", "str-or-num", "auto"),
    ("max-ram", "max_ram", "num", 32),
    ("turbo-strict", "turbo_strict", "bool", True),
    ("out", "out", "str", "out/"),
    ("compress", "compress", "str", "ZSTD"),
    ("force", "force", "bool", True),
    ("json", "json", "bool", True),
)
_CONFIG_KEYS: frozenset[str] = frozenset(key for key, _, _, _ in _CONFIG_OPTIONS)
_CONFIG_KEY_TO_DEST: dict[str, str] = {key: dest for key, dest, _, _ in _CONFIG_OPTIONS}
_CONFIG_KIND: dict[str, str] = {key: kind for key, _, kind, _ in _CONFIG_OPTIONS}

# Built-in defaults for the config-able options — the parser defaults
# pre-#63. Single source for the sentinel fallback and the `ppgrid init`
# template values. --ram-gb is CLI-only (its alias is canonicalised into
# max-ram by preflight) but resolves here too.
_CONFIG_DEFAULTS: dict[str, Any] = {
    "value_col": "value",
    "lng_col": "longitude",
    "lat_col": "latitude",
    "src_crs": SRC_CRS,
    "work_crs": WORK_CRS,
    "out_crs": OUT_CRS,
    "res": DEFAULT_RES,
    "cap_km": "auto",
    "block": DEFAULT_BLOCK,
    "transform": "auto",
    "saturation": DEFAULT_SATURATION,
    "scale": DEFAULT_SCALE,
    "percentile_step": None,
    "skip_calibration": False,
    "seed": 0,
    "calib_max_points": CALIB_MAX_POINTS_DEFAULT,
    "calibration": None,
    "workers": DEFAULT_WORKERS,
    "max_band_parallel": None,
    "turbo": None,
    "max_ram": None,
    "ram_gb": None,
    "turbo_strict": False,
    "out": "out/",
    "compress": "ZSTD",
    "force": False,
    "json": False,
}

_CONFIG_FILE_NAME = "ppgrid.toml"

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
    active. The object is passed to every worker task as-is (no per-field
    copy; audit #83: it used to ride a module-global _CTX that two
    concurrent Pipeline.run() calls could clobber mid-run), so a new field
    needs exactly one edit here.
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


# ---------------------------------------------------------------------------
# Presentation layer (issue #101). `_CURRENT_UI` is set by main() for the
# duration of the CLI run; module-level helpers route their stderr lines
# through it. Library mode (no main) and pre-UI paths fall back to a plain
# stderr print, byte-identical to the old behavior.
# ---------------------------------------------------------------------------

# A dict holder (not a bare global) so main() can bind/clear it without a
# module-level `global` statement (ruff PLW0603).
_CURRENT_UI: dict[str, _ui.UI | None] = {"ui": None}

# Library-mode markers (D-7): without an active UI the line goes to stderr
# exactly as the pre-#102 raw prints emitted it, so the library surface
# stays byte-identical. With a UI, the marker is a display concern owned by
# the UI layer (live markers / logging level letter) and the messages
# below carry no state word.
# Most warn sites embedded ``[warn] `` and most error sites ``error: ``; two
# families carried a different legacy prefix (the stale-output sites embed
# ``warning: ``; the strict precheck line keeps its ``error: `` marker
# inside the message from turbop), so they pass an explicit ``legacy``
# override instead of relying on the per-kind table.
_LEGACY_MARKERS = {"error": "error: ", "warn": "[warn] ", "ok": "ok: ", "note": ""}


def _ui_note(kind: str, text: str, legacy: str | None = None) -> None:
    """Emit a CLI stderr line via the presentation layer (issue #101).

    kind: ``note`` | ``ok`` | ``warn`` | ``error``. ``text`` carries the
    message only (D-7): the state word is re-added at display time — live
    shows ``ok: `` / ``[warn] `` / ``error: `` prefixes, plain shows the
    level letter of the unified ``L ppgrid.pipeline: msg`` line, and
    library mode (no active UI) routes the line to the ppgrid logger
    (issue #109: the library never prints directly), keeping the legacy
    raw marker in the message text. ``legacy`` overrides that marker for
    sites whose pre-#102 raw print carried a different prefix (``"warning: "``
    for the stale-output family, ``""`` for the strict precheck line);
    display modes never see it.
    """
    ui = _CURRENT_UI["ui"]
    if ui is None:
        marker = _LEGACY_MARKERS.get(kind, "") if legacy is None else legacy
        text = marker + text
        if kind == "error":
            log.error("%s", text)
        elif kind == "warn":
            log.warning("%s", text)
        else:
            log.info("%s", text)
        return
    if kind == "ok":
        ui.ok(text)
    elif kind == "warn":
        ui.warn(text)
    elif kind == "error":
        ui.error(text)
    else:
        ui.note(text)


def _ui_note_precheck(message: str) -> None:
    """Emit one turbo precheck decision message (D-7).

    The non-strict families carry an embedded ``[warn] `` prefix (stripped
    here, re-added at display time); the strict line keeps its embedded
    ``error: `` marker, so the library-mode re-append is empty for that
    family (fix-round F2: ``[warn] error: ...`` was a double marker).
    """
    legacy = "" if message.startswith("error: ") else None
    _ui_note("warn", message.removeprefix("[warn] "), legacy=legacy)


def _ru_maxrss_gb() -> float | None:
    """Process peak RSS in GB via resource.getrusage, or None (issue #102).

    ru_maxrss is KB on Linux and bytes on macOS; Windows has no stdlib
    resource module, so callers label the value ``estimated`` instead.

    """
    try:
        import resource
    except ImportError:
        return None
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError, ValueError):
        return None
    if rss <= 0:
        return None
    return rss / 1e9 if sys.platform == "darwin" else rss / 1024.0 / 1024.0


def _human_size(n: int) -> str:
    """Bytes as a compact decimal size: 856 KB, 1.2 MB (issue #102)."""
    if n >= 1e9:
        return f"{n / 1e9:.1f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.1f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.0f} KB"
    return f"{n} B"


class _NoPhase:
    """Progress handle for library mode (no UI): every method a no-op."""

    __slots__ = ("name", "summary")

    def __init__(self, name: str) -> None:
        self.name = name
        self.summary: str | None = None

    def set_total(self, total: float) -> None:
        pass

    def advance(self, n: float) -> None:
        pass

    def set_count(self, n: float) -> None:
        pass


def _env_report_line(preset: float | None) -> str:
    """Build the ``--verbose`` one-line environment report (issue #101).

    Version, Python, OS, arch, RAM detection source + resolved cap,
    rasterio/GDAL/pyproj/numpy versions, and the C-interp status. Must
    never raise: every probe is best-effort with an ``unknown`` fallback.
    """
    parts: list[str] = [f"ppgrid={__version__}", f"python={platform.python_version()}"]
    try:
        parts.append(f"os={platform.system().lower()}-{platform.release()}")
    except Exception:  # ruff: ignore[blind-except] - report line must not break the run
        parts.append("os=unknown")
    parts.append(f"arch={platform.machine() or 'unknown'}")
    try:
        physical = turbop._read_physical_gb()  # ruff: ignore[private-member-access]
    except (ValueError, OSError):
        physical = None
    try:
        cgroup = turbop._read_cgroup_max_gb()  # ruff: ignore[private-member-access]
    except (ValueError, OSError):
        cgroup = float("inf")
    try:
        cap, source = turbop.resolve_cap(preset, physical_gb=physical, cgroup_gb=cgroup)
        parts.append(f"ram={cap:g}GB ({source})")
    except Exception as e:  # ruff: ignore[blind-except] - report line must not break the run
        parts.append(f"ram=unknown ({e})")
    # Local imports, not the module-lazy names: the report runs in
    # _main_preflight, before _load_native(), and a bare global lookup
    # misses the PEP 562 __getattr__ (the lazy names are not in the module
    # dict yet), so rasterio/gdal/pyproj would read as placeholders on every
    # fresh --verbose run (F1, #115 review). Each field gets its own try so
    # one failed probe cannot swallow the next field.
    try:
        import rasterio as _rasterio

        parts.append(f"rasterio={_rasterio.__version__}")
    except Exception:  # ruff: ignore[blind-except]
        parts.append("rasterio=unknown")
    try:
        import rasterio as _rasterio

        parts.append(f"gdal={_rasterio.gdal_version()}")
    except Exception:  # ruff: ignore[blind-except]
        parts.append("gdal=unknown")
    try:
        import pyproj as _pyproj

        parts.append(f"pyproj={_pyproj.__version__}")
    except Exception:  # ruff: ignore[blind-except]
        parts.append("pyproj=unknown")
    try:
        parts.append(f"numpy={np.__version__}")
    except Exception:  # ruff: ignore[blind-except]
        parts.append("numpy=unknown")
    try:
        if _cinterp.available():
            parts.append("cinterp=c")
        else:
            parts.append(f"cinterp=numpy ({_cinterp.reason() or 'unknown'})")
    except Exception:  # ruff: ignore[blind-except]
        parts.append("cinterp=unknown")
    return "env: " + ", ".join(parts)


def _die(msg: str) -> NoReturn:
    """Print a clean CLI error (no traceback) and exit 1.

    Args:
        msg: Error message, routed to the UI layer as an error (D-7):
            live shows ``error: <msg>``, plain ``E ppgrid.pipeline: <msg>``,
            library ``error: <msg>``.

    Raises:
        SystemExit: Always, with status code 1.

    """
    _ui_note("error", msg)
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
    _ui_note(
        "warn",
        f"this run failed before writing outputs; {', '.join(existing)} in {out} "
        "are from an earlier run and were left untouched",
        legacy="warning: ",  # pre-#102 raw print embedded 'warning: ' (F1 family)
    )


def _interrupted_exit(out: Path | None, p: Pipeline | None) -> NoReturn:
    """Ctrl-C (SIGINT) mid-run: sweep temps, warn stale rasters, exit 130 (#106).

    Replaces the default KeyboardInterrupt traceback with one clean line.
    run()'s finally already sweeps run temps on a write-phase interrupt;
    this sweep covers the rest (ingest, calibrate, preflight, a second
    Ctrl-C mid-sweep). Any temp it cannot remove is named in the message.
    The stale-outputs warning fires exactly once here: the in-run
    BaseException handlers in main() skip it for KeyboardInterrupt.

    Args:
        out: Output directory; None only when the interrupt landed during
            argument parsing, before the out path was read.
        p: The Pipeline if constructed (its published set keeps the stale
            warning from naming this run's own rasters, as in #77).

    Raises:
        SystemExit: code 130 (the SIGINT convention: 128 + SIGINT(2)).

    """
    retained: list[str] = []
    if out is not None:
        for fname in _RUN_TEMP_FILES:
            fpath = out / fname
            try:
                fpath.unlink(missing_ok=True)
            except OSError:
                if fpath.is_file():
                    retained.append(fname)
        for scratch in sorted(out.glob("_stock_scratch_*.tif")):
            try:
                scratch.unlink(missing_ok=True)
            except OSError:
                if scratch.is_file():
                    retained.append(scratch.name)
    if retained:
        print(f"interrupted; retained partials: {', '.join(retained)}", file=sys.stderr)  # ruff: ignore[print]
    else:
        print("interrupted", file=sys.stderr)  # ruff: ignore[print]
    if out is not None:
        _warn_stale_outputs(out, p.published if p is not None else None)
    raise SystemExit(130)


# O_NOFOLLOW and ELOOP are POSIX-only (Windows has no symlinks): getattr
# defaults keep the same code path total on every platform (PR #97, OS matrix).
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_STAGING_RETRY_ERRNOS: tuple[int, ...] = (errno.EEXIST, getattr(errno, "ELOOP", errno.EEXIST))


def _staging_fd(path: Path) -> int:
    """Create `path` as a fresh 0600 regular file and return an open write fd.

    Symlinks are replaced, never followed (issues #15, #77): the path is
    unlinked, then recreated with O_CREAT | O_EXCL | O_NOFOLLOW — a symlink
    (or any file) planted in that window makes os.open fail (ELOOP / EEXIST)
    instead of being followed, and one retry resolves it. O_NOFOLLOW and
    ELOOP are POSIX-only; on Windows (no symlinks) the _O_NOFOLLOW /
    _STAGING_RETRY_ERRNOS getattr defaults degrade them to no-ops without
    changing the Linux open/retry semantics (PR #97).

    Args:
        path: Staging path in the output dir.

    Returns:
        Open fd on the fresh regular file.

    Raises:
        OSError: If the path still cannot be created as a regular file
            after one retry.

    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW
    path.unlink(missing_ok=True)
    try:
        return os.open(path, flags, 0o600)
    except OSError as e:
        if e.errno not in _STAGING_RETRY_ERRNOS:
            raise
        # Planted in the unlink->open window: replace it once, try again.
        path.unlink(missing_ok=True)
        return os.open(path, flags, 0o600)


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


def _nonneg_int(text: str) -> int:
    """Argparse type: a non-negative integer (`compare --max-diff`, #67).

    Args:
        text: Raw CLI argument string.

    Returns:
        The parsed integer.

    Raises:
        argparse.ArgumentTypeError: If text is not a non-negative int.

    """
    try:
        val = int(text)
    except ValueError as exc:
        msg = f"invalid non-negative int: {text!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if val < 0:
        msg = f"must be >= 0: {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return val


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
# Re-allocation when a block's shape differs (right/bottom raster edges) is
# expected and perf-only: at most one realloc per thread per distinct shape.
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
    args: tuple[_WorkerConfig, tuple[np.ndarray, np.ndarray] | None, int, int],
) -> tuple[int, int, tuple[np.ndarray, np.ndarray] | None]:
    """Interpolate a single output block using pull-push.

    Args:
        args: (cfg, dn, bx, by). cfg is the run's _WorkerConfig — per-
            Pipeline instance state passed with every task (audit #83: the
            worker used to read a module-global _CTX, which two concurrent
            Pipeline.run() calls in one process would clobber mid-run).
            dn is the (value, support) tmpfs DN-field pair when the turbo
            write path is active, else None.

    Returns:
        Tuple of block coords and the (value, support) DN grids, or None when
        the block neighbourhood has no points (written as nodata).

        When dn is set (the turbo write path), the DN grids are written from
        this worker thread straight into the block's disjoint window. The
        main thread used to write them back as results arrived and sat on
        the critical path ahead of the pool (issue #84); block windows are
        disjoint, so the writes are thread-safe and the final fields are
        byte-identical.

    """
    cfg, dn, bx, by = args
    if cfg.shared:
        grids = _block_shared(cfg, bx, by)
    else:
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
        grids = _quantize(cfg, val[sl], sup[sl], near[sl])

    if dn is not None:
        vq, rq = grids
        val_dn, sup_dn = dn
        w = _block_window(bx, by, cfg)
        rows = slice(w.row_off, w.row_off + w.height)
        cols = slice(w.col_off, w.col_off + w.width)
        val_dn[rows, cols] = vq.T[::-1, :]
        sup_dn[rows, cols] = rq.T[::-1, :]
    return bx, by, grids


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
    _ensure_warp_ready()
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
            init_dest=NODATA,
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
    _ensure_warp_ready()
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
        init_dest=NODATA,
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
    progress: Callable[[int], None] | None = None,
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
        progress: Optional callback with the cells flushed in this band
            (issue #101 progress); the caller owns throttling.

    """
    _ensure_warp_ready()
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
                    if progress is not None:
                        progress(w_w * w_h)


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
    progress: Callable[[int], None] | None = None,
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
    _load_native()
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
            progress=progress,
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
    progress: Callable[[int], None] | None = None,
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
        progress: Optional callback with the cells flushed so far in this
            band (issue #101 progress); the caller owns throttling.

    """
    _load_native()

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
        progress=progress,
    )


# TIFF base type -> bytes per element (Y-1). The parser and the head-
# overlap guard share it so a type can never be sized in one place and
# mis-sized in the other. 16/17/18 = the BigTIFF 64-bit types (LONG8,
# SLONG8, IFD8).
def _tif_parse_ifd(data: bytes) -> dict[str, Any]:
    """Parse the first IFD of a single-band classic or BigTIFF tiled raster.

    Built on the shared zstdmt primitives (parse_tiff_header +
    parse_ifd_tags) so there is exactly one IFD parser rule in the repo
    (audit #83: this used to be an independent, subtly divergent copy).

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
        ValueError: On unknown endian/magic, missing 324/325, a
            non-LONG/LONG8 tile array type, inline tile arrays, or a
            count mismatch. (Tags with a type the shared parser does not
            size are now skipped rather than a hard error: the caller only
            needs 324/325 and the head-overlap guard. Previously:
            "unsupported TIFF type" ValueError.)

    """
    e, big, off_fmt, entry_size, cnt_size, inline, ifd = zstdmt.parse_tiff_header(data)
    try:
        tags: dict[int, tuple[int, int, int]] = zstdmt.parse_ifd_tags(
            data, e, ifd, off_fmt, cnt_size, entry_size, inline
        )
    except struct.error as exc:
        msg = f"corrupt TIFF structure: {exc}"
        raise ValueError(msg) from exc
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
    per_typ = zstdmt.TIFF_TYPE_SIZES
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
    scales: tuple[float, ...] | None,
    offsets: tuple[float, ...] | None,
    n_threads: int,
    progress: Callable[[int], None] | None = None,
    progress_refresh: Callable[[], None] | None = None,
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
    _ensure_warp_ready()
    warp_tile = 2048
    # The parallel ZSTD writer resolves the GDAL CPL zstd function pointer
    # from /proc/self/maps (Linux only). Where it is unavailable (macOS,
    # Windows, or a GDAL build without the CPL zstd compressor) this writer
    # cannot run: return False so the caller falls back to the stock-codec
    # path instead of a ZstdmtError. The production dispatch already gates
    # on _turbo_zstd_ok() (== zstdmt.available()), so this only makes the
    # direct call total; Linux behaviour is unchanged (available() is True).
    if not zstdmt.available():
        return False
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
                for i0, buf in ex.map(warp_i0, range(0, dst_width, warp_tile)):
                    coarse[i0] = buf
                    if progress is not None:
                        progress(min(warp_tile, dst_width - i0) * band_h)
                    if progress_refresh is not None:
                        progress_refresh()
        else:
            for i0 in range(0, dst_width, warp_tile):
                _, buf = warp_i0(i0)
                coarse[i0] = buf
                if progress is not None:
                    progress(min(warp_tile, dst_width - i0) * band_h)
                if progress_refresh is not None:
                    progress_refresh()
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
        # No `del zero` here (audit #78 P1-1, as in the stock writer): if the
        # allocation fails `zero` is unbound and the del would mask the root
        # error. The 512^2 tile local dies at function return anyway.

    try:
        info = _tif_parse_ifd(data)
    except ValueError as e:
        _ui_note("warn", f"turbo zstd head parse: {e}; serial write")
        return False
    t324, t325, n_tiles = info["t324"], info["t325"], info["n_tiles"]
    if n_tiles != len(tiles):
        _ui_note("warn", f"turbo zstd head: tile count {n_tiles} != {len(tiles)}; serial write")
        return False
    # All external value arrays must end before the first stored tile, so
    # the head cut keeps every array and drops every reference tile.
    head_end = info["first"]
    for typ, count, voff in info["tags"].values():
        total = zstdmt.TIFF_TYPE_SIZES[typ] * count
        if total > info["inline"] and voff + total > info["first"]:
            _ui_note("warn", "turbo zstd head: value array overlaps tiles; serial write")
            return False
    if not info["big"] and head_end + sum(len(t) for t in tiles) > _TIFF_CLASSIC_MAX_BYTES:
        _ui_note("warn", "turbo zstd head: assembled file beyond classic 4 GiB; serial write")
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
    progress: Callable[[int], None] | None = None,
    progress_refresh: Callable[[], None] | None = None,
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
    _ensure_warp_ready()
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
        _ui_note("warn", f"turbo stock head parse: {e}; serial write")
        return False
    t324, t325, n_tiles = info["t324"], info["t325"], info["n_tiles"]
    head_end = info["first"]
    for typ, count, voff in info["tags"].values():
        total = zstdmt.TIFF_TYPE_SIZES[typ] * count
        if total > info["inline"] and voff + total > info["first"]:
            _ui_note("warn", "turbo stock head: value array overlaps tiles; serial write")
            return False
    # Compressed frames are never longer than the raw band, so raw size is
    # a sound classic-4GiB upper bound.
    if not info["big"] and head_end + arr.nbytes > _TIFF_CLASSIC_MAX_BYTES:
        _ui_note("warn", "turbo stock head: assembled file may exceed classic 4 GiB; serial write")
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
        if progress is not None:
            progress(bw * bh)
        return frames

    block_frames: list[list[bytes]] = []
    if n_threads > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            for frames in ex.map(lambda jb: _block(*jb), band_jobs):
                block_frames.append(frames)
                if progress_refresh is not None:
                    progress_refresh()
    else:
        for jb in band_jobs:
            block_frames.append(_block(*jb))
            if progress_refresh is not None:
                progress_refresh()

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
        _ui_note("warn", f"turbo stock: tile count {len(tiles)} != {n_tiles}; serial write")
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
    _load_native()  # pd is lazy; forkserver workers run this function directly
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


def _sniff_compression(data: bytes) -> str | None:
    """Compression magic from a file's leading bytes (None = uncompressed).

    The magics the parallel CSV split sniffs (issue #80): gzip, bzip2, zip,
    xz. `ppgrid probe` (issue #62) reuses this to report the input format,
    so the two surfaces can never drift.

    Args:
        data: Leading bytes of the file (5+ recommended; shorter is fine).

    Returns:
        'gzip' / 'bzip2' / 'zip' / 'xz', or None when uncompressed.

    """
    if data[:2] == b"\x1f\x8b":
        return "gzip"
    if data[:3] == b"BZh":
        return "bzip2"
    if data[:4] == b"PK\x03\x04":
        return "zip"
    if data[:5] == b"\xfd7zXZ":
        return "xz"
    return None


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
    _load_native()  # pd below is lazy (PEP 562); idempotent once the stack is up
    try:
        data = Path(path).read_bytes()
        # Empty or compressed input cannot be split by raw byte offsets: pandas
        # cannot sniff compression from a byte buffer, so the serial parse
        # (which reads via the path) is the only correct read (issue #80).
        if not data or _sniff_compression(data) is not None:
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


class TurboInfeasibleError(Exception):
    """Turbo strict mode is infeasible (library mode, issue #109).

    Raised by ``Pipeline.run()`` / ``Pipeline.plan()`` (via the turbo
    precheck) when ``turbo_strict`` is True, no UI is attached
    (``ui=None``, i.e. library mode), and the full-grid turbo path does
    not fit the RAM budget — the condition the CLI reports as exit code
    3 with the identical message on stderr.

    Library callers should catch this and surface it in their own way;
    the library never raises SystemExit or prints directly.

    Attributes:
        message: The precheck decision message — the exact line the CLI
            prints, e.g. "error: --turbo est peak 30.5 GB > budget 13.6 GB;
            shared path infeasible; per-box est peak 8.6 GB fits in the
            budget". Also available as ``args[0]`` via Exception.

    """

    def __init__(self, message: str) -> None:
        """Store the precheck line as ``message`` (and ``args[0]``)."""
        super().__init__(message)
        self.message = message


class Pipeline:
    """Pull-push scattered-data interpolation pipeline.

    Turns a CSV/Parquet table of geolocated point values into a pair of
    int16 GeoTIFFs (value percentiles + support km) at an arbitrary
    resolution and CRS, on CPU only.

    Units and CRS expectations:
        - ``res``: output grid resolution in metres (working CRS).
        - ``cap_km``: support (influence) radius in kilometres; "auto"
          resolves a data-driven cap from the point density.
        - ``saturation``: per-cell count saturation (points per cell).
        - ``scale``: DN per percentile point (default 100 -> DN 0..10000,
          the int16 value band; raw values come back via calibration.json).
        - ``src_crs`` / ``work_crs`` / ``out_crs``: positive EPSG codes.
          Defaults: input lon/lat 4326 (WGS 84), interpolation in 6933
          (Wagner VII, equal-area metres), output 3857 (Web Mercator).

    Side effects:
        ``run()`` creates ``out_dir`` and writes ``value.tif``,
        ``support_km.tif`` and ``calibration.json`` into it (each via an
        atomic rename), plus staging files (``_points.npy``, ``_tmp``
        rasters) that are removed on every exit path. With ``dry_run``
        True (``plan()``) nothing is written. Progress and warnings go
        to the ``ppgrid`` logger (stderr under the CLI); with ``ui=None``
        nothing is printed directly (issue #109).

    Exceptions:
        ``__init__`` raises ``ValueError`` on invalid parameters (scale * 100
        > int16 max, percentile_step outside (0, 100], saturation <= 0,
        n_threads < 1, unknown turbo preset, ...).
        ``run()`` / ``plan()`` raise ``ValueError`` on validation failures
        (missing input columns, no valid points, non-positive geometry
        values), ``OSError``/``FileNotFoundError`` on I/O errors (missing
        input file, unwritable out dir, disk full), pyproj
        ``CRSError`` on an unknown EPSG code, and ``TurboInfeasibleError``
        in library mode (``ui=None``) when ``turbo_strict`` is set and the
        full-grid turbo path does not fit the RAM budget. In CLI mode the
        same condition is ``SystemExit(3)``.
    """

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
        ui: _ui.UI | None = None,
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
        _load_native()
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
        # Write-path fallbacks actually taken this run (issue #102); the
        # final run summary surfaces them at default verbosity, not only in
        # DEBUG. Reset per run() so re-runs on one instance stay honest.
        self._fallbacks: list[str] = []
        # Set by _turbo_zstd_ok() when the A.7 gate fails: the short reason
        # the summary's fallbacks line names.
        self._zstd_gate_reason: str | None = None
        # Presentation (issue #101): the active phase handle for the current
        # _phase() scope (None in library mode or outside a phase).
        self.ui = ui
        self._active_phase: _ui.Phase | None = None
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
        # None until grid() builds the worker config: run()'s finally
        # cleanup must not raise AttributeError when grid() failed first
        # (review rev-os2 M1: the mask hid the original ValueError).
        self.cfg: _WorkerConfig | None = None
        self._cal: dict[str, Any] | None

    @property
    def published(self) -> set[str]:
        """Final-output names published (via os.replace) this run (issues #40, #77)."""
        return self._published

    @contextmanager
    def _phase(self, name: str, **kwargs: Any) -> Iterator[_ui.Phase | _NoPhase]:
        """Open a UI phase scope (issue #101).

        No-op in library mode (ui is None): yields a no-op handle. In CLI
        mode the handle is also stored on self._active_phase so inner
        helpers (the parallel CSV read) can advance it without a parameter.

        Args:
            name: Phase name shown in the progress display
                (read/calibrate/grid/interpolate/reproject/write).
            **kwargs: Passed to the UI (e.g. total, unit).

        Yields:
            The phase handle (set_total / advance / set_count / summary).

        """
        if self.ui is None:
            yield _NoPhase(name)
            return
        with self.ui.phase(name, **kwargs) as ph:
            self._active_phase = ph
            try:
                yield ph
            finally:
                self._active_phase = None

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
        ph = self._active_phase
        if tasks is None:
            if ph is not None:
                ph.set_total(0)  # serial parse: no reliable total -> indeterminate
            return serial()
        task_list, expected = tasks
        if ph is not None:
            ph.set_total(expected)
        chunks: list[dict[str, np.ndarray]] = []
        with ProcessPoolExecutor(max_workers=len(task_list)) as ex:
            for chunk in ex.map(_read_csv_chunk, task_list):
                chunks.append(chunk)
                if ph is not None:
                    ph.advance(int(chunk[self.value_col].size))
                    if self.ui is not None:
                        self.ui.maybe_refresh()
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
            # Single source for the name lookup (audit #83): a renamed or
            # unknown transform is a named ValueError, not a raw
            # StopIteration (the CLI validates against the same factory).
            self.tf = transform_named(self.tname).fit(self.v)

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
            self.tf = transform_named("identity")

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
            # Drop the reference now, not at frame end: the write phase
            # re-opens the file (mmap read). On Windows the OS handle
            # stays open while the mmap object is alive, and a frame
            # pinned by an in-flight traceback would keep _points.npy
            # locked past _remove_run_temps (WinError 32, CI run
            # 38207801346 W1). numpy 2.4+ has no memmap.close() API;
            # CPython refcounting releases the handle.
            pts = None

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

        Raises:
            ValueError: validation failures (see the class docstring).
            OSError: I/O errors (missing input file, unwritable out dir,
                disk full).
            CRSError: unknown EPSG code on src_crs/work_crs/out_crs.
            TurboInfeasibleError: turbo_strict infeasible in library mode
                (ui=None); the CLI reports the same condition as exit 3.

        """
        self._phase_wall = {}
        self._fallbacks = []
        self._zstd_gate_reason = None
        t_total = time.perf_counter()
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        scope = self.ui.run_scope() if self.ui is not None else nullcontext()
        with scope:
            t0 = time.perf_counter()
            with _prof.phase("ingest"), self._phase("read", unit="rows") as ph:
                self.ingest()
                ph.summary = f"{self.n_total:,} rows -> {self.n:,} points"
            t_ingest = time.perf_counter() - t0
            t0 = time.perf_counter()
            with _prof.phase("calibrate"), self._phase("calibrate") as ph:
                self.calibrate()
                ph.summary = f"transform={self.tname} cap_km={self.cap_km_val:.4g}"
            t_calibrate = time.perf_counter() - t0
            try:
                t0 = time.perf_counter()
                with _prof.phase("grid"), self._phase("grid") as ph:
                    self.grid()
                    ph.summary = f"{self.nx}x{self.ny} cells @ {self.res:g} m"
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
        # Final ok line is live-table-mode only: plain stderr (non-TTY,
        # NO_COLOR, TERM=dumb, --quiet) stays byte-identical to the pre-#101
        # logging output (F3/F4, #115 review).
        if self.ui is not None and self.ui.live:
            self.ui.ok(f"{t_total:.2f}s -> {Path(vpath).name}, {Path(spath).name}")
        # Final run summary (issue #102): default verbosity in both plain
        # and live modes; --quiet keeps the pre-#101 surface (zero stderr).
        if self.ui is not None and not self.ui.quiet:
            self.ui.summary("run summary", self._run_summary_rows(t_total, vpath, spath))
        return vpath, spath

    def plan(self) -> None:
        """Resolve the full run plan without the write phase (--plan, #53).

        Runs ingest -> calibrate -> grid -> turbo precheck; with dry_run
        active no files or directories are created. A --turbo-strict
        infeasibility in the precheck exits 3 (SystemExit propagates) in
        CLI mode, or raises TurboInfeasibleError in library mode (ui=None).

        """
        scope = self.ui.run_scope() if self.ui is not None else nullcontext()
        with scope:
            with self._phase("read", unit="rows") as ph:
                self.ingest()
                ph.summary = f"{self.n_total:,} rows -> {self.n:,} points"
            with self._phase("calibrate") as ph:
                self.calibrate()
                ph.summary = f"transform={self.tname} cap_km={self.cap_km_val:.4g}"
            with self._phase("grid") as ph:
                self.grid()
                ph.summary = f"{self.nx}x{self.ny} cells @ {self.res:g} m"
            self._turbo_precheck()

    def plan_data(self, args: argparse.Namespace) -> dict[str, Any]:
        """Return the resolved run plan as a JSON-ready dict (single source of truth, #105).

        The human --plan printer (:meth:`plan_text`) and the --plan --json
        writer both read this dict, so the two surfaces cannot drift. Field
        names mirror run_summary.json (#55/#117) where the same data
        exists: the grid dict (incl. src_crs/work_crs/out_crs), ingest
        counts, workers, the turbo decision/plan (cap_gb, budget_gb,
        est_peak_gb — the fallback status), and outputs. Every value under
        ``estimated`` is an estimate, not a measurement (s30 convention;
        the measured peak-RAM value only exists after a run).

        Call after :meth:`plan` (ingest/calibrate/grid/precheck resolved).

        Args:
            args: Parsed CLI namespace (input/out paths, --force).

        Returns:
            JSON-safe dict with schema_version 1.

        """
        d = self._turbo_decision
        t = self._turbo_plan
        turbo: dict[str, Any] | None = None
        if d is not None and t is not None:
            turbo = {
                "enabled": True,
                "decision": asdict(d),
                "plan": asdict(t),
                "budget_bytes": self._turbo_budget_bytes,
            }
        if d is not None:
            est_peak_gb = float(d.est_peak_gb)
        else:
            # Non-turbo: same estimate the run summary's peak-RAM row uses.
            est_peak_gb = float(turbop.est_peak(self.nx_padded * self.ny_padded, self.n) / turbop.GB)
        out = Path(args.out)
        # Overwrite behaviour of a real (non-plan) run: mirrors the #54
        # guard (guarded names present without --force -> exit 2).
        guarded = ("value.tif", "support_km.tif", "calibration.json", "run_summary.json")
        existing = [name for name in guarded if (out / name).is_file()]
        if out.is_dir() and existing:
            overwrite = "force" if args.force else "guarded"
        elif out.is_dir():
            overwrite = "existing"
        else:
            overwrite = "create"
        return {
            "schema_version": 1,
            "status": "plan",
            "ppgrid_version": __version__,
            "input": args.input,
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
            "workers": self.workers,
            "strategy": self._strategy_text(),
            "turbo": turbo,
            "estimated": {
                "peak_ram_gb": est_peak_gb,
                # Raw int16 payload of the value + support bands (2 B/cell
                # each); the compressed on-disk size is codec-dependent and
                # unknown at plan time.
                "disk_bytes": int(2 * self.nx * self.ny * 2),
            },
            "outputs": {
                "value": str(out / "value.tif"),
                "support": str(out / "support_km.tif"),
            },
            "out_dir": str(out),
            "overwrite": overwrite,
        }

    def plan_text(self, args: argparse.Namespace) -> str:
        """Human-readable run plan for --plan (issue #53); printed on stdout.

        Formats :meth:`plan_data` (single source of truth, #105); the
        --plan --json writer serializes the same dict.

        Args:
            args: Parsed CLI namespace (input/columns/out paths, --json).

        Returns:
            The multi-line plan text.

        """
        data = self.plan_data(args)
        g = data["grid"]
        ing = data["ingest"]
        out = Path(args.out)
        pct_step = f"{g['percentile_step']:g}" if g["percentile_step"] is not None else "off"
        mbp = f"{args.max_band_parallel}" if args.max_band_parallel is not None else "default (--workers)"
        json_extra = " + run_summary.json" if args.json else ""
        cells_line = (
            f"  grid cells:    {g['nx']} x {g['ny']} = {g['cells']:,} cells @ res={g['res_m']:g} m "
            f"(padded {g['nx_padded']} x {g['ny_padded']}, step {g['step']})"
        )
        blocks_line = (
            f"  blocks:        {g['nbx']} x {g['nby']} of bsize={g['bsize']} "
            f"({g['n_tasks']} tasks, {g['n_empty_blocks']} empty)"
        )
        lines = [
            "ppgrid run plan (--plan: nothing written, exit 0)",
            f"  input:         {args.input}",
            f"  columns:       value={args.value_col!r} lng={args.lng_col!r} lat={args.lat_col!r}",
            f"  out dir:       {args.out} ({'exists' if out.is_dir() else 'does not exist yet'})",
            f"  ingest:        {ing['rows_read']} rows -> {ing['points']} points ({self._drop_summary()})",
            f"  grid extent:   x0={g['x0_m']:.3f} m, y0={g['y0_m']:.3f} m (work CRS EPSG:{g['work_crs']}, metres)",
            cells_line,
            f"  levels:        {g['levels']} (halo {g['halo']}, cap {data['cap_km']:g} km)",
            blocks_line,
            f"  transform:     {data['transform']}",
            f"  saturation:    {g['saturation']:g}",
            f"  scale:         {g['scale']:g} (DN = percentile * scale), percentile-step: {pct_step}",
            f"  seed:          {data['seed']}",
            f"  workers:       {data['workers']}, max-band-parallel: {mbp}",
            f"  compress:      {g['compress']}",
            f"  outputs:       value.tif (EPSG:{g['out_crs']}) + support_km.tif{json_extra}",
        ]
        if data["turbo"] is not None:
            dec, pl = data["turbo"]["decision"], data["turbo"]["plan"]
            mode = turbop.regime_name(dec["regime"])
            if dec["path"] == "shared":
                mode += " (shared path)"
            lines.append(
                f"  turbo:         on — {mode}, cap {pl['cap_gb']:g} GB, "
                f"budget {dec['budget_gb']:g} GB, est peak {dec['est_peak_gb']:g} GB, workers {pl['workers']}"
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
            # #105: schema version, shared with the --plan --json document.
            # Additive only: existing keys keep their names and order.
            "schema_version": 1,
            # Only JSON delta of issue #102: written only after a successful
            # run(), so it is always 'ok' (error surface = the exit code).
            "status": "ok",
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

    def _strategy_text(self) -> str:
        """Plain-language strategy line for the run summary (issue #102).

        E.g. ``full in-RAM turbo, cap 16 GB, budget 13.6 GB, workers 2``;
        the per-box regime names itself as the low-RAM fallback.
        """
        d = self._turbo_decision
        if d is not None and self._turbo_plan is not None:
            t = self._turbo_plan
            return (
                f"{turbop.regime_name(d.regime)} turbo, cap {t.cap_gb:g} GB, "
                f"budget {d.budget_gb:g} GB, workers {t.workers}"
            )
        if self.turbo:
            return f"turbo, workers {self.workers}"
        return f"standard, workers {self.workers}"

    def _peak_ram_text(self) -> str:
        """Peak-RAM line: measured ru_maxrss, else visibly labelled estimated (#102)."""
        gb = _ru_maxrss_gb()
        if gb is not None:
            return f"{gb:.1f} GB (measured)"
        d = self._turbo_decision
        if d is not None:
            return f"~{d.est_peak_gb:.1f} GB (estimated)"
        est = turbop.est_peak(self.nx_padded * self.ny_padded, self.n)
        return f"~{est / turbop.GB:.1f} GB (estimated)"

    def _run_summary_rows(self, t_total: float, vpath: str, spath: str) -> list[tuple[str, str] | str]:
        """Rows for the final run summary block (issue #102).

        Rows read/accepted/dropped (with reasons), grid WxH + cell count,
        strategy (the per-box regime names itself as a fallback),
        write-path fallbacks when any were taken, elapsed, peak RAM, output
        sizes, and the output paths one per line (bare, copyable).
        """
        rows: list[tuple[str, str] | str] = [
            ("rows read", f"{self.n_total:,}"),
            ("rows accepted", f"{self.n:,}"),
        ]
        if self.n_dropped:
            detail = []
            if self.n_dropped_nan_inf:
                detail.append(f"{self.n_dropped_nan_inf} non-finite value")
            if self.n_dropped_out_of_domain:
                detail.append(f"{self.n_dropped_out_of_domain} out-of-domain")
            if self.n_dropped_nonfinite_proj:
                detail.append(f"{self.n_dropped_nonfinite_proj} non-finite projection")
            rows.append(("rows dropped", f"{self.n_dropped} ({'; '.join(detail)})" if detail else str(self.n_dropped)))
        else:
            rows.append(("rows dropped", "0"))
        rows.extend(
            [
                ("grid", f"{self.nx}x{self.ny} ({self.nx * self.ny:,} cells)"),
                ("strategy", self._strategy_text()),
            ]
        )
        if self._fallbacks:
            rows.append(("fallbacks", "; ".join(dict.fromkeys(self._fallbacks))))
        rows.extend(
            [
                ("elapsed", f"{t_total:.2f}s"),
                ("peak RAM", self._peak_ram_text()),
                ("value.tif", _human_size(Path(vpath).stat().st_size)),
                ("support_km.tif", _human_size(Path(spath).stat().st_size)),
                str(vpath),
                str(spath),
            ]
        )
        return rows

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
                        f"on ({turbop.regime_name(self._turbo_decision.regime)}, "
                        f"{self._turbo_decision.path.replace('_', '-')} path, cap {self._turbo_plan.cap_gb:g} GB)"
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

    def _remove_run_temps(self, keep: tuple[str, ...] = ()) -> None:
        """Remove every run temp file in out_dir (idempotent, best effort).

        `keep` names files that the caller still holds open (the turbo DN
        fields are live memmaps through the write phase; the final cleanup
        from run()'s finally removes them once the run ends).

        The main process's own memmaps are released first: on Windows an
        mmap holds an OS file handle until close(), and unlinking a held
        file raises WinError 32, which would mask the real run error and
        break the CLI exit codes (CI run 38207801346 W1). The unlink
        errors themselves are never swallowed.

        """
        self._release_run_fields()
        out = Path(self.out_dir)
        for fname in _RUN_TEMP_FILES:
            if fname in keep:
                continue
            (out / fname).unlink(missing_ok=True)

    def _release_run_fields(self) -> None:
        """Drop the main-process references held on cfg (idempotent).

        Worker-pool handles are per-process and die with the workers; only
        the main process's fields can still be open at cleanup time. The
        fields are plain ndarrays in the in-RAM regime and memmaps in the
        memmap regime; dropping the last reference releases the OS handle
        (numpy 2.4+ has no memmap.close() API; CPython refcounting).

        cfg is None when grid() raised before building it (halo > bsize
        etc.); the finally cleanup must release what exists, not raise
        AttributeError on the absent config (review rev-os2 M1).

        """
        if self.cfg is None:
            return
        for field in ("pts", "val_full", "sup_full", "near_full"):
            setattr(self.cfg, field, None)

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
        actually used. Shared infeasible: strict -> stderr error + exit 3
        (CLI) or TurboInfeasibleError (library mode, issue #109);
        non-strict -> [warn] + continue on the per-box path (exit 0,
        correct output, turbo warp/write still apply). When the per-box
        est peak itself exceeds the budget the warning is explicit:
        best-effort, OOM risk, output may be partial (M-3).

        Library mode (``ui=None``) never prints directly: the summary and
        any message travel the ``ppgrid`` logger (stderr only if the
        caller configured it), and the strict infeasibility raises.

        Raises:
            SystemExit: code 3 in CLI mode when --turbo-strict and the
                shared path does not fit under the budget.
            TurboInfeasibleError: the same condition in library mode
                (issue #109: the library never raises SystemExit).

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
        # precheck names the OOM risk instead of staying green. Only finite,
        # known bounds count; with none (non-Linux: no /proc/meminfo, cgroup
        # unlimited) available_gb is None and precheck skips the ceiling note.
        physical_gb = turbop._read_physical_gb()  # ruff: ignore[private-member-access]
        cgroup_gb = turbop._read_cgroup_max_gb()  # ruff: ignore[private-member-access]
        finite = [g for g in (physical_gb, cgroup_gb) if g is not None and math.isfinite(g)]
        available_gb = min(finite) if finite else None
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
        if self.ui is None:
            # Library mode (issue #109): no direct prints. The summary and
            # any message travel the ppgrid logger (stderr only if the
            # caller configured it; the default lastResort handler shows
            # WARNING and above), and a strict infeasibility becomes a
            # TurboInfeasibleError below instead of SystemExit(3).
            log.info("%s", decision.summary)
            if decision.message:
                if decision.exit_code == 3:
                    log.error("%s", decision.message)
                else:
                    log.warning("%s", decision.message.removeprefix("[warn] "))
        elif self.dry_run:
            # --plan: the plan printout carries the decision; only a hard
            # failure (--turbo-strict infeasible) says its piece on stderr.
            if decision.exit_code == 3 and decision.message:
                _ui_note_precheck(decision.message)
        else:
            # stdout stays clean (the --json/--help contract): the summary
            # goes through the run logger (stderr, INFO; -q suppresses it,
            # like every other progress line) (#77).
            log.info("%s", decision.summary)
            if decision.message:
                _ui_note_precheck(decision.message)
        if decision.exit_code == 3:
            if self.ui is None:
                raise TurboInfeasibleError(decision.message)
            raise SystemExit(3)

    def _turbo_zstd_ok(self) -> bool:
        """A.7 calibration oracle gate for the parallel ZSTD write.

        Writes one deterministic _ORACLE_PROBE_W x TILE_PX int16 probe
        (partial width, so the gate also covers GDAL's zero-padding of edge
        tiles - audit #81 F-1) with the stock codec settings, then asks
        zstdmt whether the turbo-compressed frame is byte-equal to the bytes
        GDAL stored. Mismatch (or unavailable) -> False: the caller writes
        with the stock-codec parallel path (_turbo_write_stock_parallel);
        the serial per-band write runs only if that writer's layout guards
        fail. Either way the output is byte-exact.
        """
        tile = _oracle_probe_tile()
        probe = Path(self.out_dir) / "_zstd_oracle.tif"
        try:
            # zstdmt.available() inside the try (audit #81 F-3): a gate
            # failure must mean "stock parallel write", never a crash.
            if not zstdmt.available():
                self._zstd_gate_reason = "CPL zstd unavailable"
                _ui_note("warn", "turbo zstd oracle: CPL zstd unavailable; stock parallel write")
                return False
            with rasterio.open(
                probe,
                "w",
                driver="GTiff",
                dtype="int16",
                width=_ORACLE_PROBE_W,
                height=TILE_PX,
                count=1,
                nodata=NODATA,
                crs=f"EPSG:{self.work_crs}",
                transform=from_origin(0, TILE_PX * self.res, self.res, self.res),
                compress=self.compress,
                tiled=True,
                blockxsize=TILE_PX,
                blockysize=TILE_PX,
                predictor=2,
                BIGTIFF="IF_SAFER",
            ) as dst:
                dst.write(tile, 1)
            verdict = zstdmt.oracle_check(probe, tile=0)
            if verdict.ok:
                return True
            self._zstd_gate_reason = "oracle mismatch"
            _ui_note("warn", f"{verdict.detail}")
        except Exception as e:  # ruff: ignore[blind-except] - any probe failure means the stock parallel writer runs
            self._zstd_gate_reason = "oracle probe error"
            _ui_note("warn", f"turbo zstd oracle: {e}; stock parallel write")
        finally:
            probe.unlink(missing_ok=True)
        return False

    def _is_regime_a(self) -> bool:
        """Return True when the pre-check selected regime A (full in-RAM)."""
        d = self._turbo_decision
        return d is not None and d.regime == "A"

    def _prepare_shared(self, progress: Callable[[int], None] | None = None) -> bool:
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

        # pts is fully consumed above (ix/iy copies + the binned TV row).
        # Drop the reference here, not at frame end: on Windows the OS
        # handle stays open while the mmap object is alive, and a frame
        # pinned by an in-flight traceback would keep _points.npy locked
        # past _remove_run_temps (WinError 32, CI run 38207801346 W1).
        pts = None

        with _prof.phase("shared.pyramid"):
            sums: list[np.ndarray] = [s0]
            for _ in range(self.levels):
                sums.append(downsample_sum(sums[-1]))
            counts: list[np.ndarray] = [c0]
            for _ in range(self.levels):
                counts.append(downsample_sum(counts[-1]))
        if progress is not None:
            progress(int(self.nx_padded * self.ny_padded))

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
                    progress=progress,
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
                progress=progress,
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
                    progress=progress,
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
        interrupted = False
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
                    # per-field dict copy). Rebuilt fresh each run and passed
                    # with every task: stale state from a previous run in this
                    # process cannot reach the workers (audit #83).
                    cfg = self.cfg
                    cfg.pts = np.load(cfg.pts_path, mmap_mode="r")
                    cfg.tf = make_transform(cfg.transform_state)
                    cfg.pct = PercentileTransform(cfg.pct_quantiles)
                    # Progress (issue #101): the interpolate bar budgets the
                    # shared descent plus every task block, and advances as
                    # each block is written.
                    task_cells: list[int] = []
                    for bx, by in self.tasks:
                        w = _block_window(bx, by, self.cfg)
                        task_cells.append(w.width * w.height)
                    with self._phase("interpolate", unit="cells") as iph:
                        # D-1: the shared-descent total is MEASURED, not
                        # estimated. The descent advances its own cell
                        # counts (pyramid proxy + per-level work); the old
                        # 2.5x padded-grid estimate overshot, so the bar
                        # froze under 100% (and was added even on the
                        # per-block path, where nothing advanced it).
                        # Accumulate what actually advanced; total = that +
                        # the task blocks, so the phase ends at exactly
                        # 100% (no forced finish percent).
                        desc: list[int] = [0]

                        def _desc_advance(n: int) -> None:
                            desc[0] += n
                            iph.advance(n)

                        with _WallPhase("descent", self._phase_wall):
                            if self._prepare_shared(progress=_desc_advance):
                                cfg.shared = True
                        iph.set_total(sum(task_cells) + desc[0])
                        with (
                            ThreadPoolExecutor(max_workers=self.workers) as ex,
                            _prof.phase("write.serial"),
                            _WallPhase("interpolate", self._phase_wall),
                        ):
                            results = zip(
                                self.tasks,
                                ex.map(_process_block, [(cfg, None, bx, by) for bx, by in self.tasks]),
                                task_cells,
                                strict=True,
                            )
                            for task, res, tc in results:
                                bx, by = task
                                _, _, out = res
                                w = _block_window(bx, by, self.cfg)

                                if out is None:
                                    blank = np.full((w.height, w.width), NODATA, np.int16)
                                    vd.write(blank, 1, window=w)
                                    sd.write(blank, 1, window=w)
                                else:
                                    vq, rq = out
                                    vd.write(vq.T[::-1, :], 1, window=w)
                                    sd.write(rq.T[::-1, :], 1, window=w)
                                iph.advance(tc)
                                if self.ui is not None:
                                    self.ui.maybe_refresh()
                        iph.summary = f"{self.nx}x{self.ny} grid"
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
                with self._phase("reproject", total=2 * dst_width * dst_height, unit="cells") as rph:

                    def _reproj_progress(done: int) -> None:
                        rph.advance(done)
                        if self.ui is not None:
                            self.ui.maybe_refresh()

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
                            progress=_reproj_progress,
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
                            progress=_reproj_progress,
                        )
                    rph.summary = f"-> {dst_width}x{dst_height} EPSG:{self.out_crs}"
            with self._phase("write") as wph:
                # Publish: the finals only ever change via atomic rename.
                ftmp_v.replace(vpath)
                self._published.add(vpath.name)
                ftmp_s.replace(spath)
                self._published.add(spath.name)
                mb_v = Path(vpath).stat().st_size / 1e6
                mb_s = Path(spath).stat().st_size / 1e6
                wph.summary = f"value.tif {mb_v:.1f} MB, support_km.tif {mb_s:.1f} MB"
            if wtmp_v is not ftmp_v:
                wtmp_v.unlink(missing_ok=True)
                wtmp_s.unlink(missing_ok=True)
        except BaseException as exc:
            # Mid-run failure (worker error, OOM, disk full): the finals were
            # not reached (os.replace is last). Warn so nobody consumes an
            # earlier run's files as this run's output (issues #15, #40).
            # A Ctrl-C suppresses the warn: main() reports the interrupt
            # itself, one line, exit 130 (issue #106).
            partial = True
            interrupted = isinstance(exc, KeyboardInterrupt)
            raise
        finally:
            self._remove_run_temps()
            if partial and not interrupted:
                _ui_note(
                    "warn",
                    f"run failed during write; {vpath}, {spath} were not fully written "
                    "(any existing files are from an earlier run, left untouched)",
                    legacy="warning: ",  # pre-#102 raw print embedded 'warning: ' (F1)
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
        _load_native()  # mid-run failure path references _CLI_MAPPED_EXCEPTIONS
        budget = self._turbo_budget_bytes
        if budget is not None and self.nx * self.ny * 4 > budget // 2:
            self._fallbacks.append("DN field over half budget -> serial write")
            self._write_rasters_serial(vprof, sprof, vpath, spath, vtags, stags)
            return

        # Staging names (issue #40): the finals are only ever published via
        # os.replace() on success, so a mid-write crash leaves .tmp files
        # instead of truncating value.tif / support_km.tif.
        ftmp_v = Path(self.out_dir) / (vpath.name + ".tmp")
        ftmp_s = Path(self.out_dir) / (spath.name + ".tmp")
        out_dir = Path(self.out_dir)

        partial = False
        interrupted = False
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
                    # per-field dict copy). Rebuilt fresh each run and passed
                    # with every task: stale state from a previous run in this
                    # process cannot reach the workers (audit #83).
                    cfg = self.cfg
                    cfg.pts = np.load(cfg.pts_path, mmap_mode="r")
                    cfg.tf = make_transform(cfg.transform_state)
                    cfg.pct = PercentileTransform(cfg.pct_quantiles)
                    # Progress (issue #101): as the serial writer, but the
                    # blocks write DN grids straight into the tmpfs fields.
                    task_cells: list[int] = []
                    for bx, by in self.tasks:
                        w = _block_window(bx, by, self.cfg)
                        task_cells.append(w.width * w.height)
                    with self._phase("interpolate", unit="cells") as iph:
                        # D-1: measured shared-descent total (see the serial
                        # block) — the phase ends at exactly 100%.
                        desc: list[int] = [0]

                        def _desc_advance(n: int) -> None:
                            desc[0] += n
                            iph.advance(n)

                        with _WallPhase("descent", self._phase_wall):
                            if self._prepare_shared(progress=_desc_advance):
                                cfg.shared = True
                        iph.set_total(sum(task_cells) + desc[0])
                        # The workers write the DN grids straight into the tmpfs
                        # fields (issue #84): the main thread used to do those
                        # copies as results arrived and sat on the critical path
                        # ahead of the pool. Block windows are disjoint, so the
                        # writes are thread-safe and the fields are byte-identical.
                        # The pair rides in each task tuple (audit #83): it is
                        # scoped to this block pool and cannot leak into a
                        # later serial run.
                        dn = (val_dn, sup_dn)
                        with (
                            _WallPhase("interpolate", self._phase_wall),
                            ThreadPoolExecutor(max_workers=self.workers) as ex,
                        ):
                            for tc, _ in zip(
                                task_cells,
                                ex.map(_process_block, [(cfg, dn, bx, by) for bx, by in self.tasks]),
                                strict=True,
                            ):
                                iph.advance(tc)
                                if self.ui is not None:
                                    self.ui.maybe_refresh()
                        # Drop the holder before the final unlink (Windows:
                        # an unlink of a held mmap is WinError 32, as in the
                        # finally's val_dn/sup_dn/arr release). The workers
                        # have consumed the pair; nothing else needs it.
                        dn = None
                        iph.summary = f"{self.nx}x{self.ny} grid"
                    # The shared field is dead after the block loop: free it
                    # before the reproject/write (val/sup f32 + near bool,
                    # ~13.7 GB at full-AU scale) (issue #39).
                    cfg.val_full = cfg.sup_full = cfg.near_full = None
            # Same position as before the restructure: the DN fields are
            # still open memmaps (the inode outlives the unlink), but the
            # tmpfs space is freed while the write phase runs. The two DN
            # files stay out of this mid-removal: on Windows an unlink of a
            # held mmap is WinError 32; this method's finally releases the
            # local holders, then the final cleanup removes the files.
            self._remove_run_temps(keep=("_val_dn.npy", "_sup_dn.npy"))

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
            if not cpl:
                self._fallbacks.append(
                    f"zstd parallel write -> stock parallel ({self._zstd_gate_reason or 'oracle mismatch'})"
                )
            # The writers below create ftmp_v / ftmp_s fresh (O_EXCL |
            # O_NOFOLLOW) and never follow a pre-planted symlink (issues #15, #77).
            for band, arr, prof, path, tags, scales in (
                ("value", val_dn, vprof, ftmp_v, vtags, (1.0 / self.scale,)),
                ("support", sup_dn, sprof, ftmp_s, stags, (1.0,)),
            ):
                # Per-file total: the phase opens inside the per-file loop,
                # so each bar must end at 100%, not 50% (F5, #115 review).
                with self._phase("write", total=dst_width * dst_height, unit="cells") as wph:

                    def _wprogress(done: int) -> None:
                        wph.advance(done)

                    def _wrefresh() -> None:
                        if self.ui is not None:
                            self.ui.maybe_refresh()

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
                                    progress=_wprogress,
                                    progress_refresh=_wrefresh,
                                )
                            except Exception as e:  # ruff: ignore[blind-except] - layout surprise: fall back
                                _ui_note("warn", f"turbo zstd parallel: {e}; stock parallel write")
                            if not wrote:
                                self._fallbacks.append("zstd parallel write guard -> stock parallel")
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
                                    progress=_wprogress,
                                    progress_refresh=_wrefresh,
                                )
                            except Exception as e:  # ruff: ignore[blind-except] - any surprise: serial
                                _ui_note("warn", f"turbo stock parallel: {e}; serial write")
                            if not wrote:
                                self._fallbacks.append("stock parallel write guard -> serial write")
                        if not wrote:
                            # Layout guard tripped (see the parallel writers): the
                            # serial array reproject is byte-identical by
                            # construction.

                            def _serial_progress(done: int) -> None:
                                _wprogress(done)
                                _wrefresh()

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
                                progress=_serial_progress,
                            )
                        wph.summary = f"{Path(path).name} {Path(path).stat().st_size / 1e6:.1f} MB"
            # Publish: the finals only ever change via atomic rename.
            ftmp_v.replace(vpath)
            self._published.add(vpath.name)
            ftmp_s.replace(spath)
            self._published.add(spath.name)
        except BaseException as exc:
            # Mid-run failure (worker error, OOM, disk full): the finals were
            # not reached (os.replace is last). Warn so nobody consumes an
            # earlier run's files as this run's output (issues #15, #40).
            # A Ctrl-C suppresses the warn and the stack copy below: main()
            # reports the interrupt itself, one line, exit 130 (issue #106).
            partial = True
            interrupted = isinstance(exc, KeyboardInterrupt)
            # Copy the full stack to stderr for exceptions the CLI does not
            # map to a clean 'error: <msg>' (unmapped -> exit 1 + raw
            # traceback): the tb is dropped below because it pins the
            # writer frames, whose args still reference the DN memmaps
            # (Windows: a held mmap keeps the OS file handle and the
            # finally's unlink raises WinError 32, CI run 38215575734 C1).
            # Without the print, the displayed stack for unmapped errors is
            # flattened (review rev-os2 M2). Mapped exceptions (#16: clean
            # error line, no stack) are untouched. Exception type/message
            # are unchanged, so the CLI exit-code mapping is unaffected.
            if not interrupted and not isinstance(exc, _CLI_MAPPED_EXCEPTIONS):
                traceback.print_exception(exc, file=sys.stderr)
            exc.__traceback__ = None
            raise
        finally:
            # Release the local DN memmap holders before the final unlink
            # (same WinError 32 on Windows): `val_dn` / `sup_dn` would
            # otherwise live to the function end, `arr` is the warp-loop
            # variable still bound to the last band's field, and `dn` is
            # the (val_dn, sup_dn) tuple the task args rode on — the happy
            # path releases it after the pool, an exception inside the pool
            # skips that line.
            val_dn = sup_dn = arr = dn = None
            self._remove_run_temps()
            for p in out_dir.glob("_stock_scratch_*.tif"):
                p.unlink(missing_ok=True)
            if partial and not interrupted:
                _ui_note(
                    "warn",
                    f"run failed during write; {vpath}, {spath} were not fully written "
                    "(any existing files are from an earlier run, left untouched)",
                    legacy="warning: ",  # pre-#102 raw print embedded 'warning: ' (F1)
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
        _ui_note("error", f"{flag} {name!r} not found in {path}")
        # Auxiliary detail lines, not standalone events: raw indented print
        # right after the unified error line (D-7).
        detail = [f"  columns found: {shown}"]
        close = difflib.get_close_matches(name, cols, n=3)
        if close:
            detail.append(f"  did you mean: {', '.join(close)}?")
        print("\n".join(detail), file=sys.stderr)  # ruff: ignore[print]
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


def _make_pipeline(args: argparse.Namespace, *, dry_run: bool, ui: _ui.UI | None = None) -> Pipeline:
    """Build the Pipeline from parsed CLI args (shared by --plan and run).

    Resolves the turbo flag/cap pair in one place so the two paths cannot
    drift.

    Args:
        args: Parsed CLI namespace.
        dry_run: True for --plan (no files created).
        ui: Presentation handle for progress display (issue #101); None for
            library-style calls.

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
        ui=ui,
    )


# Exception types the CLI maps to a clean named exit (see
# _map_pipeline_errors below): reported as an 'error: <msg>' line + exit
# code, no stack trace (#16). Shared with _write_rasters_turbo's failure
# path, which must know which exceptions already get clean reporting (so
# no full stack is pre-printed before the traceback is released). SystemExit
# is included too: pass-through control flow (e.g. --turbo-strict exit 3),
# not a failure warranting a stack.
# _CLI_MAPPED_EXCEPTIONS: populated by _load_native() (it references lazy
# names, so it cannot be built at import time); empty until first native use.
_CLI_MAPPED_EXCEPTIONS: tuple[type[BaseException], ...] = ()


def _map_pipeline_errors(fn: Callable[[], Any], input_path: str) -> Any:
    """Run a pipeline phase, mapping exceptions to the CLI exit-code contract.

    Shared by the --plan phases and the full run so both report identically:
    exit 1 for pipeline/IO errors, exit 2 for validation errors (bad
    columns, no valid points), exit 3 passes through (--turbo-strict).

    Default mapping (audit #83): an exception type NOT explicitly mapped
    below is a pipeline failure — a named one-liner and exit 1, never a raw
    traceback. The contract no longer depends on each new failure mode
    being born as a mapped class.

    Args:
        fn: Zero-arg callable running the phase (or the full run).
        input_path: Input path for the FileNotFoundError message.

    Returns:
        The callable's return value.

    Raises:
        SystemExit: 1 (pipeline/IO, including the default mapping) or 2
            (validation, unknown EPSG).

    """
    _load_native()  # the except clauses below reference CRSError (pyproj)
    try:
        return fn()
    except (CRSError, RasterioCRSError) as e:
        # Unknown EPSG: pyproj's CRSError (--src-crs/--work-crs) and
        # rasterio's CRSError (--out-crs, a ValueError subclass) both land
        # here — one exit code for all three CRS flags: 2, "bad flag value"
        # per the README exit-code table (#77). The message names the code.
        _ui_note("error", str(e))  # D-7: unified error line, exit 2 for all CRS flags
        raise SystemExit(2) from None
    except ImportError as e:
        # e.g. parquet input without the optional pyarrow extra.
        _die(str(e))
    except FileNotFoundError as e:
        # Only the user-supplied input renders as "input file not found";
        # any other missing path is an internal resource (meminfo, cgroup,
        # run temp) and names itself (issue #106, attribution audit).
        fname = e.filename
        if fname is None or Path(fname).resolve() == Path(input_path).resolve():
            _die(f"input file not found: {fname or input_path}")
        _die(f"cannot open {fname}: {e.strerror or 'I/O error'}")
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
        _ui_note("error", str(e))  # D-7: unified error line
        raise SystemExit(2) from None
    except OSError as e:
        # NotADirectoryError, disk full, etc. The actually failing path is
        # named (issue #106): e.filename when the OS reports one, else the
        # raw error (str(e) carries the path when it has one).
        if e.filename:
            _die(f"cannot access {e.filename}: {e.strerror or 'I/O error'}")
        _die(str(e))
    except Exception as e:  # ruff: ignore[blind-except] - default mapping (audit #83)
        # Any failure mode NOT mapped above is a pipeline failure: a named
        # one-liner and exit 1, never a raw traceback. A new exception type
        # cannot silently drift the exit-code contract; it is named on the
        # same line as a mapped one. KeyboardInterrupt/SystemExit are
        # BaseException and keep their own contracts.
        _die(f"{type(e).__name__}: {e} (input: {input_path})")


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


def _effective_params(args: argparse.Namespace) -> dict[str, Any]:
    """All effective pipeline params post-precedence-merge (issue #64).

    Config key -> final value for every config-able option, in template
    order. Call after _apply_config, so no UNSET sentinel can leak in.

    """
    return {key: getattr(args, _CONFIG_KEY_TO_DEST[key]) for key, _, _, _ in _CONFIG_OPTIONS}


def _params_fingerprint(params: dict[str, Any]) -> str:
    """sha256 of the canonical parameter JSON (issue #64).

    Canonical form: sorted keys, compact separators, finite values
    (_sanitize_json), and no timestamp — two runs with identical effective
    params share one fingerprint regardless of when they ran.

    """
    canonical = json.dumps(_sanitize_json(params), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _version_report() -> dict[str, str]:
    """Tool versions for the run fingerprint sidecar (issue #64).

    Called after a successful run, so the native stack is loaded and the
    version reads are plain attribute lookups (the lazy contract is on
    module scope, not on call-time imports).

    """
    import pyproj
    import rasterio as _rasterio  # function-local: loaded at this point by the run itself

    return {
        "ppgrid": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rasterio": _rasterio.__version__,
        "gdal": _rasterio.__gdal_version__,
        "pyproj": pyproj.__version__,
    }


def _write_run_sidecar(args: argparse.Namespace, out: Path) -> None:
    """Write <out>/ppgrid_run.json after a successful run (issue #64).

    The run fingerprint sidecar: schema_version, all effective params
    (post-precedence-merge), the config file used (or null), the input
    file name, tool versions, an ISO-8601 UTC timestamp, and fingerprint
    (sha256 of the canonical, timestamp-free parameter JSON). The file is
    new metadata: it does not enter the overwrite guard's raster list.
    Atomic like the --json summary (fresh 0600 tmp + os.replace); a write
    failure is a named error — exit 1 — never a raw traceback.

    Args:
        args: Parsed namespace after _apply_config (real values only).
        out: Output directory (already created, run already published).

    """
    params = _effective_params(args)
    payload = {
        "schema_version": 1,
        "params": params,
        "config": args.config_path,
        "input": Path(args.input).name,
        "versions": _version_report(),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "fingerprint": _params_fingerprint(params),
    }
    path = out / "ppgrid_run.json"
    tmp = out / "ppgrid_run.json.tmp"
    try:
        with os.fdopen(_staging_fd(tmp), "wb") as f:
            f.write((json.dumps(_sanitize_json(payload), indent=2, allow_nan=False) + "\n").encode("utf-8"))
        tmp.replace(path)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        _die(f"cannot write run fingerprint: {e}")
    log.info("run fingerprint: %s", path)


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
    # Stash, don't emit: this runs before _configure_logging installs the
    # unified handler, and the warning must land in the same format as
    # every other CLI event (D-7).
    _PENDING_LOG_WARN.append(f"unrecognized LOGGING={raw!r}; using 'info'")
    return "info"


# Unrecognized-LOGGING warnings stashed by _resolve_log_level and emitted by
# _configure_logging (the unified line needs the handler to exist first).
_PENDING_LOG_WARN: list[str] = []


def _configure_logging(level_name: str, *, color: bool = False) -> None:
    """Configure the `ppgrid` logger for one CLI run (stderr, unified format).

    Only the `ppgrid` logger is touched — never the root logger — so library
    users' logging setup is undisturbed and stdout stays clean for any
    machine-readable output. Every CLI event (phase lines, notes, warnings,
    errors) shares this one handler and the `L ppgrid.pipeline: msg` format
    (D-7); WARNING/ERROR lines carry semantic SGR when ``color`` is on.
    Idempotent: repeated main() calls in one process (tests) update the
    level + formatter + stream instead of stacking handlers.

    Args:
        level_name: Resolved level name from _resolve_log_level.
        color: Emit semantic SGR on WARNING/ERROR (``--color always`` on a
            non-TTY).

    """
    levels = {"error": logging.ERROR, "warning": logging.WARNING, "info": logging.INFO, "debug": logging.DEBUG}
    from . import ui as _ui_mod  # local: LOAD_GLOBAL does not consult the module __getattr__ (#109)

    root = logging.getLogger("ppgrid")
    root.setLevel(levels[level_name])
    handler = None
    for h in root.handlers:
        if getattr(h, "ppgrid_handler", False):
            handler = h
            break
    if handler is None:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.ppgrid_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    elif handler.stream is not sys.stderr:  # type: ignore[attr-defined]
        # Rebind to the current stderr: tests (capsys) and TTY swaps
        # replace sys.stderr between runs; holding the first run's
        # stream object sends later records into a closed file
        # (logging-error spam, #77).
        # Assign directly: setStream() flushes the old stream first,
        # and that raises on a closed file (capsys test teardown).
        handler.stream = sys.stderr  # type: ignore[attr-defined]
    handler.setFormatter(_ui_mod.plain_log_formatter(color=color))
    # The handler's own level always mirrors the resolved stderr level: a
    # previous --log-file run may have lowered the logger below it (the file
    # handler wants INFO on a -q run), and handler-level gating is what keeps
    # each output independent.
    handler.setLevel(levels[level_name])
    if _PENDING_LOG_WARN:
        for warn in _PENDING_LOG_WARN:
            _ui_note("warn", warn)
        _PENDING_LOG_WARN.clear()


def _configure_log_file(path_str: str | None) -> None:
    """Route a second copy of the run log to a file (issue #65).

    Created/truncated per run, level INFO (independent of the stderr level,
    so --quiet runs still get a full INFO copy), same unified format as the
    stderr handler (plain, no SGR — a log file carries no color). stderr
    output is unchanged. Idempotent across main() calls in one process
    (tests): a previous file handler is closed and replaced, and a run
    without --log-file drops the handler.

    Args:
        path_str: --log-file value (None = no file logging).

    Raises:
        SystemExit: 1 (cannot create or open the log file).

    """
    root = logging.getLogger("ppgrid")
    for h in list(root.handlers):
        if getattr(h, "ppgrid_log_file", False):
            root.removeHandler(h)
            h.close()
    if path_str is None:
        return
    path = Path(path_str)
    try:
        f = path.open("w", encoding="utf-8")
    except OSError as e:
        _die(f"cannot open log file {path}: {e.strerror or str(e)}")
    handler = logging.StreamHandler(stream=f)
    handler.ppgrid_log_file = True  # type: ignore[attr-defined]
    handler.setLevel(logging.INFO)
    from . import ui as _ui_mod  # local: LOAD_GLOBAL does not consult the module __getattr__ (#109)

    handler.setFormatter(_ui_mod.plain_log_formatter(color=False))
    root.addHandler(handler)
    # The logger level gates every record before the handlers: a -q run
    # (WARNING) would drop the file's INFO records. Lower the logger to the
    # file level and let the stderr handler's own level keep the terminal
    # output unchanged.
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)


def _config_kind_ok(kind: str, value: Any) -> bool:
    """Gate a TOML value's Python type against the option's accepted kind (#63).

    bool is checked first: it is an int subclass, and a ``res = true``-style
    typo must not sneak past the numeric kinds.

    """
    if isinstance(value, bool):
        return kind == "bool"
    if isinstance(value, int):
        return kind in ("int", "num", "str-or-num", "crs")
    if isinstance(value, float):
        return kind in ("num", "str-or-num")
    if isinstance(value, str):
        return kind in ("str", "str-or-num", "crs")
    return False


_CONFIG_KIND_ERROR: dict[str, str] = {
    "str": "a string",
    "int": "an integer",
    "num": "a number",
    "bool": "true or false",
    "str-or-num": "a string or a number",
    "crs": "an EPSG code (integer) or an 'EPSG:<code>' string",
}


def _set_config_value(
    parser: argparse.ArgumentParser, args: argparse.Namespace, cfg: Path, key: str, value: Any
) -> None:
    """Validate one config value against its flag's own checker and store it (#63).

    The value runs through the same type callable and choices as the CLI
    flag, so a config value and the flag spelling that maps to it are
    accepted identically; a rejection is a usage error (exit 2) naming the
    file and the key. A dest the CLI flag already set is left alone
    (flag > config).

    Args:
        parser: The run parser (provides the flag's type callable).
        args: Parsed namespace (mutated on success).
        cfg: Config file path (for the error message).
        key: Config key (a member of _CONFIG_KEYS).
        value: TOML value (str/int/float/bool).

    Raises:
        SystemExit: 2 (wrong TOML kind, or the flag's checker rejects it).

    """
    dest = _CONFIG_KEY_TO_DEST[key]
    if getattr(args, dest) is not UNSET:
        return  # CLI flag already set it (flag > config)
    kind = _CONFIG_KIND[key]
    if not _config_kind_ok(kind, value):
        parser.error(f"config: {key!r} must be {_CONFIG_KIND_ERROR[kind]}: {value!r} ({cfg})")
    action = next(a for a in parser._actions if a.dest == dest)  # ruff: ignore[private-member-access]
    resolved: Any
    if isinstance(value, bool):
        resolved = value
    else:
        text = value if isinstance(value, str) else str(value)
        if action.type is not None:
            try:
                resolved = action.type(text)
            except (ValueError, argparse.ArgumentTypeError) as e:
                parser.error(f"config: {key!r}: {e} ({cfg})")
        else:
            resolved = text
    if action.choices is not None and resolved not in action.choices:
        parser.error(f"config: {key!r} must be one of {', '.join(action.choices)}: {value!r} ({cfg})")
    if key == "turbo" and resolved not in ("auto", "16", "32", "64", "128"):
        # --turbo carries no type checker (optional-arg const); the preset
        # set is validated the same way _main_preflight does.
        parser.error(f"config: 'turbo' must be auto|16|32|64|128: {value!r} ({cfg})")
    setattr(args, dest, resolved)


def _resolve_sentinels(args: argparse.Namespace) -> None:
    """Fill every config-able UNSET dest with its built-in default (#63).

    Runs at the end of _apply_config, so after it returns every config-able
    option holds a real value even when no config file is present.

    """
    for dest, default in _CONFIG_DEFAULTS.items():
        if getattr(args, dest) is UNSET:
            setattr(args, dest, default)


def _apply_config(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Path | None:
    """Merge the ppgrid.toml config into the parsed args (issue #63).

    Precedence: CLI flag > config > built-in default — a config key only
    fills an option whose dest is still the UNSET sentinel (the flag was
    not passed). Discovery: --config PATH (explicit; a missing file is a
    usage error), else ./ppgrid.toml, else $HOME/.config/ppgrid/ppgrid.toml;
    an absent file is a silent no-op. Unknown keys (including TOML
    table headers) warn and are ignored; invalid values are usage errors
    (exit 2).

    Args:
        parser: The run parser (provides the flags' type checkers).
        args: Parsed namespace (mutated: config values + sentinel
            resolution; every config-able dest is real on return).

    Returns:
        The config file applied, or None (no file present).

    Raises:
        SystemExit: 2 (missing explicit --config file, invalid TOML, bad value).

    """
    cfg: Path | None
    if args.config is not None:
        cfg = Path(args.config)
        if not cfg.is_file():
            parser.error(f"config file not found: {args.config}")
    else:
        home_cfg = Path.home() / ".config" / "ppgrid" / _CONFIG_FILE_NAME
        cfg = next((c for c in (Path(_CONFIG_FILE_NAME), home_cfg) if c.is_file()), None)
    if cfg is None:
        _resolve_sentinels(args)
        return None
    try:
        with cfg.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        parser.error(f"invalid TOML in {cfg}: {e}")
    except OSError as e:
        parser.error(f"cannot read config file {cfg}: {e.strerror or str(e)}")
    for key, value in data.items():
        if key not in _CONFIG_KEYS:
            _ui_note("warn", f"config: unknown key {key!r} ignored ({cfg})")
            continue
        _set_config_value(parser, args, cfg, key, value)
    _resolve_sentinels(args)
    return cfg


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser (tests exercise the flags here).

    Layout (issue #103): options live in six groups — input/columns, CRS/grid,
    interpolation, memory/turbo, output, logging — and `--help` is snapshot-
    tested. Metavars carry units (METRES, KM, GB, CELLS). Every option whose
    built-in default is non-None shows it in its help text as "(default: X)";
    since issue #63 the config-able options carry the UNSET sentinel as their
    parser default instead (so "not passed" stays distinguishable from
    "passed"), and their built-in default comes from _CONFIG_DEFAULTS. The
    mechanical check is tests/test_cli_help.py.
    """
    parser = argparse.ArgumentParser(
        prog="ppgrid",  # fixed prog: help/snapshot are identical under -m, -c, and the console script
        description="Pull-push scattered-data interpolation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    parser.add_argument("--version", action="version", version=f"ppgrid {__version__}")
    parser.add_argument("input", nargs="?", help="CSV or Parquet input path")

    g_input = parser.add_argument_group("input/columns")
    g_input.add_argument("--value-col", default=UNSET, help="Value column name (default: value)")
    g_input.add_argument("--lng-col", default=UNSET, help="Longitude column name (default: longitude)")
    g_input.add_argument("--lat-col", default=UNSET, help="Latitude column name (default: latitude)")

    g_crs = parser.add_argument_group("CRS/grid")
    g_crs.add_argument(
        "--src-crs",
        type=_crs_arg,
        default=UNSET,
        help=f"Input CRS as a bare int or 'EPSG:<code>' string (default: {SRC_CRS})",
    )
    g_crs.add_argument(
        "--work-crs",
        type=_crs_arg,
        default=UNSET,
        help=f"Working CRS for interpolation, bare int or 'EPSG:<code>' (default: {WORK_CRS})",
    )
    g_crs.add_argument(
        "--out-crs",
        type=_crs_arg,
        default=UNSET,
        help=f"Output CRS, bare int or 'EPSG:<code>' (default: {OUT_CRS})",
    )
    g_crs.add_argument(
        "--res",
        type=_pos_float,
        default=UNSET,
        metavar="METRES",
        help=f"Cell size in metres (default: {DEFAULT_RES:g})",
    )
    g_crs.add_argument(
        "--cap-km",
        type=_cap_km_arg,
        default=UNSET,
        metavar="KM",
        help="Fill cap in km, or 'auto' (default: auto)",
    )
    g_crs.add_argument(
        "--block",
        type=int,
        default=UNSET,
        metavar="CELLS",
        help=f"Block size in cells (default: {DEFAULT_BLOCK})",
    )

    g_interp = parser.add_argument_group("interpolation")
    g_interp.add_argument(
        "--transform",
        default=UNSET,
        # Pinned to the calibrate.transforms() factory names (audit #83):
        # adding a transform there adds its name here, nothing else to update.
        choices=["auto", *(t.name for t in transforms())],
        help="Value transform for the interpolation (default: auto)",
    )
    g_interp.add_argument(
        "--saturation",
        type=_pos_float,
        default=UNSET,
        help=f"Point counts for a cell to fully self-trust (default: {DEFAULT_SATURATION:g})",
    )
    g_interp.add_argument(
        "--scale", type=_pos_float, default=UNSET, help=f"DN = percentile * scale (default: {DEFAULT_SCALE:g})"
    )
    g_interp.add_argument(
        "--percentile-step",
        type=_pos_float,
        default=UNSET,
        help="Round output percentiles to the nearest step (e.g. 5 -> 90/95/100) (default: off)",
    )
    g_interp.add_argument(
        "--skip-calibration",
        action="store_true",
        default=UNSET,
        help="Skip calibration, use defaults",
    )
    g_interp.add_argument(
        "--seed",
        type=int,
        default=UNSET,
        help="RNG seed for the calibration subsampling and blocked-CV bootstrap (default: 0; "
        "same seed + same input -> reproducible outputs)",
    )
    g_interp.add_argument(
        "--calib-max-points",
        type=int,
        default=UNSET,
        help=f"Max points used for the calibration subsampling (default: {CALIB_MAX_POINTS_DEFAULT})",
    )
    g_interp.add_argument("--calibration", default=UNSET, help="Calibration JSON path (default: none)")

    g_mem = parser.add_argument_group("memory/turbo")
    g_mem.add_argument(
        "--workers",
        type=_workers_arg,
        default=UNSET,
        help=f"Number of workers (positive integer, or 'auto' = os.cpu_count()) (default: {DEFAULT_WORKERS})",
    )
    g_mem.add_argument(
        "--max-band-parallel",
        type=int,
        default=UNSET,
        help="Cap parallel row-band workers for the low-RAM shared descent (default: --workers).",
    )
    g_mem.add_argument(
        "--turbo",
        nargs="?",
        const="auto",
        default=UNSET,
        metavar="GB",
        help="Turbo mode: parallel phases sized to a RAM budget. "
        "Optional RAM cap preset 16|32|64|128 (GB, decimal 1e9 bytes); bare --turbo = auto-detect. "
        "Presets are common sizes; use --max-ram GB for any value.",
    )
    g_mem.add_argument(
        "--max-ram",
        type=_pos_float,
        default=UNSET,
        metavar="GB",
        help="Explicit RAM cap in GB (decimal, 1e9 bytes; overrides the --turbo preset). Implies --turbo.",
    )
    g_mem.add_argument("--ram-gb", type=_pos_float, default=UNSET, metavar="GB", help="Alias of --max-ram.")
    g_mem.add_argument(
        "--turbo-strict",
        action="store_true",
        default=UNSET,
        help="Shared path infeasible under the budget: error + exit 3 instead of [warn] + per-box fallback (exit 0).",
    )

    g_out = parser.add_argument_group("output")
    g_out.add_argument("-o", "--out", default=UNSET, help="Output directory (default: out/)")
    g_out.add_argument("--compress", default=UNSET, help="GeoTIFF tile compression codec (default: ZSTD)")
    g_out.add_argument(
        "--force",
        action="store_true",
        default=UNSET,
        help="Overwrite existing outputs in the out dir (default: exit 2 when value.tif / "
        "support_km.tif / calibration.json / run_summary.json are already present)",
    )
    g_out.add_argument(
        "--json",
        action="store_true",
        default=UNSET,
        help="Write a machine-readable run summary to <out>/run_summary.json "
        "(phase timings, volumetrics, grid, calibration, CLI params; stdout stays clean)",
    )
    g_out.add_argument(
        "--plan",
        action="store_true",
        help="Resolve the full run plan, print it, and exit 0 before running; creates no files or directories",
    )
    g_out.add_argument(
        "--check",
        action="store_true",
        help="After writing, reopen the outputs and verify them against the run plan "
        "(files, band readability, shape, transform, CRS, nodata tag, tiles, out-of-coverage "
        "corners); on mismatch log the exact field, keep the outputs, and exit 1 (default: off)",
    )

    g_log = parser.add_argument_group("logging")
    g_log.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress stderr below WARNING (beats --verbose and LOGGING; an explicit --log-level wins)",
    )
    g_log.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default=None,
        help="Color output: auto (TTY, no NO_COLOR/TERM=dumb), always, never. "
        "Live progress needs a TTY stderr plus --color auto/always and no --quiet.",
    )
    g_log.add_argument(
        "--verbose",
        action="store_true",
        help="Debug logging: per-phase wall times + volumetric detail (shorthand for --log-level debug)",
    )
    g_log.add_argument(
        "--log-level",
        choices=list(_LOG_LEVELS),
        default=None,
        help="Progress log level on stderr (default: info; env LOGGING; 'verbose' = debug)",
    )
    g_log.add_argument(
        "--log-file",
        metavar="PATH",
        default=None,
        help="Also write the run log to this file: created fresh each run, level info, "
        "same unified format as stderr (no color); stderr output is unchanged",
    )

    # --- config file (issue #63; single append block — keep new flags here) ---
    g_cfg = parser.add_argument_group("config")
    g_cfg.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help=f"Pipeline options file, TOML (default: discover ./{_CONFIG_FILE_NAME}, then "
        "~/.config/ppgrid/ppgrid.toml; `ppgrid init` writes a commented template; "
        "precedence: CLI flag > config > built-in default)",
    )

    # Subcommand discoverability (#61 #62 #66 #69): one-line append; the
    # subcommand parsers live in _build_subcommand_parser (dispatch) and the
    # display-only subparsers action at the end of this parser (see the
    # section note above _SUBCOMMANDS).
    parser.description += (
        " Read-only subcommands (they write no output files): `ppgrid doctor` (stack health), "
        "`ppgrid probe INPUT` (input diagnostic), `ppgrid list-crs` (CRS codes), "
        "`ppgrid compare OUT1 OUT2` (diff two run outputs), "
        "`ppgrid completions bash|zsh` (shell completions). "
        "`ppgrid doctor --help` etc. for detail."
    )

    # Display-only subcommand append (#61 #62 #66 #69): the run parser shows
    # the six subcommands in --help (identical to `ppgrid help`). The
    # parsers that actually dispatch live in _build_subcommand_parser; the
    # run form parses with this action stripped (_parse_args_cli), so a bare
    # input path stays a path and every run-path stays byte identical.
    # prog="ppgrid" pins the subcommand usage lines to `ppgrid <command>`.
    sub = parser.add_subparsers(
        dest="command", metavar="{doctor,probe,list-crs,init,compare,completions}", required=False, prog="ppgrid"
    )
    _add_subcommand_parsers(sub)
    return parser


def _parse_args_cli(parser: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    """Parse the run CLI argv against the run parser with subcommand awareness.

    The run parser carries a display-only subparsers action (so `ppgrid
    --help` shows the subcommand section); the run form itself parses with
    that action STRIPPED, so a bare input path stays a path — `ppgrid x.csv`
    and `ppgrid --cap-km 5 x.csv` parse byte identical to the pre-subcommand
    CLI, and `ppgrid help` is the run CLI's own pseudo-command (the
    subcommand parsers live in _build_subcommand_parser, dispatched by
    main() before this parser is built). `-h`/`--help` parse on the full
    parser (the action is what renders the help section; its required=False
    never rejects argv). main() token-wins subcommand tokens before this
    runs, so this parses run-form argv only.

    Args:
        parser: The run CLI parser (the `_build_parser()` return).
        argv: The run CLI argv (no leading `ppgrid`; not a subcommand form).

    Returns:
        The parsed run-CLI namespace.

    Raises:
        SystemExit: 0 on `-h`/`--help` (help printed); 2 on parse error.

    """
    if not any(a in ("-h", "--help") for a in argv):
        sub = next((a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction"), None)  # ruff: ignore[private-member-access]
        if sub is not None and argv and argv[0] not in sub.choices:
            parser._actions.remove(sub)  # ruff: ignore[private-member-access]
            try:
                return parser.parse_args(argv)
            finally:
                parser._actions.append(sub)  # ruff: ignore[private-member-access]
    return parser.parse_args(argv)


def _main_preflight(parser: argparse.ArgumentParser, args: argparse.Namespace, out: Path) -> None:
    """Preflight phase of main(): logging, input check, validation, --plan.

    Runs the whole pre-run sequence: logging setup, the missing-input
    error, the `help` pseudo-command, all parser.error flag validations,
    the out-is-a-file check, and the --plan branch (which exits 0). main()
    wraps the call so every non-zero exit here can warn about stale
    outputs (issues #40, #77). Logging is configured FIRST so every later
    event lands in the unified `L ppgrid.pipeline: msg` format (D-7).

    Args:
        parser: The argument parser (for print_help / error).
        args: Parsed namespace (mutated: max_ram alias resolution).
        out: Output directory path.

    Raises:
        SystemExit: 0 (help, plan), 1 (out is a file), 2 (validation).

    """
    _configure_logging(
        _resolve_log_level(verbose=args.verbose, log_level=args.log_level, quiet=args.quiet),
        color=_CURRENT_UI["ui"].color if _CURRENT_UI["ui"] is not None else False,
    )
    _configure_log_file(args.log_file)  # #65: second copy of the run log, level INFO
    if args.input is None:
        _ui_note("error", "missing input file (CSV or Parquet)")
        _ui_note(
            "note",
            "hint: run 'ppgrid help' for usage and examples",
        )
        raise SystemExit(2)
    if args.input == "help" and not Path("help").exists():
        # A literal file named "help" wins over the pseudo-command (use --help).
        parser.print_help()
        raise SystemExit(0)

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
        parser.error(
            f"--turbo preset must be auto|16|32|64|128: {args.turbo} "
            "(presets are common sizes; use --max-ram GB for any value)"
        )
    if args.max_ram is None:
        args.max_ram = args.ram_gb
    if args.ram_gb is not None and args.max_ram != args.ram_gb:
        parser.error("--max-ram and --ram-gb are aliases and must match")
    if args.turbo_strict and args.turbo is None and args.max_ram is None:
        parser.error("--turbo-strict requires --turbo or --max-ram")
    if args.verbose:
        # One-line environment report (issue #101): versions, RAM detection
        # source + resolved cap, C-interp status. Must never break the run.
        if args.max_ram is not None:
            preset = args.max_ram
        elif args.turbo not in (None, "auto"):
            preset = float(args.turbo)
        else:
            preset = None
        _ui_note("note", _env_report_line(preset))

    if out.exists() and not out.is_dir():
        _die(f"output path is a file, not a directory: {args.out}")

    if args.plan:
        # #53: resolve the full plan (ingest -> calibrate -> grid -> turbo
        # precheck) in memory, print it, and exit 0. dry_run keeps calibrate
        # and grid from writing files, so nothing is created.
        # #57: column preflight first (same error as the run path).
        _check_columns(args.input, args.value_col, args.lng_col, args.lat_col)
        p = _map_pipeline_errors(lambda: _make_pipeline(args, dry_run=True, ui=_CURRENT_UI["ui"]), args.input)
        _map_pipeline_errors(p.plan, args.input)
        plan_ui = _CURRENT_UI["ui"]
        if args.json:
            # #105: machine-readable plan. stdout is exactly one JSON object
            # (schema_version 1) and nothing else; the human printer is
            # skipped. stderr (phase logs, the strict precheck line) keeps
            # the run --json contract.
            payload = json.dumps(_sanitize_json(p.plan_data(args)), indent=2, allow_nan=False) + "\n"
            sys.stdout.write(payload)
            sys.stdout.flush()
        elif plan_ui is not None:
            # Concept X2: live mode boxes only the key/value block (the header
            # line, env report and progress rows stay unboxed); plain mode is
            # byte-identical to the pre-#102 print.
            header, _, body = p.plan_text(args).partition("\n")
            plan_ui.print_plan(header, body.split("\n"))
        else:
            print(p.plan_text(args))  # ruff: ignore[print]
        raise SystemExit(0)


# ---------------------------------------------------------------------------
# Subcommands: read-only diagnostics (doctor #61, probe #62, list-crs #69)
# plus the verify surface (compare #67, completions #66) and init (#63,
# writes the ppgrid.toml template).
#
# Dispatched in main() on the literal first token (token-wins) BEFORE the
# run parser is built: a data file named like a subcommand runs as
# `ppgrid ./<name>`. They are a separate top-level parser, not subparsers
# of the run parser: the run CLI's optional `input` positional (nargs="?")
# sits before any subparser positional in the same parser and would swallow
# the command token ("ppgrid doctor" -> input="doctor", "invalid choice").
# The run parser keeps a display-only subparsers action so `ppgrid --help`
# shows the subcommand section (identical to `ppgrid help`); the run form
# parses with it stripped (_parse_args_cli), leaving every run-path byte
# identical. Read-only subcommands write no output files (doctor's oracle
# probe writes one throwaway tile under the temp dir; init writes the
# template). Their reports go to stdout like --plan; unexpected handler
# errors are one named line on stderr + exit 1, like the run CLI. Future
# subcommands append here.
# ---------------------------------------------------------------------------

_SUBCOMMANDS: tuple[str, ...] = ("doctor", "probe", "list-crs", "init", "compare", "completions")


_DOCTOR_MARK: dict[str, str] = {"pass": "[ok]", "warn": "[warn]", "fail": "[fail]"}

# Curated ppgrid CRS set for `ppgrid list-crs` (issue #69):
# (code, role, one-line note). Names are resolved at runtime from the
# pyproj database (they vary by PROJ version); the notes are the ppgrid
# guidance. UTM examples: 326xx = northern hemisphere, 327xx = southern.
_CURATED_CRS: tuple[tuple[int, str, str], ...] = (
    (4326, "input default (--src-crs)", "WGS 84 lon/lat in degrees; probe assumes this for lon/lat-looking columns"),
    (3577, "example --work-crs (Australia)", "GDA94 / Australian Albers; national equal-area work CRS (AU scale)"),
    (6933, "working CRS default (--work-crs)", "WGS 84 / NSIDC EASE-Grid 2.0 Global (Wagner VII); global equal-area"),
    (3857, "output CRS default (--out-crs)", "WGS 84 / Pseudo-Mercator (Web Mercator); the final GeoTIFF CRS"),
    (32633, "UTM example (US West)", "WGS 84 / UTM zone 33N; 120-114 W (California)"),
    (32755, "UTM example (Melbourne)", "WGS 84 / UTM zone 55S; 150-144 E (Victoria)"),
    (32756, "UTM example (Sydney)", "WGS 84 / UTM zone 56S; 156-150 E (New South Wales)"),
)

# probe limits: the candidate sample (first N rows, all columns) and the
# streaming row-count / validation chunk size (one chunk held in memory).
_PROBE_SAMPLE: int = 100_000
_PROBE_CHUNK: int = 1_000_000
_LON_NAMES: frozenset[str] = frozenset({"lon", "lng", "long", "longitude"})
_LAT_NAMES: frozenset[str] = frozenset({"lat", "latitude"})


def _add_subcommand_parsers(sub: argparse._SubParsersAction) -> None:
    """Add the six subcommand parsers to a subparsers action (#61 #62 #63 #66 #67 #69).

    Single definition of the subcommand surface: the dispatch parser
    (_build_subcommand_parser) and the run parser's display-only append
    (_build_parser) both call this, so the two cannot drift.

    Args:
        sub: The subparsers action to add the subcommand parsers to.

    """
    sub.add_parser(
        "doctor",
        help="runtime stack health report",
        description=(
            "Runtime stack health report (issue #61): python/ppgrid/numpy/"
            "rasterio/GDAL/pyproj versions, the zstd turbo oracle status, the C "
            "interp kernel state, RAM, temp dir. Exit 0 = all pass, 1 = failing "
            "checks named on stderr. Writes one throwaway oracle tile under the "
            "temp dir; no output files."
        ),
    )

    probe = sub.add_parser(
        "probe",
        help="input diagnostic (format, columns, rows, CRS, projection)",
        description=(
            "Input diagnostic (issue #62): detects the format (incl. compressed "
            "CSV magic), reports columns, the streamed row count, numeric "
            "--value-col candidates and a CRS guess, then runs the SAME "
            "projection validation the pipeline runs (non-finite / out-of-"
            "domain / non-finite-projection drops, kept-point domain) for the "
            "given work CRS, reporting what a real run would do. Writes no "
            "output files. Exit 0 = clean, 1 = problem(s) found, 2 = usage "
            "error (unknown EPSG code)."
        ),
    )
    probe.add_argument("input", help="CSV or Parquet input path")
    probe.add_argument(
        "--work-crs",
        type=_crs_arg,
        default=WORK_CRS,
        help=f"Work CRS to validate the projection against, bare int or 'EPSG:<code>' (default: {WORK_CRS})",
    )
    probe.add_argument(
        "--value-col",
        default=None,
        help="Value column to validate (default: the run default 'value'; numeric candidates are reported)",
    )

    list_crs = sub.add_parser(
        "list-crs",
        help="list the supported and known CRS codes",
        description=(
            "List CRS codes (issue #69): the curated ppgrid set by default "
            "(input/working/output defaults plus UTM examples), the full "
            "pyproj/EPSG CRS database with --all. pyproj loads after argument "
            "parsing, so `list-crs --help` stays fast."
        ),
    )
    list_crs.add_argument(
        "--all",
        action="store_true",
        help="List the full pyproj/EPSG CRS database (default: the curated ppgrid set)",
    )

    p_init = sub.add_parser(
        "init",
        help="write a fully commented ppgrid.toml template",
        description=(
            "Write a fully commented ppgrid.toml template covering every "
            "pipeline option, each set to its built-in default (the file as "
            "written changes nothing about a run). Refuses to clobber an "
            "existing file without --force."
        ),
    )
    p_init.add_argument("dir", nargs="?", default=".", help="Directory to write the template into (default: .)")
    p_init.add_argument("--force", action="store_true", help="Overwrite an existing ppgrid.toml (default: exit 2)")

    c = sub.add_parser(
        "compare",
        help="Compare two ppgrid outputs (out dirs or GeoTIFFs); exit 0 identical / 1 different / 2 usage",
        description=(
            "Compare two ppgrid outputs, band by band.\n\n"
            "Two output dirs are compared as the (value, support_km) band pair; two "
            "GeoTIFFs as the given pair. The report covers structure (shape, "
            "transform, CRS), per-cell exact-match counts, the max abs DN difference, "
            "and the mismatch bounding box. The report goes to stdout (progress stays "
            "on stderr).\n\n"
            "Exit codes: 0 identical (or within --max-diff), 1 different, "
            "2 usage error (missing files, mixed dir/file, unreadable raster)."
        ),
    )
    c.add_argument("OUT1", help="First ppgrid output dir (value.tif + support_km.tif) or value GeoTIFF")
    c.add_argument("OUT2", help="Second ppgrid output dir or value GeoTIFF (same form as OUT1)")
    c.add_argument(
        "--max-diff",
        type=_nonneg_int,
        default=0,
        metavar="N",
        help="Tolerated max abs DN difference (a band counts as identical up to this) (default: 0)",
    )
    z = sub.add_parser(
        "completions",
        help="Print shell completions (bash or zsh), generated from the live parser",
        description=(
            "Print shell completions to stdout, generated from the live argparse "
            "parser of this ppgrid version (no external completion framework, no "
            "timestamps: the output is deterministic for a given version). Install "
            'with eval "$(ppgrid completions bash)" or eval "$(ppgrid completions zsh)" '
            "(or append to your shell rc file)."
        ),
    )
    z.add_argument("shell", choices=["bash", "zsh"], help="Target shell")


def _build_subcommand_parser() -> argparse.ArgumentParser:
    """Build the subcommand parser (diagnostics #61 #62 #69 + init #63 + verify #66 #67).

    A separate top-level parser, dispatched on the literal first token
    in main() (token-wins) before the run parser is built (see the
    section note above): the run CLI's optional `input` positional would
    otherwise swallow the command token before a subparser could
    validate it. The top parser is add_help=False (the run CLI owns
    `ppgrid --help`); each subparser keeps its own -h. The run parser
    mirrors this surface display-only for --help (see _build_parser).
    Future subcommands append to _add_subcommand_parsers.

    Returns:
        The parser; `parse_args(argv)` with argv[0] a subcommand name.

    """
    parser = argparse.ArgumentParser(
        prog="ppgrid",
        add_help=False,
        description="ppgrid subcommands. The default form is the run CLI: ppgrid INPUT [flags].",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True, title="subcommands")
    _add_subcommand_parsers(sub)
    return parser


def _run_subcommand(argv: list[str]) -> None:
    """Dispatch argv[0] in _SUBCOMMANDS to its handler (see main()).

    Handler exit codes become the SystemExit code (0/1/2 per the
    subcommand contract); init is the exception — it returns on success
    (a plain return from main()) and _cmd_init raises its own
    SystemExit 1 (IO) / 2 (usage). argparse usage errors exit 2 on
    their own; any other unexpected handler exception — diag, compare
    or completions — is one named error line + exit 1, not a traceback
    (the run CLI's _run_cli contract).

    Args:
        argv: The full subcommand argv, argv[0] = the subcommand name.

    Raises:
        SystemExit: the subcommand's exit code (0/1/2); init success returns.

    """
    parser = _build_subcommand_parser()
    args = parser.parse_args(argv)
    if args.command == "init":
        # init (#63): the handler owns its exit contract — success is a
        # plain return from main(); it raises SystemExit 1 (IO) /
        # 2 (usage) itself. Its tests pin that success form.
        _cmd_init(args)
        return
    try:
        if args.command == "doctor":
            code = _cmd_doctor()
        elif args.command == "probe":
            code = _cmd_probe(args)
        elif args.command == "list-crs":
            code = _cmd_list_crs(args)
        elif args.command == "compare":
            code = _compare_outputs(args.OUT1, args.OUT2, args.max_diff)
        else:  # completions
            # Generated from the live RUN parser (display subparsers
            # included) so the scripts cover the full CLI surface, not
            # just the subcommands.
            code = _completions_print(args.shell, _build_parser())
    except Exception as e:  # ruff: ignore[blind-except] — named error, no traceback
        print(f"error: {args.command}: {type(e).__name__}: {e}", file=sys.stderr)  # ruff: ignore[print]
        code = 1
    raise SystemExit(code)


# ---------------------------------------------------------------------------
# compare (#67)
# ---------------------------------------------------------------------------


def _compare_resolve(out1: str, out2: str) -> list[tuple[str, Path, Path]]:
    """Map two compare arguments to (label, path_a, path_b) band pairs (#67).

    Both dirs -> the (value, support_km) band pair; both files -> the given
    pair (label = the filenames). Mixed dir/file, or a dir missing either
    final raster, is a usage error (exit 2).

    Args:
        out1: First argument (output dir or GeoTIFF).
        out2: Second argument (same form as out1).

    Returns:
        The band pair list (two entries for dirs, one for files).

    Raises:
        ValueError: Usage error (message is printed, exit 2).

    """
    a, b = Path(out1), Path(out2)
    if not a.exists():
        msg = f"not found: {out1}"
        raise ValueError(msg)
    if not b.exists():
        msg = f"not found: {out2}"
        raise ValueError(msg)
    if a.is_dir() != b.is_dir():
        msg = "cannot mix directory and file arguments (pass two output dirs or two GeoTIFFs)"
        raise ValueError(msg)
    if a.is_dir():
        pairs = [
            ("value", a / "value.tif", b / "value.tif"),
            ("support_km", a / "support_km.tif", b / "support_km.tif"),
        ]
        missing = [f.name for _, pa, pb in pairs for f in (pa, pb) if not f.is_file()]
        if missing:
            msg = f"not a ppgrid output dir (missing: {', '.join(missing)})"
            raise ValueError(msg)
        return pairs
    return [(a.name, a, b)]


def _compare_band(label: str, pa: Path, pb: Path, max_diff: int) -> tuple[list[str], bool, int]:
    """Compare one band pair: structure + per-cell diff report (#67).

    Identical = same shape, same transform, same CRS, and max abs DN diff
    <= max_diff. The pixel sweep is row-chunked (1024 rows) so the temp
    arrays stay small even for global rasters.

    Args:
        label: Band label for the report (value / support_km / filename).
        pa: First GeoTIFF.
        pb: Second GeoTIFF.
        max_diff: Tolerated max abs DN difference.

    Returns:
        (report lines, identical, mismatch cell count).

    Raises:
        OSError: Unreadable raster (mapped to exit 2 by the caller).

    """
    with rasterio.open(pa) as da, rasterio.open(pb) as db:
        lines = [f"compare: {label} ({pa} vs {pb})"]
        shape_ok = da.width == db.width and da.height == db.height
        transform_ok = da.transform == db.transform
        crs_a = da.crs.to_string() if da.crs is not None else "None"
        crs_b = db.crs.to_string() if db.crs is not None else "None"
        crs_ok = crs_a == crs_b
        lines.extend(
            [
                f"  shape:     {da.height}x{da.width} {'=' if shape_ok else '!='} {db.height}x{db.width}",
                f"  transform: {'equal' if transform_ok else 'differ'}",
                f"  crs:       {crs_a} {'=' if crs_ok else '!='} {crs_b}",
            ]
        )
        if not shape_ok:
            lines.extend(["  cells:     n/a (shape mismatch)", "  max diff:  n/a (shape mismatch)"])
            return lines, False, -1
        exact = 0
        mismatch = 0
        max_d = 0
        min_row = min_col = -1
        max_row = max_col = -1
        h, w = da.height, da.width
        for r0 in range(0, h, 1024):
            r1 = min(r0 + 1024, h)
            win = Window(0, r0, w, r1 - r0)
            d = da.read(1, window=win).astype(np.int32) - db.read(1, window=win).astype(np.int32)
            n_neq = int((d != 0).sum())
            exact += d.size - n_neq
            mismatch += n_neq
            if n_neq:
                max_d = max(max_d, int(np.abs(d).max()))
                rr, cc = np.nonzero(d)
                rmin, rmax = int(rr.min()) + r0, int(rr.max()) + r0
                cmin, cmax = int(cc.min()), int(cc.max())
                min_row = rmin if min_row < 0 else min(min_row, rmin)
                max_row = rmax if max_row < 0 else max(max_row, rmax)
                min_col = cmin if min_col < 0 else min(min_col, cmin)
                max_col = cmax if max_col < 0 else max(max_col, cmax)
        lines.extend([f"  cells:     {h * w} total, {exact} exact, {mismatch} mismatch", f"  max diff:  {max_d}"])
        if mismatch:
            bbox = f"rows {min_row}..{max_row}, cols {min_col}..{max_col}"
            if transform_ok:
                mwin = Window(min_col, min_row, max_col - min_col + 1, max_row - min_row + 1)
                left, bottom, right, top = rasterio.windows.bounds(mwin, da.transform)
                bbox += f"; georef ({crs_a}): left={left:.3f} bottom={bottom:.3f} right={right:.3f} top={top:.3f}"
            lines.append(f"  mismatch bbox: {bbox}")
        return lines, shape_ok and transform_ok and crs_ok and max_d <= max_diff, mismatch


def _compare_outputs(out1: str, out2: str, max_diff: int) -> int:
    """Run `ppgrid compare OUT1 OUT2` (#67); returns the exit code.

    0 = identical (or within --max-diff), 1 = different, 2 = usage error.
    The report goes to stdout (like --plan); usage errors go to stderr.

    Args:
        out1: First output dir or GeoTIFF.
        out2: Second output dir or GeoTIFF (same form as out1).
        max_diff: Tolerated max abs DN difference.

    Returns:
        The exit code (0/1/2).

    """
    _load_native()
    try:
        pairs = _compare_resolve(out1, out2)
    except ValueError as e:
        print(f"compare: {e}", file=sys.stderr)  # ruff: ignore[print]
        return 2
    lines: list[str] = []
    n_bad = 0
    total_mismatch = 0
    for label, pa, pb in pairs:
        try:
            band_lines, ok, mismatch = _compare_band(label, pa, pb, max_diff)
        except OSError as e:
            print(f"compare: read error ({pa} vs {pb}): {e}", file=sys.stderr)  # ruff: ignore[print]
            return 2
        lines.extend(band_lines)
        n_bad += 0 if ok else 1
        total_mismatch += max(mismatch, 0)
    for line in lines:
        print(line)  # ruff: ignore[print]
    if n_bad:
        print(f"compare: different ({n_bad} of {len(pairs)} bands)")  # ruff: ignore[print]
        return 1
    if total_mismatch:
        print(f"compare: within --max-diff ({total_mismatch} cell(s) differ, all <= {max_diff})")  # ruff: ignore[print]
    else:
        print("compare: identical")  # ruff: ignore[print]
    return 0


# ---------------------------------------------------------------------------
# completions (#66)
# ---------------------------------------------------------------------------


def _completions_collect(parser: argparse.ArgumentParser) -> dict[str, Any]:
    """Collect completion data from a live argparse parser, recursively (#66).

    Deterministic: options sorted by alias, subcommands sorted by name,
    choice words sorted. Positionals keep definition order.

    Args:
        parser: A parser from _build_parser (or one of its subparsers).

    Returns:
        {"options": [(alias, help), ...],
         "positionals": [word, ...],
         "subcommands": [(name, help, sub-data), ...]}

    """
    options: list[tuple[str, str]] = []
    positionals: list[str] = []
    subcommands: list[tuple[str, str, dict[str, Any]]] = []
    actions = parser._actions  # ruff: ignore[private-member-access]
    for action in actions:
        if action.option_strings:
            help_text = " ".join((action.help or "").split())
            options.extend((alias, help_text) for alias in sorted(action.option_strings))
        elif isinstance(action, argparse._SubParsersAction):  # ruff: ignore[private-member-access]
            for name in sorted(action.choices):
                sub = action.choices[name]
                desc = " ".join((sub.description or "").split())
                subcommands.append((name, desc, _completions_collect(sub)))
        elif action.choices:
            positionals.extend(sorted(action.choices))
        else:
            positionals.append(action.metavar or action.dest)
    return {"options": options, "positionals": positionals, "subcommands": subcommands}


def _completions_bash(items: dict[str, Any]) -> str:
    """Render the bash completion script from a collect() tree (#66)."""

    def menu(it: dict[str, Any]) -> str:
        words = [alias for alias, _ in it["options"]]
        words.extend(it["positionals"])
        words.extend(name for name, _, _ in it["subcommands"])
        return " ".join(sorted(words))

    cases: list[tuple[str, str]] = [(name, menu(sub)) for name, _, sub in items["subcommands"]]
    cases.append(("*", menu(items)))
    parts = [
        "# ppgrid bash completions (generated: `ppgrid completions bash`)",
        '# Install: eval "$(ppgrid completions bash)"  (or append to ~/.bashrc)',
        "#",
        "# Deterministic: sorted words, no timestamps; regenerate after a ppgrid upgrade.",
        "",
        "_ppgrid_completions() {",
        "    local cur cmd",
        '    cur="${COMP_WORDS[COMP_CWORD]}"',
        '    cmd="${COMP_WORDS[1]:-}"',
        '    case "$cmd" in',
    ]
    for name, words in cases:
        parts.extend(
            [
                f"        {name})",
                f'            COMPREPLY=( $(compgen -W "{words}" -- "$cur") )',
                "            return 0",
                "            ;;",
            ]
        )
    parts.extend(
        [
            "    esac",
            "    return 0",
            "}",
            "complete -o default -F _ppgrid_completions ppgrid",
            "",
        ]
    )
    return "\n".join(parts)


def _zsh_word(text: str) -> str:
    """Quote a word for a zsh single-quoted string (escape embedded quotes)."""
    return "'" + text.replace("'", "'\\''") + "'"


def _completions_zsh(items: dict[str, Any]) -> str:
    """Render the zsh completion script from a collect() tree (#66)."""

    def menu(it: dict[str, Any]) -> list[str]:
        words: list[str] = []
        for alias, help_text in it["options"]:
            desc = f":{help_text}" if help_text else ""
            words.append(_zsh_word(alias + desc))
        words.extend(_zsh_word(w) for w in it["positionals"])
        for name, desc, _ in it["subcommands"]:
            d = f":{desc}" if desc else ""
            words.append(_zsh_word(name + d))
        return words

    cases: list[tuple[str, list[str]]] = [(name, menu(sub)) for name, _, sub in items["subcommands"]]
    cases.append(("*", menu(items)))
    parts = [
        "#compdef ppgrid",
        "# ppgrid zsh completions (generated: `ppgrid completions zsh`)",
        '# Install: eval "$(ppgrid completions zsh)"  (or save and source from .zshrc)',
        "#",
        "# Deterministic: sorted words, no timestamps; regenerate after a ppgrid upgrade.",
        "",
        "_ppgrid() {",
        "    local -a opts",
        '    case "${words[2]:-}" in',
    ]
    for name, words in cases:
        parts.extend([f"        {name})", "            opts=("])
        parts.extend(f"                {w}" for w in words)
        parts.extend(["            )", "            ;;"])
    parts.extend(
        [
            "    esac",
            "    _describe 'ppgrid' opts",
            "}",
            "",
            '_ppgrid "$@"',
            "",
        ]
    )
    return "\n".join(parts)


def _completions_print(shell: str, parser: argparse.ArgumentParser) -> int:
    """Print `ppgrid completions <shell>` to stdout; returns the exit code (#66).

    Generated from the live parser of this ppgrid version (no external
    completion framework); deterministic for a given version.
    """
    items = _completions_collect(parser)
    text = _completions_bash(items) if shell == "bash" else _completions_zsh(items)
    sys.stdout.write(text)
    sys.stdout.flush()
    return 0


# ---------------------------------------------------------------------------
# post-write verify (--check, #68)
# ---------------------------------------------------------------------------

_CHECK_WIN_PX = 16  # --check corner-window side
_CHECK_PT_CHUNK = 50_000  # point-chunk size for the coverage sweep


def _corner_windows(w: int, h: int, size: int = _CHECK_WIN_PX) -> tuple[tuple[str, Window], ...]:
    """Return the four corner windows of a raster, clamped to its size (#68)."""
    s = min(size, w, h)
    return (
        ("top-left", Window(0, 0, s, s)),
        ("top-right", Window(w - s, 0, s, s)),
        ("bottom-left", Window(0, h - s, s, s)),
        ("bottom-right", Window(w - s, h - s, s, s)),
    )


def _window_out_of_coverage(
    win: Window,
    ds: rasterio.DatasetBase,
    out_crs: str,
    work_crs: str,
    p: Pipeline,
    cap_cells: int,
) -> np.ndarray:
    """Per-pixel out-of-coverage mask for one window, window-shaped (#68).

    Coverage = the fill's own definition: a cell is covered when at least one
    input point falls in its (2*cap_cells+1)^2 cell window. The near mask the
    quantizer applies is exactly this box count (pullpush.box_count: "the
    exact answer to 'is there data within cap of this cell?'"), so an
    uncovered cell never receives a fill value and can only hold NODATA: a 0
    there is a zero-filled buffer, not a decoded value.

    Pixel centers in the raster CRS are transformed to the work CRS (true
    metres, equal-area) and mapped to cell indices with the same
    (x - x0) // res truncation the point binning uses, so the mask
    reproduces the fill's per-cell decision. The sweep is point-chunked
    (temp arrays at ~n_px x 50k int32; a pixel drops out as soon as a chunk
    proves it covered).

    Args:
        win: The window (raster pixel coords).
        ds: Open raster (transform + CRS).
        out_crs: The raster CRS (EPSG string).
        work_crs: The point CRS (EPSG string).
        p: The Pipeline (work-CRS points + grid origin/res).
        cap_cells: The fill cap in cells (same round() as the pipeline).

    Returns:
        bool array (win.height, win.width): True = out of coverage.

    """
    tf = ds.transform
    cols = win.col_off + np.arange(win.width) + 0.5
    rows = win.row_off + np.arange(win.height) + 0.5
    xx, yy = np.meshgrid(cols, rows)
    tx = tf.a * xx + tf.b * yy + tf.c
    ty = tf.d * xx + tf.e * yy + tf.f
    if out_crs != work_crs:
        inv = Transformer.from_crs(out_crs, work_crs, always_xy=True)
        tx, ty = inv.transform(tx.ravel(), ty.ravel())
    else:
        tx, ty = tx.ravel(), ty.ravel()
    tx = np.asarray(tx, dtype=np.float64)
    ty = np.asarray(ty, dtype=np.float64)
    n = tx.size
    px = ((tx - p.x0) // p.res).astype(np.int32)
    py = ((ty - p.y0) // p.res).astype(np.int32)
    sx_all = ((p.x - p.x0) // p.res).astype(np.int32)
    sy_all = ((p.y - p.y0) // p.res).astype(np.int32)
    best = np.full(n, np.iinfo(np.int32).max, dtype=np.int32)
    for s in range(0, sx_all.size, _CHECK_PT_CHUNK):
        sx = sx_all[s : s + _CHECK_PT_CHUNK]
        sy = sy_all[s : s + _CHECK_PT_CHUNK]
        d = np.maximum(np.abs(px[:, None] - sx[None, :]), np.abs(py[:, None] - sy[None, :]))
        best = np.minimum(best, d.min(axis=1))
        if (best <= cap_cells).all():
            break
    return (best > cap_cells).reshape(win.height, win.width)


def _post_write_check(p: Pipeline, out: Path) -> None:
    """Reopen the published outputs and verify them against the run plan (#68).

    Per final raster (value.tif, support_km.tif):
      - the file exists;
      - band 1 is readable (center window reads back; count 1, int16,
        512x512 tiled);
      - shape, transform and CRS match the plan (the exact geometry the
        write phase used: the work grid, or calculate_default_transform
        for a reprojected output);
      - the nodata tag is NODATA;
      - every pixel of the four 16x16 corner windows that has no input
        point within its fill-cap window (the fill's own box-count near
        mask) holds NODATA, not 0 (a zero-filled buffer instead of a
        NoData init shows up there first).

    Args:
        p: The Pipeline that just ran (plan geometry + work-CRS points).
        out: The run's output directory.

    Raises:
        SystemExit: 1 when any check fails — the exact field is logged and
            the outputs are kept (verify, don't repair).

    """
    _load_native()
    problems: list[str] = []
    work_crs = f"EPSG:{p.work_crs}"
    out_crs = f"EPSG:{p.out_crs}"
    if p.out_crs != p.work_crs:
        # Same call the write phase made (single source of truth).
        exp_tf, exp_w, exp_h = rasterio.warp.calculate_default_transform(
            work_crs,
            out_crs,
            p.nx,
            p.ny,
            p.x0,
            p.y0,
            p.x0 + p.nx * p.res,
            p.y0 + p.ny * p.res,
        )
    else:
        exp_w, exp_h, exp_tf = p.nx, p.ny, from_origin(p.x0, p.y0 + p.ny * p.res, p.res, p.res)
    exp_crs = CRS.from_user_input(out_crs)
    cap_cells = round(p.cap_km_val * M_PER_KM / p.res)
    for name in ("value.tif", "support_km.tif"):
        path = out / name
        if not path.is_file():
            problems.append(f"{name}: file missing (expected {path})")
            continue
        try:
            with rasterio.open(path) as ds:
                if (ds.height, ds.width) != (exp_h, exp_w):
                    problems.append(f"{name}: shape {ds.height}x{ds.width} != planned {exp_h}x{exp_w}")
                if ds.transform != exp_tf:
                    problems.append(f"{name}: transform {tuple(ds.transform)} != planned {tuple(exp_tf)}")
                got_crs = ds.crs.to_string() if ds.crs is not None else "None"
                if ds.crs is None or ds.crs != exp_crs:
                    problems.append(f"{name}: crs {got_crs} != planned {out_crs}")
                if ds.count != 1:
                    problems.append(f"{name}: band count {ds.count} != 1")
                    continue
                if ds.dtypes[0] != "int16":
                    problems.append(f"{name}: dtype {ds.dtypes[0]} != int16")
                bsize = (ds.profile.get("blockxsize", 1), ds.profile.get("blockysize", 1))
                tiled = bool(ds.profile.get("tiled", False)) and all(b > 1 for b in bsize)
                if not tiled or bsize != (TILE_PX, TILE_PX):
                    problems.append(f"{name}: tiles {bsize} (tiled={tiled}) != planned {TILE_PX}x{TILE_PX} tiled")
                nd = ds.nodata
                if nd is None or int(nd) != NODATA:
                    problems.append(f"{name}: nodata tag {nd!r} != {NODATA}")
                s = min(_CHECK_WIN_PX, ds.width, ds.height)
                cwin = Window(ds.width // 2 - s // 2, ds.height // 2 - s // 2, s, s)
                try:
                    ds.read(1, window=cwin)
                except OSError as e:
                    problems.append(f"{name}: center window unreadable: {e}")
                    continue
                for corner, win in _corner_windows(ds.width, ds.height):
                    arr = ds.read(1, window=win)
                    ooc = _window_out_of_coverage(win, ds, out_crs, work_crs, p, cap_cells)
                    zeros = int(np.count_nonzero((arr == 0) & ooc))
                    if zeros:
                        problems.append(
                            f"{name}: {corner} corner (rows {win.row_off}..{win.row_off + win.height - 1}, "
                            f"cols {win.col_off}..{win.col_off + win.width - 1}): {zeros} zero pixel(s) "
                            f"with no input point within the {p.cap_km_val:g} km fill-cap window "
                            f"(expected NODATA {NODATA})"
                        )
        except OSError as e:
            problems.append(f"{name}: cannot open: {e}")
    if problems:
        for line in problems:
            _ui_note("error", f"check: {line}")
        _ui_note("error", f"check failed: {len(problems)} mismatch(es); outputs kept in {out}")
        raise SystemExit(1)
    _ui_note(
        "ok",
        "check: value.tif + support_km.tif verified against the run plan "
        "(shape, transform, CRS, nodata tag, tiles, corners)",
    )


# ---------------------------------------------------------------------------
# doctor (issue #61): runtime stack health report.
# Each check is a small named function returning (label, status, detail);
# tests monkeypatch one to force a failure. status: pass / warn / fail —
# only 'fail' flips the exit code (warn = degraded but correct behaviour,
# e.g. the np.interp fallback and the zstd oracle mismatch are byte-identical
# paths, so they warn, not fail).
# ---------------------------------------------------------------------------


def _doc_python() -> tuple[str, str, str]:
    """Check: python version (fail below 3.11, the requires-python floor)."""
    version = platform.python_version()
    ok = sys.version_info >= (3, 11)
    return ("python", "pass" if ok else "fail", version if ok else f"{version} (3.11+ required)")


def _doc_ppgrid() -> tuple[str, str, str]:
    """Check: installed ppgrid version (module-level; cheap, always present)."""
    return ("ppgrid", "pass", __version__)


def _doc_numpy() -> tuple[str, str, str]:
    """Check: numpy version (module-level import; reaching here means it loaded)."""
    return ("numpy", "pass", np.__version__)


def _doc_rasterio() -> tuple[str, str, str]:
    """Check: rasterio import (local import; doctor is a diagnostic, not hot)."""
    try:
        import rasterio
    except Exception as e:  # ruff: ignore[blind-except] — any import failure is the report
        return ("rasterio", "fail", f"import failed: {type(e).__name__}: {e}")
    return ("rasterio", "pass", rasterio.__version__)


def _doc_gdal() -> tuple[str, str, str]:
    """Check: GDAL version via rasterio.gdal_version (warn if unreadable)."""
    try:
        import rasterio

        version = rasterio.gdal_version()
    except Exception as e:  # ruff: ignore[blind-except]
        return ("gdal", "warn", f"version unreadable: {type(e).__name__}: {e}")
    return ("gdal", "pass", version)


def _doc_pyproj() -> tuple[str, str, str]:
    """Check: pyproj import + PROJ version string."""
    try:
        import pyproj
    except Exception as e:  # ruff: ignore[blind-except]
        return ("pyproj", "fail", f"import failed: {type(e).__name__}: {e}")
    proj = getattr(pyproj, "proj_version_str", None)
    detail = pyproj.__version__ + (f" (PROJ {proj})" if proj else "")
    return ("pyproj", "pass", detail)


def _doc_zstdmt() -> tuple[str, str, str]:
    """Check: CPL zstd availability + the pipeline's A.7 byte-exact oracle.

    Writes one throwaway _ORACLE_PROBE_W x TILE_PX ZSTD tile under the temp
    dir — the same shape the pipeline's turbo precheck probe
    (Pipeline._turbo_zstd_ok) uses — and runs zstdmt.oracle_check on it:
    ok -> the turbo parallel ZSTD write path is live on this stack;
    unavailable / mismatch -> the stock-codec parallel path (warn, not
    fail: bytes are identical).
    """
    if not zstdmt.available():
        return ("zstdmt", "warn", "CPL zstd unavailable; turbo writes use the stock-codec parallel path")
    tmp = Path(tempfile.mkdtemp(prefix="ppgrid-doctor-"))
    try:
        import rasterio
        from rasterio.transform import from_origin

        probe = tmp / "_zstd_oracle.tif"
        tile = _oracle_probe_tile()
        with rasterio.open(
            probe,
            "w",
            driver="GTiff",
            dtype="int16",
            width=_ORACLE_PROBE_W,
            height=TILE_PX,
            count=1,
            nodata=NODATA,
            crs=f"EPSG:{WORK_CRS}",
            transform=from_origin(0, TILE_PX * DEFAULT_RES, DEFAULT_RES, DEFAULT_RES),
            compress="ZSTD",
            tiled=True,
            blockxsize=TILE_PX,
            blockysize=TILE_PX,
            predictor=2,
            BIGTIFF="IF_SAFER",
        ) as dst:
            dst.write(tile, 1)
        verdict = zstdmt.oracle_check(probe, tile=0)
    except Exception as e:  # ruff: ignore[blind-except]
        return (
            "zstdmt",
            "warn",
            f"oracle probe failed: {type(e).__name__}: {e}; turbo falls back to the stock-codec parallel path",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if verdict.ok:
        return ("zstdmt", "pass", "CPL zstd available; oracle byte-exact; turbo parallel ZSTD write is live")
    return (
        "zstdmt",
        "warn",
        (
            f"oracle mismatch on this stack ({verdict.detail}); turbo uses the stock-codec parallel path; "
            "output bytes identical"
        ),
    )


def _doc_cinterp() -> tuple[str, str, str]:
    """Check: C interp kernel state (warn on the bit-identical np.interp fallback)."""
    if _cinterp.available():
        return ("cinterp", "pass", "C kernel active (bit-identical to np.interp)")
    return ("cinterp", "warn", f"np.interp fallback ({_cinterp.reason() or 'unknown'}); output bytes unchanged")


def _doc_ram() -> tuple[str, str, str]:
    """Check: physical RAM + cgroup ceiling (the pipeline's own helpers)."""
    try:
        physical: float | None = turbop._read_physical_gb()  # ruff: ignore[private-member-access]
    except (ValueError, OSError):
        physical = None
    try:
        cgroup: float = turbop._read_cgroup_max_gb()  # ruff: ignore[private-member-access]
    except (ValueError, OSError):
        cgroup = float("inf")
    parts: list[str] = []
    if physical is None:
        parts.append("physical unknown")
    else:
        parts.append(f"physical {physical:.2f} GB")
    if math.isinf(cgroup):
        parts.append("cgroup unlimited")
    else:
        parts.append(f"cgroup {cgroup:.2f} GB")
    finite = [p for p in (physical, cgroup) if p is not None and math.isfinite(p)]
    if finite:
        parts.append(f"effective cap {min(finite):.2f} GB")
    return ("ram", "pass" if finite else "warn", "; ".join(parts))


def _doc_tempdir() -> tuple[str, str, str]:
    """Check: the process temp dir is writable (staging, memmaps, scratch live there)."""
    d = tempfile.gettempdir()
    try:
        fd, name = tempfile.mkstemp(prefix="ppgrid-doctor-", dir=d)
        os.close(fd)
        Path(name).unlink()
    except OSError as e:
        return ("tempdir", "fail", f"{d} not writable: {e.strerror or e}")
    return ("tempdir", "pass", d)


def _doctor_checks() -> list[tuple[str, str, str]]:
    """Run every doctor check; a crashing check reports (label, 'fail', why).

    The check callables are looked up from the module globals per call, so
    tests can monkeypatch a single _doc_* function to force a failure.

    Returns:
        (label, status, detail) rows in report order.

    """
    checks = (
        _doc_python,
        _doc_ppgrid,
        _doc_numpy,
        _doc_rasterio,
        _doc_gdal,
        _doc_pyproj,
        _doc_zstdmt,
        _doc_cinterp,
        _doc_ram,
        _doc_tempdir,
    )
    rows: list[tuple[str, str, str]] = []
    for check in checks:
        try:
            rows.append(check())
        except Exception as e:  # ruff: ignore[blind-except]
            rows.append((check.__name__.removeprefix("_doc_"), "fail", f"check crashed: {type(e).__name__}: {e}"))
    return rows


def _cmd_doctor() -> int:
    """`ppgrid doctor` (issue #61). Report to stdout; returns the exit code."""
    rows = _doctor_checks()
    width = max(len(label) for label, _, _ in rows)
    failing: list[str] = []
    for label, status, detail in rows:
        if status == "fail":
            failing.append(label)
        print(f"{label:<{width}}  {detail}  {_DOCTOR_MARK[status]}")  # ruff: ignore[print]
    if failing:
        print(f"doctor: {len(failing)} check(s) failing: {', '.join(failing)}", file=sys.stderr)  # ruff: ignore[print]
        return 1
    return 0


# ---------------------------------------------------------------------------
# probe (issue #62): input diagnostic. Writes no output files; the
# projection validation mirrors the pipeline ingest (float64 pin, the same
# non-finite / out-of-domain / non-finite-projection drops, the same
# Transformer) so its numbers are what a real run would do.
# ---------------------------------------------------------------------------


def _probe_size(n: float) -> str:
    """Human file size (probe report line)."""
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.2f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.1f} KB"
    return f"{int(n)} B"


def _probe_format(path: Path, head: bytes) -> tuple[str, list[str]]:
    """Classify the input (issue #62): format label + magic/extension problems.

    Reuses _sniff_compression (the issue-#80 planner magic) for the CSV
    family; parquet is extension-driven (pyarrow sniffs internally).
    pandas infers CSV compression from the file NAME, so a magic that
    disagrees with the extension is reported as a problem a real run would
    choke on.

    Args:
        path: The input path (suffix drives the parquet vs csv split).
        head: Leading bytes (>= 5; the magic length).

    Returns:
        (label, problems): 'csv' / 'csv (gzip)' / 'parquet' + mismatch list.

    """
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return ("parquet", [])
    compression = _sniff_compression(head)
    label = "csv" if compression is None else f"csv ({compression})"
    problems: list[str] = []
    expected = {".gz": "gzip", ".gzip": "gzip", ".bz2": "bzip2", ".zip": "zip", ".xz": "xz", ".lzma": "xz"}
    if suffix in expected and expected[suffix] != compression:
        problems.append(
            f"extension {suffix} implies {expected[suffix]} but the file starts with "
            f"{compression or 'uncompressed'} magic: pandas infers compression from the name; "
            "a real run would fail to parse",
        )
    elif suffix not in expected and compression is not None:
        problems.append(
            f"file starts with {compression} magic under a plain {suffix or '(no)'} extension: pandas infers "
            "compression from the name; a real run would fail to parse (rename to e.g. .csv.gz)",
        )
    return (label, problems)


def _probe_coordinate_cols(columns: list[str]) -> tuple[str | None, str | None, list[str]]:
    """Guess the lon/lat columns from names (issue #62).

    Args:
        columns: Input column names, file order.

    Returns:
        (lon, lat, problems): the first exact-name match per role
        (case-insensitive); a missing set is a problem (a real run would
        need --lng-col/--lat-col), multiple matches report which one the
        probe uses.

    """
    lowered = {c.lower(): c for c in columns}
    lon_hits = [lowered[k] for k in lowered if k in _LON_NAMES]
    lat_hits = [lowered[k] for k in lowered if k in _LAT_NAMES]
    problems: list[str] = []
    if not lon_hits and not lat_hits:
        problems.append(
            f"no column named like lon/lat (tried {sorted(_LON_NAMES | _LAT_NAMES)}); "
            "a real run needs --lng-col/--lat-col",
        )
    if len(lon_hits) > 1:
        problems.append(
            f"multiple longitude-like columns {lon_hits}; probe uses {lon_hits[0]!r}, a real run needs --lng-col",
        )
    if len(lat_hits) > 1:
        problems.append(
            f"multiple latitude-like columns {lat_hits}; probe uses {lat_hits[0]!r}, a real run needs --lat-col",
        )
    return (lon_hits[0] if lon_hits else None, lat_hits[0] if lat_hits else None, problems)


def _degree_verdict(vals: list[np.ndarray]) -> str:
    """One-line degree-likeness verdict from coordinate arrays (issue #62)."""
    pooled = np.concatenate(vals) if vals else np.array([], dtype=np.float64)
    finite = pooled[np.isfinite(pooled)]
    if finite.size == 0:
        return "coordinate values are all empty or non-finite in the sample; cannot guess the CRS"
    mx = float(np.abs(finite).max())
    if mx <= 180.0 and mx > 0.0:
        return f"sampled values look like degrees (max |x| = {mx:.6g}); src CRS guess EPSG:{SRC_CRS}"
    return f"sampled values exceed +/-180 (max |x| = {mx:.6g}); not degrees — a real run needs a matching --src-crs"


def _probe_degree_sample(sample: pd.DataFrame, lon: str | None, lat: str | None) -> str:
    """Degree-likeness verdict on the sampled coordinate columns (csv path)."""
    vals: list[np.ndarray] = []
    for col in (lon, lat):
        if col is None:
            continue
        vals.append(pd.to_numeric(sample[col], errors="coerce").to_numpy(dtype=np.float64))
    return _degree_verdict(vals)


def _probe_value_candidates(sample: pd.DataFrame) -> list[str]:
    """Numeric --value-col candidates (issue #62): >= 50% numeric sample.

    The pipeline pins wanted columns to float64 (issue #80); a column whose
    sampled values mostly survive pd.to_numeric(errors='coerce') would
    parse under that pin.

    Args:
        sample: First _PROBE_SAMPLE rows, all columns, inferred dtypes.

    Returns:
        Candidate column names in file order.

    """
    out: list[str] = []
    for col in sample.columns:
        non_null = sample[col].dropna()
        if non_null.empty:
            continue
        if pd.to_numeric(non_null, errors="coerce").notna().mean() >= 0.5:
            out.append(col)
    return out


def _cmd_probe(args: argparse.Namespace) -> int:
    """`ppgrid probe` (issue #62). Diagnostic to stdout; returns the exit code.

    Exit codes: 0 clean, 1 problem(s) found (named), 2 usage error
    (unknown work-CRS code — the run CLI's bad-flag-value convention).
    """
    _load_native()
    # Usage check first: an unknown work-CRS code is a bad flag value
    # (exit 2) regardless of the file state, like the run CLI — both CRS
    # error types, like _map_pipeline_errors (rasterio's global CRS here,
    # pyproj's from the Transformer below).
    try:
        CRS.from_epsg(args.work_crs)
    except (CRSError, RasterioCRSError):
        print(f"error: unknown work CRS 'EPSG:{args.work_crs}' (bad flag value)", file=sys.stderr)  # ruff: ignore[print]
        return 2
    path = Path(args.input)
    problems: list[str] = []
    print(f"input:      {args.input}")  # ruff: ignore[print]
    if not path.exists():
        problems.append("input file not found; a real run would exit 1 (pipeline error)")
        print(f"verdict:    {len(problems)} problem(s) found — a real run would fail")  # ruff: ignore[print]
        for p in problems:
            print(f"  - {p}")  # ruff: ignore[print]
        return 1
    if not path.is_file():
        problems.append("input is not a regular file; a real run would exit 1 (pipeline error)")
        print(f"verdict:    {len(problems)} problem(s) found — a real run would fail")  # ruff: ignore[print]
        for p in problems:
            print(f"  - {p}")  # ruff: ignore[print]
        return 1

    with path.open("rb") as f:
        head = f.read(8)
    label, fmt_problems = _probe_format(path, head)
    problems += fmt_problems
    print(f"format:     {label} ({_probe_size(path.stat().st_size)})")  # ruff: ignore[print]

    # Columns (+ exact row count for parquet; the csv row count comes from
    # the streaming pass below).
    columns: list[str] = []
    rows = 0
    pq_file = None
    if label == "parquet":
        try:
            import pyarrow.parquet as pq

            pq_file = pq.ParquetFile(path)
            columns = list(pq_file.schema_arrow.names)
            rows = int(pq_file.metadata.num_rows)
        except ImportError:
            problems.append(
                "parquet input needs pyarrow (install the [parquet] extra); a real run would fail the same way"
            )
        except Exception as e:  # ruff: ignore[blind-except]
            problems.append(f"cannot read parquet metadata: {type(e).__name__}: {e}")
    else:
        try:
            header = pd.read_csv(path, nrows=0)
            columns = [str(c) for c in header.columns]
        except Exception as e:  # ruff: ignore[blind-except]
            problems.append(
                f"cannot read header: {type(e).__name__}: {e} (empty file?); a real run would fail to parse"
            )

    shown = ", ".join(columns[:12]) + (f", ... ({len(columns)} total)" if len(columns) > 12 else "")
    if columns:
        print(f"columns:    ({len(columns)}) {shown}")  # ruff: ignore[print]

    value_col = args.value_col or "value"  # the run default
    lon, lat, coord_problems = _probe_coordinate_cols(columns)
    problems += coord_problems

    # Preflight parity with the run (#57): the run matches column names
    # case-sensitively (_check_columns) against the exact defaults
    # "longitude"/"latitude" (g_input) unless overridden with
    # --lng-col/--lat-col. Alias names — and a single missing coordinate
    # column — exit 2 there, so they are problems here too.
    needed_flags: list[str] = []
    if lon is not None and lon != "longitude":
        needed_flags.append(f"--lng-col {lon!r}")
    if lat is not None and lat != "latitude":
        needed_flags.append(f"--lat-col {lat!r}")
    if needed_flags:
        problems.append(
            f"a real run needs {' / '.join(needed_flags)} — the column names do not match the run "
            "defaults 'longitude'/'latitude' (case-sensitive); it would exit 2 at preflight",
        )
    if (lon is None) != (lat is None):
        missing = "longitude" if lon is None else "latitude"
        flag = "--lng-col" if lon is None else "--lat-col"
        problems.append(
            f"only one coordinate column present — no {missing!r} column; a real run needs "
            f"{flag} {missing!r} (the default) and would exit 2 at preflight",
        )

    # Candidate sample (csv) + value-column check.
    sample: pd.DataFrame | None = None
    if label != "parquet" and columns:
        try:
            sample = pd.read_csv(path, nrows=_PROBE_SAMPLE)
        except Exception as e:  # ruff: ignore[blind-except]
            problems.append(f"sample read failed: {type(e).__name__}: {e}")
    if columns and value_col not in columns:
        close = difflib.get_close_matches(value_col, columns, n=3)
        hint = f"; did you mean: {', '.join(close)}" if close else ""
        problems.append(
            f"value column {value_col!r} not found{hint}; a real run would exit 2 before ingesting (issue #57)",
        )

    # Row count + the wanted-column arrays (the same float64 pin the run uses).
    # The pipeline reads all wanted columns in one call: if the read fails,
    # a real run fails before projection, so the validation below is skipped
    # for that reason (not for missing coordinate columns).
    wanted: list[str] = []
    for c in (value_col, lon, lat):
        if c and c in columns and c not in wanted:
            wanted.append(c)
    arrays: dict[str, np.ndarray] = {}
    read_failed = False
    if label != "parquet":
        if columns:
            pin = dict.fromkeys(wanted, np.float64) if wanted else None
            try:
                if wanted:
                    pieces: dict[str, list[np.ndarray]] = {c: [] for c in wanted}
                    with pd.read_csv(path, usecols=wanted, dtype=pin, chunksize=_PROBE_CHUNK) as r:
                        for chunk in r:
                            rows += len(chunk)
                            for c in wanted:
                                pieces[c].append(chunk[c].to_numpy())
                    for c in wanted:
                        arrays[c] = np.concatenate(pieces[c]) if pieces[c] else np.array([], dtype=np.float64)
                else:
                    with pd.read_csv(path, chunksize=_PROBE_CHUNK) as r:
                        for chunk in r:
                            rows += len(chunk)
            except Exception as e:  # ruff: ignore[blind-except]
                read_failed = True
                problems.append(
                    f"row read failed: {type(e).__name__}: {e}; a real run would fail to parse "
                    "the wanted columns as float64",
                )
    elif pq_file is not None and wanted:
        try:
            df = pd.read_parquet(path, columns=wanted)
            for c in wanted:
                arrays[c] = df[c].to_numpy()
        except Exception as e:  # ruff: ignore[blind-except]
            read_failed = True
            problems.append(f"parquet read failed: {type(e).__name__}: {e}")

    if read_failed:
        print("rows:       unknown (row read failed)")  # ruff: ignore[print]
    elif rows == 0:
        problems.append("no data rows; a real run would fail with 'no valid points'")
        print("rows:       0")  # ruff: ignore[print]
    else:
        print(f"rows:       {rows}")  # ruff: ignore[print]

    # Value candidates + degree guess (the report lines, not problems).
    if label == "parquet" and pq_file is not None and columns:
        try:
            import pyarrow as pa

            schema = pq_file.schema_arrow
            # is_integer/is_floating cover every numeric pyarrow type the run
            # pins to float64 (pa.types.is_numeric does not exist; pyarrow 25).
            cands = [
                name
                for name, t in zip(schema.names, schema.types, strict=True)
                if pa.types.is_integer(t) or pa.types.is_floating(t)
            ]
        except Exception:  # ruff: ignore[blind-except]
            cands = []
        cand_line = ", ".join(cands) if cands else "none"
        if "value" in cands:
            cand_line += " (run default 'value' present)"
    elif sample is not None:
        cands = _probe_value_candidates(sample)
        cand_line = ", ".join(cands) if cands else "none"
        if "value" in cands:
            cand_line += " (run default 'value' present)"
    else:
        cand_line = "n/a (no sample)"
    print(f"value col:  {value_col!r} | numeric candidates: {cand_line}")  # ruff: ignore[print]
    if lon or lat:
        if label == "parquet" and lon in arrays and lat in arrays:
            verdict = _degree_verdict([arrays[lon], arrays[lat]])
        elif sample is not None:
            verdict = _probe_degree_sample(sample, lon, lat)
        else:
            verdict = "n/a (no data read)"
        print(f"coordinates: {lon or 'n/a'} / {lat or 'n/a'} — {verdict}")  # ruff: ignore[print]
    elif not coord_problems:
        print("coordinates: n/a")  # ruff: ignore[print]

    # Projection validation: the same drops the pipeline ingest makes.
    # The pipeline reads all wanted columns in one call: if the read fails
    # or a problem would stop the run first (a missing value column exits
    # 2 at preflight), the real run never reaches projection — validate
    # only on the clean path, and say why otherwise.
    kept = 0
    n_total = 0
    if lon is None or lat is None:
        print(f"projection: skipped (no coordinate columns; --work-crs {args.work_crs})")  # ruff: ignore[print]
    elif read_failed:
        print("projection: skipped (row read failed — a real run would fail before projecting)")  # ruff: ignore[print]
    elif problems:
        print("projection: skipped (the problems above stop a real run before projecting)")  # ruff: ignore[print]
    else:
        print(f"projection: EPSG:{SRC_CRS} -> EPSG:{args.work_crs} (work CRS; src assumed EPSG:{SRC_CRS})")  # ruff: ignore[print]
        lon_a, lat_a = arrays[lon], arrays[lat]
        v_a = arrays.get(value_col)
        n_total = int(lon_a.size)
        good = np.isfinite(lon_a) & np.isfinite(lat_a)
        if v_a is not None:
            good &= np.isfinite(v_a)
        n_nan_inf = int(np.count_nonzero(~good))
        ood = good & ((lat_a < -90.0) | (lat_a > 90.0) | (lon_a < -180.0) | (lon_a > 180.0))
        n_ood = int(ood.sum())
        domain = good & ~ood
        first_ood = int(np.argmax(ood)) if n_ood else -1
        tr = Transformer.from_crs(SRC_CRS, args.work_crs, always_xy=True)
        x, y = tr.transform(lon_a[domain], lat_a[domain], radians=False)
        fin = np.isfinite(x) & np.isfinite(y)
        n_np = int(np.count_nonzero(~fin))
        first_np = int(np.nonzero(~fin)[0][0]) if n_np else -1
        kept = int(fin.sum())
        print(f"  rows read:               {n_total}")  # ruff: ignore[print]
        print(f"  dropped nan/inf:         {n_nan_inf}")  # ruff: ignore[print]
        print(f"  dropped out-of-domain:   {n_ood}" + (f" (first at row {first_ood})" if n_ood else ""))  # ruff: ignore[print]
        suffix = f" (first from lon/lat {lon_a[domain][first_np]:.6f}, {lat_a[domain][first_np]:.6f})" if n_np else ""
        print(f"  dropped non-finite proj: {n_np}{suffix}")  # ruff: ignore[print]
        if kept:
            print(f"  x domain:                {float(x[fin].min()):.3f} .. {float(x[fin].max()):.3f} m")  # ruff: ignore[print]
            print(f"  y domain:                {float(y[fin].min()):.3f} .. {float(y[fin].max()):.3f} m")  # ruff: ignore[print]
        print(f"  points kept:             {kept}")  # ruff: ignore[print]
        if kept == 0:
            problems.append(
                "no valid points remain after the same drops a real run would make; a real run would fail "
                "with 'No valid points found in input'",
            )

    if problems:
        print(f"verdict:    {len(problems)} problem(s) found — a real run would fail or drop points")  # ruff: ignore[print]
        for p in problems:
            print(f"  - {p}")  # ruff: ignore[print]
        return 1
    if n_total:
        print(f"verdict:    ok — a real run would keep {kept}/{n_total} points")  # ruff: ignore[print]
    else:
        print("verdict:    ok")  # ruff: ignore[print]
    return 0


# ---------------------------------------------------------------------------
# list-crs (issue #69): the curated ppgrid set by default, the full
# pyproj/EPSG database with --all. pyproj loads inside the handler, after
# parse_args, so `list-crs --help` stays fast.
# ---------------------------------------------------------------------------


def _cmd_list_crs(args: argparse.Namespace) -> int:
    """`ppgrid list-crs` (issue #69). List to stdout; returns the exit code."""
    import pyproj  # ruff: ignore[unused-import] — the local import IS the lazy load (post-parse_args)
    from pyproj import CRS

    if args.all:
        import pyproj.database as pdb

        rows = pdb.query_crs_info(
            auth_name="EPSG",
            pj_types=[
                pdb.PJType.GEOGRAPHIC_2D_CRS,
                pdb.PJType.GEOGRAPHIC_3D_CRS,
                pdb.PJType.PROJECTED_CRS,
                pdb.PJType.COMPOUND_CRS,
            ],
            allow_deprecated=True,
        )
        for r in sorted(rows, key=lambda x: int(x.code)):
            dep = " (deprecated)" if r.deprecated else ""
            print(f"{r.code}  {r.name}{dep}")  # ruff: ignore[print]
        return 0
    failures: list[int] = []
    for code, role, note in _CURATED_CRS:
        try:
            name = CRS.from_epsg(code).name
        except Exception:  # ruff: ignore[blind-except]
            failures.append(code)
            print(f"{code}  <failed to load>  ({role})")  # ruff: ignore[print]
            continue
        print(f"{code}  {name}  ({role}) — {note}")  # ruff: ignore[print]
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# init (issue #63): write the commented ppgrid.toml template.
#
# Stdlib-only: the subcommand path runs before _load_native(), so init
# (like --help) pays no native-stack cost. The template is rendered from
# _build_parser, so a new flag joins via _CONFIG_OPTIONS.
# ---------------------------------------------------------------------------


def _cmd_init(args: argparse.Namespace) -> None:
    """`ppgrid init [DIR]`: write the commented ppgrid.toml template (#63).

    Stdlib-only: the subcommand path runs before _load_native(), so init
    (like --help) pays no native-stack cost. A missing DIR is created with
    parents; the success line goes to stderr (stdout stays reserved for
    the run CLI's machine output).

    Args:
        args: Parsed subcommand namespace (dir, force).

    Raises:
        SystemExit: 2 (existing file without --force), 1 (IO error).

    """
    target_dir = Path(args.dir)
    toml = target_dir / _CONFIG_FILE_NAME
    if toml.exists() and not args.force:
        sys.stderr.write(f"error: {toml} already exists (use --force to overwrite)\n")
        raise SystemExit(2)
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        toml.write_text(_init_template(), encoding="utf-8")
    except OSError as e:
        sys.stderr.write(f"error: cannot write {toml}: {e.strerror or str(e)}\n")
        raise SystemExit(1) from e
    sys.stderr.write(f"ok: wrote {toml}\n")


def _toml_scalar(value: Any) -> str:
    """Render a Python scalar as a TOML literal (str/int/float/bool) (#63)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, float):
        return repr(value)  # TOML floats need a dot/exponent: repr(500.0) = '500.0'
    return str(value)


def _init_template() -> str:
    """Render the ppgrid.toml template (issue #63).

    Every config-able option in --help group order, commented with the
    flag's own help text and set to its built-in default; a None default
    renders commented out with an example value (the option is off by
    default). Deterministic: the same ppgrid version renders the same
    file. A new flag joins by one line in _CONFIG_OPTIONS.

    """
    parser = _build_parser()
    by_dest = {a.dest: a for a in parser._actions}  # ruff: ignore[private-member-access]
    lines: list[str] = [
        f"# {_CONFIG_FILE_NAME} — ppgrid run config (issue #63)",
        "#",
        f"# Generated by `ppgrid init` (ppgrid {__version__}).",
        "# Precedence: CLI flag > this file > built-in default.",
        "# Every value below equals the built-in default, so this file as",
        "# written changes nothing about a run. Edit a value to override it.",
        "# Unknown keys warn and are ignored; an invalid value is a usage",
        "# error (exit 2).",
        "",
    ]
    for group in parser._action_groups:  # ruff: ignore[private-member-access]
        if group.title in ("positional arguments", "options"):
            continue
        dests_in_group = {a.dest for a in group._group_actions}  # ruff: ignore[private-member-access]
        rows = [row for row in _CONFIG_OPTIONS if row[1] in dests_in_group]
        if not rows:
            continue
        group_lines = [f"# --- {group.title} " + "-" * max(0, 66 - len(group.title)), ""]
        for key, dest, _kind, example in rows:
            action = by_dest[dest]
            default = _CONFIG_DEFAULTS[dest]
            group_lines.append(f"# {action.help}")
            if default is None:
                group_lines.append(f"# {key} = {_toml_scalar(example)}")
            else:
                group_lines.append(f"{key} = {_toml_scalar(default)}")
            group_lines.append("")
        lines.extend(group_lines)
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point (the ``ppgrid`` console script).

    Parses ``argv`` (defaults to ``sys.argv[1:]``), runs the pipeline
    end-to-end, and reports through the CLI presentation layer. stdout is
    reserved for ``--plan``, ``help``, ``--version``, the subcommand
    reports (compare / doctor / probe / list-crs, #61 #62 #66 #69) and the
    completion scripts (completions, #66); every other line (progress,
    warnings, errors, the run summary) goes to stderr.

    Subcommands are dispatched on argv[0] before the run parser is built
    (separate parser, see the section note): ``ppgrid doctor``,
    ``ppgrid probe INPUT``, ``ppgrid list-crs``, ``ppgrid compare OUT1
    OUT2`` and ``ppgrid completions bash|zsh``. ``ppgrid init [DIR]``
    (issue #63) writes the config template and exits before any of the run
    machinery runs.

    Side effects: discovers and merges the ppgrid.toml config (issue #63,
    CLI flag > config > built-in default), creates the output directory and
    writes ``value.tif``, ``support_km.tif`` and ``calibration.json`` into
    it (plus ``ppgrid_run.json`` run-fingerprint sidecar, issue #64, every
    successful run, and ``run_summary.json`` with ``--json``); mirrors the
    run log to ``--log-file`` when given (issue #65); configures the
    ``ppgrid`` logger for the rest of the process. Diag subcommands write
    no output files (doctor's oracle probe uses a throwaway temp-dir tile).

    Args:
        argv: Argument list (defaults to sys.argv[1:]; tests pass their own).

    Raises:
        SystemExit: 0 (ok, help, --plan, init), 1 (pipeline/IO error),
            2 (validation error), 3 (--turbo-strict infeasible) — as
            documented on _run_cli (propagated unchanged). Subcommands:
            doctor 0/1, probe 0/1/2, list-crs 0/1, init 0/1/2,
            compare 0/1/2, completions 0/2 (see _run_subcommand). SIGINT
            on either path is the run contract: exit 130, one line, no
            traceback.

    """
    if argv is None:
        argv = sys.argv[1:]
    # Subcommands dispatch on the literal first token (token-wins) before
    # the run parser is built: the run CLI's positional/flag surface stays
    # byte-identical (argparse cannot mix a bare positional with
    # subparsers). A data file named like a subcommand runs as
    # `ppgrid ./<name>`. Interrupts keep the run contract (130).
    if argv and argv[0] in _SUBCOMMANDS:
        try:
            _run_subcommand(argv)
        except KeyboardInterrupt:
            _interrupted_exit(None, None)
        return
    parser = _build_parser()
    out: Path | None = None
    try:
        args = _parse_args_cli(parser, argv)
        # -h/--help/--version exit inside parse_args, so the lazy stack
        # (pandas, rasterio, pyproj) and the presentation layer (rich) are
        # paid for only by real runs, not by help/version (fast CLI).
        _load_native()
        from . import ui as _ui

        # Presentation layer (issue #101): resolve live/color modes once for
        # this invocation. main() is the only builder; library callers pass
        # their own (or no) UI straight to Pipeline.
        ui = _ui.UI.from_cli(quiet=args.quiet, color=args.color)
        _CURRENT_UI["ui"] = ui
        out: Path | None = None  # bound after the config merge (it may set out); None = interrupt during parse/merge
        try:
            # #63: config file merge (CLI flag > config > built-in
            # default). After the UI is up (unknown-key warnings use the
            # unified line) and before the out path is read (the config may
            # set it); on return every config-able option holds a real value.
            cfg = _apply_config(parser, args)
            args.config_path = str(cfg.resolve()) if cfg is not None else None
            out = Path(args.out)
            _run_cli(parser, args, out, ui)
        finally:
            _CURRENT_UI["ui"] = None
    except KeyboardInterrupt:
        # Interrupt arg-parse through the _run_cli entry (issue #106): the
        # mid-run contract applies here too — sweep, one line, exit 130.
        # Interrupts inside _run_cli's body are handled there.
        _interrupted_exit(out, None)


def _run_cli(parser: argparse.ArgumentParser, args: argparse.Namespace, out: Path, ui: _ui.UI) -> None:
    """Run the parsed-args CLI sequence: preflight -> guard -> run -> summary.

    Split from main() (issue #101) so the presentation handle is built
    once per invocation and routes every stderr line (errors, warnings,
    hints, the env report) through the same live console that hosts the
    progress table. main() clears the _CURRENT_UI slot when this returns.
    Exit codes propagate as documented on main().

    Args:
        parser: The argument parser (for print_help / error).
        args: Parsed namespace (mutated: max_ram alias resolution).
        out: Output directory path.
        ui: Presentation handle for this invocation.

    Raises:
        SystemExit: exit 0 for `ppgrid help` (and `--plan`), exit 1 for
            pipeline/IO errors (`_die`), exit 2 for missing input,
            validation errors, missing input columns (#57), or a non-empty
            out dir without --force (#54), exit 3 for `--turbo-strict`
            shared-path infeasibility. Ctrl-C mid-run exits 130 with one
            `interrupted` line, no traceback, run temps swept, retained
            partials named (issue #106). Any non-zero exit over an earlier
            run's rasters also prints the stale-outputs warning
            (issues #40, #77).

    """
    p: Pipeline | None = None
    try:
        try:
            # Preflight (input check, logging, flag validation, --plan
            # branch): any non-zero exit over an earlier run's rasters warns
            # instead of staying silent (issues #40, #77).
            _main_preflight(parser, args, out)
        except SystemExit as e:
            if e.code not in (0, None):
                _warn_stale_outputs(out)
            raise
        except BaseException as e:
            # KeyboardInterrupt is handled once, at the outer level below:
            # sweep, stale warn, one line, exit 130 (issue #106).
            if not isinstance(e, KeyboardInterrupt):
                _warn_stale_outputs(out)
            raise

        # #54: overwrite guard — a non-empty out dir means an earlier run's
        # outputs; require --force to clobber them.
        if not args.force:
            guarded = ("value.tif", "support_km.tif", "calibration.json", "run_summary.json")
            existing = [name for name in guarded if (out / name).is_file()]
            if existing:
                _ui_note("error", f"output directory {args.out} already contains: {', '.join(existing)}")
                _ui_note(
                    "note",
                    "hint: pass --force to overwrite them",
                )
                raise SystemExit(2)

        try:
            # #57: friendly missing-column error before any pipeline work;
            # inside the stale-outputs handler so a failed run over an
            # earlier run's outputs still warns (issue #40).
            _check_columns(args.input, args.value_col, args.lng_col, args.lat_col)
            out.mkdir(parents=True, exist_ok=True)
        except SystemExit:
            _warn_stale_outputs(out)
            raise
        except OSError as e:
            _die(f"cannot create output directory {args.out}: {e}")
        try:
            # Construction (init validation, e.g. --scale range) gets the
            # same exit-code mapping as the run phase (review M-1, #56
            # contract).
            p = _map_pipeline_errors(lambda: _make_pipeline(args, dry_run=False, ui=ui), args.input)
            _map_pipeline_errors(p.run, args.input)
            if args.check:
                _post_write_check(p, out)
            _write_run_sidecar(args, out)  # #64: run fingerprint sidecar
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
        except BaseException as e:
            # KeyboardInterrupt is handled once, at the outer level below:
            # sweep, stale warn, one line, exit 130 (issue #106).
            if not isinstance(e, KeyboardInterrupt):
                _warn_stale_outputs(out, p.published if p is not None else None)
            raise
    except KeyboardInterrupt:
        # One clean line, no traceback (issue #106): sweep run temps, name
        # any retained partial, warn about an earlier run's rasters, exit 130.
        _interrupted_exit(out, p)


if __name__ == "__main__":
    main()
