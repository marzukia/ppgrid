"""ppgrid — pull-push scattered-data interpolation pipeline + CLI.

Turns scattered geolocated point values into a continent-scale, gapless,
capped raster surface, in minutes, on a single machine, with no GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

from . import __version__
from .calibrate import (
    M_PER_KM,
    PERCENTILE_MAX,
    PercentileTransform,
    calibrate_fill_cap,
    choose_transform,
    make_transform,
    transforms,
)
from .pullpush import (
    _descent_banded,
    _pull_push_descent,
    bin_points,
    bin_points_banded,
    box_count,
    box_count_banded,
    downsample_sum,
    pull_push,
)

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


def _quantize(
    cfg: _WorkerConfig,
    v_out: np.ndarray,
    r_out: np.ndarray,
    near_out: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform interpolated (value, support) to int16 percentile DN grids.

    Shared by the per-block and shared-field paths so the rounding is
    structurally identical.

    Returns:
        Tuple of (value, support) int16 arrays with NODATA where not near data.

    """
    # Transform: interpolate space -> raw -> percentile
    tf = cfg.tf
    pct = cfg.pct
    pv = pct.fwd(tf.inv(v_out))
    if cfg.pct_step is not None:
        # Round to the nearest step (e.g. 5 -> 90/95/100), clamp to 0-100.
        pv = np.clip(np.round(pv / cfg.pct_step) * cfg.pct_step, 0.0, PERCENTILE_MAX)
    vq = np.where(near_out, np.round(pv * cfg.scale), NODATA).astype(np.int16)
    rq = np.where(
        near_out,
        np.clip(
            np.round(np.log2(np.maximum(r_out, SUPPORT_KM_FLOOR)) * SUPPORT_LOG2_SCALE),
            -SUPPORT_DN_MAX,
            SUPPORT_DN_MAX,
        ),
        NODATA,
    ).astype(np.int16)
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
    return _quantize(cfg, cfg.val_full[sl], cfg.sup_full[sl] / M_PER_KM, cfg.near_full[sl])


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
    return bx, by, _quantize(cfg, val[sl], sup[sl] / M_PER_KM, near[sl])


def _block_neighbourhood_nonempty(
    bx: int,
    by: int,
    starts: np.ndarray,
    nbx: int,
    nby: int,
) -> bool:
    """Return True if any block in the 3x3 neighbourhood holds at least one point."""
    return any(starts[bid + 1] > starts[bid] for bid in _neighbour_block_ids(bx, by, nbx, nby))


def _reproject_band(
    src_path: str,
    dst_path: str,
    profile: dict[str, Any],
    dst_crs: str,
    dst_transform: Any,
    dst_width: int,
    dst_height: int,
) -> None:
    """Nearest-neighbour reproject one band into a fresh output-CRS GeoTIFF.

    Vectorised. The nearest-neighbour warp is a per-pixel function of the dst
    grid, so the *warp* chunk size never changes the output values. The
    *write* chunk size, however, changes GDAL's on-disk block layout (and
    thus the file bytes). So: warp in coarse 2048px tiles (fewer, larger gdal
    calls -> ~15% faster) but flush TILE_PX blocks in raster-scan order,
    exactly as the 512px baseline does. The result is byte-identical to the
    baseline.
    """
    with rasterio.open(src_path) as src:
        dst_profile = dict(
            profile,
            width=dst_width,
            height=dst_height,
            transform=dst_transform,
            crs=dst_crs,
        )
        with rasterio.open(dst_path, "w", **dst_profile) as dst:
            dst.update_tags(**src.tags())
            if hasattr(src, "scales") and src.scales:
                dst.scales = src.scales
            if hasattr(src, "offsets") and src.offsets:
                dst.offsets = src.offsets
            warp_tile = 2048
            for j0 in range(0, dst_height, warp_tile):
                band_h = min(warp_tile, dst_height - j0)
                # Warp the coarse tiles across this row band.
                coarse: dict[int, np.ndarray] = {}
                for i0 in range(0, dst_width, warp_tile):
                    band_w = min(warp_tile, dst_width - i0)
                    w = Window(i0, j0, band_w, band_h)
                    w_bounds = rasterio.windows.bounds(w, dst_transform)
                    local_dst_transform = from_bounds(*w_bounds, band_w, band_h)
                    buf = np.zeros((band_h, band_w), dtype=np.int16)
                    reproject(
                        rasterio.band(src, 1),
                        buf,
                        src_transform=src.transform,
                        dst_transform=local_dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest,
                        nodata=NODATA,
                    )
                    coarse[i0] = buf
                # Flush TILE_PX blocks in raster-scan order.
                for j in range(j0, j0 + band_h, TILE_PX):
                    w_h = min(TILE_PX, j0 + band_h - j)
                    for i in range(0, dst_width, TILE_PX):
                        w_w = min(TILE_PX, dst_width - i)
                        i0 = (i // warp_tile) * warp_tile
                        sub = coarse[i0][j - j0 : j - j0 + w_h, i - i0 : i - i0 + w_w]
                        dst.write(sub, 1, window=Window(i, j, w_w, w_h))


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
    ) -> None:
        """Initialise the interpolation pipeline.

        Raises:
            ValueError: If scale * PERCENTILE_MAX exceeds int16 max.
            ValueError: If percentile_step is outside (0, PERCENTILE_MAX].
            ValueError: If saturation is not > 0.

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

    def ingest(self) -> None:
        """Read input, filter, project to working CRS.

        Raises:
            ValueError: If no valid points remain after filtering.
            ImportError: If pyarrow is missing for Parquet input.

        """
        if self.input_path.endswith((".parquet", ".pq")):
            try:
                import pyarrow as pa  # ruff: ignore[unused-import]
            except ImportError:
                msg = "pyarrow is required for Parquet files. Install with: pip install ppgrid[parquet]"
                raise ImportError(msg) from None
            df = pd.read_parquet(self.input_path, columns=[self.value_col, self.lng_col, self.lat_col])
        else:
            df = pd.read_csv(self.input_path, usecols=[self.value_col, self.lng_col, self.lat_col])

        v = df[self.value_col].to_numpy(dtype=np.float64)
        lon = df[self.lng_col].to_numpy(dtype=np.float64)
        lat = df[self.lat_col].to_numpy(dtype=np.float64)

        good = np.isfinite(v) & np.isfinite(lon) & np.isfinite(lat)
        v, lon, lat = v[good], lon[good], lat[good]
        self.n = len(v)

        if self.n == 0:
            msg = "No valid points found in input. Check columns and data."
            raise ValueError(msg)

        tr = Transformer.from_crs(self.src_crs, self.work_crs, always_xy=True)
        x, y = tr.transform(lon, lat)
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.v = v

    def calibrate(self) -> None:
        """Load or run calibration. Sets transform, percentile, cap.

        If an explicitly forced transform is invalid for the data (e.g. log10
        with negative values), a ValueError is raised. For the ``auto`` path,
        an invalid transform is rejected with a warning and identity is used.

        Raises:
            ValueError: If an explicit --transform cannot be applied to the data.

        """
        cal: dict[str, Any] | None = None
        cpath_obj = Path(self.calib_path) if self.calib_path else None
        if cpath_obj and cpath_obj.exists():
            with cpath_obj.open(encoding="utf-8") as f:
                cal = json.load(f)
        elif not self.skip_calibration:
            if self.n > self.calib_max_points:
                sub = np.random.default_rng(0).choice(self.n, self.calib_max_points, replace=False)
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
                cap, detail = calibrate_fill_cap(cx, cy, tf.fwd(cv))

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
            Path(self.out_dir).mkdir(parents=True, exist_ok=True)
            cal_path = cpath_obj or Path(self.out_dir) / "calibration.json"
            with cal_path.open("w", encoding="utf-8") as f:
                json.dump(cal, f, indent=2)
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

        self.levels = max(1, math.ceil(math.log2(max(self.cap_km_val * M_PER_KM / self.res, 2.0))))
        self.step = 1 << self.levels
        self.halo = max(math.ceil(self.cap_km_val * M_PER_KM / self.res) + 2, self.step)
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

        # Memmap for workers (bands PTS_X/PTS_Y/PTS_TV — see constants above)
        self.pts_path = Path(self.out_dir) / "_points.npy"
        pts = np.lib.format.open_memmap(self.pts_path, mode="w+", dtype=np.float64, shape=(PTS_NBANDS, self.n))
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

        Returns:
            Tuple of value and support GeoTIFF file paths.

        """
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        self.ingest()
        self.calibrate()
        self.grid()
        return self._write_rasters()

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
        if self.nx_padded * self.ny_padded > _SHARED_MAX_CELLS:
            return False
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
        ix = ((pts[PTS_X][:] - self.x0) // self.res).astype(np.int64)
        iy = ((pts[PTS_Y][:] - self.y0) // self.res).astype(np.int64)
        cap_cells = round(self.cap_km_val * M_PER_KM / self.res)
        out_dir = Path(self.out_dir)
        if self.nx_padded * self.ny_padded > _SHARED_MEMMAP_CELLS:
            # Keep the finest grids off the RAM budget: memmap + row banded
            # bin/near (bit-identical to the in-RAM pass, see pullpush).
            shape = (self.nx_padded, self.ny_padded)
            s0 = np.lib.format.open_memmap(out_dir / "_s0.npy", mode="w+", dtype=np.float32, shape=shape)
            c0 = np.lib.format.open_memmap(out_dir / "_c0.npy", mode="w+", dtype=np.float32, shape=shape)
            bin_points_banded(s0, c0, ix, iy, pts[PTS_TV][:], self.ny_padded)
            near_full = np.lib.format.open_memmap(out_dir / "_near.npy", mode="w+", dtype=bool, shape=shape)
            box_count_banded(c0, cap_cells, near_full)
        else:
            s0, c0 = bin_points(ix, iy, pts[PTS_TV][:], self.nx_padded, self.ny_padded)
            near_full = box_count(c0, cap_cells) > 0

        sums: list[np.ndarray] = [s0]
        for _ in range(self.levels):
            sums.append(downsample_sum(sums[-1]))
        counts: list[np.ndarray] = [c0]
        for _ in range(self.levels):
            counts.append(downsample_sum(counts[-1]))

        if self.nx_padded * self.ny_padded <= _SHARED_MEMMAP_CELLS:
            val_full, sup_full = _pull_push_descent(sums, counts, self.res, self.levels, saturation=self.cfg.sat)
        else:
            # Full descent to level 2 (small arrays), then band levels 2 -> 1
            # -> 0 through memmap to bound RAM on multi-billion-cell grids.
            # levels == 1 (cap_km*1000/res <= 2) has no level 2: stop/start the
            # descent at level 1 instead, else the banded k=1 step blends a
            # level-1 local with a level-2 upsample and broadcast-crashes.
            lvl2 = min(2, self.levels)
            val2, sup2 = _pull_push_descent(
                sums, counts, self.res, self.levels, saturation=self.cfg.sat, stop_level=lvl2, free_levels=True
            )
            val_full = np.lib.format.open_memmap(
                Path(self.out_dir) / "_val_full.npy", mode="w+", dtype=np.float32, shape=s0.shape
            )
            sup_full = np.lib.format.open_memmap(
                Path(self.out_dir) / "_sup_full.npy", mode="w+", dtype=np.float32, shape=s0.shape
            )
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

        with (
            rasterio.open(vpath, "w", **vprof) as vd,
            rasterio.open(spath, "w", **sprof) as sd,
        ):
            vd.update_tags(
                transform=self.tname,
                cap_km=str(self.cap_km_val),
                res_m=str(self.res),
                scale=str(self.scale),
                units="percentile",
                decode=f"percentile = DN/{self.scale:g}",
            )
            if self.percentile_step is not None:
                vd.update_tags(percentile_step=str(self.percentile_step))
            vd.scales = (1.0 / self.scale,)
            sd.update_tags(decode=f"support_km = 2**(DN/{SUPPORT_LOG2_SCALE})")

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
                if self._prepare_shared():
                    cfg.shared = True
                with ThreadPoolExecutor(max_workers=self.workers) as ex:
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

        try:
            self.pts_path.unlink()
            for fname in (
                "_val_full.npy",
                "_sup_full.npy",
                "_val_lvl1.npy",
                "_sup_lvl1.npy",
                "_s0.npy",
                "_c0.npy",
                "_near.npy",
            ):
                (Path(self.out_dir) / fname).unlink(missing_ok=True)

            if self.out_crs != self.work_crs:
                tmp_v = Path(self.out_dir) / "_value_tmp.tif"
                tmp_s = Path(self.out_dir) / "_support_tmp.tif"
                vpath.rename(tmp_v)
                spath.rename(tmp_s)
                try:
                    # Both bands share the same work-CRS source grid (identical
                    # bounds/size), so the output grid is identical too: compute
                    # the default transform once instead of per band.
                    with rasterio.open(tmp_v) as grid_src:
                        dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
                            grid_src.crs,
                            f"EPSG:{self.out_crs}",
                            grid_src.width,
                            grid_src.height,
                            *grid_src.bounds,
                        )

                    out_crs = f"EPSG:{self.out_crs}"
                    _reproject_band(str(tmp_v), str(vpath), vprof, out_crs, dst_transform, dst_width, dst_height)
                    _reproject_band(str(tmp_s), str(spath), sprof, out_crs, dst_transform, dst_width, dst_height)
                except Exception:
                    if tmp_v.exists():
                        tmp_v.rename(vpath)
                    if tmp_s.exists():
                        tmp_s.rename(spath)
                    raise
                if tmp_v.exists():
                    tmp_v.unlink()
                if tmp_s.exists():
                    tmp_s.unlink()
        finally:
            if self.pts_path.exists():
                self.pts_path.unlink()

        return str(vpath), str(spath)


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


def main() -> None:
    """Parse CLI arguments and run the pipeline.

    Raises:
        SystemExit: On invalid arguments or a pipeline error (exit code 2).

    """
    parser = argparse.ArgumentParser(description="Pull-push scattered-data interpolation")
    parser.add_argument("--version", action="version", version=f"ppgrid {__version__}")
    parser.add_argument("input", help="CSV or Parquet input path")
    parser.add_argument("-o", "--out", default="out/", help="Output directory (default: ./out)")
    parser.add_argument("--value-col", default="value", help="Value column name")
    parser.add_argument("--lng-col", default="longitude", help="Longitude column name")
    parser.add_argument("--lat-col", default="latitude", help="Latitude column name")
    parser.add_argument("--res", type=float, default=DEFAULT_RES, help="Cell size in metres")
    parser.add_argument("--cap-km", default="auto", help="Fill cap km, or 'auto'")
    parser.add_argument(
        "--transform",
        default="auto",
        choices=["auto", "identity", "log10", "sqrt", "percentile"],
    )
    parser.add_argument(
        "--saturation",
        type=float,
        default=DEFAULT_SATURATION,
        help="Counts for a cell to fully self-trust",
    )
    parser.add_argument("--block", type=int, default=DEFAULT_BLOCK, help="Block size in cells")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of workers")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="DN = percentile * scale")
    parser.add_argument(
        "--percentile-step",
        type=float,
        default=None,
        help="Round output percentiles to the nearest step (e.g. 5 -> 90/95/100). Default: no rounding",
    )
    parser.add_argument("--compress", default="ZSTD")
    parser.add_argument("--calib-max-points", type=int, default=CALIB_MAX_POINTS_DEFAULT)
    parser.add_argument("--calibration", default=None, help="Calibration JSON path")
    parser.add_argument("--src-crs", type=int, default=SRC_CRS)
    parser.add_argument(
        "--work-crs",
        type=int,
        default=WORK_CRS,
        help=f"Working CRS for interpolation (default {WORK_CRS})",
    )
    parser.add_argument("--out-crs", type=int, default=OUT_CRS, help=f"Output CRS (default {OUT_CRS})")
    parser.add_argument("--skip-calibration", action="store_true", help="Skip calibration, use defaults")
    args = parser.parse_args()

    if args.res <= 0:
        parser.error("--res must be positive")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.scale <= 0:
        parser.error("--scale must be positive")
    if args.saturation <= 0:
        parser.error("--saturation must be positive")
    if args.percentile_step is not None and not (0 < args.percentile_step <= PERCENTILE_MAX):
        parser.error(f"--percentile-step must be in (0, {PERCENTILE_MAX:g}]")
    if args.block < 1:
        parser.error("--block must be at least 1")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    try:
        run(
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
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)  # ruff: ignore[print] — CLI error output
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
