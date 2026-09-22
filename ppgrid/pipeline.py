"""ppgrid — CLI driver for pull-push scattered-data interpolation.

Turns scattered geolocated point values into a continent-scale, gapless,
capped raster surface, in minutes, on a single machine, with no GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.errors import RasterioError
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window, bounds

from .calibrate import (
    PERCENTILE_MAX,
    PercentileTransform,
    calibrate_fill_cap,
    choose_transform,
    make_transform,
    transforms,
)
from .pullpush import (
    bin_points,
    box_count,
    pull_push,
)

WORK_CRS: int = 6933  # Wagner VII — global equal-area, metres are true
SRC_CRS: int = 4326  # WGS 84 lon/lat, the expected input CRS
OUT_CRS: int = 3857  # Web Mercator, the expected output CRS
NODATA: int = -32768
INT16_MAX: int = 32767
DEFAULT_CAP_KM: float = 64.0  # fill cap when no calibration produces one
DEFAULT_SCALE: float = 100.0  # DN = percentile * scale (percentiles are 0-100)
DEFAULT_RES: float = 500.0  # default cell size, metres
DEFAULT_BLOCK: int = 2048  # default block size, cells
DEFAULT_WORKERS: int = 4  # default parallel worker threads
DEFAULT_SATURATION: float = 1.0  # counts for a cell to fully self-trust
DEFAULT_CALIB_MAX_POINTS: int = 2_000_000  # calibration subsample cap
TILE_PX: int = 512  # GeoTIFF tile size, also the reproject tile size
# Support-band encoding: DN = clamp(round(log2(max(km, SUPPORT_FLOOR_KM)) * SUPPORT_SCALE)).
# Decode: support_km = 2**(DN / SUPPORT_SCALE). See encode/decode_support_km.
SUPPORT_SCALE: int = 8
SUPPORT_FLOOR_KM: float = 1e-3
SUPPORT_DN_MIN: int = -32000
SUPPORT_DN_MAX: int = 32000

# Shared context for the worker threads: one module dict, read by all threads.
_CTX: dict[str, Any] = {}


@dataclass
class _WorkerConfig:
    """Configuration shared with the worker threads (one instance per run)."""

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


def _block_neighbours(bx: int, by: int, nbx: int, nby: int) -> list[tuple[int, int]]:
    """Block coords of the 3x3 neighbourhood of (bx, by), clipped to the grid.

    Returns:
        List of (x_block, y_block) tuples.

    """
    x_range = range(max(0, bx - 1), min(nbx, bx + 2))
    y_range = range(max(0, by - 1), min(nby, by + 2))
    return [(jx, jy) for jx in x_range for jy in y_range]


def _block_points(bx: int, by: int) -> np.ndarray:
    """Gather points from the 3x3 block neighbourhood.

    Returns:
        Stacked point array of shape (3, n_points): rows are x, y, value.

    """
    c = _CTX
    cfg = c["cfg"]
    starts = cfg.starts
    out: list[np.ndarray] = []
    for jx, jy in _block_neighbours(bx, by, cfg.nbx, cfg.nby):
        lo = starts[jx * cfg.nby + jy]
        hi = starts[jx * cfg.nby + jy + 1]
        if hi > lo:
            out.append(np.asarray(c["pts"][:, lo:hi].copy()))
    return np.concatenate(out, axis=1) if out else np.empty((3, 0))


def encode_support_km(support_km: np.ndarray) -> np.ndarray:
    """Encode support in km to int16 DNs: clamp(round(log2(max(km, 1e-3)) * 8), ±32000).

    Returns:
        int16 array of support DNs. Inverse of decode_support_km.

    """
    enc = np.round(np.log2(np.maximum(support_km, SUPPORT_FLOOR_KM)) * SUPPORT_SCALE)
    return np.clip(enc, SUPPORT_DN_MIN, SUPPORT_DN_MAX).astype(np.int16)


def decode_support_km(dn: np.ndarray | int) -> np.ndarray | float:
    """Decode support DNs back to km: support_km = 2**(DN / 8).

    Returns:
        Support in km. Inverse of encode_support_km.

    """
    return 2.0 ** (np.asarray(dn, dtype=np.float64) / SUPPORT_SCALE)


def _check_percentile_step(pct_step: float | None) -> None:
    """Validate a percentile step: must be in (0, PERCENTILE_MAX].

    Raises:
        ValueError: If percentile_step is outside (0, 100].

    """
    if pct_step is not None and not (0 < pct_step <= PERCENTILE_MAX):
        msg = f"percentile_step must be in (0, {PERCENTILE_MAX:g}]: {pct_step}"
        raise ValueError(msg)


def _block_cell_range(bx: int, by: int, bsize: int, nx: int, ny: int) -> tuple[int, int, int, int]:
    """Output cell range (i0, j0, i1, j1) for a block, clipped to the grid.

    Returns:
        Tuple of (i0, j0, i1, j1) cell indices.

    """
    i0 = bx * bsize
    j0 = by * bsize
    i1 = min(i0 + bsize, nx)
    j1 = min(j0 + bsize, ny)
    return i0, j0, i1, j1


def _block_window(bx: int, by: int, bsize: int, nx: int, ny: int) -> Window:
    """Output raster window for a block (row-flipped, clipped to the grid).

    Returns:
        Rasterio Window covering the block's output cells.

    """
    i0, j0, i1, j1 = _block_cell_range(bx, by, bsize, nx, ny)
    return Window(i0, ny - j1, i1 - i0, j1 - j0)


def _process_block(
    args: tuple[int, int],
) -> tuple[int, int, tuple[np.ndarray, np.ndarray] | None]:
    """Interpolate a single output block using pull-push.

    Returns:
        Tuple of block coords and optional (value, support) grids; the grids
        are None when the block neighbourhood has no points.

    """
    bx, by = args
    ctx = _CTX
    cfg = ctx["cfg"]
    step = 1 << cfg.levels
    i0, j0, i1, j1 = _block_cell_range(bx, by, cfg.bsize, cfg.nx, cfg.ny)

    # Snap halo origin to multiple of step for exact alignment
    hi0 = max(0, ((i0 - cfg.halo) // step) * step)
    hj0 = max(0, ((j0 - cfg.halo) // step) * step)
    hi1 = min(cfg.nx_padded, -(-(i1 + cfg.halo) // step) * step)
    hj1 = min(cfg.ny_padded, -(-(j1 + cfg.halo) // step) * step)

    sel = _block_points(bx, by)
    if sel.shape[1] == 0:
        return bx, by, None

    ix = ((sel[0] - cfg.x0) // cfg.res).astype(np.int64)
    iy = ((sel[1] - cfg.y0) // cfg.res).astype(np.int64)
    m = (ix >= hi0) & (ix < hi1) & (iy >= hj0) & (iy < hj1)
    if not m.any():
        return bx, by, None

    iloc = ix[m] - hi0
    jloc = iy[m] - hj0
    tv = sel[2][m]

    s, c_grid = bin_points(iloc, jloc, tv, hi1 - hi0, hj1 - hj0)
    val, sup = pull_push(s, c_grid, cfg.res, cfg.levels, saturation=cfg.sat)

    # Mask from exact radius query
    cap_cells = round(cfg.cap_km * 1000.0 / cfg.res)
    near = box_count(c_grid, cap_cells) > 0

    # Extract output window
    a0 = i0 - hi0
    b0 = j0 - hj0
    sl = (slice(a0, a0 + (i1 - i0)), slice(b0, b0 + (j1 - j0)))
    v_out = val[sl]
    r_out = sup[sl] / 1000.0
    near_out = near[sl]

    # Transform: interpolate space -> raw -> percentile
    pv = ctx["pct"].fwd(ctx["tf"].inv(v_out))
    if cfg.pct_step is not None:
        # Round to the nearest step (e.g. 5 -> 90/95/100), clamp to 0-100.
        pv = np.clip(np.round(pv / cfg.pct_step) * cfg.pct_step, 0.0, PERCENTILE_MAX)
    vq = np.where(near_out, np.round(pv * cfg.scale), NODATA).astype(np.int16)
    rq = np.where(near_out, encode_support_km(r_out), NODATA).astype(np.int16)

    return bx, by, (vq, rq)


def _block_neighbourhood_nonempty(
    bx: int,
    by: int,
    starts: np.ndarray,
    nbx: int,
    nby: int,
) -> bool:
    """Report whether the 3x3 block neighbourhood holds at least one point.

    Returns:
        Whether the neighbourhood is non-empty.

    """
    return any(starts[jx * nby + jy + 1] > starts[jx * nby + jy] for jx, jy in _block_neighbours(bx, by, nbx, nby))


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
        calib_max_points: int = DEFAULT_CALIB_MAX_POINTS,
        src_crs: int = SRC_CRS,
        work_crs: int = WORK_CRS,
        out_crs: int = OUT_CRS,
        skip_calibration: bool = False,
    ) -> None:
        """Initialise the interpolation pipeline.

        Raises:
            ValueError: If scale * 100 exceeds int16 max.
            ValueError: If percentile_step is outside (0, 100].
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
        _check_percentile_step(percentile_step)
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
                f"scale * {PERCENTILE_MAX:g} exceeds int16 max: {self.scale * PERCENTILE_MAX} > {INT16_MAX}. "
                "Reduce scale."
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

        If the requested transform is invalid for the data (e.g. log10 with
        negative values), a warning is issued and identity is used instead.
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
                "cap_km": float(self.cap_km) if self.cap_km != "auto" else DEFAULT_CAP_KM,
            }

        self.tname = (
            self.transform if self.transform != "auto" else (cal.get("transform", "identity") if cal else "identity")
        )
        if cal and "transform_state" in cal and cal["transform_state"].get("name") == self.tname:
            self.tf = make_transform(cal["transform_state"])
        else:
            self.tf = next(t for t in transforms() if t.name == self.tname).fit(self.v)

        # Same valid() guard choose_transform applies: a forced transform can
        # be invalid for this data (log10 on negatives -> NaN -> silent all-0
        # surface). Reject the transform, fall back to identity with a warning.
        if not self.tf.valid(self.v):
            msg = (
                f"transform {self.tname!r} is invalid for this data "
                f"(min={np.min(self.v):.6g}); falling back to identity"
            )
            warnings.warn(msg, stacklevel=2)
            self.tname = "identity"
            self.tf = next(t for t in transforms() if t.name == "identity")

        # Keep the worker-side transform in sync with the fitted transform
        # (a forced transform can differ from the calibration's choice).
        if cal is not None:
            cal["transform_state"] = self.tf.state()

        self.pct_q = (
            PercentileTransform(cal["percentile_quantiles"])
            if cal and "percentile_quantiles" in cal
            else PercentileTransform().fit(self.v)
        )

        self.cap_km_val = (
            float(self.cap_km)
            if self.cap_km != "auto"
            else float(cal.get("cap_km", DEFAULT_CAP_KM) if cal else DEFAULT_CAP_KM)
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

        self.levels = max(1, math.ceil(math.log2(max(self.cap_km_val * 1000.0 / self.res, 2.0))))
        self.step = 1 << self.levels
        self.halo = max(math.ceil(self.cap_km_val * 1000.0 / self.res) + 2, self.step)
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

        # Memmap for workers (3 bands: x, y, value in transform space)
        self.pts_path = Path(self.out_dir) / "_points.npy"
        pts = np.lib.format.open_memmap(self.pts_path, mode="w+", dtype=np.float64, shape=(3, self.n))
        pts[0] = self.x[order]
        pts[1] = self.y[order]
        pts[2] = self.tv[order]
        pts.flush()

        # Task list: one pass over the blocks, split into work / empty.
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

    def _write_rasters(self) -> tuple[str, str]:
        """Interpolate all blocks and write the value/support GeoTIFFs.

        Runs after grid(); the block loop is the hot phase, exposed
        separately for benchmarking.

        Returns:
            Tuple of value and support GeoTIFF file paths.

        Raises:
            OSError: If a raster cannot be written or reprojected.
            RasterioError: If a raster cannot be opened or reprojected.

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
            sd.update_tags(decode=f"support_km = 2**(DN/{SUPPORT_SCALE})")

            # Write empty blocks as nodata
            for bx, by in self.empty_blocks:
                w = _block_window(bx, by, self.bsize, self.nx, self.ny)
                blank = np.full((w.height, w.width), NODATA, np.int16)
                vd.write(blank, 1, window=w)
                sd.write(blank, 1, window=w)

            # Process blocks with parallel workers
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Setting the shape on a NumPy array")
                # Shared read-only context for the worker threads: all threads
                # read this one module dict; nothing is written after this point.
                _CTX.update(
                    {
                        "pts": np.load(self.cfg.pts_path, mmap_mode="r"),
                        "tf": make_transform(self.cfg.transform_state),
                        "pct": PercentileTransform(self.cfg.pct_quantiles),
                        "cfg": self.cfg,
                    },
                )
                with ThreadPoolExecutor(max_workers=self.workers) as ex:
                    for bx, by, out in ex.map(_process_block, self.tasks):
                        w = _block_window(bx, by, self.bsize, self.nx, self.ny)

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

            if self.out_crs != self.work_crs:
                tmp_v = Path(self.out_dir) / "_value_tmp.tif"
                tmp_s = Path(self.out_dir) / "_support_tmp.tif"
                vpath.rename(tmp_v)
                spath.rename(tmp_s)
                try:
                    _reproject_band(tmp_v, vpath, vprof, f"EPSG:{self.out_crs}")
                    _reproject_band(tmp_s, spath, sprof, f"EPSG:{self.out_crs}")
                except (OSError, RasterioError):
                    # Reprojection failed: restore the working-CRS rasters so
                    # the caller still gets valid (unprojected) output.
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


def _reproject_band(src_path: Path, dst_path: Path, profile: dict[str, Any], dst_crs: str) -> None:
    """Reproject one int16 band tile-by-tile into `dst_crs`.

    Args:
        src_path: Source GeoTIFF (working CRS).
        dst_path: Destination GeoTIFF path.
        profile: Raster profile (nodata, compression, ...); width, height,
            transform and crs are overwritten from the source bounds.
        dst_crs: Destination CRS, e.g. "EPSG:3857".

    """
    with rasterio.open(src_path) as src:
        dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
        )
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
            for j in range(0, dst_height, TILE_PX):
                for i in range(0, dst_width, TILE_PX):
                    w_h = min(TILE_PX, dst_height - j)
                    w_w = min(TILE_PX, dst_width - i)
                    dst_w = Window(i, j, w_w, w_h)
                    # Compute local transform for this window
                    w_bounds = bounds(dst_w, dst_transform)
                    local_dst_transform = from_bounds(*w_bounds, w_w, w_h)
                    dst_arr = np.zeros((w_h, w_w), dtype=np.int16)
                    reproject(
                        rasterio.band(src, 1),
                        dst_arr,
                        src_transform=src.transform,
                        dst_transform=local_dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest,
                        nodata=NODATA,
                    )
                    dst.write(dst_arr, 1, window=dst_w)


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
    """Parse CLI arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="Pull-push scattered-data interpolation")
    parser.add_argument("--version", action="version", version=f"ppgrid {_pkg_version('ppgrid')}")
    parser.add_argument("input", help="CSV or Parquet input path")
    parser.add_argument("-o", "--out", required=True, help="Output directory (required)")
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
    parser.add_argument("--calib-max-points", type=int, default=DEFAULT_CALIB_MAX_POINTS)
    parser.add_argument("--calibration", default=None, help="Calibration JSON path")
    parser.add_argument("--src-crs", type=int, default=SRC_CRS)
    parser.add_argument(
        "--work-crs", type=int, default=WORK_CRS, help=f"Working CRS for interpolation (default {WORK_CRS})"
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


if __name__ == "__main__":
    main()
