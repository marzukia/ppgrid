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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

from .calibrate import (
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
NODATA: int = -32768
INT16_MAX: int = 32767

# Per-worker context (module-level so it's local to each process)
_CTX: dict[str, Any] = {}


@dataclass
class _WorkerConfig:
    """Configuration passed to each worker process."""

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


def _block_points(bx: int, by: int) -> np.ndarray:
    """Gather points from the 3x3 block neighbourhood.

    Returns:
        Stacked point array of shape (4, n_points).

    """
    c = _CTX
    out: list[np.ndarray] = []
    for jx in range(max(0, bx - 1), min(c["nbx"], bx + 2)):
        for jy in range(max(0, by - 1), min(c["nby"], by + 2)):
            lo = c["starts"][jx * c["nby"] + jy]
            hi = c["starts"][jx * c["nby"] + jy + 1]
            if hi > lo:
                out.append(np.asarray(c["pts"][:, lo:hi].copy()))
    return np.concatenate(out, axis=1) if out else np.empty((4, 0))


def _process_block(
    args: tuple[int, int],
) -> tuple[int, int, tuple[np.ndarray, np.ndarray] | None]:
    """Interpolate a single output block using pull-push. Returns nodata if the block neighbourhood has no points.

    Returns:
        Tuple of block coords and optional (value, support) grids.

    """
    bx, by = args
    c = _CTX
    res = c["res"]
    halo = c["halo"]
    bsize = c["bsize"]
    step = 1 << c["levels"]
    i0 = bx * bsize
    j0 = by * bsize
    i1 = min(i0 + bsize, c["nx"])
    j1 = min(j0 + bsize, c["ny"])

    # Snap halo origin to multiple of step for exact alignment
    hi0 = max(0, ((i0 - halo) // step) * step)
    hj0 = max(0, ((j0 - halo) // step) * step)
    hi1 = min(c["nx_padded"], -(-(i1 + halo) // step) * step)
    hj1 = min(c["ny_padded"], -(-(j1 + halo) // step) * step)

    sel = _block_points(bx, by)
    if sel.shape[1] == 0:
        return bx, by, None

    ix = ((sel[0] - c["x0"]) // res).astype(np.int64)
    iy = ((sel[1] - c["y0"]) // res).astype(np.int64)
    m = (ix >= hi0) & (ix < hi1) & (iy >= hj0) & (iy < hj1)
    if not m.any():
        return bx, by, None

    iloc = ix[m] - hi0
    jloc = iy[m] - hj0
    tv = sel[2][m]

    s, c_grid = bin_points(iloc, jloc, tv, hi1 - hi0, hj1 - hj0)
    val, sup = pull_push(s, c_grid, res, c["levels"], saturation=c["sat"])

    # Mask from exact radius query
    cap_cells = round(c["cap_km"] * 1000.0 / res)
    near = box_count(c_grid, cap_cells) > 0

    # Extract output window
    a0 = i0 - hi0
    b0 = j0 - hj0
    sl = (slice(a0, a0 + (i1 - i0)), slice(b0, b0 + (j1 - j0)))
    v_out = val[sl]
    r_out = sup[sl] / 1000.0
    near_out = near[sl]

    # Transform: interpolate space -> raw -> percentile
    scale = c["scale"]
    pv = c["pct"].fwd(c["tf"].inv(v_out))
    vq = np.where(near_out, np.round(pv * scale), NODATA).astype(np.int16)
    rq = np.where(
        near_out,
        np.clip(np.round(np.log2(np.maximum(r_out, 1e-3)) * 8), -32000, 32000),
        NODATA,
    ).astype(np.int16)

    return bx, by, (vq, rq)


def _block_neighbourhood_nonempty(
    bx: int,
    by: int,
    starts: np.ndarray,
    nbx: int,
    nby: int,
) -> bool:
    for jx in range(max(0, bx - 1), min(nbx, bx + 2)):
        for jy in range(max(0, by - 1), min(nby, by + 2)):
            lo = starts[jx * nby + jy]
            hi = starts[jx * nby + jy + 1]
            if hi > lo:
                return True
    return False


class Pipeline:
    """Pull-push interpolation pipeline."""

    def __init__(
        self,
        input_path: str,
        value_col: str,
        lng_col: str,
        lat_col: str,
        out_dir: str,
        res: float = 500.0,
        cap_km: float | str = "auto",
        transform: str = "auto",
        saturation: float = 1.0,
        block_size: int = 2048,
        workers: int = 4,
        calib_path: str | None = None,
        *,
        scale: float = 100.0,
        compress: str = "ZSTD",
        calib_max_points: int = 2_000_000,
        src_crs: int = 4326,
        work_crs: int = 6933,
        out_crs: int = 3857,
        skip_calibration: bool = False,
    ) -> None:
        """Initialise the interpolation pipeline.

        Raises:
            ValueError: If scale * 100 exceeds int16 max.

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
        self.calib_max_points = calib_max_points
        self.src_crs = src_crs
        self.work_crs = work_crs
        self.out_crs = out_crs
        self.skip_calibration = skip_calibration

        if self.scale * 100 > INT16_MAX:
            msg = f"scale * 100 exceeds int16 max: {self.scale * 100} > {INT16_MAX}. Reduce scale."
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
        """Load or run calibration. Sets transform, percentile, cap."""
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
                "cap_km": float(self.cap_km) if self.cap_km != "auto" else 64.0,
            }

        self.tname = (
            self.transform if self.transform != "auto" else (cal.get("transform", "identity") if cal else "identity")
        )
        if cal and "transform_state" in cal and cal["transform_state"].get("name") == self.tname:
            self.tf = make_transform(cal["transform_state"])
        else:
            self.tf = next(t for t in transforms() if t.name == self.tname).fit(self.v)

        self.pct_q = (
            PercentileTransform(cal["percentile_quantiles"])
            if cal and "percentile_quantiles" in cal
            else PercentileTransform().fit(self.v)
        )

        self.cap_km_val = (
            float(self.cap_km) if self.cap_km != "auto" else float(cal.get("cap_km", 64.0) if cal else 64.0)
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

        # Memmap for workers
        self.pts_path = Path(self.out_dir) / "_points.npy"
        pts = np.lib.format.open_memmap(self.pts_path, mode="w+", dtype=np.float64, shape=(4, self.n))
        pts[0] = self.x[order]
        pts[1] = self.y[order]
        pts[2] = self.tv[order]
        pts[3] = self.v[order]
        pts.flush()

        # Task list
        self.tasks: list[tuple[int, int]] = [
            (i, j)
            for i in range(self.nbx)
            for j in range(self.nby)
            if _block_neighbourhood_nonempty(i, j, self.starts, self.nbx, self.nby)
        ]
        self.empty_blocks: list[tuple[int, int]] = [
            (i, j)
            for i in range(self.nbx)
            for j in range(self.nby)
            if not _block_neighbourhood_nonempty(i, j, self.starts, self.nbx, self.nby)
        ]

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
        )

    def run(self) -> tuple[str, str]:
        """Execute the full pipeline.

        Returns:
            Tuple of value and support GeoTIFF file paths.

        """
        self.ingest()
        self.calibrate()
        self.grid()

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
            "blockxsize": 512,
            "blockysize": 512,
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
            vd.scales = (1.0 / self.scale,)
            sd.update_tags(decode="support_km = 2**(DN/8)")

            # Write empty blocks as nodata
            for bx, by in self.empty_blocks:
                i0, j0 = bx * self.bsize, by * self.bsize
                i1, j1 = min(i0 + self.bsize, self.nx), min(j0 + self.bsize, self.ny)
                w = Window(i0, self.ny - j1, i1 - i0, j1 - j0)
                blank = np.full((w.height, w.width), NODATA, np.int16)
                vd.write(blank, 1, window=w)
                sd.write(blank, 1, window=w)

            # Process blocks with parallel workers
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Setting the shape on a NumPy array")
                _CTX.update(
                    {
                        "pts": np.load(self.cfg.pts_path, mmap_mode="r"),
                        "tf": make_transform(self.cfg.transform_state),
                        "pct": PercentileTransform(self.cfg.pct_quantiles),
                        "starts": self.cfg.starts,
                        "nbx": self.cfg.nbx,
                        "nby": self.cfg.nby,
                        "bsize": self.cfg.bsize,
                        "halo": self.cfg.halo,
                        "res": self.cfg.res,
                        "levels": self.cfg.levels,
                        "cap_km": self.cfg.cap_km,
                        "x0": self.cfg.x0,
                        "y0": self.cfg.y0,
                        "nx": self.cfg.nx,
                        "ny": self.cfg.ny,
                        "nx_padded": self.cfg.nx_padded,
                        "ny_padded": self.cfg.ny_padded,
                        "sat": self.cfg.sat,
                        "scale": self.cfg.scale,
                    },
                )
                with ThreadPoolExecutor(max_workers=self.workers) as ex:
                    for bx, by, out in ex.map(_process_block, self.tasks):
                        i0, j0 = bx * self.bsize, by * self.bsize
                        i1, j1 = min(i0 + self.bsize, self.nx), min(j0 + self.bsize, self.ny)
                        w = Window(i0, self.ny - j1, i1 - i0, j1 - j0)

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

                    def _reproject_band(
                        src_path: str,
                        dst_path: str,
                        profile: dict[str, Any],
                        dst_crs: str,
                    ) -> None:
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
                                from rasterio.transform import from_bounds

                                tile = 512
                                for j in range(0, dst_height, tile):
                                    for i in range(0, dst_width, tile):
                                        w_h = min(tile, dst_height - j)
                                        w_w = min(tile, dst_width - i)
                                        dst_w = rasterio.windows.Window(i, j, w_w, w_h)
                                        # Compute local transform for this window
                                        w_bounds = rasterio.windows.bounds(dst_w, dst_transform)
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

                    _reproject_band(str(tmp_v), str(vpath), vprof, f"EPSG:{self.out_crs}")
                    _reproject_band(str(tmp_s), str(spath), sprof, f"EPSG:{self.out_crs}")
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
    """Parse CLI arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="Pull-push scattered-data interpolation")
    parser.add_argument("--version", action="version", version="ppgrid 0.1.4")
    parser.add_argument("input", help="CSV or Parquet input path")
    parser.add_argument("-o", "--out", default="examples/", help="Output directory")
    parser.add_argument("--value-col", default="value", help="Value column name")
    parser.add_argument("--lng-col", default="longitude", help="Longitude column name")
    parser.add_argument("--lat-col", default="latitude", help="Latitude column name")
    parser.add_argument("--res", type=float, default=500.0, help="Cell size in metres")
    parser.add_argument("--cap-km", default="auto", help="Fill cap km, or 'auto'")
    parser.add_argument(
        "--transform",
        default="auto",
        choices=["auto", "identity", "log10", "sqrt", "percentile"],
    )
    parser.add_argument(
        "--saturation",
        type=float,
        default=1.0,
        help="Counts for a cell to fully self-trust",
    )
    parser.add_argument("--block", type=int, default=2048, help="Block size in cells")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--scale", type=float, default=100.0, help="DN = percentile * scale")
    parser.add_argument("--compress", default="ZSTD")
    parser.add_argument("--calib-max-points", type=int, default=2_000_000)
    parser.add_argument("--calibration", default=None, help="Calibration JSON path")
    parser.add_argument("--src-crs", type=int, default=4326)
    parser.add_argument("--work-crs", type=int, default=6933, help="Working CRS for interpolation (default 6933)")
    parser.add_argument("--out-crs", type=int, default=3857, help="Output CRS (default 3857)")
    parser.add_argument("--skip-calibration", action="store_true", help="Skip calibration, use defaults")
    args = parser.parse_args()

    if args.res <= 0:
        parser.error("--res must be positive")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.scale <= 0:
        parser.error("--scale must be positive")
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
        compress=args.compress,
        calib_max_points=args.calib_max_points,
        src_crs=args.src_crs,
        work_crs=args.work_crs,
        out_crs=args.out_crs,
        skip_calibration=args.skip_calibration,
    )


if __name__ == "__main__":
    main()
