"""idwgrid — CLI driver for pull-push scattered-data interpolation.

Turns scattered geolocated point values into a continent-scale, gapless,
capped raster surface, in minutes, on a single machine, with no GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
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

WORK_CRS: int = 3577
NODATA: int = -32768

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
    baseline: bool
    scale: float


def _init_worker(cfg: _WorkerConfig) -> None:
    """Initialise per-worker context."""
    global _CTX
    _CTX = {
        "pts": np.load(cfg.pts_path, mmap_mode="r"),
        "tf": make_transform(cfg.transform_state),
        "pct": PercentileTransform(cfg.pct_quantiles),
        "starts": cfg.starts,
        "nbx": cfg.nbx,
        "nby": cfg.nby,
        "bsize": cfg.bsize,
        "halo": cfg.halo,
        "res": cfg.res,
        "levels": cfg.levels,
        "cap_km": cfg.cap_km,
        "x0": cfg.x0,
        "y0": cfg.y0,
        "nx": cfg.nx,
        "ny": cfg.ny,
        "nx_padded": cfg.nx_padded,
        "ny_padded": cfg.ny_padded,
        "sat": cfg.sat,
        "baseline": cfg.baseline,
        "scale": cfg.scale,
    }


def _block_points(bx: int, by: int) -> np.ndarray:
    """Gather points from the 3x3 block neighbourhood (valid while halo <= bsize).

    Returns:
        Concatenated point array of shape (4, n_points).

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
    """Process a single block.

    Returns:
        Tuple of (bx, by, output) where output is (vq, rq) or None.

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
    """Check if any of the 3x3 neighbourhood has points.

    Returns:
        True if the block neighbourhood contains data.

    """
    for jx in range(max(0, bx - 1), min(nbx, bx + 2)):
        for jy in range(max(0, by - 1), min(nby, by + 2)):
            lo = starts[jx * nby + jy]
            hi = starts[jx * nby + jy + 1]
            if hi > lo:
                return True
    return False


def run(
    input_path: str,
    value_col: str,
    lng_col: str,
    lat_col: str,
    out_dir: str,
    res: float = 500.0,
    cap_km: float | str = "auto",
    transform: str = "auto",
    saturation: float = 1.0,
    block_size: int = 8192,
    workers: int = 4,
    calib_path: str | None = None,
    *,
    baseline: bool = False,
    scale: float = 100.0,
    compress: str = "ZSTD",
    calib_max_points: int = 2_000_000,
    src_crs: int = 4326,
    skip_calibration: bool = False,
) -> tuple[str, str]:
    """Full pipeline: load, calibrate, interpolate, write.

    Returns:
        Tuple of (value_tiff_path, support_tiff_path).

    """
    # --- Ingest ---
    if input_path.endswith((".parquet", ".pq")):
        df = pd.read_parquet(input_path, columns=[value_col, lng_col, lat_col])
    else:
        df = pd.read_csv(input_path, usecols=[value_col, lng_col, lat_col])

    v = df[value_col].to_numpy(np.float64)
    lon = df[lng_col].to_numpy(np.float64)
    lat = df[lat_col].to_numpy(np.float64)

    good = np.isfinite(v) & np.isfinite(lon) & np.isfinite(lat)
    v, lon, lat = v[good], lon[good], lat[good]
    n = len(v)

    tr = Transformer.from_crs(src_crs, WORK_CRS, always_xy=True)
    x, y = tr.transform(lon, lat)
    x = np.asarray(x)
    y = np.asarray(y)

    # --- Calibration ---
    cal: dict[str, Any] | None = None
    cpath_obj = Path(calib_path) if calib_path else None
    if cpath_obj and cpath_obj.exists():
        with cpath_obj.open(encoding="utf-8") as f:
            cal = json.load(f)
    elif not skip_calibration:
        if n > calib_max_points:
            sub = np.random.default_rng(0).choice(n, calib_max_points, replace=False)
            cx, cy, cv = x[sub], y[sub], v[sub]
        else:
            cx, cy, cv = x, y, v

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
            "cv": {str(k): {"overall_skill": d["overall_skill"], "cap_km": d.get("cap_km")} for k, d in detail.items()},
        }
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        cal_path = cpath_obj or Path(out_dir) / "calibration.json"
        with cal_path.open("w", encoding="utf-8") as f:
            json.dump(cal, f, indent=2)
    else:
        cal = {
            "transform": "identity",
            "cap_km": float(cap_km) if cap_km != "auto" else 64.0,
        }

    tname = transform if transform != "auto" else (cal["transform"] if cal else "identity")
    if cal and "transform_state" in cal and cal["transform_state"].get("name") == tname:
        tf = make_transform(cal["transform_state"])
    else:
        tf = next(t for t in transforms() if t.name == tname).fit(v)

    pct_q = (
        PercentileTransform(cal["percentile_quantiles"])
        if cal and "percentile_quantiles" in cal
        else PercentileTransform().fit(v)
    )

    cap_km_val = float(cap_km) if cap_km != "auto" else float(cal["cap_km"] if cal else 64.0)

    tv = tf.fwd(v)

    # --- Grid geometry ---
    x0 = x.min()
    y0 = y.min()
    nx = int((x.max() - x0) // res) + 1
    ny = int((y.max() - y0) // res) + 1

    levels = max(1, math.ceil(math.log2(max(cap_km_val * 1000.0 / res, 2.0))))
    step = 1 << levels
    halo = max(math.ceil(cap_km_val * 1000.0 / res) + 2, step)
    nx_padded = -(-nx // step) * step
    ny_padded = -(-ny // step) * step
    bsize = max(block_size, step)
    nbx = math.ceil(nx / bsize)
    nby = math.ceil(ny / bsize)

    # --- Spatial index ---
    bx_ = np.clip(((x - x0) // res // bsize).astype(np.int64), 0, nbx - 1)
    by_ = np.clip(((y - y0) // res // bsize).astype(np.int64), 0, nby - 1)
    bid = bx_ * nby + by_
    order = np.argsort(bid, kind="stable")
    starts = np.searchsorted(bid[order], np.arange(nbx * nby + 1))

    # Memmap for workers
    pts_path = Path(out_dir) / "_points.npy"
    pts = np.lib.format.open_memmap(pts_path, mode="w+", dtype=np.float64, shape=(4, n))
    pts[0] = x[order]
    pts[1] = y[order]
    pts[2] = tv[order]
    pts[3] = v[order]
    pts.flush()

    # --- Task list ---
    tasks: list[tuple[int, int]] = [
        (i, j) for i in range(nbx) for j in range(nby) if _block_neighbourhood_nonempty(i, j, starts, nbx, nby)
    ]
    empty_blocks: list[tuple[int, int]] = [
        (i, j) for i in range(nbx) for j in range(nby) if not _block_neighbourhood_nonempty(i, j, starts, nbx, nby)
    ]

    # --- Worker config ---
    cfg = _WorkerConfig(
        pts_path=str(pts_path),
        transform_state=cal.get("transform_state", {"name": tname}),
        pct_quantiles=cal.get("percentile_quantiles", pct_q.q.tolist()),
        starts=starts.tolist(),
        nbx=nbx,
        nby=nby,
        bsize=bsize,
        halo=halo,
        res=res,
        levels=levels,
        cap_km=cap_km_val,
        x0=float(x0),
        y0=float(y0),
        nx=nx,
        ny=ny,
        nx_padded=nx_padded,
        ny_padded=ny_padded,
        sat=saturation,
        baseline=baseline,
        scale=scale,
    )

    # --- Raster profile ---
    xform = from_origin(x0, y0 + ny * res, res, res)
    common = {
        "driver": "GTiff",
        "height": ny,
        "width": nx,
        "count": 1,
        "crs": f"EPSG:{WORK_CRS}",
        "transform": xform,
        "compress": compress,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
    }
    vprof = dict(common, dtype="int16", nodata=NODATA, predictor=2)
    sprof = dict(common, dtype="int16", nodata=NODATA, predictor=2)

    vpath = Path(out_dir) / "value.tif"
    spath = Path(out_dir) / "support_km.tif"

    t_start = time.perf_counter()

    with (
        rasterio.open(vpath, "w", **vprof) as vd,
        rasterio.open(spath, "w", **sprof) as sd,
    ):
        vd.update_tags(
            transform=tname,
            cap_km=str(cap_km_val),
            res_m=str(res),
            scale=str(scale),
            units="percentile",
            decode=f"percentile = DN/{scale:g}",
        )
        vd.scales = (1.0 / scale,)
        sd.update_tags(decode="support_km = 2**(DN/8)")

        # Write empty blocks as nodata
        for bx, by in empty_blocks:
            i0, j0 = bx * bsize, by * bsize
            i1, j1 = min(i0 + bsize, nx), min(j0 + bsize, ny)
            w = Window(i0, ny - j1, i1 - i0, j1 - j0)
            blank = np.full((w.height, w.width), NODATA, np.int16)
            vd.write(blank, 1, window=w)
            sd.write(blank, 1, window=w)

        # Process blocks with parallel workers
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(cfg,),
        ) as ex:
            for bx, by, out in ex.map(_process_block, tasks, chunksize=1):
                i0, j0 = bx * bsize, by * bsize
                i1, j1 = min(i0 + bsize, nx), min(j0 + bsize, ny)
                w = Window(i0, ny - j1, i1 - i0, j1 - j0)

                if out is None:
                    blank = np.full((w.height, w.width), NODATA, np.int16)
                    vd.write(blank, 1, window=w)
                    sd.write(blank, 1, window=w)
                    continue

                vq, rq = out
                vd.write(vq.T[::-1, :], 1, window=w)
                sd.write(rq.T[::-1, :], 1, window=w)

    time.perf_counter() - t_start

    # Cleanup memmap
    pts_path.unlink()
    return str(vpath), str(spath)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Pull-push scattered-data interpolation")
    parser.add_argument("input", help="CSV or Parquet input path")
    parser.add_argument("-o", "--out", default="outputs/", help="Output directory")
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
    parser.add_argument("--block", type=int, default=8192, help="Block size in cells")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--scale", type=float, default=100.0, help="DN = percentile * scale")
    parser.add_argument("--compress", default="ZSTD")
    parser.add_argument("--calib-max-points", type=int, default=2_000_000)
    parser.add_argument("--calibration", default=None, help="Calibration JSON path")
    parser.add_argument("--src-crs", type=int, default=4326)
    parser.add_argument("--skip-calibration", action="store_true", help="Skip calibration, use defaults")
    args = parser.parse_args()

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
        baseline=args.baseline,
        scale=args.scale,
        compress=args.compress,
        calib_max_points=args.calib_max_points,
        src_crs=args.src_crs,
        skip_calibration=args.skip_calibration,
    )


if __name__ == "__main__":
    main()
