"""
idwgrid — CLI driver for pull-push scattered-data interpolation.

Turns scattered geolocated point values into a continent-scale, gapless,
capped raster surface, in minutes, on a single machine, with no GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

import numpy as np
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


def _init_worker(
    pts_path: str,
    transform_state: dict[str, Any],
    pct_quantiles: list[float],
    starts: list[int],
    nbx: int,
    nby: int,
    bsize: int,
    halo: int,
    res: float,
    levels: int,
    cap_km: float,
    x0: float,
    y0: float,
    NX: int,
    NY: int,
    NXp: int,
    NYp: int,
    sat: float,
    baseline: bool,
    scale: float,
) -> None:
    """Initialise per-worker context."""
    global _CTX
    _CTX = {
        "pts": np.load(pts_path, mmap_mode="r"),
        "tf": make_transform(transform_state),
        "pct": PercentileTransform(pct_quantiles),
        "starts": starts,
        "nbx": nbx,
        "nby": nby,
        "bsize": bsize,
        "halo": halo,
        "res": res,
        "levels": levels,
        "cap_km": cap_km,
        "x0": x0,
        "y0": y0,
        "NX": NX,
        "NY": NY,
        "NXp": NXp,
        "NYp": NYp,
        "sat": sat,
        "baseline": baseline,
        "scale": scale,
    }


def _block_points(bx: int, by: int) -> np.ndarray:
    """Gather points from the 3x3 block neighbourhood (valid while halo <= bsize)."""
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
    """Process a single block. Returns (bx, by, (Vq, Rq) or None)."""
    bx, by = args
    c = _CTX
    res = c["res"]
    halo = c["halo"]
    bsize = c["bsize"]
    step = 1 << c["levels"]
    i0 = bx * bsize
    j0 = by * bsize
    i1 = min(i0 + bsize, c["NX"])
    j1 = min(j0 + bsize, c["NY"])

    # Snap halo origin to multiple of step for exact alignment
    hi0 = max(0, ((i0 - halo) // step) * step)
    hj0 = max(0, ((j0 - halo) // step) * step)
    hi1 = min(c["NXp"], -(-(i1 + halo) // step) * step)
    hj1 = min(c["NYp"], -(-(j1 + halo) // step) * step)

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

    S, C = bin_points(iloc, jloc, tv, hi1 - hi0, hj1 - hj0)
    V, R = pull_push(S, C, res, c["levels"], saturation=c["sat"])

    # Mask from exact radius query
    cap_cells = round(c["cap_km"] * 1000.0 / res)
    near = box_count(C, cap_cells) > 0

    # Extract output window
    a0 = i0 - hi0
    b0 = j0 - hj0
    sl = (slice(a0, a0 + (i1 - i0)), slice(b0, b0 + (j1 - j0)))
    V_out = V[sl]
    R_out = R[sl] / 1000.0
    near_out = near[sl]

    # Transform: interpolate space -> raw -> percentile
    scale = c["scale"]
    pv = c["pct"].fwd(c["tf"].inv(V_out))
    Vq = np.where(near_out, np.round(pv * scale), NODATA).astype(np.int16)
    Rq = np.where(
        near_out,
        np.clip(np.round(np.log2(np.maximum(R_out, 1e-3)) * 8), -32000, 32000),
        NODATA,
    ).astype(np.int16)

    return bx, by, (Vq, Rq)


def _block_neighbourhood_nonempty(
    bx: int,
    by: int,
    starts: np.ndarray,
    nbx: int,
    nby: int,
) -> bool:
    """True if any of the 3x3 neighbourhood has points."""
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
    baseline: bool = False,
    scale: float = 100.0,
    compress: str = "ZSTD",
    calib_max_points: int = 2_000_000,
    src_crs: int = 4326,
    skip_calibration: bool = False,
) -> tuple[str, str]:
    """Full pipeline: load, calibrate, interpolate, write."""

    # --- Ingest ---
    print("Reading input...", file=sys.stderr)
    if input_path.endswith((".parquet", ".pq")):
        import pandas as pd

        df = pd.read_parquet(input_path, columns=[value_col, lng_col, lat_col])
    else:
        import pandas as pd

        df = pd.read_csv(input_path, usecols=[value_col, lng_col, lat_col])

    v = df[value_col].to_numpy(np.float64)
    lon = df[lng_col].to_numpy(np.float64)
    lat = df[lat_col].to_numpy(np.float64)

    good = np.isfinite(v) & np.isfinite(lon) & np.isfinite(lat)
    v, lon, lat = v[good], lon[good], lat[good]
    N = len(v)
    print(f"  {N:,} valid points", file=sys.stderr)

    tr = Transformer.from_crs(src_crs, WORK_CRS, always_xy=True)
    x, y = tr.transform(lon, lat)
    x = np.asarray(x)
    y = np.asarray(y)

    # --- Calibration ---
    cpath = calib_path
    cal: dict[str, Any] | None = None
    if cpath and os.path.exists(cpath):
        print(f"Load calibration from {cpath}", file=sys.stderr)
        with open(cpath) as f:
            cal = json.load(f)
    elif not skip_calibration:
        print("Calibrating...", file=sys.stderr)
        n = len(v)
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
            "cv": {
                str(k): {"overall_skill": d["overall_skill"], "cap_km": d.get("cap_km")}
                for k, d in detail.items()
            },
        }
        os.makedirs(out_dir, exist_ok=True)
        with open(cpath or os.path.join(out_dir, "calibration.json"), "w") as f:
            json.dump(cal, f, indent=2)
    else:
        print("Skip calibration, use defaults...", file=sys.stderr)
        cal = {
            "transform": "identity",
            "cap_km": float(cap_km) if cap_km != "auto" else 64.0,
        }

    tname = (
        transform if transform != "auto" else (cal["transform"] if cal else "identity")
    )
    if cal and "transform_state" in cal and cal["transform_state"].get("name") == tname:
        tf = make_transform(cal["transform_state"])
    else:
        tf = next(t for t in transforms() if t.name == tname).fit(v)

    pct_q = (
        PercentileTransform(cal["percentile_quantiles"])
        if cal and "percentile_quantiles" in cal
        else PercentileTransform().fit(v)
    )

    cap_km_val = (
        float(cap_km) if cap_km != "auto" else float(cal["cap_km"] if cal else 64.0)
    )
    print(f"  Transform: {tname}, cap: {cap_km_val} km", file=sys.stderr)

    tv = tf.fwd(v)

    # --- Grid geometry ---
    x0 = x.min()
    y0 = y.min()
    NX = int((x.max() - x0) // res) + 1
    NY = int((y.max() - y0) // res) + 1

    levels = max(1, math.ceil(math.log2(max(cap_km_val * 1000.0 / res, 2.0))))
    step = 1 << levels
    halo = max(math.ceil(cap_km_val * 1000.0 / res) + 2, step)
    NXp = -(-NX // step) * step
    NYp = -(-NY // step) * step
    bsize = max(block_size, step)
    nbx = math.ceil(NX / bsize)
    nby = math.ceil(NY / bsize)

    print(f"  Grid: {NX}×{NY}, {(NX * NY):,} cells", file=sys.stderr)
    print(f"  Pyramid: {levels} levels, step={step}, halo={halo}", file=sys.stderr)
    print(f"  Blocks: {nbx}×{nby} ({nbx * nby} total), block={bsize}", file=sys.stderr)

    # --- Spatial index ---
    bx_ = np.clip(((x - x0) // res // bsize).astype(np.int64), 0, nbx - 1)
    by_ = np.clip(((y - y0) // res // bsize).astype(np.int64), 0, nby - 1)
    bid = bx_ * nby + by_
    order = np.argsort(bid, kind="stable")
    starts = np.searchsorted(bid[order], np.arange(nbx * nby + 1))

    # Memmap for workers
    pts_path = os.path.join(out_dir, "_points.npy")
    pts = np.lib.format.open_memmap(pts_path, mode="w+", dtype=np.float64, shape=(4, N))
    pts[0] = x[order]
    pts[1] = y[order]
    pts[2] = tv[order]
    pts[3] = v[order]
    pts.flush()

    # --- Task list ---
    tasks: list[tuple[int, int]] = [
        (i, j)
        for i in range(nbx)
        for j in range(nby)
        if _block_neighbourhood_nonempty(i, j, starts, nbx, nby)
    ]
    empty_blocks: list[tuple[int, int]] = [
        (i, j)
        for i in range(nbx)
        for j in range(nby)
        if not _block_neighbourhood_nonempty(i, j, starts, nbx, nby)
    ]

    print(
        f"  {len(tasks)} blocks to process, {len(empty_blocks)} empty, using {workers} workers",
        file=sys.stderr,
    )

    # --- Worker init args ---
    initargs = (
        pts_path,
        cal.get("transform_state", {"name": tname}),
        cal.get("percentile_quantiles", pct_q.q.tolist()),
        starts.tolist(),
        nbx,
        nby,
        bsize,
        halo,
        res,
        levels,
        cap_km_val,
        float(x0),
        float(y0),
        NX,
        NY,
        NXp,
        NYp,
        saturation,
        baseline,
        scale,
    )

    # --- Raster profile ---
    import rasterio

    xform = from_origin(x0, y0 + NY * res, res, res)
    common = {
        "driver": "GTiff",
        "height": NY,
        "width": NX,
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

    vpath = os.path.join(out_dir, "value.tif")
    spath = os.path.join(out_dir, "support_km.tif")

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
            i1, j1 = min(i0 + bsize, NX), min(j0 + bsize, NY)
            w = Window(i0, NY - j1, i1 - i0, j1 - j0)
            blank = np.full((w.height, w.width), NODATA, np.int16)
            vd.write(blank, 1, window=w)
            sd.write(blank, 1, window=w)

        # Process blocks with parallel workers
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=initargs,
        ) as ex:
            for bx, by, out in ex.map(_process_block, tasks, chunksize=1):
                i0, j0 = bx * bsize, by * bsize
                i1, j1 = min(i0 + bsize, NX), min(j0 + bsize, NY)
                w = Window(i0, NY - j1, i1 - i0, j1 - j0)

                if out is None:
                    blank = np.full((w.height, w.width), NODATA, np.int16)
                    vd.write(blank, 1, window=w)
                    sd.write(blank, 1, window=w)
                    continue

                Vq, Rq = out
                vd.write(Vq.T[::-1, :], 1, window=w)
                sd.write(Rq.T[::-1, :], 1, window=w)

    dt = time.perf_counter() - t_start
    print(f"Done in {dt:.0f}s", file=sys.stderr)
    print(f"  {vpath}", file=sys.stderr)
    print(f"  {spath}", file=sys.stderr)

    # Cleanup memmap
    os.remove(pts_path)
    return vpath, spath


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull-push scattered-data interpolation"
    )
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
    parser.add_argument(
        "--scale", type=float, default=100.0, help="DN = percentile * scale"
    )
    parser.add_argument("--compress", default="ZSTD")
    parser.add_argument("--calib-max-points", type=int, default=2_000_000)
    parser.add_argument("--calibration", default=None, help="Calibration JSON path")
    parser.add_argument("--src-crs", type=int, default=4326)
    parser.add_argument(
        "--skip-calibration", action="store_true", help="Skip calibration, use defaults"
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
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
