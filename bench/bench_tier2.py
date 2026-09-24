"""Tier 2 perf harness (issue #3).

Generates fixed synthetic inputs (seeded), runs representative pipeline
configs, and reports per-phase wall time + peak RSS. Outputs land in
--outdir/<tag>/<config> for bit-identity comparison (see compare_tier2.py).

Usage:
    uv run python bench/bench_tier2.py --tag baseline --outdir /tmp/tier2
    uv run python bench/bench_tier2.py --tag after    --outdir /tmp/tier2

Acceptance run: `fine1m` = 1M pts, ~3281x4698 grid at res=10m, cap 64.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import resource
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

import ppgrid.idwgrid as ig
from ppgrid.idwgrid import Pipeline

TR = Transformer.from_crs(4326, 6933, always_xy=True)

# (config, csv, kwargs)
CONFIGS: list[tuple[str, str, dict]] = [
    # Acceptance: 1M pts, ~3281x4698 grid @ 10m, cap 64 (issue #3).
    ("fine1m", "synth_fine.csv", {"res": 10.0, "cap_km": 64.0, "workers": 4}),
    # Multi-block halo stress: 1M pts, 100km x 100km, res 10m, cap 2km.
    ("wide1m", "synth_wide.csv", {"res": 10.0, "cap_km": 2.0, "workers": 4}),
    # Anchors: must bit-match the committed examples (see compare_tier2.py).
    (
        "melb10",
        "data/melb_houses.csv",
        {"value_col": "price", "res": 10.0, "cap_km": 10.0, "skip_calibration": True, "workers": 4},
    ),
    (
        "melb500",
        "data/melb_houses.csv",
        {"value_col": "price", "res": 500.0, "cap_km": 10.0, "skip_calibration": True, "workers": 4},
    ),
    (
        "equakes500",
        "data/all_equakes.csv",
        {
            "value_col": "mag",
            "res": 500.0,
            "cap_km": 25.0,
            "transform": "identity",
            "skip_calibration": True,
            "out_crs": 6933,
            "workers": 4,
        },
    ),
    # Auto calibration: exercises choose_transform + calibrate_fill_cap (hot path #2).
    (
        "equakes_auto",
        "data/all_equakes.csv",
        {"value_col": "mag", "res": 500.0, "cap_km": "auto", "workers": 4},
    ),
]

_PHASES: dict[str, float] = {}
_TIMERS: dict[str, float] = {"block_cpu": 0.0}


def _fit_box(clon: float, clat: float, target_dx: float, target_dy: float) -> tuple[float, float]:
    """Lon/lat half-boxes whose projected extent hits the target metres."""
    half_lon = target_dx / 87000.0 / 2
    half_lat = target_dy / 110000.0 / 2
    for _ in range(8):
        lons = np.array([clon - half_lon, clon, clon + half_lon])
        lats = np.array([clat - half_lat, clat, clat + half_lat])
        xx, yy = np.meshgrid(lons, lats)
        x, y = TR.transform(xx.ravel(), yy.ravel())
        dx, dy = x.max() - x.min(), y.max() - y.min()
        half_lon *= target_dx / dx
        half_lat *= target_dy / dy
    return half_lon, half_lat


def _make_synth(path: Path, clon: float, clat: float, target_dx: float, target_dy: float, n: int, seed: int) -> None:
    """Write a seeded synthetic CSV: smooth positive field + lognormal noise."""
    rng = np.random.default_rng(seed)
    hl, hh = _fit_box(clon, clat, target_dx, target_dy)
    lon = clon + rng.uniform(-hl, hl, n)
    lat = clat + rng.uniform(-hh, hh, n)
    x, y = TR.transform(lon, lat)
    ex, ey = x.max() - x.min(), y.max() - y.min()
    gx = x - x.min()
    gy = y - y.min()
    g1 = np.exp(-((gx - 0.3 * ex) ** 2 + (gy - 0.4 * ey) ** 2) / 4e6)
    g2 = np.exp(-((gx - 0.8 * ex) ** 2 + (gy - 0.7 * ey) ** 2) / 6e6)
    value = 5.0 + 40.0 * (g1 + g2) + rng.lognormal(0.0, 0.5, n)
    pd.DataFrame({"value": value, "longitude": lon, "latitude": lat}).to_csv(path, index=False)
    print(f"  synth: {n} pts -> {path.name}, projected extent {ex:.0f} x {ey:.0f} m")


def gen_data(outdir: Path) -> None:
    """Generate both synthetic CSVs (deterministic, fixed seeds)."""
    outdir.mkdir(parents=True, exist_ok=True)
    _make_synth(outdir / "synth_fine.csv", 144.995, -37.75, 32805.0, 46975.0, 1_000_000, seed=1)
    _make_synth(outdir / "synth_wide.csv", 145.05, -37.9, 100_000.0, 100_000.0, 1_000_000, seed=2)


def _patch_timers() -> None:
    """Wrap the block loop to record wall + CPU time. Call once per process."""
    orig_map = cf.ThreadPoolExecutor.map

    def timed_map(
        self: cf.ThreadPoolExecutor,
        fn: Callable,
        *iterables: Iterable,
        chunksize: int | None = None,
    ) -> list:
        t0 = time.perf_counter()
        res = list(orig_map(self, fn, *iterables, chunksize=chunksize))
        _PHASES["blockloop_wall"] = time.perf_counter() - t0
        return res

    cf.ThreadPoolExecutor.map = timed_map  # type: ignore[method-assign]

    orig_bp = ig._process_block  # pyright: ignore[reportPrivateUsage]

    def timed_bp(args: tuple[int, int]) -> tuple[int, int, tuple[np.ndarray, np.ndarray] | None]:
        t0 = time.perf_counter()
        r = orig_bp(args)
        _TIMERS["block_cpu"] += time.perf_counter() - t0
        return r

    ig._process_block = timed_bp  # type: ignore[assignment]


def run_config(name: str, csv: str, kw: dict, outdir: Path) -> dict:
    """Run one config with phase timing. Returns the timing record."""
    _PHASES.clear()
    _TIMERS["block_cpu"] = 0.0
    csv_path = csv if csv.startswith("data/") else str(outdir / "data" / csv)
    cdir = outdir / name
    cdir.mkdir(parents=True, exist_ok=True)
    kw = dict(kw)
    value_col = kw.pop("value_col", "value")

    p = Pipeline(csv_path, value_col, "longitude", "latitude", str(cdir), **kw)
    t0 = time.perf_counter()
    p.ingest()
    t_ingest = time.perf_counter() - t0

    t0 = time.perf_counter()
    p.calibrate()
    t_calibrate = time.perf_counter() - t0

    t0 = time.perf_counter()
    p.grid()
    t_grid = time.perf_counter() - t0

    t0 = time.perf_counter()
    p._write_rasters()  # pyright: ignore[reportPrivateUsage]
    t_write = time.perf_counter() - t0
    total = time.perf_counter() - t0

    return {
        "config": name,
        "n_points": p.n,
        "grid": f"{p.nx}x{p.ny}",
        "grid_padded": f"{p.nx_padded}x{p.ny_padded}",
        "blocks": f"{p.nbx}x{p.nby}",
        "cap_km": p.cap_km_val,
        "transform": p.tname,
        "t_ingest": round(t_ingest, 3),
        "t_calibrate": round(t_calibrate, 3),
        "t_grid": round(t_grid, 3),
        "t_blockloop_wall": round(_PHASES.get("blockloop_wall", 0.0), 3),
        "t_blockloop_cpu": round(_TIMERS["block_cpu"], 3),
        "t_write_total": round(t_write, 3),
        "t_total": round(total, 3),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }


def main() -> None:
    """Bench entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--configs", default=None, help="Comma-separated subset of config names")
    args = ap.parse_args()

    outdir = Path(args.outdir) / args.tag
    t0 = time.perf_counter()
    gen_data(outdir / "data")
    t_gen = time.perf_counter() - t0

    _patch_timers()
    names = set(args.configs.split(",")) if args.configs else None
    results = []
    for name, csv, kw in CONFIGS:
        if names and name not in names:
            continue
        print(f"== {name} ==", flush=True)
        rec = run_config(name, csv, kw, outdir)
        results.append(rec)
        print(
            f"  n={rec['n_points']} grid={rec['grid']} padded={rec['grid_padded']} "
            f"blocks={rec['blocks']} cap={rec['cap_km']} tf={rec['transform']}",
            flush=True,
        )
        print(
            f"  ingest {rec['t_ingest']}s  calib {rec['t_calibrate']}s  grid {rec['t_grid']}s  "
            f"blocks(wall) {rec['t_blockloop_wall']}s  write+reproj {rec['t_write_total']}s  "
            f"TOTAL {rec['t_total']}s  peakRSS {rec['peak_rss_mb']}MB",
            flush=True,
        )

    rep = {"tag": args.tag, "t_gen_data": round(t_gen, 2), "results": results}
    (outdir / "timing.json").write_text(json.dumps(rep, indent=2))
    print(f"\ntiming report -> {outdir / 'timing.json'}")


if __name__ == "__main__":
    main()
