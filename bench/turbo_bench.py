"""Turbo read-side MT (issue #32): speed evidence for reproject warp + ingest.

Repro: uv run python -m bench.turbo_bench
"""

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform

from ppgrid.pipeline import NODATA, TILE_PX, Pipeline, _reproject_band
from tests.test_turbo import _write_src_tif

TMP = Path(tempfile.mkdtemp(prefix="turbo_bench"))


def median(f, runs=3):
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def bench_reproject():
    src = TMP / "src.tif"
    _write_src_tif(src, 5000, 4000)
    with rasterio.open(src) as ds:
        t, w, h = calculate_default_transform(ds.crs, "EPSG:3857", ds.width, ds.height, *ds.bounds)
    profile = {
        "driver": "GTiff",
        "dtype": "int16",
        "nodata": NODATA,
        "compress": "ZSTD",
        "tiled": True,
        "blockxsize": TILE_PX,
        "blockysize": TILE_PX,
        "predictor": 2,
        "count": 1,
    }
    out = TMP / "out.tif"
    t1 = median(lambda: _reproject_band(str(src), str(out), profile, "EPSG:3857", t, w, h))
    t4 = median(lambda: _reproject_band(str(src), str(out), profile, "EPSG:3857", t, w, h, n_threads=4))
    print(f"reproject band {w}x{h}: serial {t1:.3f}s, 4t {t4:.3f}s, speedup {t1 / t4:.2f}x")


def bench_ingest(n_rows=1_000_000):
    rng = np.random.default_rng(0)
    csv = TMP / "big.csv"
    pd.DataFrame(
        {
            "value": rng.normal(0.0, 100.0, n_rows),
            "longitude": rng.uniform(110.0, 155.0, n_rows),
            "latitude": rng.uniform(-50.0, -10.0, n_rows),
        },
    ).to_csv(csv, index=False)
    t1 = median(lambda: Pipeline(str(csv), "value", "longitude", "latitude", str(TMP / "o1"), n_threads=1).ingest())
    t4 = median(lambda: Pipeline(str(csv), "value", "longitude", "latitude", str(TMP / "o4"), n_threads=4).ingest())
    print(f"ingest {n_rows} rows: serial {t1:.3f}s, 4t {t4:.3f}s, speedup {t1 / t4:.2f}x")


if __name__ == "__main__":
    bench_reproject()
    bench_ingest()
