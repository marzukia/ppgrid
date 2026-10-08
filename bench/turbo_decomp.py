"""Decompose turbo speedup: warp-only (GIL release) + parse-only (pool).

The full-path numbers in turbo_bench.py include the serial tail (the 512px
flush loop, pyproj transform) which Amdahl-caps the gain. This harness
measures the parallelisable part in isolation.

Repro: uv run python -m bench.turbo_decomp
"""

import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform

from ppgrid import pipeline as pm
from tests.test_turbo import _write_src_tif

TMP = Path(tempfile.mkdtemp(prefix="turbo_bench2"))


def median(f, runs=3):
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def warp_only():
    src = TMP / "src.tif"
    _write_src_tif(src, 5000, 4000)
    with rasterio.open(src) as ds:
        t, w, h = calculate_default_transform(ds.crs, "EPSG:3857", ds.width, ds.height, *ds.bounds)

    def warp_all(n_threads):
        warp_tile = 2048
        for j0 in range(0, h, warp_tile):
            band_h = min(warp_tile, h - j0)
            i0s = list(range(0, w, warp_tile))
            p = partial(
                pm._warp_dst_tile,
                str(src),
                j0=j0,
                band_h=band_h,
                warp_tile=warp_tile,
                dst_width=w,
                dst_transform=t,
                dst_crs="EPSG:3857",
            )
            if n_threads > 1:
                with ThreadPoolExecutor(max_workers=n_threads) as ex:
                    for _ in ex.map(p, i0s):
                        pass
            else:
                for i0 in i0s:
                    p(i0)

    t1 = median(lambda: warp_all(1))
    t4 = median(lambda: warp_all(4))
    print(f"warp-only {w}x{h}: serial {t1:.3f}s, 4t {t4:.3f}s, speedup {t1 / t4:.2f}x")


def parse_only(n_rows=1_000_000):
    rng = np.random.default_rng(0)
    csv = TMP / "big.csv"
    pd.DataFrame(
        {
            "value": rng.normal(0.0, 100.0, n_rows),
            "longitude": rng.uniform(110.0, 155.0, n_rows),
            "latitude": rng.uniform(-50.0, -10.0, n_rows),
        },
    ).to_csv(csv, index=False)
    t1 = median(lambda: pd.read_csv(csv))

    wanted = ["value", "longitude", "latitude"]
    tasks, _ = pm._csv_chunk_tasks(str(csv), wanted, 4)

    def p4():
        with ProcessPoolExecutor(max_workers=4) as ex:
            chunks = list(ex.map(pm._read_csv_chunk, tasks))
        return np.concatenate([c[0] for c in chunks])

    t4 = median(p4)
    print(f"parse-only {n_rows} rows: serial {t1:.3f}s, 4t {t4:.3f}s, speedup {t1 / t4:.2f}x")


if __name__ == "__main__":
    warp_only()
    parse_only()
