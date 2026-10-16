"""Shared helpers for the bench harnesses.

Kept separate from tests/ so the bench scripts do not import the test
package (issue #84 a8): a bench run must not require the test suite to be
collectable, and a test-helper change must not silently move bench numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from ppgrid.pipeline import NODATA, TILE_PX, WORK_CRS


def write_src_tif(path: Path, width: int, height: int, res: float = 100.0, seed: int = 0) -> None:
    """Write a deterministic int16 source raster in the work CRS (512px blocks)."""
    rng = np.random.default_rng(seed)
    a = ((np.arange(width, dtype=np.int64) * 13 + np.arange(height, dtype=np.int64)[:, None] * 7) % 9000 + 1000).astype(
        np.int16
    )
    a[rng.random((height, width)) < 0.05] = NODATA
    transform = from_origin(0.0, height * res, res, res)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="int16",
        crs=f"EPSG:{WORK_CRS}",
        transform=transform,
        nodata=NODATA,
        compress="ZSTD",
        tiled=True,
        blockxsize=TILE_PX,
        blockysize=TILE_PX,
        predictor=2,
    ) as dst:
        dst.write(a, 1)
