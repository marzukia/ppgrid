"""Regenerate zstd_oracle_512.tif through the stock GDAL write path.

512x512 int16, incompressible seeded random data (seed 42), ZSTD,
predictor=2, 512px tiles, EPSG:3857, nodata -32768: the same profile the
pipeline writes. Run from the repo root with:

    uv run python tests/fixtures/make_zstd_oracle_512.py

The committed raster is the stock-path reference for the zstdmt oracle
tests. Regenerate only when the rasterio/GDAL stack changes, then review
the diff (bytes are deterministic for a fixed stack + seed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

TILE_PX = 512
NODATA = -32768
SEED = 42
OUT = Path(__file__).with_name("zstd_oracle_512.tif")


def make() -> Path:
    """Write the fixture raster and return its path."""
    rng = np.random.default_rng(SEED)
    tile = rng.integers(-32768, 32767, size=(TILE_PX, TILE_PX), dtype=np.int16)
    xform = from_origin(0.0, TILE_PX * 10.0, 10.0, 10.0)
    with rasterio.open(
        OUT,
        "w",
        driver="GTiff",
        width=TILE_PX,
        height=TILE_PX,
        count=1,
        dtype="int16",
        crs="EPSG:3857",
        transform=xform,
        compress="ZSTD",
        nodata=NODATA,
        tiled=True,
        blockxsize=TILE_PX,
        blockysize=TILE_PX,
        predictor=2,
        BIGTIFF="IF_SAFER",
    ) as dst:
        dst.write(tile, 1)
    return OUT


if __name__ == "__main__":
    path = make()
    print(f"wrote {path} ({path.stat().st_size} B)")
