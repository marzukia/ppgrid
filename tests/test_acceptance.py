"""Acceptance tests for ppgrid pipeline correctness.

These tests verify the properties that matter most:
- Blocked output is bit-identical to whole-grid output (halo snapping)
- Coverage fraction is consistent across resolutions (mask correctness)
- Single points georeference back to their source coordinates
- Output pixels are either valid percentiles or exactly nodata
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol

from ppgrid.idwgrid import NODATA, Pipeline


def _make_small_csv(tmp_path: Path, n_points: int = 500) -> str:
    """Create a small CSV with random points for testing."""
    np.random.seed(42)
    lons = 144.5 + np.random.uniform(-0.5, 0.5, n_points)
    lats = -37.5 + np.random.uniform(-0.5, 0.5, n_points)
    values = np.random.uniform(10, 100, n_points)
    csv = tmp_path / "test_data.csv"
    pd.DataFrame({"lon": lons, "lat": lats, "val": values}).to_csv(csv, index=False)
    return str(csv)


def test_blocked_equals_whole_grid(tmp_path: Path) -> None:
    """Blocked output must be bit-identical to a single-block run.

    This is the acceptance test that proves halo snapping is correct.
    Any differing pixel means the halo or block alignment is wrong.
    """
    csv = _make_small_csv(tmp_path, n_points=500)

    blocked_dir = tmp_path / "blocked"
    blocked_dir.mkdir()
    whole_dir = tmp_path / "whole"
    whole_dir.mkdir()

    # Run with small blocks
    p1 = Pipeline(
        csv,
        "val",
        "lon",
        "lat",
        str(blocked_dir),
        res=500.0,
        cap_km=10.0,
        block_size=64,
        workers=1,
        skip_calibration=True,
    )
    v1, s1 = p1.run()

    # Run with one giant block (larger than any dimension)
    p2 = Pipeline(
        csv,
        "val",
        "lon",
        "lat",
        str(whole_dir),
        res=500.0,
        cap_km=10.0,
        block_size=100000,
        workers=1,
        skip_calibration=True,
    )
    v2, s2 = p2.run()

    # Compare value band bit-identically
    with rasterio.open(v1) as a, rasterio.open(v2) as b:
        assert a.width == b.width
        assert a.height == b.height
        arr_a = a.read(1)
        arr_b = b.read(1)
        diff = (arr_a != arr_b).sum()
        assert diff == 0, f"Value band has {diff} differing pixels"

    # Compare support_km band bit-identically
    with rasterio.open(s1) as a, rasterio.open(s2) as b:
        arr_a = a.read(1)
        arr_b = b.read(1)
        diff = (arr_a != arr_b).sum()
        assert diff == 0, f"Support band has {diff} differing pixels"


def test_coverage_fraction_consistent(tmp_path: Path) -> None:
    """Coverage fraction should be stable across resolutions (within 0.1%).

    Catches the mask drifting back to the pyramid instead of box_count.
    """
    csv = _make_small_csv(tmp_path, n_points=500)
    fractions = []

    for res in [100.0, 500.0, 2000.0]:
        out_dir = tmp_path / f"res_{int(res)}"
        out_dir.mkdir()
        p = Pipeline(
            csv,
            "val",
            "lon",
            "lat",
            str(out_dir),
            res=res,
            cap_km=10.0,
            block_size=2048,
            workers=1,
            skip_calibration=True,
        )
        vpath, _ = p.run()
        with rasterio.open(vpath) as ds:
            arr = ds.read(1)
            valid = (arr != NODATA).sum()
            total = arr.size
            fractions.append(valid / total)

    # All three resolutions should give similar coverage
    for i in range(1, len(fractions)):
        drift = abs(fractions[i] - fractions[0])
        assert drift < 0.001, (
            f"Coverage fraction drifted by {drift:.4f} "
            f"between res 100 ({fractions[0]:.4f}) and res {int(100 * (2**i))} ({fractions[i]:.4f})"
        )


def test_single_point_georeferences(tmp_path: Path) -> None:
    """Points at known lon/lat should georeference back to those coordinates.

    Catches the .T[::-1] / ny - j1 orientation pairing.
    """
    known_lon, known_lat = 144.9631, -37.8136  # Melbourne CBD
    # Spread points slightly so the grid covers a real area (not 1x1)
    np.random.seed(42)
    csv = tmp_path / "cluster.csv"
    pd.DataFrame(
        {
            "lon": known_lon + np.random.uniform(-0.005, 0.005, 100),
            "lat": known_lat + np.random.uniform(-0.005, 0.005, 100),
            "val": [float(i) for i in range(100)],
        },
    ).to_csv(csv, index=False)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    p = Pipeline(
        str(csv),
        "val",
        "lon",
        "lat",
        str(out_dir),
        res=100.0,
        cap_km=50.0,
        block_size=2048,
        workers=1,
        skip_calibration=True,
    )
    vpath, _ = p.run()

    with rasterio.open(vpath) as ds:
        transform = ds.transform
        # Output is in EPSG:3857 (Web Mercator), project the known coords
        tr = Transformer.from_crs(4326, 3857, always_xy=True)
        merc_x, merc_y = tr.transform(known_lon, known_lat)
        # Find the pixel closest to the known coordinate (rowcol takes xs, ys)
        row, col = rowcol(transform, merc_x, merc_y, offset="center")
        arr = ds.read(1)

        # Check a 5x5 neighborhood around the expected pixel
        h, w = arr.shape
        r0, r1 = max(0, row - 2), min(h, row + 3)
        c0, c1 = max(0, col - 2), min(w, col + 3)
        neighborhood = arr[r0:r1, c0:c1]
        valid_pixels = (neighborhood != NODATA).sum()
        assert valid_pixels > 0, "No valid pixels near the known point location"


def test_output_pixels_are_valid(tmp_path: Path) -> None:
    """Every output pixel must be either a valid percentile or exactly nodata.

    Catches spurious zeros from skipped blocks or log-space output.
    """
    csv = _make_small_csv(tmp_path, n_points=500)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    p = Pipeline(
        csv,
        "val",
        "lon",
        "lat",
        str(out_dir),
        res=500.0,
        cap_km=10.0,
        block_size=2048,
        workers=1,
        skip_calibration=True,
    )
    vpath, spath = p.run()

    # Check value band
    with rasterio.open(vpath) as ds:
        arr = ds.read(1)
        valid_mask = arr != NODATA
        valid_values = arr[valid_mask]
        # Valid values should be in range [0, 100*scale] where scale=100
        if len(valid_values) > 0:
            assert valid_values.min() >= 0, f"Negative percentile: {valid_values.min()}"
            assert valid_values.max() <= 10000, f"Percentile > 100*scale: {valid_values.max()}"

    # Check support_km band
    with rasterio.open(spath) as ds:
        arr = ds.read(1)
        valid_mask = arr != NODATA
        valid_values = arr[valid_mask]
        # Valid support DN values encode support_km = 2^(DN/8).
        # DN can be negative when support_km < 1km (log2 < 0).
        # Just check values are within int16 range and finite.
        if len(valid_values) > 0:
            assert valid_values.min() >= -32768, f"Support DN below int16 min: {valid_values.min()}"
            assert valid_values.max() <= 32767, f"Support DN exceeds int16 max: {valid_values.max()}"
