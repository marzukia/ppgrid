"""Turbo read-side MT (issue #32): parallel reproject warp + process-pool ingest.

- parallel _reproject_band (n_threads=4) vs serial (n_threads=1): output
  file sha256 equal, band arrays equal (A6 unit, design S3.3)
- parallel ingest vs single whole-file parse: bit-identical float64 arrays,
  original row order preserved, on a fixture with edge rows (int64 vs
  float64 inference, |x| >= 2**53, NaN/inf, quoted fields, non-adjacent
  columns) (S3.5, review F3)
- fallbacks: small files and quoted-newline files stay on the serial parse
- n_threads=1 regression: the default code path is the serial one (A8)
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform

from ppgrid import pipeline as pipeline_mod
from ppgrid.pipeline import (
    NODATA,
    OUT_CRS,
    TILE_PX,
    WORK_CRS,
    Pipeline,
    _csv_chunk_tasks,
    _reproject_band,
)

_WANTED = ["value", "longitude", "latitude"]


def _write_src_tif(path: Path, width: int, height: int, res: float = 100.0, seed: int = 0) -> None:
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
        dst.scales = (1.0 / 100.0,)


def test_reproject_band_mt_sha256_equal(tmp_path: Path) -> None:
    """The warp thread pool must not change a single output byte (A6 unit)."""
    src = tmp_path / "src.tif"
    _write_src_tif(src, 5000, 4000)
    with rasterio.open(src) as ds:
        dst_crs = f"EPSG:{OUT_CRS}"
        dst_transform, dst_width, dst_height = calculate_default_transform(
            ds.crs, dst_crs, ds.width, ds.height, *ds.bounds
        )
    assert dst_width > 2048  # >= 2 warp tiles per row band
    assert dst_height > 2048
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
    out1 = tmp_path / "out1.tif"
    out4 = tmp_path / "out4.tif"
    _reproject_band(str(src), str(out1), profile, dst_crs, dst_transform, dst_width, dst_height)
    _reproject_band(str(src), str(out4), profile, dst_crs, dst_transform, dst_width, dst_height, n_threads=4)
    h1 = hashlib.sha256(out1.read_bytes()).hexdigest()
    h4 = hashlib.sha256(out4.read_bytes()).hexdigest()
    assert h1 == h4
    with rasterio.open(out1) as a, rasterio.open(out4) as b:
        assert np.array_equal(a.read(1), b.read(1))


def _write_edge_csv(path: Path, n: int) -> None:
    """Write n rows covering the per-chunk inference edge cases (review F3).

    value: floats plus int-looking rows, 2**53 +/- 1, an empty (NaN) and an
    inf row. longitude: floats plus an int-looking row, a 17-significant-
    digit row, and an empty (NaN) row. latitude: integers only (a whole-file
    parse infers int64, the chunk parses must still agree). Columns are
    non-adjacent and non-leading, with a quoted-comma note column, so the
    reader must map by name.
    """
    rng = np.random.default_rng(123)
    value = rng.normal(0.0, 100.0, n).astype(object)
    value[0] = 2**53 + 1
    value[1] = -(2**53 + 1)
    value[2] = 42
    value[3] = ""
    value[4] = np.inf
    longitude = rng.uniform(144.0, 145.0, n).astype(object)
    longitude[5] = ""
    longitude[6] = 144.123456789012345
    latitude = rng.integers(-38, -37, n)
    pd.DataFrame(
        {
            "id": np.arange(n),
            "value": value,
            "longitude": longitude,
            "latitude": latitude,
            "note": ["a,b" if i % 97 == 0 else "plain" for i in range(n)],
        },
    ).to_csv(path, index=False)


def test_ingest_mt_matches_single_parse(tmp_path: Path) -> None:
    """Parallel ingest must be bit-identical to the single whole-file parse."""
    csv = tmp_path / "edge.csv"
    n = 500_000
    _write_edge_csv(csv, n)
    tasks = _csv_chunk_tasks(str(csv), _WANTED, 4)
    assert tasks is not None  # the fixture must actually drive the pool
    assert len(tasks[0]) == 4  # 250k rows -> 4 chunks at n_threads=4

    # Reference: the single whole-file parse, exactly as the serial path does it.
    df = pd.read_csv(csv, usecols=_WANTED)
    v_ref = df["value"].to_numpy(dtype=np.float64)
    lon_ref = df["longitude"].to_numpy(dtype=np.float64)
    lat_ref = df["latitude"].to_numpy(dtype=np.float64)
    good = np.isfinite(v_ref) & np.isfinite(lon_ref) & np.isfinite(lat_ref)
    v_ref, lon_ref, lat_ref = v_ref[good], lon_ref[good], lat_ref[good]
    tr = Transformer.from_crs(4326, 6933, always_xy=True)
    x_ref, y_ref = tr.transform(lon_ref, lat_ref)

    prev = None
    for nt in (1, 4):
        p = Pipeline(str(csv), "value", "longitude", "latitude", str(tmp_path / f"out{nt}"), n_threads=nt)
        p.ingest()
        assert p.n == v_ref.size
        assert np.array_equal(p.v, v_ref)  # bit-identical values, original row order
        assert np.array_equal(p.x, np.asarray(x_ref))
        assert np.array_equal(p.y, np.asarray(y_ref))
        if prev is not None:
            assert np.array_equal(prev[0], p.v)
            assert np.array_equal(prev[1], p.x)
            assert np.array_equal(prev[2], p.y)
        prev = (p.v, p.x, p.y)


def test_ingest_mt_fallbacks(tmp_path: Path) -> None:
    """Small files stay on the serial parse and still ingest correctly."""
    rng = np.random.default_rng(7)
    small = tmp_path / "small.csv"
    pd.DataFrame(
        {
            "value": rng.normal(0.0, 1.0, 100),
            "longitude": rng.uniform(144.0, 145.0, 100),
            "latitude": rng.uniform(-38.0, -37.0, 100),
        },
    ).to_csv(small, index=False)
    assert _csv_chunk_tasks(str(small), _WANTED, 4) is None  # below the min chunk size
    p = Pipeline(str(small), "value", "longitude", "latitude", str(tmp_path / "o1"), n_threads=4)
    p.ingest()
    assert p.n == 100


def test_ingest_mt_quoted_newline_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A quoted field with an embedded newline must refuse the split (serial parse)."""
    monkeypatch.setattr(pipeline_mod, "_INGEST_MIN_CHUNK_ROWS", 1)
    qnl = tmp_path / "qnl.csv"
    qnl.write_text(
        'value,longitude,latitude,note\n1.5,144.1,-37.5,"line1\nline2"\n2.5,144.2,-37.6,plain\n',
        encoding="utf-8",
    )
    assert _csv_chunk_tasks(str(qnl), _WANTED, 4) is None  # chunk boundary splits a quoted field
    p = Pipeline(str(qnl), "value", "longitude", "latitude", str(tmp_path / "o2"), n_threads=4)
    p.ingest()
    assert p.n == 2  # the serial parse reads both rows
    assert np.array_equal(p.v, np.array([1.5, 2.5]))


def test_pipeline_mt_end_to_end_sha256_equal(tmp_path: Path) -> None:
    """n_threads=4 must not change a single output byte of a full run (A8)."""
    rng = np.random.default_rng(1)
    n = 40
    csv = tmp_path / "pts.csv"
    pd.DataFrame(
        {
            "value": rng.uniform(1.0, 10.0, n),
            "longitude": 144.6 + rng.uniform(-0.05, 0.05, n),
            "latitude": -37.7 + rng.uniform(-0.05, 0.05, n),
        },
    ).to_csv(csv, index=False)

    digests = {}
    for nt in (1, 4):
        out = tmp_path / f"o{nt}"
        p = Pipeline(
            str(csv),
            "value",
            "longitude",
            "latitude",
            str(out),
            res=500.0,
            cap_km=10.0,
            workers=1,
            skip_calibration=True,
            n_threads=nt,
        )
        vpath, spath = p.run()
        digests[nt] = (
            hashlib.sha256(Path(vpath).read_bytes()).hexdigest(),
            hashlib.sha256(Path(spath).read_bytes()).hexdigest(),
        )
    assert digests[1] == digests[4]


def test_pipeline_n_threads_validation() -> None:
    """n_threads must be >= 1."""
    with pytest.raises(ValueError, match="n_threads"):
        Pipeline("in.csv", "value", "longitude", "latitude", "out", n_threads=0)
