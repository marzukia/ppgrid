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
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform

from ppgrid import pipeline as pipeline_mod
from ppgrid import turbop
from ppgrid.pipeline import (
    NODATA,
    OUT_CRS,
    TILE_PX,
    WORK_CRS,
    Pipeline,
    _build_parser,
    _csv_chunk_tasks,
    _reproject_band,
    _reproject_band_array,
    _tif_parse_ifd,
    _turbo_write_parallel,
)
from ppgrid.pullpush import bin_points, box_count_banded

DATA_CSV = Path(pipeline_mod.__file__).resolve().parent.parent / "data" / "melb_houses.csv"


def _sha(p: str | Path) -> str:
    """sha256 hex digest of a file."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


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


# ---------------------------------------------------------------------------
# Issue #33: turbo write path (S3.4a) + budget cap (S3.3) + CLI (S3.6)
# ---------------------------------------------------------------------------

# A9 anchor geometry (design 3.6.1): full-AU 16M pts @ 100 m, cap 25 km.
_A9_CELLS = 1_592_524_800  # Wc 41472 x ny_padded 38400
_A9_WC = 41472
_A9_RADIUS = 250  # cap_km 25 -> radius cells


def _identity_cap(preset: float | None, *_args: float | None, **_kwargs: float | None) -> tuple[float | None, str]:
    """resolve_cap stub: the explicit preset stands, source labelled 'explicit'."""
    return preset, "explicit"


def _a9_pipeline(tmp_path: Path, cap: float, *, strict: bool = False) -> Pipeline:
    """Build a full-AU-geometry Pipeline with the A9 anchor grid (no run)."""
    p = Pipeline(
        str(DATA_CSV),
        "price",
        "longitude",
        "latitude",
        str(tmp_path / f"o_{cap}_{int(strict)}"),
        res=100.0,
        cap_km=25.0,
        turbo=True,
        turbo_cap_gb=cap,
        turbo_strict=strict,
    )
    p.n, p.nx_padded, p.ny_padded, p.res, p.cap_km_val = (
        400_000,
        _A9_WC,
        _A9_CELLS // _A9_WC,
        100.0,
        25,
    )
    return p


def test_a9_pin_20gb_regime_c() -> None:
    """20 GB full-AU: regime C, c=2, b=2, est ~16.6 GB, shared, exit 0."""
    p = turbop.plan(20.0, _A9_CELLS, _A9_WC, _A9_RADIUS)
    assert p.regime == "C"
    assert p.box_chunks == 2
    assert p.descent_bands == 2
    assert p.workers == 2
    assert abs(p.est_peak_bytes / 1e9 - 16.6) < 0.2
    d = turbop.precheck(p)
    assert d.path == "shared"
    assert d.exit_code == 0
    assert "--turbo: cap" in d.summary  # the budget value used is printed
    assert "budget 0.85*C = 17.0 GB" in d.summary


def test_a9_pin_16gb_per_box() -> None:
    """16 GB full-AU: per_box, est ~4.7 GB; non-strict warns + exit 0, strict exit 3."""
    p = turbop.plan(16.0, _A9_CELLS, _A9_WC, _A9_RADIUS)
    assert p.regime == "per_box"
    assert abs(p.est_peak_bytes / 1e9 - 4.7) < 0.2
    d = turbop.precheck(p)
    assert d.path == "per_box"
    assert d.exit_code == 0  # [warn] + continue on the per-box path
    d3 = turbop.precheck(p, strict=True)
    assert d3.exit_code == 3


def test_a9_pin_32gb_regime_b() -> None:
    """32 GB full-AU: regime B, c=3, b=4, workers 3, est ~27.2 GB, exit 0."""
    p = turbop.plan(32.0, _A9_CELLS, _A9_WC, _A9_RADIUS)
    assert p.regime == "B"
    assert p.box_chunks == 3
    assert p.descent_bands == 4
    assert p.workers == 3
    assert abs(p.est_peak_bytes / 1e9 - 27.2) < 0.2
    assert turbop.precheck(p).exit_code == 0


def test_turbo_precheck_strict_exit3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--turbo-strict: shared infeasible (per_box) -> SystemExit(3)."""
    monkeypatch.setattr(pipeline_mod.turbop, "resolve_cap", _identity_cap)
    p = _a9_pipeline(tmp_path, 16.0, strict=True)
    with pytest.raises(SystemExit) as ei:
        p._turbo_precheck()  # ruff: ignore[private-member-access]
    assert ei.value.code == 3
    assert p._turbo_decision is not None  # ruff: ignore[private-member-access]
    assert p._turbo_decision.path == "per_box"  # ruff: ignore[private-member-access]


def test_turbo_precheck_nonstrict_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-strict: per_box warns and continues; the 0.85*C budget is stored."""
    monkeypatch.setattr(pipeline_mod.turbop, "resolve_cap", _identity_cap)
    p = _a9_pipeline(tmp_path, 16.0, strict=False)
    p._turbo_precheck()  # ruff: ignore[private-member-access] - must not raise
    assert p._turbo_decision is not None  # ruff: ignore[private-member-access]
    assert p._turbo_decision.path == "per_box"  # ruff: ignore[private-member-access]
    assert p._turbo_decision.exit_code == 0  # ruff: ignore[private-member-access]
    assert p._turbo_budget_bytes == turbop.precheck_budget_bytes(16.0)  # ruff: ignore[private-member-access]


def test_turbo_cap_gb_validation() -> None:
    """turbo_cap_gb must be > 0."""
    with pytest.raises(ValueError, match="turbo_cap_gb"):
        Pipeline(str(DATA_CSV), "price", "longitude", "latitude", "out", turbo=True, turbo_cap_gb=0.0)


def _demo_field(width: int, height: int, res: float, seed: int) -> tuple[np.ndarray, Any]:
    """Deterministic int16 work-CRS field with a 5% NODATA scatter."""
    rng = np.random.default_rng(seed)
    a = ((np.arange(width, dtype=np.int64) * 13 + np.arange(height, dtype=np.int64)[:, None] * 7) % 9000 + 100).astype(
        np.int16
    )
    a[rng.random((height, width)) < 0.05] = NODATA
    return a, from_origin(0.0, height * res, res, res)


def _dst_grid(width: int, height: int, res: float) -> tuple[Any, int, int]:
    """Default output-CRS grid for a work-CRS field of the given size."""
    work_crs, dst_crs = f"EPSG:{WORK_CRS}", f"EPSG:{OUT_CRS}"
    return calculate_default_transform(work_crs, dst_crs, width, height, 0.0, 0.0, width * res, height * res)


def _demo_profile() -> dict[str, Any]:
    """Return the base write profile (zstd tiled int16), identical to the pipeline's."""
    return {
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


def test_array_reproject_matches_file_reproject(tmp_path: Path) -> None:
    """_reproject_band_array == _reproject_band on the same work-CRS file (A6 unit)."""
    a, xform = _demo_field(3000, 2400, 100.0, seed=42)
    dst_transform, dst_width, dst_height = _dst_grid(3000, 2400, 100.0)
    work_crs, dst_crs = f"EPSG:{WORK_CRS}", f"EPSG:{OUT_CRS}"
    tags = {"transform": "identity", "res_m": "100.0"}
    scales, offsets = (1.0 / 100.0,), (0.0,)
    work = tmp_path / "work.tif"
    with rasterio.open(
        work,
        "w",
        driver="GTiff",
        width=a.shape[1],
        height=a.shape[0],
        count=1,
        dtype="int16",
        crs=work_crs,
        transform=xform,
        nodata=NODATA,
        compress="ZSTD",
        tiled=True,
        blockxsize=TILE_PX,
        blockysize=TILE_PX,
        predictor=2,
    ) as dst:
        dst.write(a, 1)
        dst.update_tags(**tags)
        dst.scales = scales
        dst.offsets = offsets
    out_file = tmp_path / "out_file.tif"
    out_arr = tmp_path / "out_arr.tif"
    _reproject_band(str(work), str(out_file), _demo_profile(), dst_crs, dst_transform, dst_width, dst_height)
    _reproject_band_array(
        a,
        xform,
        work_crs,
        str(out_arr),
        _demo_profile(),
        dst_crs,
        dst_transform,
        dst_width,
        dst_height,
        tags=tags,
        scales=scales,
        offsets=offsets,
        n_threads=4,
    )
    assert _sha(out_file) == _sha(out_arr)


def _stock_frames(path: Path) -> list[bytes]:
    """Read the stored (compressed) tile frames of a tiled GTiff in raster-scan order."""
    data = path.read_bytes()
    info = _tif_parse_ifd(data)
    fmt = info["off_fmt"]
    n = info["n_tiles"]
    offs = struct.unpack_from(info["e"] + f"{n}{fmt}", data, info["t324"][2])
    sizes = struct.unpack_from(info["e"] + f"{n}{fmt}", data, info["t325"][2])
    frames = [data[o : o + s] for o, s in zip(offs, sizes, strict=True)]
    assert len(frames) == n
    assert all(f[:4] == b"\x28\xb5\x2f\xfd" for f in frames)  # zstd magic
    return frames


def test_turbo_write_parallel_matches_serial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Patched-head parallel assembly matches the serial array reproject.

    Simulates an oracle pass: compress_tiles returns the exact stock codec
    frames, so the assembled file must be byte-identical (A6 unit).
    """
    a, xform = _demo_field(1600, 1400, 100.0, seed=11)
    dst_transform, dst_width, dst_height = _dst_grid(1600, 1400, 100.0)
    work_crs, dst_crs = f"EPSG:{WORK_CRS}", f"EPSG:{OUT_CRS}"
    tags = {"transform": "identity", "res_m": "100.0"}
    scales, offsets = (1.0 / 100.0,), (0.0,)
    out_ser = tmp_path / "ser.tif"
    _reproject_band_array(
        a,
        xform,
        work_crs,
        str(out_ser),
        _demo_profile(),
        dst_crs,
        dst_transform,
        dst_width,
        dst_height,
        tags=tags,
        scales=scales,
        offsets=offsets,
        n_threads=1,
    )
    frames = iter(_stock_frames(out_ser))

    def stub(tiles: list[np.ndarray], n_threads: int = 1) -> list[bytes]:  # ruff: ignore[unused-function-argument]
        """Stock-frame stand-in: consume the real frames in raster-scan order."""
        return [next(frames) for _ in tiles]

    monkeypatch.setattr(pipeline_mod.zstdmt, "compress_tiles", stub)
    out_par = tmp_path / "par.tif"
    ok = _turbo_write_parallel(
        a,
        xform,
        work_crs,
        str(out_par),
        _demo_profile(),
        dst_crs,
        dst_transform,
        dst_width,
        dst_height,
        tags=tags,
        scales=scales,
        offsets=offsets,
        n_threads=4,
    )
    assert ok
    assert _sha(out_ser) == _sha(out_par)
    with rasterio.open(out_par) as ds:
        assert ds.scales == scales
        for k, v in tags.items():  # GDAL auto-adds AREA_OR_POINT; ours must be present
            assert ds.tags()[k] == v


def test_turbo_write_parallel_4gib_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Classic head + assembled size beyond 4 GiB -> False (serial fallback)."""
    a, xform = _demo_field(800, 700, 100.0, seed=12)
    dst_transform, dst_width, dst_height = _dst_grid(800, 700, 100.0)
    work_crs, dst_crs = f"EPSG:{WORK_CRS}", f"EPSG:{OUT_CRS}"
    monkeypatch.setattr(pipeline_mod, "_TIFF_CLASSIC_MAX_BYTES", 1024)  # any classic file now "exceeds"
    ok = _turbo_write_parallel(
        a,
        xform,
        work_crs,
        str(tmp_path / "big.tif"),
        _demo_profile(),
        dst_crs,
        dst_transform,
        dst_width,
        dst_height,
        tags={},
        scales=(1.0,),
        offsets=(0.0,),
        n_threads=2,
    )
    assert ok is False


def test_tif_parse_ifd_classic_and_bigtiff(tmp_path: Path) -> None:
    """Classic and BigTIFF heads both parse; tile count and offsets resolve."""
    rng = np.random.default_rng(13)
    width, height = 1200, 900
    a = (np.arange(width * height, dtype=np.int64) % 9000).reshape(height, width).astype(np.int16)
    a[rng.random((height, width)) < 0.1] = NODATA
    xform = from_origin(0.0, height * 100.0, 100.0, 100.0)
    n_tiles = math.ceil(width / TILE_PX) * math.ceil(height / TILE_PX)  # 3*2
    for big, fname in ((False, "c.tif"), (True, "b.tif")):
        path = tmp_path / fname
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype="int16",
            crs=f"EPSG:{WORK_CRS}",
            transform=xform,
            nodata=NODATA,
            compress="ZSTD",
            tiled=True,
            blockxsize=TILE_PX,
            blockysize=TILE_PX,
            predictor=2,
            BIGTIFF="YES" if big else "IF_SAFER",
        ) as dst:
            dst.write(a, 1)
        info = _tif_parse_ifd(path.read_bytes())
        assert info["big"] is big
        assert info["n_tiles"] == n_tiles
        assert info["t324"][1] == n_tiles
        assert info["t325"][1] == n_tiles
        assert info["first"] > 0


def test_turbo_e2e_inram_matches_nonturbo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full turbo run is byte-identical to the non-turbo run (A6).

    In-RAM DN field, oracle mismatch -> serial array reproject, MT
    descent/warp driven by the turbo plan.
    """
    monkeypatch.setattr(pipeline_mod.turbop, "resolve_cap", _identity_cap)
    p0 = Pipeline(str(DATA_CSV), "price", "longitude", "latitude", str(tmp_path / "ref"), res=50.0, workers=4)
    v0, s0 = p0.run()
    p1 = Pipeline(
        str(DATA_CSV),
        "price",
        "longitude",
        "latitude",
        str(tmp_path / "turbo"),
        res=50.0,
        workers=4,
        turbo=True,
        turbo_cap_gb=64.0,
    )
    v1, s1 = p1.run()
    assert _sha(v0) == _sha(v1)
    assert _sha(s0) == _sha(s1)
    assert p1._turbo_decision is not None  # ruff: ignore[private-member-access]
    assert p1._turbo_decision.path == "shared"  # ruff: ignore[private-member-access]
    assert p1._turbo_plan is not None  # ruff: ignore[private-member-access]
    assert p1._turbo_plan.workers >= 2  # ruff: ignore[private-member-access] - MT paths actually taken


def test_turbo_budget_gate_falls_back_to_serial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DN field above half the budget -> file-based serial write; bytes still match."""
    monkeypatch.setattr(pipeline_mod.turbop, "resolve_cap", _identity_cap)
    p0 = Pipeline(str(DATA_CSV), "price", "longitude", "latitude", str(tmp_path / "ref"), res=50.0, workers=4)
    v0, s0 = p0.run()
    p1 = Pipeline(
        str(DATA_CSV),
        "price",
        "longitude",
        "latitude",
        str(tmp_path / "turbo"),
        res=50.0,
        workers=4,
        turbo=True,
        turbo_cap_gb=64.0,
    )
    p1.ingest()
    p1.calibrate()
    p1.grid()
    p1._turbo_precheck()  # ruff: ignore[private-member-access]
    p1._turbo_budget_bytes = 1000  # ruff: ignore[private-member-access] - force the DN field gate
    v1, s1 = p1._write_rasters()  # ruff: ignore[private-member-access]
    assert _sha(v0) == _sha(v1)
    assert _sha(s0) == _sha(s1)


def test_turbo_out_crs_equals_work_crs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """out_crs == work_crs: the block-write file is final; turbo adds nothing."""
    monkeypatch.setattr(pipeline_mod.turbop, "resolve_cap", _identity_cap)
    p0 = Pipeline(
        str(DATA_CSV),
        "price",
        "longitude",
        "latitude",
        str(tmp_path / "ref"),
        res=50.0,
        workers=4,
        out_crs=WORK_CRS,
    )
    v0, s0 = p0.run()
    p1 = Pipeline(
        str(DATA_CSV),
        "price",
        "longitude",
        "latitude",
        str(tmp_path / "turbo"),
        res=50.0,
        workers=4,
        out_crs=WORK_CRS,
        turbo=True,
        turbo_cap_gb=64.0,
    )
    v1, s1 = p1.run()
    assert _sha(v0) == _sha(v1)
    assert _sha(s0) == _sha(s1)


def test_box_count_banded_mt_matches_serial() -> None:
    """box_count_banded n_threads=4 == n_threads=1 (the _prepare_shared wiring)."""
    rng = np.random.default_rng(3)
    nx = ny = 800
    x = rng.integers(0, nx, 20_000)
    y = rng.integers(0, ny, 20_000)
    tv = rng.uniform(0.0, 10.0, 20_000)
    _s0, c0 = bin_points(x, y, tv, nx, ny)
    cap_cells = 5
    near1 = np.zeros((nx, ny), dtype=bool)
    near4 = np.zeros((nx, ny), dtype=bool)
    box_count_banded(c0, cap_cells, near1, n_threads=1)
    box_count_banded(c0, cap_cells, near4, n_threads=4)
    assert np.array_equal(near1, near4)
    assert near1.any()
    assert not near1.all()


def test_cli_turbo_flags() -> None:
    """--turbo [preset], --max-ram, --ram-gb, --turbo-strict parse and validate."""
    base = ["x.csv", "--value-col", "v", "--lng-col", "l", "--lat-col", "la"]
    parser = _build_parser()
    a = parser.parse_args([*base, "--turbo", "16"])
    assert a.turbo == "16"
    a = parser.parse_args([*base, "--turbo"])
    assert a.turbo == "auto"
    a = parser.parse_args([*base, "--max-ram", "20"])
    assert a.max_ram == 20.0
    for extra in (
        ["--turbo", "24"],
        ["--max-ram", "16", "--ram-gb", "32"],
        ["--turbo-strict"],
        ["--max-ram", "0"],
    ):
        with pytest.raises(SystemExit):
            pipeline_mod.main([*base, *extra])


def test_cli_turbo_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--turbo 16 + --turbo-strict on an infeasible grid exits 3 (A9 CLI path)."""
    monkeypatch.setattr(pipeline_mod.turbop, "resolve_cap", _identity_cap)
    # The melb grid at 100 m is far too small to be infeasible under 16 GB;
    # swap in the A9 anchor geometry at the pre-check point.
    orig = Pipeline._turbo_precheck  # ruff: ignore[private-member-access]

    def a9_hack(self: Pipeline) -> None:
        """Inject the A9 anchor grid, then run the real pre-check."""
        self.n, self.nx_padded, self.ny_padded, self.res, self.cap_km_val = (
            400_000,
            _A9_WC,
            _A9_CELLS // _A9_WC,
            100.0,
            25,
        )
        orig(self)

    monkeypatch.setattr(Pipeline, "_turbo_precheck", a9_hack)
    with pytest.raises(SystemExit) as ei:
        pipeline_mod.main(
            [
                str(DATA_CSV),
                "--value-col",
                "price",
                "--lng-col",
                "longitude",
                "--lat-col",
                "latitude",
                "--res",
                "100",
                "--cap-km",
                "25",
                "--turbo",
                "16",
                "--turbo-strict",
                "-o",
                str(tmp_path / "o"),
            ]
        )
    assert ei.value.code == 3
