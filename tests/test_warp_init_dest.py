"""Regression test: out-of-coverage dst pixels must warp to NODATA, not 0.

rasterio.warp.reproject fills destination pixels with NO source coverage
with the GDAL INIT_DEST value. The coarse-tile warp helpers did not set it
explicitly: _warp_dst_tile (file-band source) inherits dst_nodata from the
source band on current rasterio, but _warp_dst_tile_arr (in-RAM array
source, the turbo write path) leaves src/dst nodata unset, so GDAL
initialised uncovered pixels to 0. With work CRS != output CRS the output
grid is the bounding box of the projected work-CRS rectangle; for a
geocentric work CRS (EPSG:3577) the corner wedges between that projected
footprint and the bbox are hundreds of km wide. The turbo writers
(_turbo_write_parallel / _turbo_write_stock_parallel) warp every coarse
block with no NoData-tile skip, so --turbo runs painted the entire wedge
with 0 instead of NODATA (Andryo national-run report 2026-10-15).

Geometry: a few hundred points over an AU-scale box, work CRS 3577
(geocentric metres), output CRS 4326, res 5 km. The projected work
footprint is strongly non-axis-aligned in 4326, so the wedge is
non-trivial (~80k cells on the hydrogen PROJ).

Guarantees checked:
- serial run: out-of-coverage cells are all NODATA, none 0;
- turbo + stock-parallel writer (zstdmt.available patched False): output
  byte-identical to the serial run and NODATA out-of-coverage;
- turbo + parallel-ZSTD writer (zstdmt stubbed): output pixels identical
  to the serial run and NODATA out-of-coverage.

"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

import ppgrid.pipeline as pipeline_mod
from ppgrid.pipeline import NODATA, TILE_PX, Pipeline, _tif_parse_ifd

if TYPE_CHECKING:
    import pytest

WORK = 3577  # GDA2020 geocentric, metres
OUT = 4326  # degrees: the projected work footprint is non-axis-aligned


def _write_pts(csv: Path, n: int = 500, seed: int = 7) -> None:
    """Seeded AU-scale point cloud (lon 114-153, lat -38..-12)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "val": rng.uniform(1.0, 100.0, n),
            "lon": rng.uniform(114.0, 153.0, n),
            "lat": rng.uniform(-38.0, -12.0, n),
        }
    )
    df.to_csv(csv, index=False)


def _make_pipeline(csv: Path, out_dir: Path, **kw: Any) -> Pipeline:
    return Pipeline(
        str(csv),
        "val",
        "lon",
        "lat",
        str(out_dir),
        res=5000.0,
        cap_km=1.0,
        workers=2,
        skip_calibration=True,
        work_crs=WORK,
        out_crs=OUT,
        transform="percentile",
        scale=1,
        **kw,
    )


def _coverage_mask(p: Pipeline, dst: rasterio.DatasetBase) -> np.ndarray:
    """Warp a ones raster over the pipeline work grid; True marks no coverage."""
    src_t = from_bounds(p.x0, p.y0 + p.ny * p.res, p.x0 + p.nx * p.res, p.y0, p.nx, p.ny)
    mask = np.zeros((dst.height, dst.width), dtype=np.int16)
    reproject(
        np.ones((p.ny, p.nx), dtype=np.int16),
        mask,
        src_transform=src_t,
        src_crs=f"EPSG:{p.work_crs}",
        dst_transform=dst.transform,
        dst_crs=dst.crs.to_string(),
        resampling=Resampling.nearest,
        nodata=NODATA,
        init_dest=-1,
    )
    return mask == -1


def _assert_nodata_outside(path: Path, p: Pipeline) -> None:
    """Out-of-coverage cells are NODATA (and specifically not 0)."""
    with rasterio.open(path) as dst:
        outside = _coverage_mask(p, dst)
        arr = dst.read(1)
    assert outside.sum() > 10_000  # the wedge must be non-trivial, or this test is vacuous
    bad = int((arr[outside] != NODATA).sum())
    assert bad == 0, f"{path.name}: {bad} out-of-coverage cells are not NODATA"
    assert not np.any(arr[outside] == 0), f"{path.name}: out-of-coverage cells contain 0"
    # Data survived inside the coverage.
    assert np.any(arr[~outside] != NODATA)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _identity_cap(preset: float | None, *_args: float | None, **_kwargs: float | None) -> tuple[float | None, str]:
    return preset, "explicit"


def test_warp_init_dest_serial_nodata_outside(tmp_path: Path) -> None:
    """Serial run (file-band warp): the wedge is NODATA, never 0."""
    csv = tmp_path / "pts.csv"
    _write_pts(csv)
    out = tmp_path / "ser"
    p = _make_pipeline(csv, out)
    v, s = p.run()
    _assert_nodata_outside(v, p)
    _assert_nodata_outside(s, p)


def test_warp_init_dest_turbo_stock_parallel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Turbo + _turbo_write_stock_parallel (zstdmt unavailable): byte-identical to serial, NODATA out-of-coverage."""
    csv = tmp_path / "pts.csv"
    _write_pts(csv)

    p0 = _make_pipeline(csv, tmp_path / "ser")
    v0, s0 = p0.run()

    monkeypatch.setattr(pipeline_mod.turbop, "resolve_cap", _identity_cap)
    monkeypatch.setattr(pipeline_mod.zstdmt, "available", lambda: False)
    p1 = _make_pipeline(csv, tmp_path / "turbo", turbo=True, turbo_cap_gb=64.0)
    v1, s1 = p1.run()

    # The zstd gate routed to the stock parallel writer, which succeeded
    # (no guard fallback to the serial write).
    assert any(f.startswith("zstd parallel write -> stock parallel") for f in p1._fallbacks)  # ruff: ignore[private-member-access]
    assert not any("serial write" in f for f in p1._fallbacks)  # ruff: ignore[private-member-access]
    # Byte-identical to the serial run (stock codec frames match).
    assert _sha(Path(v1)) == _sha(Path(v0))
    assert _sha(Path(s1)) == _sha(Path(s0))
    _assert_nodata_outside(v1, p1)
    _assert_nodata_outside(s1, p1)


def _gdal_zstd_frames(tiles: list[np.ndarray], width: int, height: int, probe: Path) -> list[bytes]:
    """Real GDAL ZSTD frames of the given (predictor-2-applied) tiles.

    The tiles are all full TILE_PX^2 (edge tiles zero-padded by the caller
    so they decode with the standard GDAL edge-tile crop); together they
    form an ntx x nty block grid that is a multiple of TILE_PX, which the
    stub writes as one tiled GTiff with compress=ZSTD and no predictor,
    then reads the stored frames back in raster-scan order. Works on every
    platform with a GDAL ZSTD codec (the pipeline's default output
    compression), so the parallel-ZSTD writer is exercised without the
    Linux-only CPL pfn resolution.

    """
    ntx = -(-width // TILE_PX)
    nty = -(-height // TILE_PX)
    # Tiles arrive in raster-scan order (ty*ntx + tx); arrange them as the
    # nty x ntx block grid the stub raster is written with.
    arr = (
        np.stack(tiles).reshape(nty, ntx, TILE_PX, TILE_PX).transpose(0, 2, 1, 3).reshape(nty * TILE_PX, ntx * TILE_PX)
    )
    with rasterio.open(
        probe,
        "w",
        driver="GTiff",
        width=ntx * TILE_PX,
        height=nty * TILE_PX,
        count=1,
        dtype="int16",
        tiled=True,
        blockxsize=TILE_PX,
        blockysize=TILE_PX,
        compress="ZSTD",
    ) as dst:
        dst.write(arr, 1)
    data = probe.read_bytes()
    info = _tif_parse_ifd(data)
    n = info["n_tiles"]
    assert n == len(tiles)
    offs = struct.unpack_from(info["e"] + f"{n}{'Q' if info['sz324'] == 8 else 'I'}", data, info["t324"][2])
    sizes = struct.unpack_from(info["e"] + f"{n}{'Q' if info['sz325'] == 8 else 'I'}", data, info["t325"][2])
    return [data[o : o + s] for o, s in zip(offs, sizes, strict=True)]


def test_warp_init_dest_turbo_parallel_zstd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Turbo + _turbo_write_parallel (zstdmt stubbed): pixels identical to serial, NODATA out-of-coverage.

    The stub replaces only the compression (the CPL zstd pfn is Linux-only);
    the tiles it receives are the writer's real warped content, so a
    0-filled wedge is caught here. The stock parallel writer would be
    byte-identical to the serial run on such a stack; here the parallel
    writer's own assembly (head patch + frame layout) is what runs.

    """
    csv = tmp_path / "pts.csv"
    _write_pts(csv)

    p0 = _make_pipeline(csv, tmp_path / "ser")
    v0, s0 = p0.run()
    with rasterio.open(v0) as ref:
        width, height = ref.width, ref.height

    counter = {"n": 0}

    def stub(tiles: list[np.ndarray], n_threads: int = 1) -> list[bytes]:  # ruff: ignore[unused-function-argument]
        counter["n"] += 1
        return _gdal_zstd_frames(tiles, width, height, tmp_path / f"_zstub_{counter['n']}.tif")

    monkeypatch.setattr(pipeline_mod.turbop, "resolve_cap", _identity_cap)
    monkeypatch.setattr(pipeline_mod.zstdmt, "available", lambda: True)
    monkeypatch.setattr(pipeline_mod.Pipeline, "_turbo_zstd_ok", lambda _self: True)
    monkeypatch.setattr(pipeline_mod.zstdmt, "compress_tiles", stub)
    p1 = _make_pipeline(csv, tmp_path / "turbo", turbo=True, turbo_cap_gb=64.0)
    v1, s1 = p1.run()

    # The parallel-ZSTD writer ran for both bands and did not fall back.
    assert counter["n"] == 2
    assert not p1._fallbacks  # ruff: ignore[private-member-access]

    # Pixels identical to the serial run (frames carry the warped content).
    with rasterio.open(v0) as a, rasterio.open(v1) as b, rasterio.open(s0) as c, rasterio.open(s1) as d:
        assert (a.width, a.height) == (b.width, b.height)
        assert np.array_equal(a.read(1), b.read(1))
        assert np.array_equal(c.read(1), d.read(1))
    _assert_nodata_outside(v1, p1)
    _assert_nodata_outside(s1, p1)
