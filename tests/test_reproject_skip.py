"""Tests for the NoData-tile skip in _reproject_band (C(ii)) and --max-band-parallel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import ppgrid.pipeline as pgrid
from ppgrid.pipeline import NODATA, Pipeline, _reproject_band

PROFILE = {
    "driver": "GTiff",
    "count": 1,
    "dtype": "int16",
    "nodata": NODATA,
    "tiled": True,
    "blockxsize": 512,
    "blockysize": 512,
    "compress": "LZW",
}


def _make_src(path: Path, nx: int = 6144, ny: int = 6144, res: float = 10.0, bsize: int = 1024) -> tuple:
    """Return the transform of an identity-geometry raster, NoData except block (4, 4)."""
    t = from_origin(0.0, ny * res, res, res)
    arr = np.full((ny, nx), NODATA, dtype=np.int16)
    arr[4 * bsize : 5 * bsize, 4 * bsize : 5 * bsize] = 42
    with rasterio.open(path, "w", width=nx, height=ny, transform=t, crs="EPSG:3857", **PROFILE) as dst:
        dst.write(arr, 1)
    return t


def _task_info_around_44(nby: int = 6, bsize: int = 1024) -> dict:
    """Return task_info for the 3x3 task neighbourhood of block (4, 4)."""
    bids = {bx * nby + by for bx in (3, 4, 5) for by in (3, 4, 5)}
    return {"bsize": bsize, "nby": nby, "task_bids": bids}


def test_reproject_skip_output_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Skipped tiles give pixel-identical output with fewer reproject calls."""
    src = tmp_path / "src.tif"
    t = _make_src(src)
    dst_plain = tmp_path / "plain.tif"
    dst_skip = tmp_path / "skip.tif"

    calls = {"n": 0}
    orig = pgrid.reproject

    def counting(*a: object, **kw: object) -> object:
        calls["n"] += 1
        return orig(*a, **kw)

    monkeypatch.setattr(pgrid, "reproject", counting)

    calls["n"] = 0
    _reproject_band(str(src), str(dst_plain), PROFILE, "EPSG:3857", t, 6144, 6144)
    n_plain = calls["n"]

    calls["n"] = 0
    _reproject_band(str(src), str(dst_skip), PROFILE, "EPSG:3857", t, 6144, 6144, _task_info_around_44())
    n_skip = calls["n"]

    with rasterio.open(dst_plain) as a, rasterio.open(dst_skip) as b:
        da, db = a.read(1), b.read(1)
    assert np.array_equal(da, db)  # same pixels, skip or no skip
    # 9 coarse tiles; the 5 not touching the task neighbourhood are skipped.
    assert n_plain == 9
    assert n_skip == 4
    # Every skipped tile must be pure NoData in the output.
    for r, c in [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)]:
        assert np.all(da[r * 2048 : (r + 1) * 2048, c * 2048 : (c + 1) * 2048] == NODATA)
    # Data survives in the warped tile containing block (4, 4).
    assert np.any(da[4 * 1024 : 5 * 1024, 4 * 1024 : 5 * 1024] != NODATA)


def test_reproject_skip_no_task_info_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """task_info=None (no populated blocks) behaves exactly like the old path."""
    src = tmp_path / "src.tif"
    t = _make_src(src)
    dst = tmp_path / "out.tif"
    calls = {"n": 0}
    orig = pgrid.reproject

    def counting(*a: object, **kw: object) -> object:
        calls["n"] += 1
        return orig(*a, **kw)

    monkeypatch.setattr(pgrid, "reproject", counting)
    _reproject_band(str(src), str(dst), PROFILE, "EPSG:3857", t, 6144, 6144, None)
    assert calls["n"] == 9


def test_pipeline_max_band_parallel_validation() -> None:
    """max_band_parallel is validated in Pipeline.__init__."""
    args = ("in.csv", "value", "lng", "lat", "out/")
    with pytest.raises(ValueError, match="max_band_parallel"):
        Pipeline(*args, max_band_parallel=0)
    p = Pipeline(*args, max_band_parallel=3)
    assert p.max_band_parallel == 3
    assert Pipeline(*args).max_band_parallel is None


def _tiny_argv(tmp_path: Path) -> list[str]:
    """Write a 2-point CSV and return a minimal CLI argv for it."""
    csv = tmp_path / "pts.csv"
    csv.write_text("value,lng,lat\n1.0,149.0,-35.0\n2.0,149.1,-35.1\n")
    return [
        "ppgrid",
        str(csv),
        "-o",
        str(tmp_path / "out"),
        "--value-col",
        "value",
        "--lng-col",
        "lng",
        "--lat-col",
        "lat",
        "--res",
        "100",
        "--block",
        "2048",
        "--workers",
        "2",
    ]


def test_max_band_parallel_cli_passthrough(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--max-band-parallel N reaches Pipeline as max_band_parallel=N."""
    captured: dict = {}

    class FakePipeline:
        def __init__(self, *_a: object, **kw: object) -> None:
            self.captured = captured
            captured.update(kw)

        def run(self, *_a: object, **_kw: object) -> tuple[str, str]:
            assert self.captured is not None
            return str(tmp_path / "v.tif"), str(tmp_path / "s.tif")

    monkeypatch.setattr(pgrid, "Pipeline", FakePipeline)
    argv = [*_tiny_argv(tmp_path), "--max-band-parallel", "3"]
    monkeypatch.setattr("sys.argv", argv)
    pgrid.main()
    assert captured["max_band_parallel"] == 3


def test_max_band_parallel_cli_defaults_to_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without --max-band-parallel, Pipeline receives max_band_parallel=None."""
    captured: dict = {}

    class FakePipeline:
        def __init__(self, *_a: object, **kw: object) -> None:
            self.captured = captured
            captured.update(kw)

        def run(self, *_a: object, **_kw: object) -> tuple[str, str]:
            assert self.captured is not None
            return str(tmp_path / "v.tif"), str(tmp_path / "s.tif")

    monkeypatch.setattr(pgrid, "Pipeline", FakePipeline)
    monkeypatch.setattr("sys.argv", _tiny_argv(tmp_path))
    pgrid.main()
    assert captured["max_band_parallel"] is None
