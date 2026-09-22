"""Tests for ppgrid.pipeline."""

import sys
import warnings
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pandas as pd
import pytest
import rasterio

from ppgrid.calibrate import PERCENTILE_MAX
from ppgrid.pipeline import DEFAULT_SCALE, WORK_CRS, Pipeline


def _write_neg_values_csv(tmp_path: Path, n: int = 20) -> str:
    """Write the issue #2 repro: n points with values spanning -1.0..2.7."""
    rng = np.random.default_rng(0)
    csv = tmp_path / "neg.csv"
    pd.DataFrame(
        {
            "value": np.linspace(-1.0, 2.7, n),
            "longitude": 144.6 + rng.uniform(-0.2, 0.2, n),
            "latitude": -37.7 + rng.uniform(-0.2, 0.2, n),
        },
    ).to_csv(csv, index=False)
    return str(csv)


def test_empty_input_error(tmp_path: Path) -> None:
    """Pipeline.ingest raises ValueError on empty CSV."""
    csv = tmp_path / "empty.csv"
    csv.write_text("value,longitude,latitude\n")
    p = Pipeline(str(csv), "value", "longitude", "latitude", str(tmp_path / "out"))
    with pytest.raises(ValueError, match="No valid points"):
        p.ingest()


def test_scale_overflow_error(tmp_path: Path) -> None:
    """Pipeline raises ValueError when scale * 100 > 32767."""
    csv = tmp_path / "data.csv"
    csv.write_text("value,longitude,latitude\n1.0,0.0,0.0\n")
    with pytest.raises(ValueError, match="exceeds int16 max"):
        Pipeline(str(csv), "value", "longitude", "latitude", str(tmp_path / "out"), scale=400.0)


def test_halo_exceeds_block_error(tmp_path: Path) -> None:
    """Pipeline.grid raises ValueError when halo > bsize."""
    csv = tmp_path / "data.csv"
    csv.write_text("value,longitude,latitude\n1.0,0.0,0.0\n")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # cap_km=1, res=500 => levels=1, step=2, halo=max(4,2)=4, bsize=max(3,2)=3 => 4>3
    p = Pipeline(
        str(csv),
        "value",
        "longitude",
        "latitude",
        str(out_dir),
        cap_km=1.0,
        block_size=3,
        skip_calibration=True,
    )
    p.ingest()
    p.calibrate()
    with pytest.raises(ValueError, match=r"halo.*exceeds block size"):
        p.grid()


def test_percentile_step_validation(tmp_path: Path) -> None:
    """Pipeline rejects percentile_step outside (0, 100]."""
    csv = tmp_path / "data.csv"
    csv.write_text("value,longitude,latitude\n1.0,0.0,0.0\n")
    for bad in (0.0, -5.0, 150.0):
        with pytest.raises(ValueError, match="percentile_step"):
            Pipeline(str(csv), "value", "longitude", "latitude", str(tmp_path / "out"), percentile_step=bad)
    # boundary values are accepted
    Pipeline(str(csv), "value", "longitude", "latitude", str(tmp_path / "out"), percentile_step=5.0)
    Pipeline(str(csv), "value", "longitude", "latitude", str(tmp_path / "out"), percentile_step=100.0)


def test_percentile_step_rounds_output(tmp_path: Path) -> None:
    """End-to-end: --percentile-step 5 -> every output percentile is a multiple of 5."""
    data_csv = tmp_path / "pts.csv"
    rng = np.random.default_rng(42)
    pd.DataFrame(
        {
            "value": rng.uniform(1.0, 10.0, 120),
            "longitude": 144.8 + rng.uniform(-0.3, 0.3, 120),
            "latitude": -37.6 + rng.uniform(-0.3, 0.3, 120),
        },
    ).to_csv(data_csv, index=False)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    p = Pipeline(
        str(data_csv),
        "value",
        "longitude",
        "latitude",
        str(out_dir),
        res=5000.0,
        cap_km=50.0,
        block_size=256,
        workers=1,
        skip_calibration=True,
        percentile_step=5.0,
    )
    vpath, _ = p.run()

    with rasterio.open(vpath) as ds:
        arr = ds.read(1)
        vals = arr[arr != ds.nodata].astype(np.float64) / DEFAULT_SCALE
        assert len(vals) > 0
        rem = vals % 5.0
        assert np.all(np.minimum(rem, 5.0 - rem) < 1e-6), (
            f"off-step values: {vals[(rem > 1e-6) & (rem < 5 - 1e-6)][:5]}"
        )
        assert np.all((vals >= 0.0) & (vals <= PERCENTILE_MAX))
        tags = ds.tags()
        assert tags.get("percentile_step") == "5.0"


def test_pipeline_run_with_equakes(tmp_path: Path) -> None:
    """Run the full pipeline on data/all_equakes.csv and verify output GeoTIFFs."""
    data_csv = Path(__file__).resolve().parent.parent / "data" / "all_equakes.csv"
    if not data_csv.exists():
        pytest.skip(f"Data file not found: {data_csv}")

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    p = Pipeline(
        str(data_csv),
        "mag",
        "longitude",
        "latitude",
        str(out_dir),
        res=5000.0,
        cap_km=50.0,
        block_size=2048,
        workers=1,
        skip_calibration=True,
    )
    vpath, spath = p.run()

    # Verify output files exist
    assert Path(vpath).exists()
    assert Path(spath).exists()

    # Verify they are valid GeoTIFFs with shape > 0
    with rasterio.open(vpath) as ds:
        assert ds.width > 0
        assert ds.height > 0
        arr = ds.read(1)
        assert arr.shape[0] > 0
        assert arr.shape[1] > 0

    with rasterio.open(spath) as ds:
        assert ds.width > 0
        assert ds.height > 0
        arr = ds.read(1)
        assert arr.shape[0] > 0
        assert arr.shape[1] > 0


@pytest.mark.parametrize("bad_transform", ["log10", "sqrt"])
def test_forced_transform_invalid_falls_back(tmp_path: Path, bad_transform: str) -> None:
    """Forced log10/sqrt on data with a negative value -> warning + identity, no NaN (issue #2)."""
    csv = _write_neg_values_csv(tmp_path)
    p = Pipeline(
        csv,
        "value",
        "longitude",
        "latitude",
        str(tmp_path / "out"),
        cap_km=20.0,
        skip_calibration=True,
        transform=bad_transform,
    )
    p.ingest()
    with pytest.warns(UserWarning, match="falling back to identity"):
        p.calibrate()
    assert p.tname == "identity"
    assert np.all(np.isfinite(p.tv))


@pytest.mark.parametrize("bad_transform", ["log10", "sqrt"])
def test_forced_transform_invalid_not_all_zero_surface(tmp_path: Path, bad_transform: str) -> None:
    """End-to-end: forced log10 on negative data must not yield an all-DN-0 surface (issue #2).

    This is the regression test that would have caught the silent corruption:
    before the fix, NaN in the transform space made the whole surface read
    percentile 0 (DN 0) instead of real values.
    """
    csv = _write_neg_values_csv(tmp_path)
    p = Pipeline(
        csv,
        "value",
        "longitude",
        "latitude",
        str(tmp_path / "out"),
        res=5000.0,
        cap_km=20.0,
        workers=1,
        skip_calibration=True,
        transform=bad_transform,
        out_crs=WORK_CRS,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vpath, _ = p.run()
    with rasterio.open(vpath) as ds:
        arr = ds.read(1)
    valid = arr[arr != ds.nodata]
    assert valid.size > 0, "no valid pixels at all"
    assert not np.all(valid == 0), "whole surface is DN=0 (percentile 0)"
    assert valid.mean() > 0


def test_cli_version_matches_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    """`ppgrid --version` must report the real package version, not a hardcoded one (issue #2)."""
    from ppgrid import __version__
    from ppgrid.idwgrid import main

    old_argv = sys.argv
    sys.argv = ["ppgrid", "--version"]
    try:
        with pytest.raises(SystemExit) as exc:
            main()
    finally:
        sys.argv = old_argv
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"ppgrid {__version__}"


def test_run_creates_missing_out_dir(tmp_path: Path) -> None:
    """run() with skip_calibration on a fresh/nonexistent out_dir must not crash (issue #2)."""
    csv = tmp_path / "pts.csv"
    pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
            "longitude": [144.6, 144.7, 144.8, 144.9, 144.65],
            "latitude": [-37.6, -37.7, -37.8, -37.65, -37.75],
        },
    ).to_csv(csv, index=False)
    out_dir = tmp_path / "nested" / "fresh"  # does not exist yet
    p = Pipeline(
        str(csv),
        "value",
        "longitude",
        "latitude",
        str(out_dir),
        res=5000.0,
        cap_km=20.0,
        workers=1,
        skip_calibration=True,
        out_crs=WORK_CRS,
    )
    vpath, spath = p.run()
    assert Path(vpath).exists()
    assert Path(spath).exists()


def test_saturation_zero_rejected(tmp_path: Path) -> None:
    """Pipeline rejects saturation <= 0 (c/0 = NaN corrupts the blend, issue #2)."""
    csv = tmp_path / "data.csv"
    csv.write_text("value,longitude,latitude\n1.0,0.0,0.0\n")
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="saturation"):
            Pipeline(str(csv), "value", "longitude", "latitude", str(tmp_path / "out"), saturation=bad)
    Pipeline(str(csv), "value", "longitude", "latitude", str(tmp_path / "out"), saturation=2.5)


def test_cli_saturation_zero_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    """`--saturation 0` exits with a clear CLI error (issue #2)."""
    from ppgrid.pipeline import main

    old_argv = sys.argv
    sys.argv = ["ppgrid", "in.csv", "-o", "out", "--saturation", "0"]
    try:
        with pytest.raises(SystemExit) as exc:
            main()
    finally:
        sys.argv = old_argv
    assert exc.value.code == 2
    assert "saturation" in capsys.readouterr().err


def test_explicit_cap_skips_cv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit cap_km must skip the blocked-CV fill-cap search (issue #2)."""
    import ppgrid.pipeline as ig

    rng = np.random.default_rng(0)
    csv = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "value": np.linspace(1.0, 10.0, 40),
            "longitude": 144.6 + rng.uniform(-0.2, 0.2, 40),
            "latitude": -37.7 + rng.uniform(-0.2, 0.2, 40),
        },
    ).to_csv(csv, index=False)

    def boom(*_args: Any, **_kwargs: Any) -> NoReturn:
        msg = "calibrate_fill_cap should not run when cap_km is explicit"
        raise AssertionError(msg)

    monkeypatch.setattr(ig, "calibrate_fill_cap", boom)
    p = Pipeline(str(csv), "value", "longitude", "latitude", str(tmp_path / "out"), cap_km=10.0)
    p.ingest()
    p.calibrate()
    assert p.cap_km_val == 10.0


def test_cli_rejects_bad_percentile_step(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`--percentile-step` outside (0, 100] exits with a clear CLI error."""
    from ppgrid.pipeline import main

    csv = tmp_path / "data.csv"
    csv.write_text("value,longitude,latitude\n1.0,0.0,0.0\n")
    for bad in ("0", "101", "-5"):
        old_argv = sys.argv
        sys.argv = ["ppgrid", str(csv), "-o", str(tmp_path / "out"), "--percentile-step", bad]
        try:
            with pytest.raises(SystemExit) as exc:
                main()
        finally:
            sys.argv = old_argv
        assert exc.value.code == 2
        assert "percentile-step" in capsys.readouterr().err


def test_cli_requires_out(capsys: pytest.CaptureFixture[str]) -> None:
    """`-o/--out` is required: running without it exits with a clear CLI error."""
    from ppgrid.pipeline import main

    old_argv = sys.argv
    sys.argv = ["ppgrid", "in.csv"]
    try:
        with pytest.raises(SystemExit) as exc:
            main()
    finally:
        sys.argv = old_argv
    assert exc.value.code == 2
    assert "-o/--out" in capsys.readouterr().err
