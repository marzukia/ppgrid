"""Tests for ppgrid.pipeline."""

import json
import sys
import warnings
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pandas as pd
import pytest
import rasterio

from ppgrid.pipeline import Pipeline


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


def _write_pos_values_csv(tmp_path: Path, n: int = 20) -> str:
    """Write n points with positive values spanning 1.0..9.0 (log/sqrt both valid)."""
    rng = np.random.default_rng(1)
    csv = tmp_path / "pos.csv"
    pd.DataFrame(
        {
            "value": np.linspace(1.0, 9.0, n),
            "longitude": 144.6 + rng.uniform(-0.2, 0.2, n),
            "latitude": -37.7 + rng.uniform(-0.2, 0.2, n),
        },
    ).to_csv(csv, index=False)
    return str(csv)


def _write_cal_file(tmp_path: Path, name: str, cap_km: float = 20.0) -> str:
    """Write a cal JSON as if a prior auto run had picked transform `name`."""
    cal = tmp_path / "calibration.json"
    cal.write_text(json.dumps({"transform": name, "transform_state": {"name": name}, "cap_km": cap_km}))
    return str(cal)


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
    """End-to-end: percentile_step=5 -> every output percentile is a multiple of 5."""
    rng = np.random.default_rng(0)
    csv = tmp_path / "pts.csv"
    pd.DataFrame(
        {
            "value": rng.uniform(1.0, 100.0, 100),
            "longitude": 144.6 + rng.uniform(-0.2, 0.2, 100),
            "latitude": -37.7 + rng.uniform(-0.2, 0.2, 100),
        },
    ).to_csv(csv, index=False)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    p = Pipeline(
        str(csv),
        "value",
        "longitude",
        "latitude",
        str(out_dir),
        res=500.0,
        cap_km=2.0,
        workers=1,
        skip_calibration=True,
        percentile_step=5.0,
    )
    vpath, _ = p.run()

    with rasterio.open(vpath) as ds:
        arr = ds.read(1)
        vals = arr[arr != ds.nodata].astype(np.float64) / 100.0
        assert len(vals) > 0
        rem = vals % 5.0
        assert np.all(np.minimum(rem, 5.0 - rem) < 1e-6), (
            f"off-step values: {vals[(rem > 1e-6) & (rem < 5 - 1e-6)][:5]}"
        )
        assert np.all((vals >= 0.0) & (vals <= 100.0))
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
def test_forced_transform_invalid_hard_error(tmp_path: Path, bad_transform: str) -> None:
    """Explicit --transform log10/sqrt on negative data -> ValueError, no silent fallback (issue #5)."""
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
    with pytest.raises(ValueError, match="not valid for this data"):
        p.calibrate()


@pytest.mark.parametrize("bad_transform", ["log10", "sqrt"])
def test_auto_transform_invalid_falls_back(tmp_path: Path, bad_transform: str) -> None:
    """Auto path + a cal file that picked log10/sqrt, on negative data -> warn + identity (issue #5)."""
    csv = _write_neg_values_csv(tmp_path)
    cal_path = _write_cal_file(tmp_path, bad_transform)
    p = Pipeline(
        csv,
        "value",
        "longitude",
        "latitude",
        str(tmp_path / "out"),
        cap_km=20.0,
        transform="auto",
        calib_path=cal_path,
    )
    p.ingest()
    with pytest.warns(UserWarning, match="falling back to identity"):
        p.calibrate()
    assert p.tname == "identity"
    assert np.all(np.isfinite(p.tv))
    assert p._cal["transform"] == "identity" == p._cal["transform_state"]["name"]  # ruff: ignore[private-member-access]


def test_cal_transform_name_matches_state_after_forced_transform(tmp_path: Path) -> None:
    """Cal dict: transform name must agree with transform_state.name (issue #5).

    A cal file that picked log10 is reused with an explicit --transform sqrt:
    after calibrate, the cal JSON must hold transform == transform_state.name
    == "sqrt", not a stale "log10" beside the sqrt state.
    """
    csv = _write_pos_values_csv(tmp_path)
    cal_path = _write_cal_file(tmp_path, "log10")
    p = Pipeline(
        csv,
        "value",
        "longitude",
        "latitude",
        str(tmp_path / "out"),
        cap_km=20.0,
        transform="sqrt",
        calib_path=cal_path,
    )
    p.ingest()
    p.calibrate()
    assert p.tname == "sqrt"
    assert p._cal["transform"] == "sqrt"  # ruff: ignore[private-member-access]
    assert p._cal["transform"] == p._cal["transform_state"]["name"]  # ruff: ignore[private-member-access]


@pytest.mark.parametrize("bad_transform", ["log10", "sqrt"])
def test_auto_fallback_not_all_zero_surface(tmp_path: Path, bad_transform: str) -> None:
    """End-to-end: auto + a log10/sqrt cal file on negative data must not yield an all-DN-0 surface.

    This is the regression test that would have caught the silent corruption
    (issue #2): before the fix, NaN in the transform space made the whole
    surface read percentile 0 (DN 0) instead of real values. The auto path
    rejects the invalid transform with a warning and falls back to identity.
    """
    csv = _write_neg_values_csv(tmp_path)
    cal_path = _write_cal_file(tmp_path, bad_transform)
    p = Pipeline(
        csv,
        "value",
        "longitude",
        "latitude",
        str(tmp_path / "out"),
        res=5000.0,
        cap_km=20.0,
        workers=1,
        transform="auto",
        calib_path=cal_path,
        out_crs=6933,
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
    from ppgrid.pipeline import main

    old_argv = sys.argv
    sys.argv = ["ppgrid", "--version"]
    try:
        with pytest.raises(SystemExit) as exc:
            main()
    finally:
        sys.argv = old_argv
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"ppgrid {__version__}"


def test_cli_explicit_invalid_transform_hard_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI: explicit --transform log10 on negative data -> exit 2 + clear error (issue #5)."""
    from ppgrid.pipeline import main

    csv = _write_neg_values_csv(tmp_path)
    old_argv = sys.argv
    sys.argv = ["ppgrid", csv, "-o", str(tmp_path / "out"), "--transform", "log10", "--cap-km", "20"]
    try:
        with pytest.raises(SystemExit) as exc:
            main()
    finally:
        sys.argv = old_argv
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "error:" in err
    assert "log10" in err
    assert "not valid" in err


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
        out_crs=6933,
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


@pytest.mark.parametrize(
    ("extra_args", "expect"),
    [
        (["--res", "0"], "res"),
        (["--res", "-4"], "res"),
        (["--workers", "0"], "workers"),
        (["--scale", "0"], "scale"),
        (["--saturation", "0"], "saturation"),
        (["--percentile-step", "0"], "percentile-step"),
        (["--percentile-step", "150"], "percentile-step"),
        (["--block", "0"], "block"),
    ],
)
def test_cli_arg_validation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra_args: list[str],
    expect: str,
) -> None:
    """argparse-level validation: each bad flag exits 2 with its name on stderr.

    This is where the tier-1 #2 bugs hid (CLI validation gaps).
    """
    from ppgrid.pipeline import main

    old_argv = sys.argv
    sys.argv = ["ppgrid", "in.csv", "-o", str(tmp_path / "out"), *extra_args]
    try:
        with pytest.raises(SystemExit) as exc:
            main()
    finally:
        sys.argv = old_argv
    assert exc.value.code == 2
    assert expect in capsys.readouterr().err


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
