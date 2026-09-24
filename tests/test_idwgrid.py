"""Tests for ppgrid.idwgrid."""

import importlib.util
import json
import sys
import warnings
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pandas as pd
import pytest
import rasterio

from ppgrid.idwgrid import Pipeline


def _run_cli(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    """Run ppgrid main() with argv; return (exit code, stderr)."""
    from ppgrid.idwgrid import main

    old_argv = sys.argv
    sys.argv = ["ppgrid", *argv]
    try:
        with pytest.raises(SystemExit) as exc:
            main()
    finally:
        sys.argv = old_argv
    code = exc.value.code if isinstance(exc.value.code, int) else 0
    return code, capsys.readouterr().err


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
    import numpy as np

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


def test_cli_saturation_zero_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    """`--saturation 0` exits with a clear CLI error (issue #2)."""
    from ppgrid.idwgrid import main

    old_argv = sys.argv
    sys.argv = ["ppgrid", "in.csv", "-o", "out", "--saturation", "0"]
    try:
        with pytest.raises(SystemExit) as exc:
            main()
    finally:
        sys.argv = old_argv
    assert exc.value.code == 2
    assert "saturation" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--scale", "nan"),
        ("--scale", "inf"),
        ("--scale", "-5"),
        ("--res", "nan"),
        ("--res", "inf"),
        ("--cap-km", "-5"),
        ("--cap-km", "inf"),
        ("--cap-km", "nan"),
        ("--saturation", "inf"),
        ("--percentile-step", "nan"),
        ("--percentile-step", "-5"),
        ("--calib-max-points", "0"),
        ("--calib-max-points", "-1"),
        ("--src-crs", "-1"),
        ("--work-crs", "0"),
        ("--out-crs", "-4326"),
    ],
)
def test_cli_numeric_flags_reject_nan_inf_negative(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], flag: str, value: str
) -> None:
    """#13: nan/inf/invalid negatives are rejected at parse time (exit 2), not silently rastered."""
    code, err = _run_cli(["in.csv", "-o", str(tmp_path / "out"), flag, value], capsys)
    assert code == 2, err
    assert flag in err
    assert "Traceback" not in err
    assert not (tmp_path / "out" / "value.tif").exists()


def test_cli_cap_km_auto_still_accepted(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """#13: 'auto' is not a number and must still parse; the failure then is a clean exit-1."""
    code, err = _run_cli(["nope.csv", "-o", str(tmp_path / "out"), "--cap-km", "auto"], capsys)
    assert code == 1
    assert "error:" in err


def test_cli_missing_file_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """#16: missing input file -> clean `error:`, exit 1, no traceback."""
    code, err = _run_cli(["no_such_file.csv", "-o", str(tmp_path / "out")], capsys)
    assert code == 1
    assert err.startswith("error:")
    assert "no_such_file.csv" in err
    assert "Traceback" not in err


def test_cli_missing_column_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """#16: missing value column -> clean `error:`, exit 1, no traceback."""
    csv = _write_neg_values_csv(tmp_path)
    code, err = _run_cli([csv, "-o", str(tmp_path / "out"), "--value-col", "nope"], capsys)
    assert code == 1
    assert err.startswith("error:")
    assert "nope" in err
    assert "Traceback" not in err


def test_cli_bad_crs_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """#16: invalid --src-crs -> clean `error:`, exit 1, no traceback."""
    csv = _write_neg_values_csv(tmp_path)
    code, err = _run_cli([csv, "-o", str(tmp_path / "out"), "--src-crs", "999999"], capsys)
    assert code == 1
    assert err.startswith("error:")
    assert "999999" in err
    assert "Traceback" not in err


def test_cli_non_numeric_cell_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """#16: non-numeric value cell -> clean `error:`, exit 1, no traceback."""
    csv = tmp_path / "bad.csv"
    csv.write_text("value,longitude,latitude\nabc,144.6,-37.7\n1.0,144.7,-37.8\n")
    code, err = _run_cli([str(csv), "-o", str(tmp_path / "out")], capsys)
    assert code == 1
    assert "error:" in err
    assert "abc" in err
    assert "Traceback" not in err


def test_cli_out_is_file_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """#16: -o pointing at a file -> clean `error:`, exit 1, no traceback."""
    f = tmp_path / "outfile"
    f.write_text("x")
    csv = _write_neg_values_csv(tmp_path)
    code, err = _run_cli([csv, "-o", str(f)], capsys)
    assert code == 1
    assert "error:" in err
    assert "Traceback" not in err


def test_cli_malformed_csv_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """#16: unclosed quote -> clean `error:`, exit 1, no traceback."""
    csv = tmp_path / "bad.csv"
    csv.write_text('value,longitude,latitude\n"1.0,144.6,-37.7\n2.0,144.7,-37.8\n')
    code, err = _run_cli([str(csv), "-o", str(tmp_path / "out")], capsys)
    assert code == 1
    assert "error:" in err
    assert "Traceback" not in err


def test_cli_parquet_without_pyarrow_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """#16: .parquet input without pyarrow -> clean `error:` naming the parquet extra."""
    if importlib.util.find_spec("pyarrow") is not None:
        pytest.skip("pyarrow installed; ImportError path not reachable")
    pq = tmp_path / "in.parquet"
    pq.write_bytes(b"")
    code, err = _run_cli([str(pq), "-o", str(tmp_path / "out")], capsys)
    assert code == 1
    assert "error:" in err
    assert "pyarrow" in err
    assert "Traceback" not in err


def test_cli_calibration_missing_file_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """#21: --calibration pointing at a missing file is a clean error; no file created."""
    csv = _write_neg_values_csv(tmp_path)
    cal = tmp_path / "cal" / "missing.json"
    code, err = _run_cli([csv, "-o", str(tmp_path / "out"), "--calibration", str(cal)], capsys)
    assert code == 1
    assert "error:" in err
    assert "not found" in err
    assert "Traceback" not in err
    assert not cal.exists()
    assert not (tmp_path / "out" / "calibration.json").exists()


def test_mid_run_failure_cleans_temps_warns_partial(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#15: a mid-run worker failure removes temp files and warns about partial rasters."""
    import ppgrid.idwgrid as ig

    rng = np.random.default_rng(0)
    csv = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "value": rng.uniform(1.0, 10.0, 100),
            "longitude": 144.6 + rng.uniform(-0.1, 0.1, 100),
            "latitude": -37.7 + rng.uniform(-0.1, 0.1, 100),
        },
    ).to_csv(csv, index=False)
    out = tmp_path / "out"

    def boom(_args: tuple[int, int]) -> NoReturn:
        msg = "worker boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(ig, "_process_block", boom)
    p = Pipeline(
        str(csv),
        "value",
        "longitude",
        "latitude",
        str(out),
        res=1000.0,
        cap_km=10.0,
        block_size=64,
        workers=1,
        skip_calibration=True,
    )
    with pytest.raises(RuntimeError, match="worker boom"):
        p.run()
    leftover = list(out.glob("_*.npy"))
    assert not leftover, f"temp files leaked: {leftover}"
    assert (out / "value.tif").exists()  # partial raster left on disk
    assert "partial" in capsys.readouterr().err


def test_points_temp_fresh_0600_symlink_not_followed(tmp_path: Path) -> None:
    """#15: _points.npy is a fresh 0600 file; a pre-planted symlink is replaced, not followed."""
    rng = np.random.default_rng(0)
    csv = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "value": rng.uniform(1.0, 10.0, 50),
            "longitude": 144.6 + rng.uniform(-0.1, 0.1, 50),
            "latitude": -37.7 + rng.uniform(-0.1, 0.1, 50),
        },
    ).to_csv(csv, index=False)
    out = tmp_path / "out"
    victim = tmp_path / "victim.npy"
    victim.write_bytes(b"VICTIM-DATA")
    out.mkdir()
    (out / "_points.npy").symlink_to(victim)

    p = Pipeline(
        str(csv),
        "value",
        "longitude",
        "latitude",
        str(out),
        res=1000.0,
        cap_km=10.0,
        block_size=128,
        workers=1,
        skip_calibration=True,
    )
    p.ingest()
    p.calibrate()
    p.grid()
    mode = p.pts_path.stat().st_mode & 0o777
    assert mode == 0o600, oct(mode)
    vpath, _ = p.run()
    assert victim.read_bytes() == b"VICTIM-DATA"
    assert Path(vpath).exists()
    assert not (out / "_points.npy").exists()


def test_constant_values_pipeline_percentile_50(tmp_path: Path) -> None:
    """#24: constant data decodes to the 50th percentile (DN 5000), not 0."""
    xx, yy = np.meshgrid(np.linspace(144.55, 144.65, 10), np.linspace(-37.75, -37.65, 10))
    csv = tmp_path / "const.csv"
    pd.DataFrame(
        {
            "value": np.full(100, 42.0),
            "longitude": xx.ravel(),
            "latitude": yy.ravel(),
        },
    ).to_csv(csv, index=False)
    out = tmp_path / "out"
    p = Pipeline(
        str(csv),
        "value",
        "longitude",
        "latitude",
        str(out),
        res=5000.0,
        cap_km=10.0,
        workers=1,
        skip_calibration=True,
    )
    vpath, _ = p.run()
    with rasterio.open(vpath) as ds:
        arr = ds.read(1)
    valid = arr[arr != ds.nodata]
    assert valid.size > 0
    assert np.all(valid == 5000), f"expected DN 5000 (50th pct), got {np.unique(valid)[:5]}"


def test_constant_values_calibration_strict_json(tmp_path: Path) -> None:
    """#14: constant value column -> calibration.json is strict RFC 8259 JSON.

    Zero variance (skill -inf) + extent < CV block (empty folds,
    'Mean of empty slice') must not leak NaN/Infinity into the file.
    """
    rng = np.random.default_rng(0)
    n = 40
    csv = tmp_path / "const.csv"
    pd.DataFrame(
        {
            "value": np.full(n, 5.0),
            "longitude": 144.6 + rng.uniform(-0.1, 0.1, n),
            "latitude": -37.7 + rng.uniform(-0.1, 0.1, n),
        },
    ).to_csv(csv, index=False)
    out = tmp_path / "out"
    p = Pipeline(
        str(csv),
        "value",
        "longitude",
        "latitude",
        str(out),
        res=5000.0,
        cap_km="auto",
        block_size=512,
        workers=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        p.run()
    text = (out / "calibration.json").read_text()
    assert "NaN" not in text
    assert "Infinity" not in text
    parsed = json.loads(text, parse_constant=lambda c: pytest.fail(f"non-strict JSON constant: {c}"))
    for blk in parsed["cv"].values():
        assert blk["overall_skill"] == 0.0


def test_forced_transform_recorded_in_cal_file_reload_bit_identical(tmp_path: Path) -> None:
    """#23: the cal file must record the FORCED transform; reload with auto is bit-identical."""
    rng = np.random.default_rng(0)
    n = 500
    u = rng.uniform(0.0, 1.0, n)
    csv = tmp_path / "skew.csv"
    pd.DataFrame(
        {
            "value": np.exp(4.0 * u + rng.normal(0.0, 0.25, n)),  # right-skewed
            "longitude": 144.6 + u * 0.5,
            "latitude": -37.7 + rng.uniform(0.0, 0.5, n),
        },
    ).to_csv(csv, index=False)
    cal = tmp_path / "out1" / "calibration.json"
    kwargs = {"res": 2000.0, "cap_km": 10.0, "block_size": 1024, "workers": 1}

    p1 = Pipeline(
        str(csv),
        "value",
        "longitude",
        "latitude",
        str(tmp_path / "out1"),
        transform="percentile",
        **kwargs,
    )
    v1, s1 = p1.run()
    cal = tmp_path / "out1" / "calibration.json"
    recorded = json.loads(cal.read_text())
    assert recorded["transform"] == "percentile"
    assert recorded["transform_state"]["name"] == "percentile"

    p2 = Pipeline(
        str(csv),
        "value",
        "longitude",
        "latitude",
        str(tmp_path / "out2"),
        transform="auto",
        calib_path=str(cal),
        **kwargs,
    )
    v2, s2 = p2.run()

    with rasterio.open(v1) as a, rasterio.open(v2) as b:
        np.testing.assert_array_equal(a.read(1), b.read(1))
    with rasterio.open(s1) as a, rasterio.open(s2) as b:
        np.testing.assert_array_equal(a.read(1), b.read(1))


def test_explicit_cap_skips_cv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit cap_km must skip the blocked-CV fill-cap search (issue #2)."""
    import ppgrid.idwgrid as ig

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
