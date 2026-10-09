"""Tests for ppgrid.pipeline."""

import hashlib
import json
import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import Affine

from ppgrid.calibrate import PercentileTransform
from ppgrid.pipeline import NODATA, WORK_CRS, Pipeline, _check_default_transform


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


# ---------------------------------------------------------------------------
# Issue #79: malformed user-supplied calibration.json -> friendly exit 2
# ---------------------------------------------------------------------------


def _three_point_csv(tmp_path: Path) -> str:
    csv = tmp_path / "pts3.csv"
    csv.write_text("value,lng,lat\n1.0,144.0,-37.0\n2.0,144.01,-37.01\n3.0,144.02,-37.02\n")
    return str(csv)


@pytest.mark.parametrize(
    ("cal", "match"),
    [
        # (a) percentile state without quantiles: was AttributeError (NoneType.tolist)
        (
            {"transform": "percentile", "transform_state": {"name": "percentile"}, "cap_km": 20.0},
            "transform_state.quantiles",
        ),
        # (b) unknown transform name, no transform_state: was raw StopIteration
        ({"transform": "bogus", "cap_km": 20.0}, "bogus"),
        # (c) wrong-length percentile_quantiles: was np.interp fp/xp mismatch at write time
        (
            {
                "transform": "identity",
                "transform_state": {"name": "identity"},
                "percentile_quantiles": [0.0, 1.0],
                "cap_km": 20.0,
            },
            "percentile_quantiles",
        ),
    ],
)
def test_bad_calibration_file_exit2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], cal: dict[str, Any], match: str
) -> None:
    """#79: all three malformed-file modes are one friendly exit-2 naming the file.

    No raw traceback text (AttributeError/StopIteration/np.interp) reaches stderr.
    """
    from ppgrid.pipeline import main

    csv = _three_point_csv(tmp_path)
    cal_path = tmp_path / "bad_calibration.json"
    cal_path.write_text(json.dumps(cal))
    with pytest.raises(SystemExit) as exc:
        main(
            [
                csv,
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
                "--calibration",
                str(cal_path),
            ]
        )
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert str(cal_path) in err
    assert match in err
    for needle in ("AttributeError", "StopIteration", "fp and xp", "Traceback"):
        assert needle not in err


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
        (["--res", "nan"], "res"),
        (["--res", "inf"], "res"),
        (["--workers", "0"], "workers"),
        (["--scale", "0"], "scale"),
        (["--scale", "nan"], "scale"),
        (["--scale", "inf"], "scale"),
        (["--saturation", "0"], "saturation"),
        (["--saturation", "nan"], "saturation"),
        (["--cap-km", "-5"], "cap-km"),
        (["--cap-km", "nan"], "cap-km"),
        (["--cap-km", "inf"], "cap-km"),
        (["--percentile-step", "0"], "percentile-step"),
        (["--percentile-step", "150"], "percentile-step"),
        (["--percentile-step", "nan"], "percentile-step"),
        (["--block", "0"], "block"),
        (["--calib-max-points", "0"], "calib-max-points"),
        (["--calib-max-points", "-1"], "calib-max-points"),
        (["--src-crs", "0"], "src-crs"),
        (["--work-crs", "-1"], "work-crs"),
        (["--out-crs", "0"], "out-crs"),
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

    #13 adds the numeric float guards: nan/inf/zero/negative must be rejected
    at parse time, not produce a silently wrong raster (all-0 from
    --scale nan, inverted SAT window from --cap-km -5, OverflowError from
    --cap-km inf, all-NODATA from --calib-max-points 0).
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


# ---------------------------------------------------------------------------
# Issue #13: numeric flags reject nan/inf at parse time
# ---------------------------------------------------------------------------


def test_scale_nan_does_not_exit_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """#13: --scale nan must not run to exit 0 and emit an all-zero raster."""
    from ppgrid.pipeline import main

    csv = _write_pos_values_csv(tmp_path)
    old_argv = sys.argv
    sys.argv = ["ppgrid", str(csv), "-o", str(tmp_path / "out"), "--scale", "nan"]
    try:
        with pytest.raises(SystemExit) as exc:
            main()
    finally:
        sys.argv = old_argv
    assert exc.value.code == 2
    assert "scale" in capsys.readouterr().err
    assert not (tmp_path / "out" / "value.tif").exists()


# ---------------------------------------------------------------------------
# Issue #14: strict RFC-8259 calibration.json on degenerate data
# ---------------------------------------------------------------------------


def _strict_json(text: str) -> Any:
    """json.loads that fails on NaN/Infinity (non-strict RFC-8259)."""

    def boom(name: str) -> Any:
        msg = f"non-strict JSON constant in calibration.json: {name}"
        raise AssertionError(msg)

    return json.loads(text, parse_constant=boom)


def _write_constant_csv(tmp_path: Path, n: int, spread_deg: float, value: float = 42.0) -> Path:
    """Constant-value CSV over a controlled extent (degrees around Melbourne)."""
    rng = np.random.default_rng(11)
    df = pd.DataFrame(
        {
            "value": np.full(n, value),
            "longitude": 144.96 + rng.uniform(-spread_deg / 2, spread_deg / 2, n),
            "latitude": -37.81 + rng.uniform(-spread_deg / 2, spread_deg / 2, n),
        }
    )
    p = tmp_path / "const.csv"
    df.to_csv(p, index=False)
    return p


@pytest.mark.parametrize(
    ("spread_deg", "case_id"),
    [(0.6, "zero-variance-predictions"), (0.002, "empty-folds")],
)
def test_constant_value_cal_file_is_strict_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], spread_deg: float, case_id: str
) -> None:
    """#14: constant values (zero variance) must not produce -inf/NaN in calibration.json.

    No 'Mean of empty slice' warnings, whether or not the CV folds yield any
    predictions (extent vs 50km CV block).
    """
    del case_id
    csv = _write_constant_csv(tmp_path, n=200, spread_deg=spread_deg)
    out = tmp_path / "out"
    p = Pipeline(
        str(csv),
        "value",
        "longitude",
        "latitude",
        str(out),
        res=1000.0,
        cap_km="auto",
        workers=2,
        out_crs=WORK_CRS,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        p.run()
    cal_path = out / "calibration.json"
    assert cal_path.exists()
    cal = _strict_json(cal_path.read_text(encoding="utf-8"))
    for entry in cal["cv"].values():
        assert math.isfinite(entry["overall_skill"])
    assert "Mean of empty slice" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Issue #15: temp cleanup on mid-run failure + 0600 + no symlink follow
# ---------------------------------------------------------------------------


def test_midrun_worker_failure_cleans_temps_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """#15: a mid-run worker error must warn about the failed write.

    Every temp file is removed (the write loop is wrapped in try/finally),
    and no final is left behind: the finals are only published via atomic
    rename on success (issue #40).
    """
    import ppgrid.pipeline as ig

    csv = _write_pos_values_csv(tmp_path)
    out = tmp_path / "out"
    observed: dict[str, int] = {}

    def boom(_task: tuple[int, int]) -> None:
        pts = out / "_points.npy"
        if pts.exists():
            observed["mode"] = pts.stat().st_mode & 0o777
        msg = "simulated worker OOM"
        raise RuntimeError(msg)

    monkeypatch.setattr(ig, "_process_block", boom)
    p = Pipeline(
        str(csv),
        "value",
        "longitude",
        "latitude",
        str(out),
        res=500.0,
        cap_km=2.0,
        workers=2,
        skip_calibration=True,
        out_crs=WORK_CRS,
    )
    with pytest.raises(RuntimeError, match="simulated worker OOM"):
        p.run()
    err = capsys.readouterr().err
    assert "failed during write" in err
    for fname in ig._RUN_TEMP_FILES:  # ruff: ignore[private-member-access]
        assert not (out / fname).exists(), f"temp left behind: {fname}"
    # Issue #40 (d): a mid-write crash no longer leaves truncated finals.
    assert not (out / "value.tif").exists()
    assert not (out / "support_km.tif").exists()
    assert observed.get("mode") == 0o600, f"temp was {oct(observed.get('mode'))}, expected 0600"


def test_preplanted_symlink_not_followed(tmp_path: Path) -> None:
    """#15: a pre-planted _points.npy symlink is replaced, never followed.

    The symlink target must not be overwritten.
    """
    csv = _write_pos_values_csv(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    victim = tmp_path / "victim.dat"
    victim.write_text("precious")
    (out / "_points.npy").symlink_to(victim)
    p = Pipeline(
        str(csv),
        "value",
        "longitude",
        "latitude",
        str(out),
        res=500.0,
        cap_km=2.0,
        workers=1,
        skip_calibration=True,
        out_crs=WORK_CRS,
    )
    p.run()
    assert victim.read_text() == "precious"
    assert not (out / "_points.npy").exists()


# ---------------------------------------------------------------------------
# Issue #16: clean 'error: <msg>' + non-zero exit, no traceback
# ---------------------------------------------------------------------------


def _run_cli(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str]:
    """Run ppgrid main() with sys.argv; return (exit code, stderr)."""
    from ppgrid.pipeline import main

    old_argv = sys.argv
    sys.argv = ["ppgrid", *args]
    try:
        try:
            main()
            code = 0
        except SystemExit as e:
            code = int(e.code) if e.code is not None else 1
    finally:
        sys.argv = old_argv
    return code, capsys.readouterr().err


def _assert_clean_error(code: int, err: str, needle: str) -> None:
    assert code in (1, 2), f"exit code {code}, expected non-zero"
    assert err.startswith("error: "), f"stderr must start with 'error: ', got: {err!r}"
    assert needle in err
    assert "Traceback" not in err


def test_cli_missing_file_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """#16: a missing input file is a clean error, not a traceback."""
    code, err = _run_cli(capsys, str(tmp_path / "nope.csv"), "-o", str(tmp_path / "out"))
    _assert_clean_error(code, err, "nope.csv")


def test_cli_missing_column_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """#16: a missing column is a clean error."""
    csv = tmp_path / "nocol.csv"
    csv.write_text("value,longitude\n1.0,144.0\n2.0,145.0\n")
    code, err = _run_cli(capsys, str(csv), "-o", str(tmp_path / "out"))
    _assert_clean_error(code, err, "latitude")


def test_cli_bad_crs_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """#16: an invalid --src-crs is a clean error."""
    csv = _write_pos_values_csv(tmp_path)
    code, err = _run_cli(capsys, str(csv), "-o", str(tmp_path / "out"), "--src-crs", "999999")
    _assert_clean_error(code, err, "999999")


def test_cli_o_is_file_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """#16: -o pointing at a file is a clean error."""
    csv = _write_pos_values_csv(tmp_path)
    f = tmp_path / "notadir"
    f.write_text("x")
    code, err = _run_cli(capsys, str(csv), "-o", str(f))
    _assert_clean_error(code, err, "notadir")


def test_cli_non_numeric_cell_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """#16: a non-numeric cell is a clean error."""
    csv = tmp_path / "badcell.csv"
    csv.write_text("value,longitude,latitude\n1.0,144.0,-37.0\nabc,145.0,-38.0\n")
    code, err = _run_cli(capsys, str(csv), "-o", str(tmp_path / "out"))
    _assert_clean_error(code, err, "abc")


def test_cli_malformed_csv_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """#16: a malformed CSV is a clean error."""
    csv = tmp_path / "ragged.csv"
    csv.write_text('value,longitude,latitude\n"1.0,144.0",145.0,-38.0\n"unterminated,145.0,-38.0\n')
    code, err = _run_cli(capsys, str(csv), "-o", str(tmp_path / "out"))
    _assert_clean_error(code, err, "malformed input file")


# ---------------------------------------------------------------------------
# Issue #21: --calibration <missing> is a user error, not a fresh create
# ---------------------------------------------------------------------------


def test_calibration_missing_file_errors_creates_nothing(tmp_path: Path) -> None:
    """#21: an absent --calibration path is a clean error; nothing is created."""
    csv = _write_pos_values_csv(tmp_path)
    missing = tmp_path / "nope.json"
    p = Pipeline(
        str(csv),
        "value",
        "longitude",
        "latitude",
        str(tmp_path / "out"),
        res=500.0,
        cap_km=2.0,
        workers=1,
        calib_path=str(missing),
        out_crs=WORK_CRS,
    )
    p.ingest()
    with pytest.raises(ValueError, match="--calibration file not found"):
        p.calibrate()
    assert not missing.exists()
    assert not (tmp_path / "out" / "calibration.json").exists()


def test_cli_calibration_missing_file_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """#21: an absent --calibration path via CLI is a clean error."""
    csv = _write_pos_values_csv(tmp_path)
    code, err = _run_cli(capsys, str(csv), "-o", str(tmp_path / "out"), "--calibration", str(tmp_path / "nope.json"))
    _assert_clean_error(code, err, "--calibration file not found")
    assert not (tmp_path / "nope.json").exists()
    assert not (tmp_path / "out" / "calibration.json").exists()


# ---------------------------------------------------------------------------
# Issue #23: forced transform recorded in cal file; auto path byte-identical
# ---------------------------------------------------------------------------

ANCHOR = Path(__file__).parent / "fixtures" / "auto_anchor"


def test_auto_path_byte_identical_to_af13522_anchor(tmp_path: Path) -> None:
    """#23: the auto (non-forced) path must stay byte-identical to af13522.

    The cal file write moved after transform resolution; its content must not
    have changed. Anchor was generated with the af13522 code.
    """
    out = tmp_path / "out"
    p = Pipeline(str(ANCHOR / "input.csv"), "val", "lon", "lat", str(out), res=500.0, workers=1)
    p.run()
    for name in ("value.tif", "support_km.tif", "calibration.json"):
        assert (out / name).read_bytes() == (ANCHOR / name).read_bytes(), f"{name} differs from af13522 anchor"


def _skewed_csv(path: Path, n: int = 500) -> Path:
    """Deterministic right-skewed dataset (same distribution as the anchor)."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "value": 10.0 + rng.lognormal(0.0, 1.5, n),
            "longitude": 144.6 + rng.uniform(-0.25, 0.25, n),
            "latitude": -37.7 + rng.uniform(-0.25, 0.25, n),
        }
    )
    df.to_csv(path, index=False)
    return path


def test_forced_transform_recorded_in_cal_file_and_reload_stable(tmp_path: Path) -> None:
    """#23: a forced --transform must be what calibration.json records.

    Not the auto choice; a later --transform auto run reloading that file must
    reproduce the forced output bit-identically.
    """
    csv = _skewed_csv(tmp_path / "in.csv")
    out_forced = tmp_path / "forced"
    p1 = Pipeline(
        str(csv),
        "value",
        "longitude",
        "latitude",
        str(out_forced),
        res=500.0,
        cap_km=5.0,
        workers=1,
        transform="percentile",
        out_crs=WORK_CRS,
    )
    p1.run()
    cal = json.loads((out_forced / "calibration.json").read_text(encoding="utf-8"))
    # Auto picked a different transform on this data, so the file content
    # proves the forced override is what got recorded.
    top_auto = max(cal["transform_scores"], key=cal["transform_scores"].get)
    assert top_auto != "percentile", "test data must make auto != percentile"
    assert cal["transform"] == "percentile"
    assert cal["transform_state"]["name"] == "percentile"

    out_reload = tmp_path / "reload"
    p2 = Pipeline(
        str(csv),
        "value",
        "longitude",
        "latitude",
        str(out_reload),
        res=500.0,
        cap_km=5.0,
        workers=1,
        transform="auto",
        calib_path=str(out_forced / "calibration.json"),
        out_crs=WORK_CRS,
    )
    p2.run()
    for name in ("value.tif", "support_km.tif"):
        assert (out_reload / name).read_bytes() == (out_forced / name).read_bytes(), f"{name} differs after cal reload"


# ---------------------------------------------------------------------------
# Issue #24: constant data must not quantize to 0
# ---------------------------------------------------------------------------


def _decode(pct: PercentileTransform, dn: np.ndarray) -> np.ndarray:
    """Decode int16 percentile DNs (default scale 100) back to raw values."""
    pv = dn.astype(np.float64) / 100.0
    return np.interp(pv, pct._p, pct.q)  # ruff: ignore[private-member-access]


def test_constant_csv_single_percentile_dn(tmp_path: Path) -> None:
    """#24: a constant value column must not collapse the value surface to 0.

    The pull-push mipmap blends empty cells toward its coarse (zero) ancestors,
    so the interpolated value field on constant input is 42*w with 0<w<=1
    (point cells: w=1). The flat percentile staircase must be centred on the
    constant: data cells sit at exactly the 50th percentile (DN 5000), and
    decoding the surface back through the cal LUT lands in (0, 42].
    """
    csv = _write_constant_csv(tmp_path, n=40, spread_deg=0.4, value=42.0)
    out = tmp_path / "out"
    p = Pipeline(
        str(csv),
        "value",
        "longitude",
        "latitude",
        str(out),
        res=1000.0,
        cap_km=5.0,
        workers=1,
        skip_calibration=True,
        out_crs=WORK_CRS,
    )
    vpath, _ = p.run()
    with rasterio.open(vpath) as ds:
        arr = ds.read(1).astype(np.int32)
        valid = arr[arr != NODATA]
    assert len(valid) > 0
    assert valid.max() == 5000, f"data cells must sit at exactly DN 5000, got max {valid.max()}"
    assert valid.min() > 0, f"surface collapsed toward percentile 0: min DN {valid.min()}"
    dec = _decode(p.pct_q, valid)
    assert np.isfinite(dec).all()
    assert dec.min() > 0.0
    assert abs(dec.max() - 42.0) < 0.5, f"data cells must decode to the constant, got {dec.max()}"


def test_constant_csv_negative_value(tmp_path: Path) -> None:
    """#24: same for a negative constant.

    The auto transform falls back to identity; the surface must still be
    centred on the 50th percentile, never zero-quantized.
    """
    csv = _write_constant_csv(tmp_path, n=40, spread_deg=0.4, value=-3.5)
    out = tmp_path / "out"
    p = Pipeline(
        str(csv),
        "value",
        "longitude",
        "latitude",
        str(out),
        res=1000.0,
        cap_km=5.0,
        workers=1,
        out_crs=WORK_CRS,
    )
    vpath, _ = p.run()
    with rasterio.open(vpath) as ds:
        arr = ds.read(1).astype(np.int32)
        valid = arr[arr != NODATA]
    assert len(valid) > 0
    assert valid.min() == 5000, f"data cells must sit at exactly DN 5000, got min {valid.min()}"
    assert (valid >= 5000).all()
    dec = _decode(p.pct_q, valid)
    assert np.isfinite(dec).all()
    assert dec.max() < 0.0, f"negative constant must decode negative, got max {dec.max()}"
    assert abs(dec.min() + 3.5) < 0.5, f"data cells must decode to the constant, got {dec.min()}"


# ---------------------------------------------------------------------------
# Issues #74/#76: coordinate domain validation + transform-result validation
# ---------------------------------------------------------------------------


def _write_domain_csv(tmp_path: Path, name: str, rows: list[tuple[float, float, float]]) -> str:
    """Write a (value, longitude, latitude) CSV for the domain tests."""
    csv = tmp_path / name
    pd.DataFrame(
        {
            "value": [r[0] for r in rows],
            "longitude": [r[1] for r in rows],
            "latitude": [r[2] for r in rows],
        },
    ).to_csv(csv, index=False)
    return str(csv)


def test_ingest_out_of_domain_lat_dropped_with_example(tmp_path: Path) -> None:
    """#76: lat 95 is silently mapped to inf by pyproj; the row is dropped.

    The counted, named summary replaces the pre-fix grid() crash (int64-cast
    inf: 'ix indices out of range' / 'cannot convert float NaN to integer').
    """
    csv = _write_domain_csv(
        tmp_path,
        "lat95.csv",
        [(10.0, 144.6, -37.7), (20.0, 144.7, -37.8), (30.0, 144.6, 95.0)],
    )
    p = Pipeline(csv, "value", "longitude", "latitude", str(tmp_path / "out"), res=500.0, cap_km=10.0)
    p.ingest()
    assert p.n == 2
    assert p.n_dropped_out_of_domain == 1
    assert p.n_dropped_nan_inf == 0
    assert "out-of-domain lat 95.000000 outside [-90, 90] (row 2)" in p._drop_examples[0]  # ruff: ignore[private-member-access]
    assert np.isfinite(p.x).all()
    assert np.isfinite(p.y).all()
    p.calibrate()
    p.grid()  # pre-fix: ValueError, root cause never stated
    assert p.nx * p.ny < 10000


def test_ingest_out_of_domain_lon_grid_matches_in_domain_subset(tmp_path: Path) -> None:
    """#76: lon 181 used to silently plan a global-span grid for 3 points.

    Pre-fix: 2.5M+ cells at res 500. The row is now dropped and the grid is
    identical to the one built from the in-domain points alone.
    """
    melb = [(10.0, 144.6, -37.7), (20.0, 144.7, -37.8)]
    csv3 = _write_domain_csv(tmp_path, "lon181.csv", [*melb, (30.0, 181.0, -37.9)])
    csv2 = _write_domain_csv(tmp_path, "lon181_in.csv", melb)
    p3 = Pipeline(csv3, "value", "longitude", "latitude", str(tmp_path / "o3"), res=500.0, cap_km=10.0)
    p3.ingest()
    assert p3.n == 2
    assert p3.n_dropped_out_of_domain == 1
    assert "lon 181.000000 outside [-180, 180] (row 2)" in p3._drop_examples[0]  # ruff: ignore[private-member-access]
    p2 = Pipeline(csv2, "value", "longitude", "latitude", str(tmp_path / "o2"), res=500.0, cap_km=10.0)
    p2.ingest()
    for p in (p3, p2):
        p.calibrate()
        p.grid()
    assert (p3.nx, p3.ny, p3.x0, p3.y0) == (p2.nx, p2.ny, p2.x0, p2.y0)
    assert p3.nx * p3.ny < 10000  # pre-fix: ~2.5M cells, global span


def test_ingest_nonfinite_projection_dropped(tmp_path: Path) -> None:
    """#76: in-domain 4326 input pyproj maps to non-finite in the work CRS.

    APRS pole singularity at lat -90: dropped with a count + named source
    coords, instead of reaching grid() as int64 min/max.
    """
    csv = _write_domain_csv(tmp_path, "aprs.csv", [(10.0, 0.0, -90.0), (20.0, 150.0, -37.0)])
    p = Pipeline(csv, "value", "longitude", "latitude", str(tmp_path / "out"), res=500.0, cap_km=10.0, work_crs=3408)
    p.ingest()
    assert p.n == 1
    assert p.n_dropped_nonfinite_proj == 1
    assert p.n_dropped_out_of_domain == 0  # lat -90 is in-domain for 4326
    assert "non-finite after projection to EPSG:3408" in p._drop_examples[0]  # ruff: ignore[private-member-access]
    assert "row 0: lon 0.000000, lat -90.000000" in p._drop_examples[0]  # ruff: ignore[private-member-access]
    assert np.isfinite(p.x).all()
    assert np.isfinite(p.y).all()


def test_ingest_all_rows_out_of_domain_names_first(tmp_path: Path) -> None:
    """#76: all rows out of domain -> 'No valid points' naming the first offender.

    Not just 'Check columns and data'.
    """
    csv = _write_domain_csv(
        tmp_path,
        "allbad.csv",
        [(10.0, 144.6, 95.0), (20.0, 190.0, -37.0)],
    )
    p = Pipeline(csv, "value", "longitude", "latitude", str(tmp_path / "out"), res=500.0, cap_km=10.0)
    with pytest.raises(ValueError, match="No valid points") as exc:
        p.ingest()
    assert "No valid points" in str(exc.value)
    assert "out-of-domain lat 95.000000 outside [-90, 90] (row 0)" in str(exc.value)


def test_check_default_transform_rejects_nonfinite_and_nonpositive() -> None:
    """#74: the helper rejects the INT32_MIN/NaN GDAL result and zero-size rasters.

    A sane transform is accepted.
    """
    bad = Affine(-2147483648.0, 0.0, float("nan"), 0.0, float("nan"), -2147483648.0)
    with pytest.raises(ValueError, match="exceeds the output projection's domain"):
        _check_default_transform(bad, -2147483648, -2147483648, 6933, 3857)
    with pytest.raises(ValueError, match="exceeds the output projection's domain"):
        _check_default_transform(Affine(1.0, 0.0, 0.0, 1.0, 0.0, 0.0), 0, 10, 6933, 3857)
    with pytest.raises(ValueError, match="exceeds the output projection's domain"):
        _check_default_transform(Affine(1.0, 0.0, 0.0, 1.0, float("nan"), 0.0), 10, 10, 6933, 3857)
    _check_default_transform(Affine(1.0, 0.0, 0.0, 1.0, 0.0, 0.0), 10, 10, 6933, 3857)  # no raise


@pytest.mark.parametrize("turbo", ["", "--turbo"], ids=["serial", "turbo"])
def test_cli_out_of_domain_transform_named_error(
    tmp_path: Path,
    turbo: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#74 repro: a grid extent crossing the Wagner VII pole + default out-crs 3857.

    Pre-fix: dies at the write phase with 'Attempt to create -2147483648x
    -2147483648 dataset is illegal' (exit 1). Now: a named user-data error
    (ValueError -> exit 2, the bad-data convention) that names the out-crs
    and suggests fixes. Serial and turbo write paths.
    """
    from ppgrid.pipeline import main

    csv = _write_domain_csv(
        tmp_path,
        "pole.csv",
        [(10.0, 144.6, -37.7), (20.0, 144.7, -37.8), (30.0, 145.0, 89.9)],
    )
    out = tmp_path / "out"
    old_argv = sys.argv
    sys.argv = ["ppgrid", csv, "-o", str(out), "--res", "100000", *([turbo] if turbo else [])]
    try:
        with pytest.raises(SystemExit) as exc:
            main()
    finally:
        sys.argv = old_argv
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "exceeds the output projection's domain" in err
    assert "EPSG:3857" in err
    assert "EPSG:6933" in err
    assert "--out-crs 4326" in err
    assert "coarser --res" in err
    assert "Attempt to create" not in err
    assert "Traceback" not in err


def test_cli_lat95_run_drops_named_no_int64_min(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#76 repro: the lat=95 file must run to completion (bad row dropped).

    The ingest summary names the row; no int64-min / NaN-to-int error
    anywhere in the output.
    """
    from ppgrid.pipeline import main

    csv = _write_domain_csv(
        tmp_path,
        "lat95.csv",
        [(10.0, 144.6, -37.7), (20.0, 144.7, -37.8), (30.0, 144.6, 95.0)],
    )
    out = tmp_path / "out"
    old_argv = sys.argv
    sys.argv = ["ppgrid", csv, "-o", str(out), "--res", "500", "--cap-km", "10"]
    try:
        with caplog.at_level(logging.INFO, logger="ppgrid"):
            main()  # no SystemExit: the run completes
    finally:
        sys.argv = old_argv
    assert (out / "value.tif").is_file()
    msgs = [r.getMessage() for r in caplog.records if r.name == "ppgrid.pipeline"]
    assert any(m.startswith("ingest: ") and "1 out-of-domain" in m and "lat 95.000000" in m for m in msgs)
    alltext = "\n".join(msgs) + capsys.readouterr().err
    assert "ix indices out of range" not in alltext
    assert "9223372036854775808" not in alltext
    assert "cannot convert float NaN" not in alltext


def test_in_domain_run_value_band_bitexact_anchor(tmp_path: Path) -> None:
    """#76: in-domain inputs are bit-identical before/after the drop change.

    Anchor = value-band sha256 computed on 3793736 (pre-fix) for this exact
    CSV (rng seed 1234, 50 pts, res 500, cap 10, workers 2, seed 0) and
    verified equal on the fixed code. The in-domain ingest summary keeps the
    pre-#76 format too.
    """
    rng = np.random.default_rng(1234)
    n = 50
    csv = tmp_path / "anchor.csv"
    pd.DataFrame(
        {
            "value": 10 + 90 * rng.random(n),
            "longitude": 144.6 + rng.uniform(-0.3, 0.3, n),
            "latitude": -37.7 + rng.uniform(-0.3, 0.3, n),
        },
    ).to_csv(csv, index=False)
    p = Pipeline(
        str(csv),
        "value",
        "longitude",
        "latitude",
        str(tmp_path / "out"),
        res=500.0,
        cap_km=10.0,
        workers=2,
        seed=0,
    )
    vpath, _ = p.run()
    with rasterio.open(vpath) as ds:
        digest = hashlib.sha256(ds.read(1).tobytes()).hexdigest()
    assert digest == "21a77172bfece6208e4f0b00bd900d90568e64fbff72e1c53e6ce046446025d7"
    assert p._drop_summary() == "0 NaN/inf dropped"  # ruff: ignore[private-member-access]


# Audit #83: connascence pins (transform choices, error mapping, parquet e2e)
# ---------------------------------------------------------------------------


def _small_df(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "value": rng.uniform(100000, 900000, n),
            "longitude": 144.6 + rng.uniform(-0.3, 0.3, n),
            "latitude": -37.7 + rng.uniform(-0.3, 0.3, n),
        }
    )


def test_transform_choices_pinned(tmp_path: Path) -> None:
    """--transform choices must equal auto + the calibrate.transforms() factory names."""
    from ppgrid.calibrate import transforms
    from ppgrid.pipeline import _build_parser

    parser = _build_parser()
    for name in ["auto"] + [t.name for t in transforms()]:
        args = parser.parse_args(["x", "-o", str(tmp_path / "o"), "--transform", name])
        assert args.transform == name
    with pytest.raises(SystemExit):
        parser.parse_args(["x", "-o", str(tmp_path / "o"), "--transform", "not_a_transform"])


def test_map_pipeline_errors_widened() -> None:
    """IndexError/OverflowError/ZeroDivisionError/RasterioIOError/RuntimeError -> clean exit 1."""
    from rasterio.errors import RasterioIOError

    from ppgrid.pipeline import _map_pipeline_errors

    msg = "simulated"
    io_msg = "simulated io"
    for typ in (IndexError, OverflowError, ZeroDivisionError, RuntimeError):

        def boom(t: type[Exception] = typ) -> NoReturn:
            raise t(msg)

        with pytest.raises(SystemExit) as ei:
            _map_pipeline_errors(boom, "in.csv")
        assert ei.value.code == 1

    def boom_io() -> NoReturn:
        raise RasterioIOError(io_msg)

    with pytest.raises(SystemExit) as ei:
        _map_pipeline_errors(boom_io, "in.csv")
    assert ei.value.code == 1


def test_parquet_input_e2e(tmp_path: Path) -> None:
    """Parquet input runs the full pipeline (optional pyarrow extra)."""
    pytest.importorskip("pyarrow")
    df = _small_df()
    pq = tmp_path / "pts.parquet"
    df.to_parquet(pq, index=False)
    out = tmp_path / "out"
    p = Pipeline(
        str(pq),
        "value",
        "longitude",
        "latitude",
        str(out),
        res=1000.0,
        cap_km=5.0,
        workers=1,
        out_crs=WORK_CRS,
    )
    vpath, spath = p.run()
    assert Path(vpath).exists()
    assert Path(spath).exists()
