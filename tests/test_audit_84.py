"""Intended-behavior-change pins for the 2026-10-09 hygiene audit (issue #84).

Each test pins a behavior the audit moved from "undocumented/inconsistent" to
"documented and enforced"; the area tags (a1..a8) match the issue checklist.
Behavior-preserving items need no pin (existing tests already cover them).
"""

import contextlib
import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ppgrid import pipeline as _pl
from ppgrid.pipeline import main
from ppgrid.pullpush import downsample_sum, pull_push

TINY = "value,lng,lat\n1.0,149.0,-35.0\n2.0,149.1,-35.1\n"


def _tiny_csv(tmp_path: Path, text: str = TINY) -> str:
    csv = tmp_path / "pts.csv"
    csv.write_text(text)
    return str(csv)


def _argv(csv: str, out: Path, *extra: str) -> list[str]:
    """Full-pipeline argv for the tiny CSV (test_cli_polish._run_argv shape)."""
    return [
        csv,
        "-o",
        str(out),
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
        *extra,
    ]


# ---------------------------------------------------------------------------
# a1 CLI: --compress validation + read-back
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("codec", ["BZIP2", "CCITTRLE", "JPEG", "not-a-codec"])
def test_a1_compress_rejects_unknown_codecs(tmp_path: Path, capsys: pytest.CaptureFixture[str], codec: str) -> None:
    """a1: GDAL would silently ignore these (exit 0, uncompressed file); now exit 2."""
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        main(_argv(_tiny_csv(tmp_path), out, "--compress", codec))
    assert exc.value.code == 2
    assert "--compress must be one of" in capsys.readouterr().err
    assert not out.exists()


def test_a1_compress_lowercase_accepted_and_read_back(tmp_path: Path) -> None:
    """a1: the GTiff driver is case-insensitive; lowercase is accepted and the tag read back."""
    import rasterio

    out = tmp_path / "out"
    with contextlib.suppress(SystemExit):
        main(_argv(_tiny_csv(tmp_path), out, "--compress", "zstd"))
    with rasterio.open(out / "value.tif") as ds:
        assert (ds.profile.get("compress") or "none").lower() == "zstd"


# ---------------------------------------------------------------------------
# a1 CLI: --out "" + malformed --calibration
# ---------------------------------------------------------------------------


def test_a1_out_empty_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """a1: Path("") == Path(".") — an empty --out would write into the CWD; now exit 2."""
    csv = _tiny_csv(tmp_path)
    argv = [
        csv,
        "-o",
        "",
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
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert "--out must not be empty" in capsys.readouterr().err


def test_a1_malformed_calibration_names_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """a1: a JSONDecodeError carried no file name; now the message names the path (exit 2)."""
    cal = tmp_path / "cal.json"
    cal.write_text('{"transform": "identity", oops')
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        main(_argv(_tiny_csv(tmp_path), out, "--calibration", str(cal)))
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "cal.json" in err
    assert "Traceback" not in err


# ---------------------------------------------------------------------------
# a1 ingest: non-numeric column named + corrupt parquet exit 1
# ---------------------------------------------------------------------------


def test_a1_non_numeric_value_column_named(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """a1: pandas names the value (and line) but not the column; the fix names the column."""
    csv = _tiny_csv(tmp_path, "value,lng,lat\n1.0,149.0,-35.0\nxyz,149.1,-35.1\n")
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        main(_argv(csv, out))
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "column 'value'" in err
    assert "xyz" in err  # the original parser detail survives
    assert "Traceback" not in err


def _have_pyarrow() -> bool:
    try:
        import pyarrow as pa
    except ImportError:
        return False
    return pa is not None


@pytest.mark.skipif(not _have_pyarrow(), reason="pyarrow not installed (optional [parquet] extra)")
def test_a1_corrupt_parquet_exit1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """a1: ArrowInvalid (a ValueError) on a corrupt .parquet is a pipeline/IO failure.

    Exit 1, matching the CSV ParserError path, not exit 2.
    """
    pq = tmp_path / "corrupt.parquet"
    pq.write_bytes(b"not a parquet file at all")
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        main(_argv(str(pq), out))
    assert exc.value.code == 1
    assert "Traceback" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# a1 CLI: idwgrid shim runs the pipeline under `python -m`
# ---------------------------------------------------------------------------


def test_a1_idwgrid_module_main() -> None:
    """a1: `python -m ppgrid.idwgrid` was a silent no-op; now it runs main() (no input -> exit 2)."""
    repo = Path(_pl.__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "ppgrid.idwgrid"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr


# ---------------------------------------------------------------------------
# a1 CLI: summary JSON is strict-JSON native or stringified
# ---------------------------------------------------------------------------


def test_a1_summary_json_stringifies_np_float32() -> None:
    """a1: an np.float32 (not a Python float) in the summary is stringified.

    default=str covers it instead of crashing the dump with allow_nan=False.
    """
    doc = json.dumps(_pl._sanitize_json({"k": np.float32(2.5)}), indent=2, allow_nan=False, default=str)
    assert "2.5" in doc


def test_a1_plan_header_documents_strict_exit3(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """a1: the plan header documents the --turbo-strict exit-3 case (one-word fix)."""
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        main(_argv(_tiny_csv(tmp_path), out, "--plan"))
    assert exc.value.code == 0
    assert "exit 0; 3 with --turbo-strict infeasible" in capsys.readouterr().out


def test_a1_plan_turbo_strict_infeasible_exits3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """a8-11 pin (paired with a1-12): #53's claim.

    An infeasible --turbo-strict budget under --plan exits 3 (no files
    written). resolve_cap is patched so the 0.001 GB budget is not floored
    to the 8 GB cap floor — the est peak (~4 GB) then outruns it.
    """
    from ppgrid import turbop

    monkeypatch.setattr(turbop, "resolve_cap", lambda *a, **_k: (a[0], "preset"))
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        main(_argv(_tiny_csv(tmp_path), out, "--plan", "--turbo", "16", "--ram-gb", "0.001", "--turbo-strict"))
    assert exc.value.code == 3
    assert not out.exists()


# ---------------------------------------------------------------------------
# a2 — ingest


def test_a2_duplicate_role_column_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """a2-2 pin: a role name present twice is rejected in preflight (exit 2)."""
    dup = tmp_path / "dup.csv"
    dup.write_text("value,value,lng,lat\n1.0,2.0,149.0,-35.0\n3.0,4.0,149.1,-35.1\n")
    with pytest.raises(SystemExit) as exc:
        main(_argv(str(dup), tmp_path / "o"))
    assert exc.value.code == 2
    assert "appears 2 times" in capsys.readouterr().err


def test_a2_all_missing_columns_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """a2-5 pin: every missing role column is reported in one pass."""
    with pytest.raises(SystemExit) as exc:
        main(_argv(_tiny_csv(tmp_path), tmp_path / "o", "--value-col", "nope", "--lat-col", "nope2"))
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--value-col 'nope' not found" in err
    assert "--lat-col 'nope2' not found" in err


def test_a2_kwarg_validation() -> None:
    """a2-7 pin: __init__ rejects res/scale/saturation/workers/calib_max_points early.

    Raises:
        AssertionError: when a bad kwarg is not rejected.

    """
    from ppgrid.pipeline import Pipeline

    for kw in (
        {"res": 0},
        {"res": float("nan")},
        {"res": -1000},
        {"scale": 0},
        {"scale": float("nan")},
        {"saturation": float("nan")},
        {"saturation": float("inf")},
        {"workers": 0},
        {"calib_max_points": 0},
    ):
        try:
            Pipeline("in.csv", "v", "lng", "lat", "o", **kw)
        except ValueError:
            pass
        else:
            msg = f"{kw} not rejected"
            raise AssertionError(msg)


def test_a2_underscore_int_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """a2-8 pin: '1_000' is rejected (exit 2, column named) on the dtype-pinned path."""
    csv = _tiny_csv(tmp_path, "value,lng,lat\n1_000,149.0,-35.0\n2.0,149.1,-35.1\n")
    with pytest.raises(SystemExit) as exc:
        main(_argv(csv, tmp_path / "o"))
    assert exc.value.code == 2
    assert "column 'value'" in capsys.readouterr().err


def test_a2_chunk_runtimeerror_mapped_exit1() -> None:
    """a2-3 pin: the chunk dtype RuntimeError maps to exit 1 (pipeline error)."""
    from ppgrid.pipeline import _map_pipeline_errors

    def boom() -> None:
        msg = "chunk dtype mismatch for value: object"
        raise RuntimeError(msg)

    with pytest.raises(SystemExit) as exc:
        _map_pipeline_errors(boom, "in.csv")
    assert exc.value.code == 1


def test_a2_row_mismatch_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """a2-4 pin: the silent serial fallback on a row-count mismatch now warns.

    Parallel ingest is library-only (the CLI never sets n_threads), so this
    drives Pipeline directly. A blank line makes the newline scan count one
    more row than the C parser parses.
    """
    from ppgrid.pipeline import Pipeline

    rows = ["value,lng,lat"] + [f"{i}.0,149.0,-35.0" for i in range(200_000)]
    rows.insert(500, "")  # blank line: the newline scan counts it, pandas does not
    csv = tmp_path / "big.csv"
    csv.write_text("\n".join(rows) + "\n")
    # The n_threads=2 ingest forks a process pool. By this point in the suite
    # other tests have left threads behind, so CPython's fork emits its
    # multi-threaded DeprecationWarning — a test-harness artifact, not ppgrid
    # behaviour; keep the suite's warnings summary clean (issue #84 a8).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"multiprocessing\.popen_fork")
        Pipeline(
            str(csv),
            "value",
            "lng",
            "lat",
            str(tmp_path / "o"),
            res=100,
            block_size=2048,
            workers=2,
            n_threads=2,
        ).run()
    assert any("row mismatch" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# a3 — grid / reproject


def test_a3_crs_pair_skip_warns(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """a3-8 pin: the tile-skip footprint premise is verified only for 6933 -> 3857."""
    main(_argv(_tiny_csv(tmp_path), tmp_path / "o", "--out-crs", "4326"))
    err = capsys.readouterr().err
    assert "non-default work/out CRS pair" in err


def test_a3_default_crs_pair_no_warn(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """a3-8 pin (negative): the default pair runs quiet."""
    main(_argv(_tiny_csv(tmp_path), tmp_path / "o"))
    assert "tile skip active" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# a4 — turbo


def test_a4_wall_range_brackets_estimate() -> None:
    """a4-2 pin: wall_range_s must bracket wall_s for every cap/geometry.

    The table-row ranges were measured at each row's worker count; the
    generic +/-15% range brackets by construction. A plan where the central
    estimate sits outside its own range is a reporting bug (audit #84 a4).
    """
    from ppgrid import turbop

    for cap in (8.0, 16.0, 32.0, 64.0, 128.0, 256.0):
        for n in (10**5, 10**6, 10**7, 5 * 10**7, 2 * 10**8, 10**9, 5 * 10**9, 2 * 10**10):
            p = turbop.plan(cap, n, 1000, 250)
            lo, hi = p.wall_range_s
            assert lo <= p.wall_s <= hi, (cap, n, p.regime, p.workers, p.wall_s, p.wall_range_s)


def test_a4_perbox_summary_names_actual_inflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """a4-9 pin: the per-box estimate must be announced with the in-flight count it used.

    The count is min(CLI --workers, tasks), not plan.workers. Tiny grid ->
    1 task; CLI --workers 8; plan.workers is the per-box pool (4). The
    summary line must say "x 1 in-flight".
    """
    from ppgrid import turbop

    monkeypatch.setattr(turbop, "resolve_cap", lambda *a, **_k: (a[0], "preset"))
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        main(_argv(_tiny_csv(tmp_path), out, "--turbo", "16", "--ram-gb", "0.001", "--turbo-strict", "--workers", "8"))
    assert exc.value.code == 3
    err = capsys.readouterr().err
    assert "worst snapped box x 1 in-flight" in err


def test_a4_failed_run_discards_fresh_calibration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """a4-11 pin: a failed run leaves no fresh calibration.json behind.

    A naive re-run is then not blocked by an output the run never
    completed.
    """
    from ppgrid import turbop

    monkeypatch.setattr(turbop, "resolve_cap", lambda *a, **_k: (a[0], "preset"))
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        main(_argv(_tiny_csv(tmp_path), out, "--turbo", "16", "--ram-gb", "0.001", "--turbo-strict"))
    assert exc.value.code == 3
    assert not (out / "calibration.json").exists()


def test_a4_successful_run_keeps_calibration(tmp_path: Path) -> None:
    """a4-11 pin (negative): a successful run publishes calibration.json."""
    out = tmp_path / "out"
    with contextlib.suppress(SystemExit):
        main(_argv(_tiny_csv(tmp_path), out))
    assert (out / "calibration.json").exists()


# ---------------------------------------------------------------------------
# a5 pullpush
# ---------------------------------------------------------------------------


def test_a5_grid_index_mt_wired_for_multiworker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """a5-1 pin: the classic path routes the index/scatter through grid_index_mt when workers > 1."""
    calls: list[int | None] = []
    real = _pl.grid_index_mt

    def spy(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        calls.append(kwargs.get("n_threads"))
        return real(*args, **kwargs)

    monkeypatch.setattr(_pl, "grid_index_mt", spy)
    with contextlib.suppress(SystemExit):
        main(_argv(_tiny_csv(tmp_path), tmp_path / "out"))  # --workers 2
    assert calls == [2]


def test_a5_grid_index_mt_bit_identical_to_serial(tmp_path: Path) -> None:
    """a5-1 pin: the threaded index/scatter pass yields byte-identical rasters to the serial path."""
    csv = _tiny_csv(tmp_path)
    rasters = []
    for w in ("1", "2"):
        out = tmp_path / f"out{w}"
        with contextlib.suppress(SystemExit):
            main(
                [
                    csv,
                    "-o",
                    str(out),
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
                    w,
                ]
            )
        rasters.append(((out / "value.tif").read_bytes(), (out / "support_km.tif").read_bytes()))
    assert rasters[0] == rasters[1]


@pytest.mark.parametrize("saturation", [0.0, -3.0])
def test_a5_pull_push_rejects_nonpositive_saturation(saturation: float) -> None:
    """a5-3 pin: saturation <= 0 used to silently produce NaN/Inf; now ValueError."""
    s = np.array([[1.0, 0.0], [0.0, 0.0]], np.float32)
    c = np.array([[1.0, 0.0], [0.0, 0.0]], np.float32)
    with pytest.raises(ValueError, match="saturation must be > 0"):
        pull_push(s, c, 100.0, 1, saturation=saturation)


def test_a5_downsample_sum_names_odd_shape() -> None:
    """a5-4 pin: odd dims raise a named-shape ValueError, not an opaque broadcast error."""
    with pytest.raises(ValueError, match="even dimensions"):
        downsample_sum(np.ones((3, 4), np.float32))


def test_a5_downsample_sum_even_shape_unaffected() -> None:
    """a5-4 pin (negative): even dims keep working."""
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = downsample_sum(a)
    assert out.shape == (2, 2)
    assert out[0, 0] == a[0, 0] + a[1, 0] + a[0, 1] + a[1, 1]


# ---------------------------------------------------------------------------
# a6 compression
# ---------------------------------------------------------------------------


def test_a6_memmap_dir_path_raises_valueerror(tmp_path: Path) -> None:
    """a6-1 pin: a directory at the temp path is a named ValueError, not IsADirectoryError."""
    from ppgrid._tempfiles import _open_fresh_memmap

    d = tmp_path / "_points.npy"
    d.mkdir()
    with pytest.raises(ValueError, match="is a directory"):
        _open_fresh_memmap(d, np.float64, (2, 2))


def test_a6_memmap_dir_eperm_maps_to_valueerror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """a6-1 macOS pin: unlink() on a dir raises EPERM, not EISDIR (BSD).

    Linux raises IsADirectoryError; macOS/BSD raise PermissionError(EPERM).
    Both must map to the same named ValueError. Monkeypatch Path.unlink to
    raise PermissionError so the BSD branch is exercised on every platform.
    A genuine permission error on a regular file is re-raised untouched.
    """
    from typing import NoReturn

    from ppgrid._tempfiles import _open_fresh_memmap

    d = tmp_path / "_points.npy"
    d.mkdir()

    def _raise_eperm(self: Path, *, missing_ok: bool = False) -> NoReturn:
        """Stand-in for macOS/BSD unlink(dir): raises EPERM (not EISDIR).

        Args:
            self: The path (a directory, mirroring unlink(dir)).
            missing_ok: Signature parity with Path.unlink; unused.

        Raises:
            PermissionError: Always (EPERM), as BSD unlink(2) on a directory.

        """
        del missing_ok
        raise PermissionError(1, "Operation not permitted", str(self))

    monkeypatch.setattr(Path, "unlink", _raise_eperm)
    with pytest.raises(ValueError, match="is a directory"):
        _open_fresh_memmap(d, np.float64, (2, 2))

    # A regular file with a genuine permission error is NOT a directory:
    # the PermissionError propagates, it is not swallowed into ValueError.
    f = tmp_path / "_points2.npy"
    f.touch()
    with pytest.raises(PermissionError):
        _open_fresh_memmap(f, np.float64, (2, 2))


def test_a6_cpl_struct_version_checked() -> None:
    """a6-3 pin: a CPLCompressor with nStructVersion != 1 is rejected before the pfn read."""
    import ctypes

    from ppgrid import zstdmt

    buf = ctypes.create_string_buffer(64)
    ctypes.c_int.from_address(ctypes.addressof(buf) + zstdmt._OFF_VERSION).value = 2
    with pytest.raises(zstdmt.ZstdmtError, match="nStructVersion"):
        zstdmt._extract_fn(ctypes.addressof(buf), "zstd-test")


def test_a6_decompress_tile_round_trip() -> None:
    """a6-3 pin: decompress_tile recovers the input of compress_tile (probe passes on this stack)."""
    from ppgrid import zstdmt

    if not zstdmt.available():
        pytest.skip("CPL zstd path unavailable on this stack")
    raw = np.arange(64, dtype=np.int16).tobytes()
    assert zstdmt.decompress_tile(zstdmt.compress_tile(raw)) == raw


# ---------------------------------------------------------------------------
# a7 calibrate
# ---------------------------------------------------------------------------


def _write_xy_csv(tmp_path: Path, x: np.ndarray, y: np.ndarray, v: np.ndarray, name: str = "pts.csv") -> Path:
    import pandas as pd

    p = tmp_path / name
    pd.DataFrame(
        {
            "value": v,
            "longitude": x,
            "latitude": y,
        }
    ).to_csv(p, index=False)
    return p


def test_a7_fill_cap_rejects_kw_typo() -> None:
    """a7-5 pin: a typo in a CV knob is a TypeError, not a silent default."""
    import ppgrid.calibrate as cal

    x = np.arange(100, dtype=np.float64)
    with pytest.raises(TypeError, match="seedd"):
        cal.calibrate_fill_cap(x, x, x, seedd=1)


def test_a7_transforms_removed_names_gone() -> None:
    """a7-1/a7-2 pins: dead icc() and never-instantiated CVDetail/CVCurve are gone."""
    import ppgrid.calibrate as cal

    assert not hasattr(cal, "icc")
    assert not hasattr(cal, "CVDetail")
    assert not hasattr(cal, "CVCurve")


def test_a7_p_grid_cached() -> None:
    """a7-10 pin: the 0-100 percentile grid is built once, not per call."""
    from ppgrid.calibrate import PercentileTransform

    t = PercentileTransform()
    assert t._p is t._p
    assert len(t._p) == t.NQ


def test_a7_cap_tuned_in_forced_transform_space(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """a7-3 pin: forced --transform identity runs the CV in identity space.

    Auto "chose" log10; the cap search must see identity-space values.
    """
    from ppgrid import pipeline as pl

    n = 200
    rng = np.random.default_rng(0)
    x_deg = 144.90 + rng.uniform(-0.05, 0.05, n)
    y_deg = -37.81 + rng.uniform(-0.05, 0.05, n)
    v = 10.0 ** (0.5 * ((x_deg - x_deg.min()) / np.ptp(x_deg))) * (1.0 + 0.01 * rng.standard_normal(n))
    csv = _write_xy_csv(tmp_path, x_deg, y_deg, v)
    captured: dict[str, np.ndarray] = {}

    def fake_fill_cap(_xx: np.ndarray, _yy: np.ndarray, tv: np.ndarray, **_kw: object) -> tuple[float, dict]:
        captured["tv"] = tv
        return 25.0, {}

    def fake_choose(_xx: np.ndarray, _yy: np.ndarray, vv: np.ndarray, **_kw: object) -> tuple:
        log_t = next(t for t in pl.transforms() if t.name == "log10").fit(vv)
        id_t = next(t for t in pl.transforms() if t.name == "identity").fit(vv)
        return log_t, [(log_t, 0.9, {}), (id_t, 0.1, {})]

    monkeypatch.setattr(pl, "calibrate_fill_cap", fake_fill_cap)
    monkeypatch.setattr(pl, "choose_transform", fake_choose)

    p = pl.Pipeline(str(csv), "value", "longitude", "latitude", str(tmp_path / "out"), res=1000.0, cap_km="auto")
    # Force identity: the CV must run on identity-space values (the data
    # itself), not the auto choice's log10 space.
    p.transform = "identity"
    p.ingest()
    p.calibrate()
    assert np.shares_memory(captured["tv"], p.v), "cap CV ran in the auto transform's space, not the forced one"


def test_a7_percentile_lut_pairing(tmp_path: Path) -> None:
    """a7-4/a7-7/a7-11 pins: one LUT per file; CV rows ship; no per-block cap."""
    n = 200
    rng = np.random.default_rng(1)
    x_deg = 144.90 + rng.uniform(-0.03, 0.03, n)
    y_deg = -37.81 + rng.uniform(-0.03, 0.03, n)
    v = rng.uniform(1.0, 1000.0, n)
    csv = _write_xy_csv(tmp_path, x_deg, y_deg, v)
    out = tmp_path / "out"
    # calib_max_points < n forces the sample/full-data LUT split the fix
    # removes.
    p = _pl.Pipeline(
        str(csv),
        "value",
        "longitude",
        "latitude",
        str(out),
        res=500.0,
        cap_km="auto",
        transform="percentile",
        calib_max_points=100,
        out_crs=_pl.WORK_CRS,
    )
    p.ingest()
    p.calibrate()
    cal = p._cal
    assert cal is not None
    # a7-4: transform_state and percentile_quantiles carry the SAME LUT.
    assert cal["transform_state"]["quantiles"] == cal["percentile_quantiles"]
    assert list(p.pct_q.q) == cal["percentile_quantiles"]
    # a7-7/a7-11: per-block entries have the full rows and no cap_km copy.
    for entry in cal["cv"].values():
        assert "rows" in entry
        assert "cap_km" not in entry
    assert "cap_km" in cal


def test_a7_old_format_calibration_loads(tmp_path: Path) -> None:
    """a7 review-minor pin: pre-#84 calibration files still load.

    Per-block cap_km with no rows (the pre-#84 shape) must still validate,
    rebuild, and round-trip — the CHANGELOG's "older files load unchanged"
    claim.
    """
    from ppgrid.calibrate import make_transform, validate_calibration

    n = 200
    rng = np.random.default_rng(1)
    x_deg = 144.90 + rng.uniform(-0.03, 0.03, n)
    y_deg = -37.81 + rng.uniform(-0.03, 0.03, n)
    v = rng.uniform(1.0, 1000.0, n)
    csv = _write_xy_csv(tmp_path, x_deg, y_deg, v)
    p = _pl.Pipeline(
        str(csv),
        "value",
        "longitude",
        "latitude",
        str(tmp_path / "out"),
        res=500.0,
        cap_km="auto",
        transform="percentile",
        out_crs=_pl.WORK_CRS,
    )
    p.ingest()
    p.calibrate()
    cal = p._cal
    assert cal is not None
    # Downgrade the fresh (post-#84) file to the pre-#84 shape: per-block
    # cap_km, no rows. Everything else (transform_state, quantiles, LUT)
    # was already present in the old format.
    old = json.loads(json.dumps(cal))
    for entry in old["cv"].values():
        rows = entry.pop("rows")
        entry["cap_km"] = rows[-1]["hi"] if rows else old["cap_km"]
    validate_calibration(old, "legacy.json")
    tf = make_transform(old["transform_state"])
    lo, hi = old["percentile_quantiles"][0], old["percentile_quantiles"][-1]
    sample = lo + np.linspace(0.1, 0.9, 9) * (hi - lo)
    pct = tf.fwd(sample)
    assert np.all(np.isfinite(pct))
    assert np.allclose(tf.inv(pct), sample)


# ---------------------------------------------------------------------------
# a8 — tests / docs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("codec", ["LZW", "DEFLATE", "PACKBITS", "LERC", "LZMA", "NONE"])
def test_a8_compress_non_zstd_e2e_roundtrip(tmp_path: Path, codec: str) -> None:
    """a8-11 pin: every supported non-ZSTD codec runs the full pipeline e2e.

    The zstd path is the default and heavily pinned; the other COMPRESS_CHOICES
    members only had unit coverage of the preflight. This runs each one to
    completion and reads the GTiff compress tag back, so a codec that GDAL
    accepts but silently drops (writes uncompressed) or a readback/parse
    mismatch fails here (issue #84 a8).
    """
    import rasterio

    out = tmp_path / "out"
    with contextlib.suppress(SystemExit):
        main(_argv(_tiny_csv(tmp_path), out, "--compress", codec))
    want = "none" if codec == "NONE" else codec.lower()
    for name in ("value.tif", "support_km.tif"):
        with rasterio.open(out / name) as ds:
            got = (ds.profile.get("compress") or "none").lower()
            assert got == want, f"{name}: --compress {codec} read back as {got!r}"
            # The band decodes to finite DNs (the writer + codec round-trip).
            arr = ds.read(1)
        assert arr.shape[0] > 0
