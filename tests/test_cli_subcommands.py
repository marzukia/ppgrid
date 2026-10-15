"""Subcommand tests (issues #61 #62 #69): doctor, probe, list-crs.

The subcommands are dispatched in main() on argv[0] BEFORE the run parser
is built (a separate top-level parser: the run CLI's `input` positional
would swallow the command token in the same parser). Pinned here:

- dispatch: subcommand names go to the subcommand handlers, the run path
  is untouched (positional input, --plan, missing-input error, unknown
  first token = run input, not an argparse error).
- doctor: one report line per check, exit 0 on a healthy stack, exit 1
  naming the failing check (one forced via monkeypatch); warn does not
  fail (the zstd oracle mismatch and the cinterp fallback are
  byte-identical paths).
- probe: format detection (csv, gzip magic, magic/extension mismatch),
  empty/missing file, missing value column + --value-col override, the
  projection validation (non-finite / out-of-domain / non-finite-projection
  drops mirror the pipeline ingest, kept-point domain), unknown
  --work-crs exits 2, no output files written.
- list-crs: the curated ppgrid set, --all = the full EPSG database.
- laziness: `doctor --help` (parse before the handler) loads none of
  pandas/rasterio/pyproj/rich; `--help` stays fast and mentions the
  subcommands.
"""

import gzip
import io
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ppgrid import pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]

# 5 rows, all parseable as float64: 3 good points near Melbourne, one
# out-of-domain lon (200.0), one NaN value. A clean probe keeps 3/5.
GOOD_CSV = """value,other,longitude,latitude
1.5,foo,144.96,-37.81
2.5,bar,145.0,-37.82
3.5,baz,145.1,-37.9
4.5,qux,200.0,-37.9
5.5,quux,nan,-37.9
"""

NO_VALUE_CSV = """price,other,longitude,latitude
1.5,foo,144.96,-37.81
2.5,bar,145.0,-37.82
"""

ALL_BAD_CSV = """value,longitude,latitude
1.0,200.0,-37.8
2.0,300.0,-37.9
"""

# One point at the South Pole: non-finite after projection to EPSG:3408
# (the pipeline's issue-#76 repro singularity).
POLE_CSV = """value,longitude,latitude
1.0,145.0,-37.8
2.0,145.0,-90.0
"""


def _cli(*args: str, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    """Run `python -m ppgrid <args>` in a fresh interpreter (fixed argv)."""
    return subprocess.run(
        [sys.executable, "-m", "ppgrid", *[str(a) for a in args]],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _write_gz(src: Path, dst: Path) -> Path:
    dst.write_bytes(gzip.compress(src.read_bytes()))
    return dst


# ---------------------------------------------------------------------------
# Dispatch: the run path is untouched.
# ---------------------------------------------------------------------------


def test_run_input_not_swallowed_by_subcommand_dispatch(tmp_path: Path) -> None:
    """A positional input (not a subcommand name) still runs the pipeline."""
    csv = _write_csv(tmp_path / "in.csv", GOOD_CSV)
    proc = _cli(csv, "--plan", "--quiet")
    assert proc.returncode == 0, proc.stderr
    assert "run plan" in proc.stdout


def test_unknown_first_token_is_run_input(tmp_path: Path) -> None:
    """An unknown first token is the run input, not a subcommand.

    The run reports the missing file (exit 1), not an argparse error (exit 2).
    """
    proc = _cli("nosuchfile_xyz.csv", "-o", str(tmp_path / "out"))
    assert proc.returncode == 1
    assert "nosuchfile_xyz.csv" in proc.stderr


def test_subcommand_helps() -> None:
    """Each subcommand has --help; the run --help mentions them."""
    for name in ("doctor", "probe", "list-crs"):
        proc = _cli(name, "--help")
        assert proc.returncode == 0, proc.stderr
        assert name in proc.stdout
    help_proc = _cli("--help")
    assert help_proc.returncode == 0
    for name in ("doctor", "probe", "list-crs"):
        assert name in help_proc.stdout


def test_subcommand_help_does_not_load_native_stack() -> None:
    """`doctor --help` exits in parse_args: no pandas/rasterio/pyproj/rich."""
    code = (
        "import sys\n"
        "from ppgrid.pipeline import main\n"
        "try:\n"
        "    main(['doctor', '--help'])\n"
        "except SystemExit as e:\n"
        "    assert e.code == 0, e.code\n"
        "bad = [m for m in ('pandas', 'rasterio', 'pyproj', 'rich') if m in sys.modules]\n"
        "assert not bad, bad\n"
        "print('LAZY-OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "LAZY-OK" in proc.stdout


def test_help_time_budget() -> None:
    """`--help` (parser build + print) stays well under the 0.2s fast-CLI budget."""
    t0 = time.perf_counter()
    buf = io.StringIO()
    pipeline._build_parser().print_help(file=buf)  # ruff: ignore[private-member-access]
    dt = time.perf_counter() - t0
    assert dt < 0.2, f"--help took {dt:.3f}s (budget 0.2s)"
    assert "Read-only subcommands" in buf.getvalue()


# ---------------------------------------------------------------------------
# doctor (issue #61).
# ---------------------------------------------------------------------------


def test_doctor_report_and_exit0() -> None:
    """A healthy stack: one line per check, no [fail], exit 0."""
    proc = _cli("doctor")
    assert proc.returncode == 0, proc.stderr
    for label in ("python", "ppgrid", "numpy", "rasterio", "gdal", "pyproj", "zstdmt", "cinterp", "ram", "tempdir"):
        assert re.search(rf"^{label}\s", proc.stdout, re.MULTILINE), f"missing {label!r} line in:\n{proc.stdout}"
    assert "[fail]" not in proc.stdout
    # The zstd oracle outcome is reported either way (live or fallback).
    assert "zstdmt" in proc.stdout


def test_doctor_forced_failure_exit1(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """A failing check: exit 1, the failing label named on stderr."""
    monkeypatch.setattr(pipeline, "_doc_tempdir", lambda: ("tempdir", "fail", "temp dir not writable (simulated)"))
    with pytest.raises(SystemExit) as e:
        pipeline.main(["doctor"])
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert "check(s) failing: tempdir" in err


def test_doctor_warn_does_not_fail(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Warn = degraded but byte-identical behaviour: it reports, not fails."""
    monkeypatch.setattr(pipeline, "_doc_cinterp", lambda: ("cinterp", "warn", "np.interp fallback (simulated)"))
    with pytest.raises(SystemExit) as e:
        pipeline.main(["doctor"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "[warn]" in out
    assert "np.interp fallback (simulated)" in out


def test_doctor_crashing_check_reports_fail(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A check that raises is reported as a named fail, not a traceback."""

    def _doc_gdal() -> tuple[str, str, str]:
        msg = "simulated check crash"
        raise RuntimeError(msg)

    monkeypatch.setattr(pipeline, "_doc_gdal", _doc_gdal)
    with pytest.raises(SystemExit) as e:
        pipeline.main(["doctor"])
    assert e.value.code == 1
    captured = capsys.readouterr()
    assert "check(s) failing: gdal" in captured.err
    assert "check crashed: RuntimeError: simulated check crash" in captured.out
    assert "Traceback" not in captured.err + captured.out


# ---------------------------------------------------------------------------
# probe (issue #62).
# ---------------------------------------------------------------------------


def test_probe_csv_reports_drops(tmp_path: Path) -> None:
    """A clean probe: drops, kept-point domain, exit 0.

    Format, columns, rows, candidates, and the pipeline's drop counts
    (NaN/inf + out-of-domain) are all reported.
    """
    csv = _write_csv(tmp_path / "in.csv", GOOD_CSV)
    proc = _cli("probe", csv)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "format:     csv" in out
    assert "rows:       5" in out
    assert "value, other, longitude, latitude" in out
    assert "numeric candidates: value, longitude, latitude" in out
    assert "dropped nan/inf:         1" in out
    assert "dropped out-of-domain:   1" in out
    assert "dropped non-finite proj: 0" in out
    assert "points kept:             3" in out
    assert "x domain:" in out
    assert "y domain:" in out
    assert "verdict:    ok — a real run would keep 3/5 points" in out


def test_probe_csv_gz(tmp_path: Path) -> None:
    """Gzip magic under .csv.gz: detected, read through, same result."""
    raw = _write_csv(tmp_path / "in.csv", GOOD_CSV)
    gz = _write_gz(raw, tmp_path / "in.csv.gz")
    proc = _cli("probe", gz)
    assert proc.returncode == 0, proc.stderr
    assert "format:     csv (gzip)" in proc.stdout
    assert "rows:       5" in proc.stdout
    assert "points kept:             3" in proc.stdout


def test_probe_magic_extension_mismatch(tmp_path: Path) -> None:
    """Gzip content under a plain .csv name is flagged, exit 1.

    pandas infers compression from the file name, so a real run would
    fail to parse; probe says so.
    """
    raw = _write_csv(tmp_path / "data.csv", GOOD_CSV)
    renamed = _write_gz(raw, tmp_path / "renamed.csv")
    proc = _cli("probe", renamed)
    assert proc.returncode == 1
    assert "gzip magic under a plain .csv extension" in proc.stdout


def test_probe_empty_file(tmp_path: Path) -> None:
    """A 0-byte csv is a named problem, exit 1 (EmptyDataError)."""
    empty = _write_csv(tmp_path / "empty.csv", "")
    proc = _cli("probe", empty)
    assert proc.returncode == 1
    assert "EmptyDataError" in proc.stdout


def test_probe_missing_file(tmp_path: Path) -> None:
    """A nonexistent input path is a named problem, exit 1."""
    proc = _cli("probe", tmp_path / "nope.csv")
    assert proc.returncode == 1
    assert "input file not found" in proc.stdout


def test_probe_value_col_default_missing(tmp_path: Path) -> None:
    """A missing 'value' column with no --value-col is flagged, exit 1.

    A real run exits 2 at preflight (issue #57).
    """
    csv = _write_csv(tmp_path / "in.csv", NO_VALUE_CSV)
    proc = _cli("probe", csv)
    assert proc.returncode == 1
    assert "'value' not found" in proc.stdout


def test_probe_value_col_override(tmp_path: Path) -> None:
    """--value-col selects a present column: the probe then passes, exit 0."""
    csv = _write_csv(tmp_path / "in.csv", NO_VALUE_CSV)
    proc = _cli("probe", csv, "--value-col", "price")
    assert proc.returncode == 0, proc.stderr
    assert "value col:  'price'" in proc.stdout
    assert "points kept:             2" in proc.stdout


def test_probe_unknown_work_crs_exit2(tmp_path: Path) -> None:
    """Unknown --work-crs code: bad flag value, exit 2 (the run CLI's #77 contract)."""
    csv = _write_csv(tmp_path / "in.csv", GOOD_CSV)
    proc = _cli("probe", csv, "--work-crs", "999999999")
    assert proc.returncode == 2
    assert "unknown work CRS" in proc.stderr


def test_probe_nonfinite_proj_drop(tmp_path: Path) -> None:
    """The lat -90 pole singularity drops under --work-crs 3408.

    The same non-finite-projection drop the pipeline makes (issue #76).
    """
    csv = _write_csv(tmp_path / "in.csv", POLE_CSV)
    proc = _cli("probe", csv, "--work-crs", "3408")
    assert proc.returncode == 0, proc.stderr
    assert "dropped non-finite proj: 1" in proc.stdout
    assert "points kept:             1" in proc.stdout


def test_probe_no_valid_points_exit1(tmp_path: Path) -> None:
    """Every row out of domain: 0 kept, a real run would fail, exit 1."""
    csv = _write_csv(tmp_path / "in.csv", ALL_BAD_CSV)
    proc = _cli("probe", csv)
    assert proc.returncode == 1
    assert "points kept:             0" in proc.stdout
    assert "no valid points remain" in proc.stdout


def test_probe_writes_no_files(tmp_path: Path) -> None:
    """Probe is read-only: no new files in the input dir or the repo root."""
    csv = _write_csv(tmp_path / "in.csv", GOOD_CSV)
    before_tmp = sorted(p.name for p in tmp_path.iterdir())
    before_repo = sorted(p.name for p in REPO_ROOT.iterdir())
    proc = _cli("probe", csv)
    assert proc.returncode == 0
    assert before_tmp == sorted(p.name for p in tmp_path.iterdir())
    assert before_repo == sorted(p.name for p in REPO_ROOT.iterdir())


def test_subcommand_usage_errors_exit2(tmp_path: Path) -> None:
    """Argparse usage errors keep the exit-2 convention."""
    assert _cli("probe").returncode == 2  # missing input
    assert _cli("list-crs", "--bogus").returncode == 2
    assert _cli("probe", tmp_path, "--work-crs", "notacode").returncode == 2


# ---------------------------------------------------------------------------
# list-crs (issue #69).
# ---------------------------------------------------------------------------


def test_list_crs_curated() -> None:
    """The curated set: every pinned code present, the 4326 line annotated."""
    proc = _cli("list-crs")
    assert proc.returncode == 0, proc.stderr
    for code in ("4326", "3577", "6933", "3857", "32633", "32755", "32756"):
        assert re.search(rf"^{code}\s", proc.stdout, re.MULTILINE), f"missing {code} in:\n{proc.stdout}"
    line_4326 = next(line for line in proc.stdout.splitlines() if line.startswith("4326"))
    assert "WGS 84" in line_4326
    assert "input default" in line_4326


def test_list_crs_all() -> None:
    """--all: the full EPSG CRS database (thousands of rows, 4326 included)."""
    proc = _cli("list-crs", "--all", timeout=600)
    assert proc.returncode == 0, proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert len(lines) > 5000
    assert any(line.startswith("4326") for line in lines)
