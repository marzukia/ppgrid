"""Clean Ctrl-C (exit 130), error attribution, and the exit-code table (#106).

- SIGINT mid-run -> one `interrupted` line, no traceback, exit 130, and no
  run-temp residue (retained partials named instead) — subprocess test with
  the parent sending SIGINT at a fixed wall-clock delay that lands in the
  interpolate/reproject phase (the per-phase log lines are only written in
  the end-of-run summary block, so the parent cannot gate on them).
- Attribution audit: only the user-supplied input renders as `input file
  not found`; internal resources (meminfo, cgroup, run temps) name
  themselves.
- Disk-full / permission errors on the out dir name the actually failing
  path.
- Every documented exit code (0, 1, 2, 3, 130) has an asserting test.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import ppgrid.pipeline as pipeline_mod
from ppgrid import turbop
from ppgrid.pipeline import Pipeline, _interrupted_exit, _map_pipeline_errors, main

REPO_ROOT = Path(pipeline_mod.__file__).resolve().parent.parent
TINY_CSV = REPO_ROOT / "data" / "melb_houses.csv"


def _tiny_csv(tmp_path: Path) -> Path:
    csv = tmp_path / "tiny.csv"
    csv.write_text("value,lng,lat\n100,145.0,-37.8\n200,145.01,-37.81\n300,145.02,-37.82\n")
    return csv


def _run_argv(out: Path, *extra: str, csv: Path | None = None) -> list[str]:
    csv = csv or TINY_CSV
    cols = ("price", "longitude", "latitude") if csv is TINY_CSV else ("value", "lng", "lat")
    return [str(csv), "-o", str(out), "--value-col", cols[0], "--lng-col", cols[1], "--lat-col", cols[2], *extra]


def _big_csv(tmp_path: Path, n: int = 20_000) -> Path:
    """Spread-out points: a big grid whose interpolate/write phase runs seconds."""
    rng = np.random.default_rng(106)
    csv = tmp_path / "big.csv"
    with csv.open("w") as f:
        f.write("value,lng,lat\n")
        lng = rng.uniform(140.0, 155.0, n)
        lat = rng.uniform(-44.0, -32.0, n)
        val = rng.uniform(100.0, 2000.0, n)
        for i in range(n):
            f.write(f"{val[i]:.2f},{lng[i]:.5f},{lat[i]:.5f}\n")
    return csv


# ---------------------------------------------------------------------------
# Ctrl-C: exit 130, one line, no traceback, no residue
# ---------------------------------------------------------------------------


def test_sigint_mid_run_exit130_no_traceback_no_residue(tmp_path: Path) -> None:
    """Parent sends SIGINT mid-run (fixed wall-clock delay, interpolate window).

    The per-phase log lines (`ingest:`/`grid:`/`done:` ...) are emitted only
    in the end-of-run summary block, so a child-side log gate arms far too
    late. Measured phase layout for this input (200k rows, --res 200, 2
    workers): import <= 1.5 s, ingest+calibrate+grid <= 0.3 s,
    interpolate ~3.6 s, reproject+write ~9-13 s (total ~13.5 s). SIGINT at
    parent t=3.0 s therefore lands in the interpolate/reproject window on
    any machine within 2x of that speed, well after the run started and
    long before it can finish. A second send at t=6.0 s covers a main
    thread parked in a long C call at the first one (the signal trampoline
    is lossless; only the Python handler needs an eval break).
    """
    csv = _big_csv(tmp_path)
    out = tmp_path / "out"
    argv = [
        str(csv),
        "-o",
        str(out),
        "--res",
        "200",
        "--block",
        "1024",
        "--workers",
        "2",
        "--cap-km",
        "25",
        "--value-col",
        "value",
        "--lng-col",
        "lng",
        "--lat-col",
        "lat",
    ]
    code = f"import ppgrid.pipeline as p\np.main({argv!r})\n"
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=REPO_ROOT,
    )
    for send_at in (3.0, 6.0):
        while time.time() - t0 < send_at:
            time.sleep(0.05)
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
    _out_s, err = proc.communicate(timeout=240)
    # Interrupted mid-run, not before or after: the summary block (incl.
    # `done:`) is only logged on completion, so its absence + missing finals
    # prove the run did not finish.
    assert "done:" not in err, err
    assert not (out / "value.tif").exists(), err
    # The contract: exit 130, no traceback, one `interrupted` line.
    assert proc.returncode == 130
    assert "Traceback" not in err
    interrupted_lines = [line for line in err.splitlines() if "interrupted" in line]
    assert len(interrupted_lines) == 1, err
    assert not interrupted_lines[0].startswith("warning")
    # No run-temp residue: everything the sweep covers is gone (finals were
    # never published, so the out dir holds only pre-write metadata).
    for name in ("value.tif.tmp", "support_km.tif.tmp", "_value_tmp.tif", "_support_tmp.tif", "_points.npy"):
        assert not (out / name).exists(), f"residue: {name}"
    assert not (out / "value.tif").exists()
    assert not (out / "support_km.tif").exists()


def test_sigint_over_earlier_outputs_warns_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A Ctrl-C over an earlier run's rasters warns once, sweeps temps, exits 130."""
    csv = _tiny_csv(tmp_path)
    out = tmp_path / "out"
    main(_run_argv(out, csv=csv))
    capsys.readouterr()
    (out / "value.tif.tmp").write_bytes(b"partial")
    (out / "_points.npy").write_bytes(b"partial")

    def kb(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(Pipeline, "ingest", kb)
    with pytest.raises(SystemExit) as exc:
        main(_run_argv(out, "--force", csv=csv))
    assert exc.value.code == 130
    err = capsys.readouterr().err
    assert err.count("interrupted") == 1
    assert "failed before writing outputs" in err
    assert not (out / "value.tif.tmp").exists()
    assert not (out / "_points.npy").exists()
    # Earlier finals are left in place (the warning says so).
    assert (out / "value.tif").is_file()


def test_interrupted_exit_sweeps_and_names_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """_interrupted_exit removes run temps and names any it cannot."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "_points.npy").write_bytes(b"x")
    (out / "value.tif.tmp").write_bytes(b"x")
    (out / "value.tif").write_bytes(b"final")

    orig_unlink = Path.unlink

    def no_unlink(self: Path, missing_ok: bool = False) -> None:
        if self.name == "_points.npy":
            raise PermissionError(13, "simulated lock")
        orig_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", no_unlink)
    with pytest.raises(SystemExit) as exc:
        _interrupted_exit(out, None)
    assert exc.value.code == 130
    err = capsys.readouterr().err
    assert "retained partials: _points.npy" in err
    assert "value.tif.tmp" not in err  # swept, so not named
    assert not (out / "value.tif.tmp").exists()
    assert (out / "_points.npy").exists()
    assert (out / "value.tif").is_file()  # finals untouched


def test_interrupted_exit_without_out_dir(capsys: pytest.CaptureFixture) -> None:
    """An interrupt before the out path was read: clean line, exit 130."""
    with pytest.raises(SystemExit) as exc:
        _interrupted_exit(None, None)
    assert exc.value.code == 130
    err = capsys.readouterr().err
    assert err.strip() == "interrupted"


# ---------------------------------------------------------------------------
# Attribution audit
# ---------------------------------------------------------------------------


def test_missing_input_still_says_input_file_not_found(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """The user's own input path (as pandas reports it) keeps the friendly message."""
    input_path = str(tmp_path / "data.csv")

    def boom() -> None:
        raise FileNotFoundError(2, "No such file or directory", input_path)

    with pytest.raises(SystemExit) as exc:
        _map_pipeline_errors(boom, input_path)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert f"input file not found: {input_path}" in err


@pytest.mark.parametrize("raise_fname", [None, "in.csv"])
def test_fnf_input_filename_variants_name_input(capsys: pytest.CaptureFixture, raise_fname: str | None) -> None:
    """FNF with no filename (or the input's own name) still names the input."""

    def boom() -> None:
        if raise_fname is None:
            raise FileNotFoundError(2, "No such file or directory")
        raise FileNotFoundError(2, "No such file or directory", raise_fname)

    with pytest.raises(SystemExit) as exc:
        _map_pipeline_errors(boom, "in.csv")
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "input file not found: in.csv" in err


def test_fnf_internal_resource_names_itself(capsys: pytest.CaptureFixture) -> None:
    """A missing /proc/meminfo names the resource, not the user input (#106 item 3)."""

    def boom() -> None:
        raise FileNotFoundError(2, "No such file or directory", "/proc/meminfo")

    with pytest.raises(SystemExit) as exc:
        _map_pipeline_errors(boom, "in.csv")
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "cannot open /proc/meminfo" in err
    assert "input file not found" not in err


def test_missing_meminfo_run_names_ram_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """End-to-end: absent meminfo + bare --turbo reports unknown RAM, exits 0.

    Platform-deterministic (#106): stub every physical-RAM source. The
    meminfo path covers Linux; the shared-detector stub (_read_physical_gb
    -> None) covers darwin, where the sysctl hw.memsize probe inside it
    would otherwise succeed on a real macOS runner (RAM known) and the
    unknown-RAM branch never runs.
    """
    csv = _tiny_csv(tmp_path)
    monkeypatch.setattr(turbop, "MEMINFO_PATH", str(tmp_path / "missing-meminfo"))
    monkeypatch.setattr(turbop, "_read_physical_gb", lambda: None)
    monkeypatch.setattr(turbop, "_read_cgroup_max_gb", lambda: math.inf)
    main(_run_argv(tmp_path / "out", "--turbo", csv=csv))
    err = capsys.readouterr().err
    assert "input file not found" not in err
    assert "physical RAM unknown (no /proc/meminfo)" in err


def test_osexcept_with_filename_names_path(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """An OSError carrying a filename names that path (disk-full / permission class)."""

    def boom() -> None:
        raise PermissionError(13, "Permission denied", str(tmp_path / "out" / "value.tif.tmp"))

    with pytest.raises(SystemExit) as exc:
        _map_pipeline_errors(boom, "in.csv")
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert f"cannot access {tmp_path / 'out' / 'value.tif.tmp'}: Permission denied" in err


def test_unwritable_out_dir_names_path(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Disk-full / permission class: the actually failing path is named (#106 item 4)."""
    if os.name == "nt" or os.geteuid() == 0:  # type: ignore[attr-defined]
        pytest.skip("directory permission bits are a no-op on Windows / root")
    csv = _tiny_csv(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    out.chmod(0o555)
    try:
        with pytest.raises(SystemExit) as exc:
            main(_run_argv(out, csv=csv))
    finally:
        out.chmod(0o755)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert str(out) in err
    assert "input file not found" not in err


# ---------------------------------------------------------------------------
# Exit-code table: every documented code has an asserting test
# ---------------------------------------------------------------------------


def test_exit_code_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """0 (ok), 1 (pipeline error), 2 (validation), 3 (turbo-strict infeasible)."""
    csv = _tiny_csv(tmp_path)
    base = _run_argv(tmp_path / "o0", csv=csv)

    # 0: a full run completes (main returns; no SystemExit).
    main(base)
    assert (tmp_path / "o0" / "value.tif").is_file()

    # 1: missing input file.
    with pytest.raises(SystemExit) as exc:
        main(
            [
                str(tmp_path / "nope.csv"),
                "-o",
                str(tmp_path / "o1"),
                "--value-col",
                "value",
                "--lng-col",
                "lng",
                "--lat-col",
                "lat",
            ]
        )
    assert exc.value.code == 1
    assert "input file not found" in capsys.readouterr().err

    # 2: bad flag value.
    with pytest.raises(SystemExit) as exc:
        main([*base, "--workers", "0"])
    assert exc.value.code == 2

    # 3: --turbo-strict with a shared path provably infeasible (A9 geometry,
    # same recipe as tests/test_turbo.py::test_cli_turbo_wiring).
    import ppgrid.turbop as turbop_mod

    def _identity_cap(preset: float | None, *_args: float | None, **_kwargs: float | None) -> tuple[float | None, str]:
        return preset, "preset"

    monkeypatch.setattr(turbop_mod, "resolve_cap", _identity_cap)
    orig_precheck = pipeline_mod.Pipeline._turbo_precheck

    def a9_precheck(self: Pipeline) -> None:
        # Inject the A9 anchor grid, then run the real pre-check (same recipe
        # as tests/test_turbo.py::test_cli_turbo_wiring).
        self.n, self.nx_padded, self.ny_padded, self.res, self.cap_km_val = (
            400_000,
            41472,
            1_592_524_800 // 41472,
            100.0,
            25,
        )
        orig_precheck(self)

    monkeypatch.setattr(pipeline_mod.Pipeline, "_turbo_precheck", a9_precheck)
    with pytest.raises(SystemExit) as exc:
        main(
            [
                str(csv),
                "-o",
                str(tmp_path / "o3"),
                "--value-col",
                "value",
                "--lng-col",
                "lng",
                "--lat-col",
                "lat",
                "--res",
                "100",
                "--cap-km",
                "25",
                "--turbo",
                "16",
                "--turbo-strict",
            ],
        )
    assert exc.value.code == 3


def test_exit_code_130_is_documented() -> None:
    """The epilog (printed by `ppgrid help`) documents every code incl. 130."""
    epilog = pipeline_mod._EPILOG
    for code in ("0", "1", "2", "3", "130"):
        assert f"\n  {code} " in epilog or epilog.startswith(f"  {code} "), f"exit {code} not documented"
