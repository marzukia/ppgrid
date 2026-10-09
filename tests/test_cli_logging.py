"""CLI help command + leveled logging tests.

Covers: the `ppgrid help` subcommand, the no-input error hint, the enriched
--help epilog, --verbose / --log-level / LOGGING resolution,
INFO phase progress + summary lines, DEBUG perf/volumetric lines, and
byte-identical output with and without --verbose (logging is side-channel).
"""

import hashlib
import logging
from pathlib import Path

import pytest

from ppgrid.pipeline import _build_parser, _resolve_log_level, main


def _tiny_csv(tmp_path: Path) -> str:
    """2-point CSV for the smallest possible end-to-end run (test_reproject_skip pattern)."""
    csv = tmp_path / "pts.csv"
    csv.write_text("value,lng,lat\n1.0,149.0,-35.0\n2.0,149.1,-35.1\n")
    return str(csv)


def _run_argv(tmp_path: Path, out: Path, *extra: str) -> list[str]:
    """Full-pipeline argv for the tiny CSV (res=100 keeps the grid small)."""
    return [
        _tiny_csv(tmp_path),
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


def _pipeline_msgs(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Message strings of ppgrid.pipeline records captured by caplog."""
    return [r.getMessage() for r in caplog.records if r.name == "ppgrid.pipeline"]


def _sha(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# --verbose / --log-level parser flags
# ---------------------------------------------------------------------------


def test_parser_log_flags() -> None:
    """The --verbose and --log-level flags parse; verbose is not a --log-level value."""
    p = _build_parser()
    args = p.parse_args(["in.csv", "--verbose"])
    assert args.verbose is True
    assert args.log_level is None
    args = p.parse_args(["in.csv", "--log-level", "debug"])
    assert args.verbose is False
    assert args.log_level == "debug"
    args = p.parse_args(["in.csv", "--log-level", "warning"])
    assert args.log_level == "warning"
    args = p.parse_args(["in.csv"])
    assert args.verbose is False
    assert args.log_level is None
    # The 'verbose' alias is accepted by the env vars only, not by the flag.
    with pytest.raises(SystemExit):
        p.parse_args(["in.csv", "--log-level", "verbose"])


@pytest.mark.parametrize(
    ("verbose", "flag", "env", "expect"),
    [
        (False, None, {}, "info"),  # default
        (True, None, {}, "debug"),  # --verbose shorthand
        (False, "debug", {}, "debug"),
        (False, "info", {"LOGGING": "debug"}, "info"),  # explicit flag beats env
        (True, None, {"LOGGING": "error"}, "debug"),  # --verbose beats env
        (True, "warning", {"LOGGING": "error"}, "warning"),  # explicit level beats --verbose
        (False, None, {"LOGGING": "error"}, "error"),  # env var
        (False, None, {"LOGGING": "verbose"}, "debug"),  # verbose alias value
        (False, None, {"LOGGING": "bogus"}, "info"),  # unrecognized -> info
    ],
)
def test_resolve_log_level_precedence(
    monkeypatch: pytest.MonkeyPatch,
    *,
    verbose: bool,
    flag: str | None,
    env: dict[str, str],
    expect: str,
) -> None:
    """Resolution precedence: --log-level > --verbose > LOGGING > info."""
    monkeypatch.delenv("LOGGING", raising=False)
    for key, val in env.items():
        monkeypatch.setenv(key, val)
    assert _resolve_log_level(verbose=verbose, log_level=flag) == expect


# ---------------------------------------------------------------------------
# `ppgrid help` + no-input dispatch
# ---------------------------------------------------------------------------


def test_main_help_command_prints_help_exit0(capsys: pytest.CaptureFixture[str]) -> None:
    """`ppgrid help` prints the rich help (usage + examples) and exits 0."""
    with pytest.raises(SystemExit) as exc:
        main(["help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out
    assert "examples:" in out
    assert "--log-level" in out
    assert "--verbose" in out


def test_main_no_input_error_exit2(capsys: pytest.CaptureFixture[str]) -> None:
    """Bare `ppgrid` (no input) exits 2 with a concise error + hint to `ppgrid help`."""
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "error" in err
    assert "ppgrid help" in err


def test_main_version_still_works(capsys: pytest.CaptureFixture[str]) -> None:
    """`ppgrid --version` still reports the package version and exits 0."""
    from ppgrid import __version__

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"ppgrid {__version__}"


# ---------------------------------------------------------------------------
# Leveled logging on a real (tiny) run
# ---------------------------------------------------------------------------


def test_caplog_info_phase_lines_and_summary(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Default (INFO) run logs one line per phase + the final summary, no DEBUG lines."""
    with caplog.at_level(logging.INFO, logger="ppgrid"):
        main(_run_argv(tmp_path, tmp_path / "out"))
    msgs = _pipeline_msgs(caplog)
    assert any(m.startswith("ingest: ") and "rows ->" in m for m in msgs)
    assert any(m.startswith("interpolate: done in") for m in msgs)
    assert any(m.startswith("done: ") and "rows ->" in m and "value.tif" in m for m in msgs)
    assert not any(m.startswith(("perf: ", "volumetrics: ", "plan: ")) for m in msgs)


def test_caplog_debug_verbose_extra(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """--verbose (DEBUG) run adds perf, volumetric and plan lines on top of the INFO ones."""
    with caplog.at_level(logging.DEBUG, logger="ppgrid"):
        main(_run_argv(tmp_path, tmp_path / "out", "--verbose"))
    msgs = _pipeline_msgs(caplog)
    perf = [m for m in msgs if m.startswith("perf: ")]
    vol = [m for m in msgs if m.startswith("volumetrics: ")]
    plan = [m for m in msgs if m.startswith("plan: ")]
    assert len(perf) == 1
    assert "ms" in perf[0]
    assert len(vol) == 1
    assert "rows_read=2" in vol[0]
    assert "dropped_nan_inf=" in vol[0]
    assert len(plan) == 1
    assert "workers=2" in plan[0]
    assert "transform=" in plan[0]


def test_caplog_env_legacy_logging_verbose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """LOGGING=verbose (legacy alias) enables the DEBUG perf/volumetric lines."""
    monkeypatch.setenv("LOGGING", "verbose")
    with caplog.at_level(logging.DEBUG, logger="ppgrid"):
        main(_run_argv(tmp_path, tmp_path / "out"))
    assert any(m.startswith("perf: ") for m in _pipeline_msgs(caplog))


def test_caplog_log_level_error_suppresses_info(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """--log-level error runs silently (no DEBUG/INFO ppgrid.pipeline records)."""
    with caplog.at_level(logging.ERROR, logger="ppgrid"):
        main(_run_argv(tmp_path, tmp_path / "out", "--log-level", "error"))
    info_msgs = [
        r.getMessage()
        for r in caplog.records
        if r.name == "ppgrid.pipeline" and logging.DEBUG <= r.levelno <= logging.INFO
    ]
    assert info_msgs == []


def test_verbose_and_default_output_byte_identical(tmp_path: Path) -> None:
    """Logging is side-channel: default and --verbose runs emit byte-identical outputs."""
    main(_run_argv(tmp_path, tmp_path / "out_a"))
    main(_run_argv(tmp_path, tmp_path / "out_b", "--verbose"))
    for name in ("value.tif", "support_km.tif", "calibration.json"):
        assert _sha(tmp_path / "out_a" / name) == _sha(tmp_path / "out_b" / name), name


def test_handler_idempotency_across_level_changes(tmp_path: Path) -> None:
    """Repeated main() calls with different levels must not stack handlers."""
    main(_run_argv(tmp_path, tmp_path / "o1"))
    main(_run_argv(tmp_path, tmp_path / "o2", "--log-level", "error"))
    main(_run_argv(tmp_path, tmp_path / "o3", "--verbose"))
    assert len(logging.getLogger("ppgrid").handlers) == 1
