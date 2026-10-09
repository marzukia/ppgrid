"""CLI polish smalls tests (#52-#60, #77).

Covers: --quiet suppression + precedence, --plan (exit 0, no files), the
overwrite guard + --force (#54), --json run summary (#55), the exit-code
epilog (#56), friendly missing-column errors (#57), EPSG string CRS args
(#58), --workers auto (#59), --seed (#60 — stochastic case: calibration
subsampling + blocked-CV bootstrap are seeded), byte-identical outputs
with/without each new flag, and the #77 contract fixes (stdout stays clean
on turbo + sparse-skip runs, stale-warning coverage for preflight/plan
failures, --json write-failure mapping, EPSG exit-2 alignment).
"""

import hashlib
import json
import logging
import os
from pathlib import Path

import pytest

from ppgrid.pipeline import _build_parser, _resolve_log_level, main


def _tiny_csv(tmp_path: Path) -> str:
    """2-point CSV for the smallest possible end-to-end run (test_cli_logging pattern)."""
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


def _pipeline_records(caplog: pytest.LogCaptureFixture, *, max_level: int = logging.INFO) -> list[str]:
    """Message strings of ppgrid.pipeline records up to max_level."""
    return [r.getMessage() for r in caplog.records if r.name == "ppgrid.pipeline" and r.levelno <= max_level]


def _sha(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _outputs_sha(out: Path) -> dict[str, str]:
    """SHA-256 of the three committed-style outputs in a run dir."""
    return {name: _sha(out / name) for name in ("value.tif", "support_km.tif", "calibration.json")}


# ---------------------------------------------------------------------------
# #52 --quiet
# ---------------------------------------------------------------------------


def test_quiet_suppresses_info(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """-q runs with no ppgrid.pipeline records below WARNING on stderr."""
    with caplog.at_level(logging.WARNING, logger="ppgrid"):
        main(_run_argv(tmp_path, tmp_path / "out", "-q"))
    assert _pipeline_records(caplog) == []
    # The run itself succeeded.
    assert (tmp_path / "out" / "value.tif").is_file()


def test_quiet_beats_verbose_and_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Precedence: --quiet beats --verbose and LOGGING; explicit --log-level beats --quiet."""
    monkeypatch.setenv("LOGGING", "debug")
    assert _resolve_log_level(verbose=False, log_level=None, quiet=True) == "warning"
    assert _resolve_log_level(verbose=True, log_level=None, quiet=True) == "warning"
    assert _resolve_log_level(verbose=False, log_level="info", quiet=True) == "info"
    # And end-to-end: -q --log-level info still logs INFO.
    monkeypatch.delenv("LOGGING", raising=False)
    with caplog.at_level(logging.INFO, logger="ppgrid"):
        main(_run_argv(tmp_path, tmp_path / "out2", "-q", "--log-level", "info"))
    assert any(m.startswith("ingest: ") for m in _pipeline_records(caplog))


# ---------------------------------------------------------------------------
# #53 --plan
# ---------------------------------------------------------------------------


def test_plan_exits_zero_creates_no_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--plan resolves the plan, prints it, exits 0, and creates nothing."""
    out = tmp_path / "plan_out"
    with pytest.raises(SystemExit) as exc:
        main(_run_argv(tmp_path, out, "--plan"))
    assert exc.value.code == 0
    assert not out.exists()
    assert [p.name for p in tmp_path.iterdir()] == ["pts.csv"]
    text = capsys.readouterr().out
    for needle in ("run plan", "grid cells", "levels", "transform", "seed", "turbo", "workers"):
        assert needle in text, needle


def test_plan_includes_turbo_decision(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--plan with --turbo prints the resolved regime/budget instead of running."""
    out = tmp_path / "plan_turbo"
    with pytest.raises(SystemExit) as exc:
        main(_run_argv(tmp_path, out, "--plan", "--turbo", "16"))
    assert exc.value.code == 0
    assert not out.exists()
    text = capsys.readouterr().out
    assert "turbo:         on" in text
    assert "regime" in text
    assert "budget" in text


# ---------------------------------------------------------------------------
# #54 overwrite guard + --force
# ---------------------------------------------------------------------------


def test_guard_exit2_without_force(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A re-run over an out dir containing prior outputs exits 2 and lists them."""
    out = tmp_path / "out"
    main(_run_argv(tmp_path, out))
    with pytest.raises(SystemExit) as exc:
        main(_run_argv(tmp_path, out))
    assert exc.value.code == 2
    err = capsys.readouterr().err
    for name in ("value.tif", "support_km.tif", "calibration.json"):
        assert name in err
    assert "--force" in err
    # The guard runs before the pipeline: prior outputs are untouched.
    assert (out / "value.tif").is_file()


def test_guard_triggers_on_single_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """run_summary.json (or any guarded name) alone trips the guard."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "run_summary.json").write_text("{}")
    with pytest.raises(SystemExit) as exc:
        main(_run_argv(tmp_path, out))
    assert exc.value.code == 2
    assert "run_summary.json" in capsys.readouterr().err


def test_force_overwrites(tmp_path: Path) -> None:
    """--force runs to completion over an existing out dir (exit 0, no raise)."""
    out = tmp_path / "out"
    main(_run_argv(tmp_path, out))
    main(_run_argv(tmp_path, out, "--force"))  # returns normally
    assert (out / "value.tif").is_file()


def test_plan_skips_guard(tmp_path: Path) -> None:
    """--plan is a read-only resolution: an existing out dir does not block it."""
    out = tmp_path / "out"
    main(_run_argv(tmp_path, out))
    with pytest.raises(SystemExit) as exc:
        main(_run_argv(tmp_path, out, "--plan"))
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# #55 --json
# ---------------------------------------------------------------------------


def test_json_summary(tmp_path: Path) -> None:
    """--json writes <out>/run_summary.json with the expected keys and values."""
    out = tmp_path / "out"
    main(_run_argv(tmp_path, out, "--json", "--seed", "7"))
    s = json.loads((out / "run_summary.json").read_text())
    for key in (
        "ppgrid_version",
        "seed",
        "transform",
        "cap_km",
        "grid",
        "ingest",
        "calibration",
        "timings_ms",
        "volumetrics",
        "turbo",
        "outputs",
        "cli",
        "git_commit",
    ):
        assert key in s, key
    assert s["seed"] == 7
    assert s["ingest"] == {"rows_read": 2, "points": 2, "dropped_nan_inf": 0}
    g = s["grid"]
    assert g["cells"] == g["nx"] * g["ny"] > 0
    assert g["src_crs"] == 4326
    assert g["work_crs"] == 6933
    assert g["out_crs"] == 3857
    t = s["timings_ms"]
    for key in ("ingest", "calibrate", "grid", "write", "total", "subphases"):
        assert key in t, key
    assert t["total"] >= t["ingest"] >= 0
    v = s["volumetrics"]
    assert len(v["cells_per_level"]) == g["levels"] + 1
    for band in ("value", "support"):
        assert v[band]["raw_bytes"] == g["cells"] * 2
        assert 0 < v[band]["compressed_bytes"] <= v[band]["raw_bytes"]
        assert v[band]["compression_ratio"] > 1.0
    assert s["turbo"] is None  # not a turbo run
    assert s["cli"]["seed"] == 7
    assert s["outputs"]["value"].endswith("value.tif")
    # Git commit: a string when the checkout is under git, else null.
    assert s["git_commit"] is None or isinstance(s["git_commit"], str)


def test_json_keeps_stdout_clean(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--json writes the summary to the file only; stdout stays empty."""
    out = tmp_path / "out"
    main(_run_argv(tmp_path, out, "--json"))
    captured = capsys.readouterr()
    assert not captured.out
    json.loads((out / "run_summary.json").read_text())  # valid JSON in the file


# ---------------------------------------------------------------------------
# #56 exit-code epilog
# ---------------------------------------------------------------------------


def test_help_documents_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    """--help (and `ppgrid help`) document the exit-code contract + new flags."""
    with pytest.raises(SystemExit) as exc:
        main(["help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "exit codes:" in text
    for line in ("0  ok", "1  pipeline / I/O error", "2  validation error", "3  --turbo-strict"):
        assert line in text, line
    for flag in ("--quiet", "--plan", "--force", "--json", "--seed"):
        assert flag in text, flag


# ---------------------------------------------------------------------------
# #57 friendly column errors
# ---------------------------------------------------------------------------


def test_missing_column_lists_real_columns(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A bad --value-col name exits 2, lists the actual columns, and suggests matches."""
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        main(_run_argv(tmp_path, out, "--value-col", "price"))
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--value-col 'price' not found" in err
    assert "value, lng, lat" in err
    assert (tmp_path / "out").exists() is False  # failed before creating anything


def test_missing_column_closest_match(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A typo'd --lng-col gets a 'did you mean' suggestion from difflib."""
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        main(_run_argv(tmp_path, out, "--lng-col", "lng2"))
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "did you mean: lng?" in err


def test_missing_column_in_plan_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The column preflight applies to --plan too (same exit 2)."""
    with pytest.raises(SystemExit) as exc:
        main(_run_argv(tmp_path, tmp_path / "out", "--plan", "--lat-col", "laty"))
    assert exc.value.code == 2
    assert "not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# #58 EPSG string CRS args
# ---------------------------------------------------------------------------


def test_epsg_string_parsing(tmp_path: Path) -> None:
    """--src-crs / --work-crs / --out-crs accept 'EPSG:<code>' and bare ints."""
    p = _build_parser()
    a = p.parse_args(["in.csv", "--src-crs", "EPSG:4326", "--work-crs", "epsg:6933", "--out-crs", "3857"])
    assert (a.src_crs, a.work_crs, a.out_crs) == (4326, 6933, 3857)
    a = p.parse_args(["in.csv", "--src-crs", "4326"])  # bare int still valid
    assert a.src_crs == 4326
    for bad in ("EPSG:abc", "nope", "EPSG:", ""):
        with pytest.raises(SystemExit) as exc:
            p.parse_args(["in.csv", "--out-crs", bad])
        assert exc.value.code == 2
    # The parser accepts any int (incl. negative); main()'s validation rejects
    # non-positive codes with parser.error (exit 2).
    assert p.parse_args(["in.csv", "--src-crs", "EPSG:-1"]).src_crs == -1
    csv = _tiny_csv(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main([str(csv), "-o", str(tmp_path / "out"), "--src-crs", "EPSG:-1"])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# #59 --workers auto
# ---------------------------------------------------------------------------


def test_workers_auto() -> None:
    """--workers auto resolves to os.cpu_count(); ints stay valid; junk exits 2."""
    p = _build_parser()
    assert p.parse_args(["in.csv", "--workers", "auto"]).workers == os.cpu_count()
    assert p.parse_args(["in.csv", "--workers", "Auto"]).workers == os.cpu_count()  # case-insensitive
    assert p.parse_args(["in.csv", "--workers", "7"]).workers == 7
    with pytest.raises(SystemExit) as exc:
        p.parse_args(["in.csv", "--workers", "x"])
    assert exc.value.code == 2
    # --workers 0 still fails the >= 1 validation in main (exit 2).
    with pytest.raises(SystemExit) as exc:
        main(["in.csv", "--workers", "0"])
    assert exc.value.code == 2


def test_sanitize_json_replaces_nonfinite() -> None:
    """--json summary must survive NaN/Inf from external cal files (m-2)."""
    from ppgrid.pipeline import _sanitize_json

    out = _sanitize_json({"a": float("nan"), "b": [1.0, float("inf")], "c": "x", "d": None})
    assert out == {"a": None, "b": [1.0, None], "c": "x", "d": None}


def test_scale_init_validation_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Pipeline.__init__ validation errors map to exit 2, not a traceback (M-1)."""
    csv = tmp_path / "pts.csv"
    csv.write_text("value,longitude,latitude\n1.0,149.0,-35.0\n2.0,149.1,-35.1\n")
    with pytest.raises(SystemExit) as exc:
        main([str(csv), "-o", str(tmp_path / "o"), "--scale", "400"])
    assert exc.value.code == 2
    assert "scale * 100 exceeds int16 max" in capsys.readouterr().err
    # Same contract in --plan mode (construction is mapped there too).
    with pytest.raises(SystemExit) as exc:
        main([str(csv), "-o", str(tmp_path / "o2"), "--scale", "400", "--plan"])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# #60 --seed (stochastic case: calib subsampling + blocked-CV bootstrap)
# ---------------------------------------------------------------------------


def test_seed_default_zero_in_summary(tmp_path: Path) -> None:
    """Without --seed the summary records seed 0 (the historical hardcoded seed)."""
    out = tmp_path / "out"
    main(_run_argv(tmp_path, out, "--json"))
    s = json.loads((out / "run_summary.json").read_text())
    assert s["seed"] == 0


def test_seed_explicit_zero_byte_identical(tmp_path: Path) -> None:
    """--seed 0 (the default) is byte-identical to omitting the flag."""
    main(_run_argv(tmp_path, tmp_path / "out_a"))
    main(_run_argv(tmp_path, tmp_path / "out_b", "--seed", "0"))
    assert _outputs_sha(tmp_path / "out_a") == _outputs_sha(tmp_path / "out_b")


def test_seed_nonzero_runs_and_records(tmp_path: Path) -> None:
    """--seed N runs to completion and the seed is recorded in the summary."""
    out = tmp_path / "out"
    main(_run_argv(tmp_path, out, "--json", "--seed", "12345"))
    s = json.loads((out / "run_summary.json").read_text())
    assert s["seed"] == 12345
    assert (out / "value.tif").is_file()


def test_seed_negative_rejected(tmp_path: Path) -> None:
    """--seed -1 exits 2 via parser.error."""
    with pytest.raises(SystemExit) as exc:
        main(_run_argv(tmp_path, tmp_path / "out", "--seed", "-1"))
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# Byte identity: every new flag vs the baseline (2-point CSV)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        ["-q"],  # #52
        ["--force"],  # #54
        ["--json"],  # #55 (adds run_summary.json; the three outputs are untouched)
        ["--seed", "0"],  # #60
        ["--src-crs", "EPSG:4326", "--work-crs", "EPSG:6933", "--out-crs", "EPSG:3857"],  # #58
        ["--workers", "auto"],  # #59 (MT bit-identity, S3)
        ["-q", "--log-level", "warning"],  # #52 precedence
    ],
    ids=[
        "quiet",
        "force",
        "json",
        "seed0",
        "epsg-strings",
        "workers-auto",
        "quiet-loglevel",
    ],
)
def test_new_flags_byte_identical(tmp_path: Path, extra: list[str]) -> None:
    """Each new flag (unchanged semantics) leaves the three outputs byte-identical."""
    main(_run_argv(tmp_path, tmp_path / "ref"))
    main(_run_argv(tmp_path, tmp_path / "alt", *extra))
    assert _outputs_sha(tmp_path / "ref") == _outputs_sha(tmp_path / "alt"), extra


# ---------------------------------------------------------------------------
# #77: CLI contract fixes
# ---------------------------------------------------------------------------


def test_turbo_run_keeps_stdout_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    """Turbo run: the precheck summary is a log line (stderr), stdout stays empty (#77)."""
    with caplog.at_level(logging.DEBUG, logger="ppgrid"):
        main(_run_argv(tmp_path, tmp_path / "out", "--turbo", "16"))
    captured = capsys.readouterr()
    assert not captured.out
    assert any(m.startswith("--turbo: cap 16") for m in _pipeline_records(caplog))


def _far_csv(tmp_path: Path) -> str:
    """Two far-apart points (Melbourne + Darwin) for a sparse grid (the #77 repro)."""
    csv = tmp_path / "far.csv"
    csv.write_text("value,lng,lat\n1.0,145.0,-37.8\n2.0,130.8,-12.4\n")
    return str(csv)


def test_sparse_tile_skip_keeps_stdout_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    """Sparse grid: the reproject tile-skip counter is DEBUG log, stdout stays empty (#77)."""
    out = tmp_path / "out"
    with caplog.at_level(logging.DEBUG, logger="ppgrid"):
        main(
            [
                _far_csv(tmp_path),
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
                "--cap-km",
                "20",
                "--block",
                "2048",
                "--workers",
                "2",
                "--log-level",
                "debug",  # the skip counter is a DEBUG log line (#77)
            ]
        )
    captured = capsys.readouterr()
    assert not captured.out
    assert any("all-NoData tiles" in m for m in _pipeline_records(caplog, max_level=logging.DEBUG))


def test_plan_failure_warns_stale_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--plan failure over an earlier run's outputs warns about the stale rasters (#77)."""
    out = tmp_path / "out"
    main(_run_argv(tmp_path, out))
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        main(_run_argv(tmp_path, out, "--plan", "--value-col", "nope"))
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "failed before writing outputs" in err
    assert "value.tif" in err


def test_argparse_error_warns_stale_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Argparse validation failure (exit 2) over an earlier run's outputs warns (#77)."""
    out = tmp_path / "out"
    main(_run_argv(tmp_path, out))
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        main(_run_argv(tmp_path, out, "--workers", "0"))
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--workers must be at least 1" in err
    assert "failed before writing outputs" in err


def test_json_summary_dir_conflict_exit1_no_stale_warning(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """run_summary.json as a directory: exit 1, named error, no traceback.

    No stale-outputs warning: the rasters were published this run (#77).
    """
    out = tmp_path / "out"
    main(_run_argv(tmp_path, out))
    capsys.readouterr()
    (out / "run_summary.json").mkdir()
    with pytest.raises(SystemExit) as exc:
        main(_run_argv(tmp_path, out, "--json", "--force"))
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "cannot write run summary" in err
    assert "Traceback" not in err
    assert "failed before writing outputs" not in err
    assert not (out / "run_summary.json.tmp").exists()
    assert (out / "value.tif").is_file()


@pytest.mark.parametrize("flag", ["--src-crs", "--work-crs", "--out-crs"])
def test_unknown_epsg_exit2_all_flags(tmp_path: Path, capsys: pytest.CaptureFixture[str], flag: str) -> None:
    """Unknown EPSG exits 2 (bad flag value) for all three CRS flags and names the code (#77)."""
    with pytest.raises(SystemExit) as exc:
        main(_run_argv(tmp_path, tmp_path / "out", flag, "999999999", "--force"))
    assert exc.value.code == 2, flag
    err = capsys.readouterr().err
    assert "999999999" in err
    assert "Traceback" not in err
