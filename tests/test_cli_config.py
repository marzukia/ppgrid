"""Config file, init template, run-fingerprint sidecar, --log-file (#63, #64, #65).

Covers:
- Precedence: CLI flag > config > built-in default, across six options
  (res, transform, workers, scale, value-col, cap-km) — at the merge level
  (unit) and through full runs (sidecar params).
- Discovery: --config PATH (explicit; a missing file is a usage error, exit 2)
  > ./ppgrid.toml > $HOME/.config/ppgrid/ppgrid.toml; no file = silent no-op.
- Unknown keys (including table headers): warn + continue. Invalid values and
  invalid TOML: usage error, exit 2.
- `ppgrid init [DIR]`: valid fully-commented template covering every
  config-able option at its built-in default, refuses to clobber without
  --force, loads no native stack (like --help).
- Template round-trip: init -> run with the template (discovered) == run with
  the template's active values as explicit flags (rasters byte-identical).
- `ppgrid_run.json` sidecar: contents (effective params, versions, input name,
  ISO timestamp, config path), stable fingerprint (same effective params ->
  identical across runs; one param changed -> different; timestamp excluded),
  sha256 of the canonical params JSON, not written on --plan or on failure.
- `--log-file`: per-run create/truncate, level-info copy independent of the
  stderr level, same unified format without color, stderr output unchanged,
  open failure -> exit 1.
"""

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from ppgrid.pipeline import (
    _CONFIG_DEFAULTS,
    _CONFIG_KEYS,
    _CONFIG_OPTIONS,
    UNSET,
    _apply_config,
    _build_parser,
    main,
)

# ---------------------------------------------------------------------------
# Fixtures + builders
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point cwd + HOME at clean dirs so config discovery is fully under test control."""
    work = tmp_path / "work"
    home = tmp_path / "home"
    work.mkdir()
    home.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("LOGGING", raising=False)
    return work


def _tiny_csv(tmp_path: Path) -> Path:
    """2-point CSV (default column names) for the smallest possible end-to-end run."""
    csv = tmp_path / "pts.csv"
    csv.write_text("value,longitude,latitude\n1.0,149.0,-35.0\n2.0,149.1,-35.1\n")
    return csv


def _run_argv(tmp_path: Path, out: Path, *extra: str) -> list[str]:
    """Full-pipeline argv for the tiny CSV (res=100 keeps the grid small)."""
    return [
        str(_tiny_csv(tmp_path)),
        "-o",
        str(out),
        "--res",
        "100",
        "--block",
        "2048",
        "--workers",
        "2",
        *extra,
    ]


def _home_cfg() -> Path:
    """$HOME/.config/ppgrid/ppgrid.toml under the test HOME, created."""
    cfg = Path(os.environ["HOME"]) / ".config" / "ppgrid" / "ppgrid.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    return cfg


def _sidecar(out: Path) -> dict:
    return json.loads((out / "ppgrid_run.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Precedence (unit): flag > config > default
# ---------------------------------------------------------------------------


def test_precedence_matrix(isolated: Path) -> None:
    """CLI flag > config > built-in default across six options, at the merge level."""
    config_text = """\
res = 1000
transform = "log10"
workers = 3
scale = 10
value-col = "alt"
cap-km = 20
"""
    (isolated / "ppgrid.toml").write_text(config_text)
    p = _build_parser()

    # 1. config only: every config value beats the built-in default
    a = p.parse_args(["in.csv"])
    _apply_config(p, a)
    assert (a.res, a.transform, a.workers, a.scale, a.value_col, a.cap_km) == (
        1000.0,
        "log10",
        3,
        10.0,
        "alt",
        20.0,
    )

    # 2. one flag: the flag beats its config value; the other config keys still apply
    a = p.parse_args(["in.csv", "--res", "250"])
    _apply_config(p, a)
    assert a.res == 250.0
    assert (a.transform, a.workers, a.scale, a.value_col, a.cap_km) == ("log10", 3, 10.0, "alt", 20.0)

    # 3. all six flags: every flag beats its config value
    a = p.parse_args(
        [
            "in.csv",
            "--res",
            "300",
            "--transform",
            "sqrt",
            "--workers",
            "5",
            "--scale",
            "50",
            "--value-col",
            "v2",
            "--cap-km",
            "40",
        ]
    )
    _apply_config(p, a)
    assert (a.res, a.transform, a.workers, a.scale, a.value_col, a.cap_km) == (300.0, "sqrt", 5, 50.0, "v2", 40.0)

    # 4. no config anywhere: built-in defaults, and no UNSET sentinel leaks
    (isolated / "ppgrid.toml").unlink()
    a = p.parse_args(["in.csv"])
    _apply_config(p, a)
    assert (a.res, a.transform, a.workers, a.scale, a.value_col, a.cap_km) == (500.0, "auto", 4, 100.0, "value", "auto")
    for key, dest, _, _ in _CONFIG_OPTIONS:
        assert getattr(a, dest) is not UNSET, key


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discovery_order(isolated: Path) -> None:
    """--config PATH > ./ppgrid.toml > $HOME/.config/ppgrid/ppgrid.toml > silent no-op."""
    p = _build_parser()

    def effective() -> tuple[float, Path | None]:
        a = p.parse_args(["in.csv"])
        cfg = _apply_config(p, a)
        return a.res, cfg.resolve() if cfg is not None else None

    # 1. no file anywhere: silent no-op, built-in default
    assert effective() == (500.0, None)

    # 2. home config alone: applied
    home = _home_cfg()
    home.write_text("res = 900\n")
    assert effective() == (900.0, home)

    # 3. cwd config beats home
    cwd_cfg = isolated / "ppgrid.toml"
    cwd_cfg.write_text("res = 800\n")
    assert effective() == (800.0, cwd_cfg)

    # 4. explicit --config beats cwd
    explicit = isolated / "explicit.toml"
    explicit.write_text("res = 700\n")
    a = p.parse_args(["in.csv", "--config", str(explicit)])
    cfg = _apply_config(p, a)
    assert a.res == 700.0
    assert cfg == explicit

    # 5. explicit but missing: usage error 2
    a = p.parse_args(["in.csv", "--config", str(isolated / "nope.toml")])
    with pytest.raises(SystemExit) as e:
        _apply_config(p, a)
    assert e.value.code == 2


def test_explicit_config_missing_exit2(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An explicitly named config file that does not exist exits 2 (full CLI path)."""
    with pytest.raises(SystemExit) as e:
        main(_run_argv(isolated, isolated / "out", "--config", str(isolated / "nope.toml")))
    assert e.value.code == 2
    assert "config file not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Unknown keys + invalid values
# ---------------------------------------------------------------------------


def test_unknown_key_warns_and_continues(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Unknown keys (scalar + table header) warn once each, the run continues, known keys apply."""
    (isolated / "ppgrid.toml").write_text("res = 1000\nbogus-key = 1\n[grid]\nworkers = 9\n")
    main([str(_tiny_csv(isolated)), "-o", str(isolated / "out"), "--block", "2048", "--workers", "2"])
    err = capsys.readouterr().err
    assert "unknown key 'bogus-key'" in err, err
    assert "unknown key 'grid'" in err, err
    s = _sidecar(isolated / "out")
    assert s["params"]["res"] == 1000.0  # known key applied (no --res flag)
    assert s["params"]["workers"] == 2  # from the CLI flag; the [grid] table was ignored entirely


@pytest.mark.parametrize(
    ("line", "needle"),
    [
        ("res = 'abc'", "must be a number"),  # wrong TOML type
        ("transform = 5", "must be a string"),  # wrong TOML type
        ("workers = 'x'", "workers"),  # type checker rejects the spelling
        ("turbo = 42", "turbo"),  # not a preset
        ("scale = -5", "scale"),  # out of range
        ("src-crs = 'bogus'", "EPSG"),  # unknown CRS
    ],
)
def test_invalid_value_exit2(isolated: Path, capsys: pytest.CaptureFixture[str], line: str, needle: str) -> None:
    """A config value the flag's own checker rejects is a usage error, exit 2."""
    (isolated / "ppgrid.toml").write_text(line + "\n")
    # no CLI flags at all: nothing may shadow a config value
    argv = [str(_tiny_csv(isolated)), "-o", str(isolated / "out")]
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert needle in err, err


def test_invalid_toml_exit2(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A config file that does not parse as TOML is a usage error, exit 2."""
    (isolated / "ppgrid.toml").write_text("res = \n")
    with pytest.raises(SystemExit) as e:
        main(_run_argv(isolated, isolated / "out"))
    assert e.value.code == 2
    assert "invalid TOML" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# `ppgrid init`
# ---------------------------------------------------------------------------


def test_init_writes_parseable_template(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Write the init template: valid TOML covering every config-able option, active values = built-in defaults."""
    main(["init"])
    toml = isolated / "ppgrid.toml"
    assert toml.is_file()
    err = capsys.readouterr().err
    assert any(line.startswith("ok: wrote") and line.endswith("ppgrid.toml") for line in err.splitlines()), err

    text = toml.read_text(encoding="utf-8")
    data = tomllib.loads(text)  # parses

    # every option appears, active (at its built-in default) or commented with an example
    commented = set(re.findall(r"^# ([a-z][a-z0-9-]*) = ", text, re.MULTILINE))
    assert set(data) | commented == _CONFIG_KEYS
    assert set(data) & commented == set()

    for key, dest, _, _ in _CONFIG_OPTIONS:
        if key in data:
            assert data[key] == _CONFIG_DEFAULTS[dest], key


def test_init_refuses_clobber_without_force(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Refuse a second init without --force (exit 2, file on disk); --force overwrites."""
    main(["init"])
    toml = isolated / "ppgrid.toml"
    assert toml.is_file()
    with pytest.raises(SystemExit) as e:
        main(["init"])
    assert e.value.code == 2
    assert "already exists" in capsys.readouterr().err
    main(["init", "--force"])  # no error
    assert toml.is_file()


def test_init_creates_dir(isolated: Path) -> None:
    """Create the directory tree for `init [DIR]`."""
    main(["init", "sub/dir"])
    assert (isolated / "sub" / "dir" / "ppgrid.toml").is_file()


def test_init_defers_native_modules(tmp_path: Path) -> None:
    """Run `ppgrid init` in a fresh interpreter: no native stack loads (like --help)."""
    code = (
        "import sys\n"
        "from ppgrid.pipeline import main\n"
        "main(['init'])\n"
        "for mod in ('rasterio', 'pyproj', 'pandas', 'rich'):\n"
        "    assert mod not in sys.modules, f'{mod} loaded by init'\n"
        "assert 'ppgrid.ui' not in sys.modules, 'ppgrid.ui loaded by init'\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr


def test_init_template_run_has_no_unknown_key_warnings(isolated: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Run a full pipeline over the init template (discovered): no unknown-key warnings."""
    main(["init"])
    with caplog.at_level(logging.INFO, logger="ppgrid"):
        main(_run_argv(isolated, isolated / "out"))
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert not [m for m in warns if "unknown key" in m], warns


# ---------------------------------------------------------------------------
# Template round-trip
# ---------------------------------------------------------------------------


def _template_flags(data: dict, out: Path) -> list[str]:
    """Render the template's active values as explicit CLI flags (`out` is the run target: -o)."""
    argv: list[str] = []
    for key, value in data.items():
        if key == "out":
            continue
        if isinstance(value, bool):
            if value:
                argv.append(f"--{key}")
            continue
        argv += [f"--{key}", str(value)]
    argv += ["-o", str(out)]
    return argv


def test_init_template_roundtrip(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Round-trip the init template: run with the template == run with the same values as explicit flags.

    Both runs take the same run-shape flags (res/block/workers); every other
    parameter comes from the template — discovered by run A, explicit by run B.
    Rasters must be byte-identical and sidecar params equal (modulo `out`).
    """
    csv = _tiny_csv(isolated)
    main(["init"])
    data = tomllib.loads((isolated / "ppgrid.toml").read_text(encoding="utf-8"))
    shape = ["--res", "100", "--block", "2048", "--workers", "2"]

    # run A: template discovered from cwd
    main([str(csv), "-o", str(isolated / "outA"), *shape])

    # run B: template's active values as explicit flags, from a cwd with no config
    work_b = isolated / "workB"
    work_b.mkdir()
    monkeypatch.chdir(work_b)
    main([str(csv), *_template_flags(data, work_b / "out"), *shape])

    for name in ("value.tif", "support_km.tif", "calibration.json"):
        assert (isolated / "outA" / name).read_bytes() == (work_b / "out" / name).read_bytes(), name

    a, b = _sidecar(isolated / "outA"), _sidecar(work_b / "out")
    assert a["config"] == str(isolated / "ppgrid.toml")
    assert b["config"] is None
    assert {k: v for k, v in a["params"].items() if k != "out"} == {k: v for k, v in b["params"].items() if k != "out"}


# ---------------------------------------------------------------------------
# Config through a full run
# ---------------------------------------------------------------------------


def test_config_applies_through_full_run(isolated: Path) -> None:
    """Config values reach the run (sidecar params); a CLI flag still beats config."""
    cfg = isolated / "ppgrid.toml"
    cfg.write_text('res = 100\ntransform = "log10"\nworkers = 3\n')
    out = isolated / "out"
    main([str(_tiny_csv(isolated)), "-o", str(out), "--block", "2048", "--transform", "sqrt", "--workers", "2"])
    s = _sidecar(out)
    assert s["config"] == str(cfg)
    assert s["params"]["res"] == 100.0  # config beats default
    assert s["params"]["transform"] == "sqrt"  # flag beats config
    assert s["params"]["workers"] == 2  # flag beats config


# ---------------------------------------------------------------------------
# ppgrid_run.json sidecar (#64)
# ---------------------------------------------------------------------------


def test_sidecar_contents(isolated: Path) -> None:
    """Schema, effective params, versions, input name, ISO timestamp, fingerprint = sha256(canonical params)."""
    out = isolated / "out"
    main(_run_argv(isolated, out, "--seed", "7", "--transform", "log10"))
    s = _sidecar(out)
    assert set(s) == {"schema_version", "params", "config", "input", "versions", "timestamp", "fingerprint"}
    assert s["schema_version"] == 1
    assert s["input"] == "pts.csv"  # name only, not the path
    assert s["config"] is None
    assert set(s["params"]) == _CONFIG_KEYS
    assert s["params"]["seed"] == 7
    assert s["params"]["transform"] == "log10"
    assert s["params"]["res"] == 100.0  # run-shape flag
    assert set(s["versions"]) == {"ppgrid", "python", "numpy", "rasterio", "gdal", "pyproj"}
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", s["timestamp"])
    assert re.fullmatch(r"[0-9a-f]{64}", s["fingerprint"])
    canonical = json.dumps(s["params"], sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == s["fingerprint"]


def test_sidecar_stable_fingerprint(isolated: Path) -> None:
    """Identical effective params -> identical fingerprint across runs; one param changed -> different."""
    out = isolated / "out"
    main(_run_argv(isolated, out, "--force"))
    s1 = _sidecar(out)
    main(_run_argv(isolated, out, "--force"))  # identical effective params, second run
    s2 = _sidecar(out)
    assert s1["params"] == s2["params"]
    assert s1["fingerprint"] == s2["fingerprint"]
    main(_run_argv(isolated, out, "--force", "--res", "200"))  # one param changed
    s3 = _sidecar(out)
    assert s3["fingerprint"] != s1["fingerprint"]


def test_sidecar_records_config(isolated: Path) -> None:
    """The config file path lands in the sidecar; params show the post-merge values."""
    cfg = isolated / "ppgrid.toml"
    cfg.write_text("res = 100\n")
    out = isolated / "out"
    main([str(_tiny_csv(isolated)), "-o", str(out), "--block", "2048", "--workers", "2"])
    s = _sidecar(out)
    assert s["config"] == str(cfg)
    assert s["params"]["res"] == 100.0


def test_sidecar_not_written_on_plan_or_failure(isolated: Path) -> None:
    """--plan (exit 0) and a failed run (exit 1) leave no sidecar behind."""
    out = isolated / "out"
    with pytest.raises(SystemExit) as e:
        main(_run_argv(isolated, out, "--plan"))
    assert e.value.code == 0
    assert not (out / "ppgrid_run.json").exists()

    argv = [str(isolated / "nope.csv"), "-o", str(isolated / "out2"), "--res", "100", "--workers", "2"]
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 1
    assert not (isolated / "out2" / "ppgrid_run.json").exists()


# ---------------------------------------------------------------------------
# Subcommand dispatch edge
# ---------------------------------------------------------------------------


def test_subcommand_token_wins_over_same_named_file(isolated: Path) -> None:
    """A file named `init` in the cwd: the bare token dispatches to the subcommand.

    The bare `init` writes the ppgrid.toml template (subcommand, exit 0);
    any path form (`./init`) runs the pipeline on that file. Dispatch is on
    the literal first token (token-wins, same rule as the sibling branches);
    README Config file documents it.
    """
    (isolated / "init").write_text("value,longitude,latitude\n1.0,149.0,-35.0\n2.0,149.1,-35.1\n")
    main(["init"])
    assert (isolated / "ppgrid.toml").is_file()
    out = isolated / "out"
    main(["./init", "-o", str(out), "--res", "5000", "--block", "2048", "--workers", "2"])
    assert (out / "value.tif").is_file()
    assert (out / "support_km.tif").is_file()


# ---------------------------------------------------------------------------
# --log-file (#65)
# ---------------------------------------------------------------------------


def test_log_file_is_info_copy_of_stderr(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Every file line appears on stderr; the file has no color codes; INFO phases present."""
    logfile = isolated / "run.log"
    main(_run_argv(isolated, isolated / "out", "--log-file", str(logfile)))
    err = capsys.readouterr().err
    text = logfile.read_text(encoding="utf-8")
    assert text
    for line in text.splitlines():
        assert line in err.splitlines(), line
    assert "ingest:" in text
    assert "grid:" in text
    assert "done:" in text
    assert "\x1b[" not in text


def test_log_file_writes_info_under_quiet(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A -q run suppresses INFO on stderr but the file still gets the full info copy."""
    logfile = isolated / "run.log"
    main(_run_argv(isolated, isolated / "out", "--log-file", str(logfile), "-q"))
    err = capsys.readouterr().err
    text = logfile.read_text(encoding="utf-8")
    assert not any("ingest:" in line for line in err.splitlines())
    assert "ingest:" in text


def test_log_file_truncates_per_run(isolated: Path) -> None:
    """Each run rewrites the log file fresh: only the latest run's lines remain."""
    logfile = isolated / "run.log"
    main(_run_argv(isolated, isolated / "outA", "--log-file", str(logfile)))
    main(_run_argv(isolated, isolated / "outB", "--log-file", str(logfile)))
    text = logfile.read_text(encoding="utf-8")
    assert "outB/value.tif" in text
    assert "outA/value.tif" not in text


def test_log_file_open_failure_exit1(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A log file that cannot be opened is a named pipeline error, exit 1."""
    with pytest.raises(SystemExit) as e:
        main(_run_argv(isolated, isolated / "out", "--log-file", str(isolated / "no" / "dir" / "x.log")))
    assert e.value.code == 1
    assert "cannot open log file" in capsys.readouterr().err
