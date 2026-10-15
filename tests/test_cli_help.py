"""CLI presentation tests (issue #103: grouped help, unit metavars, jargon-free strings).

Covers the six acceptance items:
1. --help option groups (input/columns, CRS/grid, interpolation, memory/turbo,
   output, logging) + a full-text snapshot test.
2. Metavars carry units (METRES, KM, GB, CELLS).
3. --help / --plan / warnings contain no design-section IDs or regime letters
   (grep over captured output).
4. Every epilog example runs verbatim with in-repo inputs (smoke test executes
   all three).
5. `ppgrid help` vs `ppgrid --help` are identical (pin test).
6. Defaults are displayed consistently in the option help text.
"""

import io
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from ppgrid import turbop
from ppgrid.pipeline import _CONFIG_DEFAULTS, _EPILOG, UNSET, _build_parser, main
from tests.test_turbop import _plan_full_au

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path(__file__).parent / "fixtures" / "help_snapshot.txt"

# (pattern, what it is, case-insensitive?) — design-doc jargon that must not
# reach user-facing output (issue #103 item 3).
JARGON_PATTERNS: tuple[tuple[str, str, bool], ...] = (
    (r"\bregime\b", "design-doc word 'regime'", True),
    (r"\bS\d+(?:\.\d+)*[A-Z]?\b", "design-section id (S3.3, S13, S28A)", False),
    (r"\b[A-Z]\d+\b", "design-section id (A9, A7)", False),
    (r"\b[A-Z]-\d+\b", "finding id (M-2, Y-1)", False),
    (r"\b[ABC]\b", "bare regime letter", False),
    (r"\b0\.85\*C\b", "sizing formula '0.85*C'", False),
    (r"\bBt\b", "sizing-budget name 'Bt'", False),
)


def _assert_no_jargon(*texts: str) -> None:
    """Fail naming the offending pattern + text when any jargon pattern matches."""
    for text in texts:
        for pattern, what, ci in JARGON_PATTERNS:
            m = re.search(pattern, text, flags=re.IGNORECASE if ci else 0)
            assert m is None, f"jargon leak ({what}): {pattern!r} matched {m.group()!r} in: {text!r}"


def _help_text() -> str:
    """Capture parser.print_help() output."""
    buf = io.StringIO()
    _build_parser().print_help(file=buf)
    return buf.getvalue()


def _metavar(parser: object, flag: str) -> str | None:
    """Metavar of the parser action owning `flag` (None when absent)."""
    for action in parser._actions:
        if flag in action.option_strings:
            return action.metavar
    return None


def _tiny_csv(tmp_path: Path) -> str:
    """2-point CSV for the smallest possible plan resolution."""
    csv = tmp_path / "pts.csv"
    csv.write_text("value,longitude,latitude\n1.0,149.0,-35.0\n2.0,149.1,-35.1\n")
    return str(csv)


def _plan_argv(tmp_path: Path, out: Path, *extra: str) -> list[str]:
    """Full-pipeline argv for the tiny CSV (res=100 keeps the grid small)."""
    return [
        _tiny_csv(tmp_path),
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


# ---------------------------------------------------------------------------
# Item 1: option groups + help snapshot
# ---------------------------------------------------------------------------


REQUIRED_GROUPS = ("input/columns", "CRS/grid", "interpolation", "memory/turbo", "output", "logging")


def test_help_has_required_groups() -> None:
    """--help is organised in the six required groups (issue #103 item 1)."""
    parser = _build_parser()
    titles = [g.title for g in parser._action_groups]
    for title in REQUIRED_GROUPS:
        assert title in titles, f"missing argument group: {title!r} (have {titles})"


def test_help_snapshot() -> None:
    """The full --help text is pinned verbatim (snapshot; item 1).

    Regenerate with:
        uv run python -c "from ppgrid.pipeline import _build_parser; _build_parser().print_help()"
    (3.13+ rendering; see the version normalisation below).
    """
    captured = _help_text()
    expected = SNAPSHOT.read_text(encoding="utf-8")
    if sys.version_info < (3, 13):
        # Pre-3.13 argparse renders the dual-alias output option as
        # "-o OUT, --out OUT" instead of "-o, --out OUT" (option field gets 4
        # wider, help field 4 narrower — same help column). The rest of the
        # help text is byte-identical across 3.11-3.14 (verified).
        expected = expected.replace("  -o, --out OUT         ", "  -o OUT, --out OUT     ")
    assert captured == expected


# ---------------------------------------------------------------------------
# Item 2: unit-carrying metavars
# ---------------------------------------------------------------------------


def test_metavars_carry_units() -> None:
    """Numeric metavars name their unit (issue #103 item 2)."""
    parser = _build_parser()
    expected = {
        "--res": "METRES",
        "--cap-km": "KM",
        "--max-ram": "GB",
        "--ram-gb": "GB",
        "--block": "CELLS",
        "--turbo": "GB",
    }
    for flag, unit in expected.items():
        assert _metavar(parser, flag) == unit, f"{flag} metavar must be {unit!r}"


# ---------------------------------------------------------------------------
# Item 3: no design-section IDs or regime letters in user-facing output
# ---------------------------------------------------------------------------


def test_help_is_jargon_free() -> None:
    """--help contains no design-section IDs or regime letters (item 3)."""
    _assert_no_jargon(_help_text())


def test_plan_output_is_jargon_free(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--plan (turbo on) prints the resolved mode in plain language (item 3)."""
    out = tmp_path / "plan_out"
    with pytest.raises(SystemExit) as exc:
        main(_plan_argv(tmp_path, out, "--plan", "--turbo", "16"))
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "turbo:         on" in text  # the turbo line is present
    _assert_no_jargon(text)


def test_precheck_strings_are_jargon_free() -> None:
    """Precheck summary/warn/error strings are plain language (item 3).

    Covers every decision shape: full in-RAM (clean), banded, file-backed,
    per-box warn, strict error, and the memory-ceiling note.
    """
    decisions = [
        turbop.precheck(_plan_full_au(64.0), cap_source="preset 64 GB"),
        turbop.precheck(_plan_full_au(32.0), cap_source="preset 32 GB"),
        turbop.precheck(_plan_full_au(20.0), cap_source="preset 20 GB"),
        turbop.precheck(_plan_full_au(16.0), cap_source="preset 16 GB"),
        turbop.precheck(_plan_full_au(16.0), strict=True, cap_source="preset 16 GB"),
        turbop.precheck(_plan_full_au(8.0), cap_source="preset 8 GB (floored to 8.0 GB)", available_gb=6.0),
    ]
    for d in decisions:
        _assert_no_jargon(d.summary)
        if d.message is not None:
            _assert_no_jargon(d.message)


def test_strict_precheck_stderr_is_jargon_free(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """End-to-end: the --turbo-strict infeasible error on stderr is plain (item 3)."""
    cmd = [
        str(REPO_ROOT / "data" / "all_equakes.csv"),
        "-o",
        str(tmp_path / "o"),
        "--res",
        "1000",
        "--max-ram",
        "8",
        "--turbo-strict",
        "--value-col",
        "mag",
        "--plan",
    ]
    with pytest.raises(SystemExit) as exc:
        main(cmd)
    assert exc.value.code == 3
    _assert_no_jargon(capsys.readouterr().err)


# ---------------------------------------------------------------------------
# Item 4: epilog examples run verbatim with in-repo inputs
# ---------------------------------------------------------------------------


def _epilog_examples() -> list[list[str]]:
    """Return argv lists of the epilog's example commands (one per 'ppgrid ...' line)."""
    examples: list[list[str]] = []
    for line in _EPILOG.splitlines():
        stripped = line.strip()
        if stripped.startswith("ppgrid "):
            examples.append(shlex.split(stripped))
    return examples


def test_epilog_examples_run_verbatim(tmp_path: Path) -> None:
    """Every epilog example executes (exit 0) with in-repo inputs (item 4).

    Slow: example 2 is a global 500 m run (minutes on a small box). Runs in a
    scratch dir with a data/ symlink into the repo, so the argv stays verbatim
    while the checkout root needs no write access (ownerless pi-bg wrapper
    runs execute as another user, review info-1). The compare example reuses
    the dir example 1 wrote, so it runs after the run examples.
    """
    examples = _epilog_examples()
    assert len(examples) == 4
    for argv in examples:
        assert argv[0] == "ppgrid"
        if argv[1] in ("compare", "completions"):
            continue  # subcommand example: argv[1] is not an input file
        assert (REPO_ROOT / argv[1]).is_file(), f"epilog example input missing from repo: {argv[1]}"
    (tmp_path / "data").symlink_to(REPO_ROOT / "data")
    for argv in examples:
        cmd = [sys.executable, "-m", "ppgrid", *argv[1:]]
        proc = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=1800)
        assert proc.returncode == 0, (
            f"epilog example failed (exit {proc.returncode}): {' '.join(cmd)}\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
        if "-o" in argv:
            d = tmp_path / argv[argv.index("-o") + 1]
            assert (d / "value.tif").is_file(), f"missing value.tif in {d}"
            assert (d / "support_km.tif").is_file(), f"missing support_km.tif in {d}"


# ---------------------------------------------------------------------------
# Item 5: `ppgrid help` == `ppgrid --help` (pin)
# ---------------------------------------------------------------------------


def test_help_command_matches_help_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """`ppgrid help` and `ppgrid --help` print identical text (pin; item 5)."""
    with pytest.raises(SystemExit) as exc:
        main(["help"])
    assert exc.value.code == 0
    via_command = capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    via_flag = capsys.readouterr().out
    assert via_command == via_flag
    assert "examples:" in via_command


# ---------------------------------------------------------------------------
# Item 6: defaults displayed consistently
# ---------------------------------------------------------------------------


def _render_default(value: object) -> str:
    """Render a default the way the help text displays it (floats via :g, else str())."""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def test_defaults_displayed_consistently() -> None:
    """Every option with a real default shows '(default: X)' in its help (item 6).

    Since #63 the config-able options carry the UNSET sentinel as their
    parser default (so "not passed" stays distinguishable from "passed");
    their built-in default comes from _CONFIG_DEFAULTS, the single source
    for the sentinel fallback and the init template.
    """
    parser = _build_parser()
    checked = 0
    for action in parser._actions:
        if not action.option_strings:
            continue  # positional
        if action.dest in ("help", "version"):
            continue
        default = _CONFIG_DEFAULTS[action.dest] if action.default is UNSET else action.default
        if default is False:
            continue  # store_true flag: absence of the flag is the default
        if default is None:
            continue  # optional: no default value to display
        assert action.help, f"{action.option_strings} has a default but no help text"
        rendered = _render_default(default)
        assert f"default: {rendered}" in action.help, (
            f"{action.option_strings[0]} default {rendered!r} not displayed consistently in help: {action.help!r}"
        )
        checked += 1
    assert checked >= 15, f"expected >= 15 default-bearing options, checked {checked}"
