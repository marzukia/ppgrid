"""CLI presentation layer tests (issue #101).

Covers: rich imported only in ppgrid/ui.py, plain-mode output with zero
escape sequences, live TTY mode (exact phase names, rates, no invented
percent, delayed spinner), --color auto|always|never precedence,
NO_COLOR / TERM=dumb / --quiet suppression, unicode/ASCII fallback, the
--verbose env: report line, and clean cursor restoration on error.
"""

import logging
import re
import sys
import time
from pathlib import Path

import pytest

from ppgrid import ui as _ui
from ppgrid.pipeline import main

_SGR_RE = re.compile(r"\x1b\[([0-9;]*)m")
# The only SGR codes the UI may emit: green=ok, yellow=warn, red=error, reset.
_ALLOWED_SGR = {"32", "33", "31", "0"}

_PHASES = ("read", "calibrate", "grid", "interpolate", "reproject", "write")


class FakeTTY:
    """Minimal writable stream that pretends to be a UTF-8 (or ASCII) TTY."""

    def __init__(self, encoding: str = "utf-8") -> None:
        """Build the fake with the given stream encoding."""
        self._buf: list[str] = []
        self.encoding = encoding

    @staticmethod
    def isatty() -> bool:
        """Report as a TTY."""
        return True

    def write(self, text: str) -> int:
        """Accumulate the text; return its length (file-like)."""
        self._buf.append(text)
        return len(text)

    def flush(self) -> None:
        """No-op (nothing is buffered beyond _buf)."""

    def getvalue(self) -> str:
        """All text written so far, in order."""
        return "".join(self._buf)


def _tiny_csv(tmp_path: Path) -> str:
    """2-point CSV for the smallest possible end-to-end run."""
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


# ---------------------------------------------------------------------------
# rich is the only import site
# ---------------------------------------------------------------------------


def test_rich_imported_only_in_ui() -> None:
    """No module other than ppgrid/ui.py may import rich (issue #101)."""
    pkg = Path(__file__).parent.parent / "ppgrid"
    offenders = set()
    for f in sorted(pkg.glob("*.py")):
        for line in f.read_text().splitlines():
            ls = line.strip()
            if ls.startswith(("import rich", "from rich")):
                offenders.add(f.name)
    assert offenders == {"ui.py"}


# ---------------------------------------------------------------------------
# Mode resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "env", "expect"),
    [
        (None, {}, True),  # auto: TTY
        ("always", {}, True),
        ("never", {}, False),
        (None, {"NO_COLOR": "1"}, False),  # auto: NO_COLOR
        ("always", {"NO_COLOR": "1"}, True),  # always beats NO_COLOR (git-style)
        (None, {"TERM": "dumb"}, False),  # auto: dumb terminal
        ("always", {"TERM": "dumb"}, True),  # always beats dumb
    ],
)
def test_resolve_color_precedence(
    monkeypatch: pytest.MonkeyPatch, *, flag: str | None, env: dict[str, str], expect: bool
) -> None:
    """--color always beats environment; never disables; auto respects TTY/NO_COLOR/TERM."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    for key, val in env.items():
        monkeypatch.setenv(key, val)
    assert _ui.resolve_color(flag=flag, stream=FakeTTY()) is expect


@pytest.mark.parametrize(
    ("quiet", "env", "expect"),
    [
        (False, {}, True),  # TTY, clean env
        (True, {}, False),  # --quiet
        (False, {"NO_COLOR": "1"}, False),
        (False, {"TERM": "dumb"}, False),
    ],
)
def test_resolve_live(
    monkeypatch: pytest.MonkeyPatch,
    *,
    quiet: bool,
    env: dict[str, str],
    expect: bool,
) -> None:
    """Live = TTY AND not --quiet AND no NO_COLOR AND not TERM=dumb."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    for key, val in env.items():
        monkeypatch.setenv(key, val)
    assert _ui.resolve_live(quiet=quiet, stream=FakeTTY()) is expect


# ---------------------------------------------------------------------------
# Columns: percent, rate, spinner
# ---------------------------------------------------------------------------


def test_percent_column_indeterminate_blank() -> None:
    """An indeterminate task renders an empty percent cell (no invented %)."""
    from rich.progress import Task

    none = _ui._PercentColumn().render(Task(0, "p", total=None, completed=0, _get_time=lambda: 0.0))  # ruff: ignore[private-member-access]
    assert not none.plain
    det = _ui._PercentColumn().render(Task(0, "p", total=200, completed=50, _get_time=lambda: 0.0))  # ruff: ignore[private-member-access]
    assert det.plain == "25%"


def test_rate_column_units_and_warmup() -> None:
    """The rate column shows <unit>/s only after a 0.25 s warmup."""
    from rich.progress import Task

    col = _ui._RateColumn()  # ruff: ignore[private-member-access]
    early = Task(0, "p", total=1000, completed=100, _get_time=lambda: 1.1)
    early.start_time = 1.0  # elapsed 0.1 s < warmup
    assert not col.render(early).plain
    late = Task(0, "p", total=100000, completed=8300, _get_time=lambda: 2.0)
    late.start_time = 1.0  # elapsed 1.0 s
    assert col.render(late).plain == "8.3K/s"  # no unit field: bare /s
    late.fields["unit"] = "rows"
    assert col.render(late).plain == "8.3Krows/s"
    late.fields["unit"] = "cells"
    assert col.render(late).plain == "8.3Kcells/s"


def test_spinner_blank_before_min_s() -> None:
    """The spinner stays blank until the task has run for min_s (no flash)."""
    from rich.progress import Task

    col = _ui._SpinnerColumn(unicode=True, min_s=2.0)  # ruff: ignore[private-member-access]
    t = [100.5]
    task = Task(0, "p", total=10, completed=1, _get_time=lambda: t[0])
    task.start_time = 100.0
    assert col.render(task).plain == " "
    t[0] = 103.5
    assert col.render(task).plain != " "


# ---------------------------------------------------------------------------
# Plain mode (non-TTY): zero escape sequences
# ---------------------------------------------------------------------------


def test_plain_run_zero_escapes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A default run (non-TTY under pytest) emits no escape sequences at all."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    main(_run_argv(tmp_path, tmp_path / "out"))
    err = capsys.readouterr().err
    assert "\x1b[" not in err


def test_plain_run_phase_lines_unchanged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Plain mode keeps the one-line-per-phase logging (pre-#101 surface)."""
    with caplog.at_level(logging.INFO, logger="ppgrid.pipeline"):
        main(_run_argv(tmp_path, tmp_path / "out"))
    msgs = [r.getMessage() for r in caplog.records if r.name == "ppgrid.pipeline"]
    for prefix in ("ingest: ", "interpolate: done in", "done: "):
        assert any(m.startswith(prefix) for m in msgs), prefix


# ---------------------------------------------------------------------------
# Color modes
# ---------------------------------------------------------------------------


def test_color_always_non_tty_semantic_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """--color always colors even on a non-TTY; only semantic SGR codes appear.

    A successful plain run has no status line left to color (the final ok
    line is live-mode only, F3), so drive an error: the 'error:' line must
    carry SGR, and only allow-listed semantic codes may appear.
    """
    monkeypatch.delenv("NO_COLOR", raising=False)
    with pytest.raises(SystemExit):
        main(["/nonexistent-input.csv", "-o", str(tmp_path / "o"), "--color", "always"])
    err = capsys.readouterr().err
    assert "\x1b[" in err
    for group in _SGR_RE.findall(err):
        for code in group.split(";"):
            assert code in _ALLOWED_SGR, f"non-semantic SGR code {code!r}"


def test_color_never_on_tty_no_escapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """--color never on a TTY: plain lines, no escapes, no live table."""
    fake = FakeTTY()
    monkeypatch.setattr(sys, "stderr", fake)
    with caplog.at_level(logging.INFO, logger="ppgrid.pipeline"):
        main(_run_argv(tmp_path, tmp_path / "out", "--color", "never"))
    text = fake.getvalue()
    assert "\x1b[" not in text
    msgs = [r.getMessage() for r in caplog.records if r.name == "ppgrid.pipeline"]
    assert any(m.startswith("ingest: ") for m in msgs)


def test_no_color_env_suppresses_live_on_tty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """NO_COLOR on a TTY with auto color: no live table, no escapes."""
    fake = FakeTTY()
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(sys, "stderr", fake)
    main(_run_argv(tmp_path, tmp_path / "out"))
    text = fake.getvalue()
    assert "\x1b[" not in text


def test_error_line_red_only_when_colored(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A CLI error is red SGR with --color always, plain without it.

    D-7: the plain line is the unified ``E ppgrid.pipeline: <msg>`` log
    line (the state word is the level letter, not the text).
    """
    with pytest.raises(SystemExit) as exc:
        main(["/nonexistent-input.csv", "-o", str(tmp_path / "o1")])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "E ppgrid.pipeline: input file not found: /nonexistent-input.csv" in err
    assert "\x1b[" not in err
    with pytest.raises(SystemExit) as exc:
        main(["/nonexistent-input.csv", "-o", str(tmp_path / "o2"), "--color", "always"])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "\x1b[31m" in err


# ---------------------------------------------------------------------------
# Live TTY mode
# ---------------------------------------------------------------------------


def test_live_run_phase_names_and_rates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Live mode renders the six exact phase names, counts, and units."""
    fake = FakeTTY()
    monkeypatch.setattr(sys, "stderr", fake)
    main(_run_argv(tmp_path, tmp_path / "out"))
    text = fake.getvalue()
    for name in _PHASES:
        assert name in text, f"phase {name!r} missing from live output"
    # read phase counts rows, interpolate/write count cells (unit columns).
    assert "rows" in text
    assert "cells" in text
    # cursor hide/show are balanced after the run scope closes.
    assert text.count("\x1b[?25l") == text.count("\x1b[?25h")
    # the logging handler swap is restored: one handler left on the logger.
    assert len(logging.getLogger("ppgrid").handlers) == 1


def test_live_run_rate_shows_throughput(monkeypatch: pytest.MonkeyPatch) -> None:
    """A phase with real elapsed time shows its throughput (rows/s, cells/s)."""
    fake = FakeTTY()
    monkeypatch.setattr(sys, "stderr", fake)
    ui = _ui.UI(live=True, color=True, unicode=True, stream=fake, spinner_min_s=0.0, refresh_interval=0.01)
    with ui.run_scope():
        _read_phase(ui)
        _interpolate_phase(ui)
    text = fake.getvalue()
    assert "rows/s" in text
    assert "cells/s" in text
    assert "100%" in text
    assert "2000 rows -> 2000 points" in text  # summary lands on the final row


def _read_phase(ui: _ui.UI) -> None:
    """Drive the read phase: 2000 rows over ~0.3 s, with a summary."""
    with ui.phase("read", unit="rows") as ph:
        ph.set_total(2000)
        ph.advance(1000)
        time.sleep(0.3)
        ui.maybe_refresh()
        ph.advance(1000)
        ph.summary = "2000 rows -> 2000 points"


def _interpolate_phase(ui: _ui.UI) -> None:
    """Drive the interpolate phase: 4000 cells, with a summary."""
    with ui.phase("interpolate", unit="cells") as ph2:
        ph2.set_total(4000)
        ph2.advance(4000)
        time.sleep(0.3)
        ui.maybe_refresh()
        ph2.summary = "4000 cells"


def test_live_indeterminate_phase_no_percent() -> None:
    """A phase whose total is never set renders no percentage for it."""
    fake = FakeTTY()
    ui = _ui.UI(live=True, color=True, unicode=True, stream=fake, spinner_min_s=0.0, refresh_interval=0.01)
    with ui.run_scope(), ui.phase("grid", total=0) as ph:
        ph.advance(123)
        ui.maybe_refresh()
        ph.summary = "indeterminate"
    # The indeterminate row shows the count but never a percent.
    row = [ln for ln in fake.getvalue().splitlines() if "indeterminate" in ln]
    assert row
    assert not any("%" in ln for ln in row)


def _failing_phase(ui: _ui.UI) -> None:
    """Run an interpolate phase that raises RuntimeError("boom") after half the work."""
    boom = RuntimeError("boom")
    with ui.phase("interpolate", unit="cells") as ph:
        ph.set_total(10)
        ph.advance(5)
        raise boom


def test_live_phase_failure_marks_row_red_and_stops() -> None:
    """A raising phase unwinds cleanly and restores the cursor (issue #101)."""
    fake = FakeTTY()
    ui = _ui.UI(live=True, color=True, unicode=True, stream=fake, spinner_min_s=0.0, refresh_interval=0.01)
    with ui.run_scope(), pytest.raises(RuntimeError, match="boom"):
        _failing_phase(ui)
    text = fake.getvalue()
    assert text.count("\x1b[?25l") == text.count("\x1b[?25h")
    assert "interpolate" in text


# ---------------------------------------------------------------------------
# Unicode / ASCII fallback
# ---------------------------------------------------------------------------


def test_ascii_tty_falls_back_to_ascii_symbols() -> None:
    """A non-UTF-8 (ASCII) TTY gets ASCII spinner/bar symbols only."""
    fake = FakeTTY(encoding="ascii")
    ui = _ui.UI(live=True, color=False, unicode=_ui._is_utf8(fake), stream=fake, spinner_min_s=0.0)  # ruff: ignore[private-member-access]
    with ui.run_scope(), ui.phase("read", total=100) as ph:
        ph.set_total(100)
        ph.advance(100)
        time.sleep(0.05)
        ui.maybe_refresh()
    text = fake.getvalue()
    assert text.isascii()


def test_utf8_tty_uses_unicode_symbols() -> None:
    """A UTF-8 TTY renders unicode bar/spinner glyphs."""
    fake = FakeTTY(encoding="utf-8")
    ui = _ui.UI(live=True, color=False, unicode=True, stream=fake, spinner_min_s=0.0)
    with ui.run_scope(), ui.phase("write", total=100) as ph:
        ph.set_total(100)
        ph.advance(100)
        time.sleep(0.05)
        ui.maybe_refresh()
    text = fake.getvalue()
    assert not text.isascii()


# ---------------------------------------------------------------------------
# --verbose env: report
# ---------------------------------------------------------------------------


def test_verbose_env_report_single_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """--verbose emits exactly one env: line with REAL native-stack versions.

    The version values must be real (digits, not 'unknown'/'not loaded'
    placeholders): the report probes with local imports, so it must work in a
    fresh process where the module-lazy native globals are not yet published
    (F1, #115 review). D-7: the line is the unified
    ``I ppgrid.pipeline: env: ...`` log line.
    """
    monkeypatch.delenv("NO_COLOR", raising=False)
    main(_run_argv(tmp_path, tmp_path / "out", "--verbose"))
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.startswith("I ppgrid.pipeline: env:")]
    assert len(lines) == 1
    line = lines[0]
    for field in (
        "ppgrid=",
        "python=",
        "os=",
        "arch=",
        "ram=",
        "rasterio=",
        "gdal=",
        "pyproj=",
        "numpy=",
        "cinterp=",
    ):
        assert field in line, f"missing {field}"
    # Real versions, not placeholders: this fails standalone if the probes
    # ever regress to module-globals lookups before the native load.
    for pat in (r"rasterio=\d", r"gdal=\d", r"pyproj=\d", r"numpy=\d"):
        assert re.search(pat, line), f"no real version matching {pat!r}"


def test_plain_run_stderr_no_ok_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plain (non-TTY) stderr stays byte-identical to the pre-#101 surface.

    The final ok line (e.g. '0.24s -> value.tif, support_km.tif') is
    live-table-mode only; plain runs keep exactly the one-line-per-phase log
    output (F3, #115 review).
    """
    monkeypatch.delenv("NO_COLOR", raising=False)
    main(_run_argv(tmp_path, tmp_path / "out"))
    err = capsys.readouterr().err
    assert not re.search(r"^\d+\.\d+s -> ", err, flags=re.MULTILINE), "ok line leaked into plain stderr"
    for prefix in ("ingest: ", "calibrate: ", "grid: ", "interpolate: ", "write: ", "done: "):
        # Log format is 'I ppgrid.pipeline: <message>' - match the message part.
        assert any(prefix in ln for ln in err.splitlines()), f"missing {prefix!r} line"


def test_live_run_emits_final_ok_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Live table mode (TTY) emits the final ok line with the output names.

    D-8: the line carries the promised ``ok: `` prefix (the docstring
    contract) — not just the bare success text.
    """
    fake = FakeTTY()
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys, "stderr", fake)
    main(_run_argv(tmp_path, tmp_path / "out"))
    text = fake.getvalue()
    assert re.search(r"ok: \d+\.\d+s -> value\.tif, support_km\.tif", text), (
        "ok line with 'ok: ' prefix (D-8) missing from live stderr"
    )


def test_quiet_tty_zero_stderr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--quiet on a TTY: zero stderr, zero escape sequences (F4, #115 review).

    quiet kills the live table, and the final ok line is live-mode only, so a
    quiet successful run produces no stderr at all (pre-#101 behavior).
    """
    fake = FakeTTY()
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys, "stderr", fake)
    main(_run_argv(tmp_path, tmp_path / "out", "--quiet"))
    assert not fake.getvalue()


def test_set_total_zero_is_indeterminate() -> None:
    """A falsy total stays indeterminate (total=None) in rich's task state.

    rich renders total=0 as a finished task (full green bar), not
    indeterminate; the UI normalizes 0 to None at both entry points (F6,
    #115 review).
    """
    fake = FakeTTY()
    ui = _ui.UI(live=True, color=False, unicode=True, stream=fake, spinner_min_s=0.0, refresh_interval=0.01)
    with ui.run_scope(), ui.phase("read", unit="rows") as ph:
        ph.set_total(0)
        assert ui._progress.tasks[0].total is None  # ruff: ignore[private-member-access]
        ph.advance(10)
    with ui.run_scope(), ui.phase("grid", total=0):
        assert ui._progress.tasks[1].total is None  # ruff: ignore[private-member-access]


def test_env_report_not_emitted_without_verbose(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Without --verbose there is no env: line (report is opt-in)."""
    main(_run_argv(tmp_path, tmp_path / "out"))
    err = capsys.readouterr().err
    assert not any(ln.startswith("env:") for ln in err.splitlines())


# ---------------------------------------------------------------------------
# Plain-mode Phase handle is a no-op
# ---------------------------------------------------------------------------


def test_plain_phase_handle_noop(tmp_path: Path) -> None:
    """With ui=None the pipeline phase handle no-ops (library mode unchanged)."""
    from ppgrid.pipeline import Pipeline

    p = Pipeline(
        _tiny_csv(tmp_path),
        "value",
        "lng",
        "lat",
        str(tmp_path / "out"),
        res=100,
        block_size=2048,
        workers=2,
    )
    with p._phase("read") as ph:  # ruff: ignore[private-member-access]
        ph.set_total(10)
        ph.advance(10)
        ph.set_count(10)
        ph.summary = "10 rows"
    assert ph.summary == "10 rows"
    assert p._active_phase is None  # ruff: ignore[private-member-access]
