r"""CLI presentation layer (issue #101).

The ONLY module in the package that imports ``rich`` (enforced by
``tests/test_ui.py``). Every other module reports state through this small
API:

- ``UI.phase(name, ...)`` — one pipeline phase (read, calibrate, grid,
  interpolate, reproject, write). On a TTY this renders a live Rich
  progress table; otherwise it is a no-op and the existing plain one-line-per-phase
  logging stays the interface.
- ``UI.ok / warn / error / note`` — status lines. Semantic color only
  (green = ok, yellow = warn, red = error); the state word is always in the
  text, so color is never the sole carrier.
- ``UI.env_report`` — the ``--verbose`` one-line environment report.

Mode resolution (documented in the ``--color`` help):

- live (animated progress) = TTY stderr AND not ``--quiet`` AND no
  ``NO_COLOR`` AND not ``TERM=dumb``.
- color = ``--color always`` (beats everything, git-style) >
  ``--color auto`` (TTY AND no NO_COLOR AND not TERM=dumb) >
  ``--color never`` (also disables the live table).
- unicode symbols = stream encoding is UTF-8; anything else falls back to
  ASCII (``|/-/\\`` spinner, ``-``/``#`` bar).

Plain (non-live) output with color off is byte-identical to the pre-#101
logging output: no escape sequences.

``rich`` is imported at module top: it is an unconditional runtime
dependency (pyproject), so there is no lazy-import cost to save.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TextColumn
from rich.spinner import Spinner
from rich.text import Text

__all__ = ["UI", "Phase", "resolve_color", "resolve_live"]

# Phase-line prefixes the progress table already renders; the UI log
# handler suppresses these INFO records so they do not double-print.
_PHASE_LINE_PREFIXES: tuple[str, ...] = (
    "ingest: ",
    "calibrate: ",
    "grid: ",
    "interpolate: ",
    "write: ",
    "done: ",
)

_LOG_FORMAT = "%(levelname).1s %(name)s: %(message)s"

# Semantic SGR color codes for plain (non-live) output: green = ok,
# yellow = warn, red = error. No other colors are ever emitted.
_SGR_OK = "32"
_SGR_WARN = "33"
_SGR_ERROR = "31"
_SGR_RESET = "0"


def _is_tty(stream: Any) -> bool:
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except Exception:  # ruff: ignore[blind-except] - broken isatty means "not a tty"
        return False


def _no_color() -> bool:
    """no-color.org: a non-empty NO_COLOR value requests no color."""
    return bool(os.environ.get("NO_COLOR"))


def _term_dumb() -> bool:
    return os.environ.get("TERM", "").strip().lower() == "dumb"


def _is_utf8(stream: Any) -> bool:
    enc = getattr(stream, "encoding", None)
    if not isinstance(enc, str) or not enc:
        return True  # in-memory stream: unicode is safe
    return enc.lower().startswith("utf")


def resolve_live(*, quiet: bool, stream: Any) -> bool:
    """Return True when stderr should show the animated progress table.

    TTY + not ``--quiet`` + no NO_COLOR + not TERM=dumb (issue #101).
    """
    if quiet or _no_color() or _term_dumb():
        return False
    return _is_tty(stream)


def resolve_color(*, flag: str | None, stream: Any) -> bool:
    """Resolve the ``--color auto|always|never`` flag to a bool.

    ``always`` beats everything (git-style); ``auto`` colors on a TTY
    without NO_COLOR / TERM=dumb; ``never`` disables. ``None`` = auto.
    """
    if flag == "always":
        return True
    if flag == "never":
        return False
    if _no_color() or _term_dumb():
        return False
    return _is_tty(stream)


# ---------------------------------------------------------------------------
# Progress columns. Semantic color only (green = running/ok, red = failed);
# everything else renders in the terminal's default colors.
# ---------------------------------------------------------------------------


def _human_rate(rate: float) -> str:
    """Compact SI-style rate: 8300 -> '8.3K', 2_000_000 -> '2.0M'."""
    if rate >= 1e6:
        return f"{rate / 1e6:.1f}M"
    if rate >= 1e4:
        return f"{rate / 1e3:.0f}K"
    if rate >= 1e3:
        return f"{rate / 1e3:.1f}K"
    return f"{rate:.0f}"


class _SpinnerColumn(ProgressColumn):
    r"""Spinner that stays blank until the task has run for ``min_s``.

    A phase shorter than ~2 s finishes before the spinner would start, so
    it never flashes (issue #101). Green while running (semantic: in
    progress / ok); a finished phase renders a blank. Unicode braille
    dots when the stream is UTF-8, ``|/-/\\`` otherwise.
    """

    def __init__(self, *, unicode: bool, min_s: float) -> None:
        super().__init__()
        self._spinner = Spinner("dots" if unicode else "line", style="green")
        self._min_s = min_s

    def render(self, task: Task) -> Text:
        if task.finished or not task.started:
            return Text(" ")
        if task.get_time() - (task.start_time or 0.0) < self._min_s:
            return Text(" ")
        return self._spinner.render(task.get_time())


class _CountColumn(ProgressColumn):
    """Completed count + unit, e.g. ``12,345 rows`` / ``2,000,000 cells``."""

    @staticmethod
    def render(task: Task) -> Text:
        if task.completed <= 0:
            return Text("")
        unit = task.fields.get("unit", "")
        return Text(f"{int(task.completed):,} {unit}".rstrip())


class _BarColumn(BarColumn):
    """Green (ok) bar; red (error) once the phase ended with a failure."""

    @staticmethod
    def render(task: Task) -> Text:
        style = "red" if task.fields.get("failed") else "green"
        bar = BarColumn(
            bar_width=24,
            style="none",
            complete_style=style,
            finished_style=style,
            pulse_style=style,
        )
        return bar.render(task)


class _PercentColumn(ProgressColumn):
    """Determinate percentage.

    Indeterminate tasks render blank: an invented percent is never shown
    (issue #101).
    """

    @staticmethod
    def render(task: Task) -> Text:
        if not task.total:
            return Text("")
        return Text(f"{task.percentage:.0f}%")


class _ElapsedColumn(ProgressColumn):
    """H:MM:SS elapsed for the task. Unstyled: semantic color only."""

    @staticmethod
    def render(task: Task) -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("-:--:--")
        return Text(str(timedelta(seconds=max(0, int(elapsed)))))


class _RateColumn(ProgressColumn):
    """Throughput, e.g. ``8.3Kcells/s``.

    Shown once the task has both counts and a little history (the first
    samples are too noisy).
    """

    @staticmethod
    def render(task: Task) -> Text:
        if task.completed <= 0:
            return Text("")
        elapsed = task.elapsed
        if elapsed is None or elapsed < 0.25:
            return Text("")
        unit = task.fields.get("unit", "")
        return Text(f"{_human_rate(task.completed / elapsed)}{unit}/s")


# ---------------------------------------------------------------------------
# Log routing while the live table is up.
# ---------------------------------------------------------------------------


class _UIStreamHandler(logging.Handler):
    """Routes ``ppgrid`` log records to the live UI console.

    Installed on the ``ppgrid`` logger in place of the plain stderr
    handler for the duration of a live run (``UI._begin_run`` /
    ``UI._end_run``). Phase INFO lines the progress table already renders
    are dropped here so they do not double-print; every other record
    (DEBUG perf lines under --verbose, WARNING, ERROR) is emitted as a
    semantic-color line above the table.
    """

    def __init__(self, ui: UI) -> None:
        super().__init__()
        self._ui = ui
        self._formatter = logging.Formatter(_LOG_FORMAT)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            if record.levelno == logging.INFO and msg.startswith(_PHASE_LINE_PREFIXES):
                return  # the progress table already renders this phase
            if record.levelno >= logging.ERROR:
                kind = "error"
            elif record.levelno >= logging.WARNING:
                kind = "warn"
            else:
                kind = "note"
            self._ui._emit(kind, self._formatter.format(record))
        except Exception:  # ruff: ignore[blind-except] - logging must never raise
            self.handleError(record)


# ---------------------------------------------------------------------------
# Phase handle + UI.
# ---------------------------------------------------------------------------


class Phase:
    """Progress handle for one pipeline phase (issue #101).

    All methods are no-ops when the UI is plain. Set ``.summary`` before
    the phase block ends to append the result to the phase's final row,
    e.g. ``read: 13580 rows -> 13580 points (0 dropped)``.
    """

    __slots__ = ("_task_id", "_ui", "failed", "name", "summary")

    def __init__(self, name: str, ui: UI, task_id: int | None) -> None:
        """Bind the phase name, UI handle and (live) task id. No-op UI: task_id None."""
        self.name = name
        self.summary: str | None = None
        self.failed = False
        self._ui = ui
        self._task_id = task_id

    def set_total(self, total: float) -> None:
        """Set (or grow) the determinate total; enables % and rate.

        A falsy total (0) marks the task indeterminate: rich renders
        total=0 as a finished task (full green bar), so the UI layer
        normalizes 0 to total=None (F6, #115 review).
        """
        self._ui._phase_set_total(self._task_id, total)

    def advance(self, n: float) -> None:
        """Add n units of work. Thread-safe; does not force a redraw."""
        self._ui._phase_advance(self._task_id, n)

    def set_count(self, n: float) -> None:
        """Set the absolute completed count (end-of-phase finalization)."""
        self._ui._phase_set_count(self._task_id, n)


class UI:
    """CLI presentation state for one process (issue #101).

    Built by ``main`` via :meth:`from_cli` and passed to the pipeline as
    ``Pipeline(ui=...)``. In plain mode every method is either a no-op
    (progress) or a plain-text line (status); in live mode the Rich
    progress table runs on stderr for the duration of a run scope.

    ``advance`` is safe from worker threads (rich's internal lock);
    console output (``maybe_refresh``, status lines) must happen on the
    main thread.
    """

    def __init__(
        self,
        *,
        live: bool,
        color: bool,
        unicode: bool,
        stream: Any | None = None,
        spinner_min_s: float = 2.0,
        refresh_interval: float = 0.15,
    ) -> None:
        """Store mode flags; build the rich Console/Progress when live."""
        self._stream = stream if stream is not None else sys.stderr
        self._live = live
        self._color = color
        self._unicode = unicode
        self._spinner_min_s = spinner_min_s
        self._refresh_interval = refresh_interval
        self._last_refresh = 0.0
        self._console: Console | None = None
        self._progress: Progress | None = None
        self._log_handler: _UIStreamHandler | None = None
        self._saved_handlers: list[logging.Handler] = []
        if live:
            self._console = Console(
                file=self._stream,
                no_color=not color,
                force_terminal=True,
                highlight=False,
                soft_wrap=False,
                emoji=False,
            )
            self._progress = Progress(
                _SpinnerColumn(unicode=unicode, min_s=spinner_min_s),
                TextColumn("{task.description}"),
                _CountColumn(),
                _BarColumn(),
                _PercentColumn(),
                _ElapsedColumn(),
                _RateColumn(),
                console=self._console,
                auto_refresh=False,
                redirect_stdout=False,
                redirect_stderr=False,
            )

    # -- construction ------------------------------------------------------

    @classmethod
    def from_cli(cls, *, quiet: bool, color: str | None = None, stream: Any | None = None) -> UI:
        """Resolve live/color/unicode modes from the CLI state + environment.

        ``--color never`` disables the live table as well as color: it is an
        explicit request for plain output (same as NO_COLOR / --quiet).
        """
        stream = stream if stream is not None else sys.stderr
        live = resolve_live(quiet=quiet, stream=stream)
        if color == "never":
            live = False
        return cls(
            live=live,
            color=resolve_color(flag=color, stream=stream),
            unicode=_is_utf8(stream),
            stream=stream,
        )

    # -- mode flags ----------------------------------------------------------

    @property
    def live(self) -> bool:
        """True when the animated progress table is active."""
        return self._live

    @property
    def color(self) -> bool:
        """True when semantic SGR/rich colors are emitted."""
        return self._color

    @property
    def unicode(self) -> bool:
        """True when the stream encoding supports unicode symbols."""
        return self._unicode

    # -- run scope -----------------------------------------------------------

    @contextmanager
    def run_scope(self) -> Iterator[None]:
        """Brackets one CLI run: up the progress table (live) or nothing.

        Yields:
            Nothing; the body runs while the run scope is open.

        """
        self._begin_run()
        try:
            yield
        finally:
            self._end_run()

    def _begin_run(self) -> None:
        if not self._live or self._progress is None:
            return
        if self._progress.live.is_started:
            return  # idempotent
        root = logging.getLogger("ppgrid")
        self._saved_handlers = [h for h in root.handlers if getattr(h, "ppgrid_handler", False)]
        for h in self._saved_handlers:
            root.removeHandler(h)
        self._log_handler = _UIStreamHandler(self)
        root.addHandler(self._log_handler)
        self._progress.start()

    def _end_run(self) -> None:
        if not self._live or self._progress is None:
            return
        try:
            if self._progress.live.is_started:
                self._progress.stop()
        finally:
            root = logging.getLogger("ppgrid")
            if self._log_handler is not None:
                root.removeHandler(self._log_handler)
                self._log_handler = None
            for h in self._saved_handlers:
                root.addHandler(h)
            self._saved_handlers = []

    # -- phases ----------------------------------------------------------------

    @contextmanager
    def phase(self, name: str, *, total: float | None = None, unit: str = "") -> Iterator[Phase]:
        """One named pipeline phase. No-op scope in plain mode.

        Yields:
            The Phase handle (advance/set_total/set_count/summary/failed).

        """
        if self._live and self._progress is not None and self._progress.live.is_started:
            # A falsy total means indeterminate (total=None): rich renders
            # total=0 as a finished task (full green bar) (F6, #115 review).
            task_id = self._progress.add_task(name, total=total or None, **({"unit": unit} if unit else {}))
        else:
            task_id = None
        ph = Phase(name, self, task_id)
        self._refresh(force=True)
        try:
            yield ph
        except BaseException:
            ph.failed = True
            self._finish_phase(ph)
            raise
        self._finish_phase(ph)

    def _finish_phase(self, ph: Phase) -> None:
        if ph._task_id is None or self._progress is None:
            return
        progress = self._progress
        task = progress.tasks[ph._task_id]
        if ph.summary is not None:
            progress.update(ph._task_id, description=f"{ph.name}: {ph.summary}")
        if ph.failed:
            task.fields["failed"] = True
        if task.finished_time is None:
            task.finished_time = task.elapsed  # duration; freezes the elapsed clock
        progress.stop_task(ph._task_id)
        self._refresh(force=True)

    def _phase_set_total(self, task_id: int | None, total: float) -> None:
        if task_id is None or self._progress is None:
            return
        if not total:
            # rich renders total=0 as a finished task (full green bar), not
            # indeterminate; a falsy total means "no reliable total" (serial
            # CSV parse), so normalize to total=None (F6, #115 review).
            task = self._progress.tasks[task_id]
            if task.total is not None:
                task.total = None
                task._reset()  # rich's internal rate/speed state
                self._refresh(force=True)
            return
        self._progress.update(task_id, total=total)
        self._refresh(force=True)

    def _phase_advance(self, task_id: int | None, n: float) -> None:
        if task_id is None or self._progress is None or n <= 0:
            return
        self._progress.advance(task_id, n)

    def _phase_set_count(self, task_id: int | None, n: float) -> None:
        if task_id is None or self._progress is None:
            return
        self._progress.update(task_id, completed=n)
        self._refresh(force=True)

    def maybe_refresh(self) -> None:
        """Throttled redraw (main thread only).

        A render failure never fails the run - the progress display is
        cosmetic. No-op unless the run scope (progress table) is active.
        """
        if self._progress is None or not self._progress.live.is_started:
            return
        if time.monotonic() - self._last_refresh < self._refresh_interval:
            return
        self._refresh(force=True)

    def _refresh(self, *, force: bool = False) -> None:
        if self._progress is None:
            return
        if not force and time.monotonic() - self._last_refresh < self._refresh_interval:
            return
        self._last_refresh = time.monotonic()
        with contextlib.suppress(Exception):  # cosmetic; see maybe_refresh
            self._progress.refresh()

    # -- status lines -------------------------------------------------------------

    def note(self, text: str) -> None:
        """Neutral information line (e.g. the ``env:`` report)."""
        self._emit("note", text)

    def ok(self, text: str) -> None:
        """Success line (green in colored output; the text says it too)."""
        self._emit("ok", text)

    def warn(self, text: str) -> None:
        """Warning line (yellow in colored output)."""
        self._emit("warn", text)

    def error(self, text: str) -> None:
        """Error line (red in colored output)."""
        self._emit("error", text)

    def env_report(self, text: str) -> None:
        """Emit the ``--verbose`` one-line environment report (issue #101)."""
        self.note(text)

    def _emit(self, kind: str, text: str) -> None:
        if self._live and self._console is not None:
            style = {"ok": "green", "warn": "yellow", "error": "red"}.get(kind)
            t = Text(text)
            if style and self._color:
                t.stylize(style)
            self._console.print(t)
            self._refresh()
            return
        code = {"ok": _SGR_OK, "warn": _SGR_WARN, "error": _SGR_ERROR}.get(kind) if self._color else None
        if code:
            print(f"\x1b[{code}m{text}\x1b[{_SGR_RESET}m", file=self._stream)
        else:
            print(text, file=self._stream)
