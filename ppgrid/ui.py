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
- ``UI.summary`` — the final run summary block (issue #102): aligned
  ``label: value`` rows + bare output paths, at most 78 columns wide so it
  never wraps or truncates at 80/120.

Mode resolution (documented in the ``--color`` help):

- live (animated progress) = TTY stderr AND not ``--quiet`` AND no
  ``NO_COLOR`` AND not ``TERM=dumb``.
- color = ``--color always`` (beats everything, git-style) >
  ``--color auto`` (TTY AND no NO_COLOR AND not TERM=dumb) >
  ``--color never`` (also disables the live table).
- unicode symbols = stream encoding is UTF-8; anything else falls back to
  ASCII (``|/-/\\`` spinner, ``-``/``#`` bar).

Plain (non-live) output is the unified logging surface: every event
(phase lines, notes, warnings, errors) is one ``L ppgrid.pipeline: msg``
line (D-7), with semantic SGR on WARNING/ERROR only when color is on.
Without color there are no escape sequences.

``rich`` is imported at module top: it is an unconditional runtime
dependency (pyproject), so there is no lazy-import cost to save.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import textwrap
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import timedelta
from typing import Any

from rich import box
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TextColumn
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Column
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

# Display markers (D-7): the state word is a presentation concern owned by
# this layer. Call sites carry the message only; live output re-adds the
# marker here, plain output carries it as the logging level letter
# (I/W/E) instead. Library mode (no UI) re-adds the legacy raw marker in
# pipeline._ui_note to stay byte-identical.
_LIVE_MARKERS = {"ok": "ok: ", "warn": "[warn] ", "error": "error: ", "note": ""}
_LIVE_STYLES = {"ok": "green", "warn": "yellow", "error": "red"}
_PLAIN_SGR = {"ok": _SGR_OK, "warn": _SGR_WARN, "error": _SGR_ERROR}


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


def _compact_count(n: int) -> str:
    """Compact SI-style count: 1_692_256 -> '1.7M', 365_000 -> '365K'.

    The narrow-terminal fallback for the count column (issue #102); full
    precision is used whenever the table fits the terminal (_plan_table).
    """
    if n >= 1e6:
        s, suf = f"{n / 1e6:.1f}", "M"
    elif n >= 1e3:
        s, suf = f"{n / 1e3:.1f}", "K"
    else:
        return str(n)
    return s.removesuffix(".0") + suf


# Column count of the progress table: spinner, description, count, bar,
# percent, elapsed, rate. Rich's Table.grid (padding=(0, 1)) renders a row
# as " ".join(cells), and every column here is no_wrap, so: row width =
# sum(column widths) + (n_cols - 1) (+ 1 trailing space when that row's
# last cell is empty). Column widths are the max content across rows.
_N_PROGRESS_COLS = 7

# Standalone column units (no console): assume a wide terminal and a
# single-row table, so the pre-#102 full-precision layout is what they render.
_WIDE_DEFAULT = 10_000

# Minimum usable bar width (issue #102): the rate column, then full count
# precision, are sacrificed to keep the bar at least this wide.
_BAR_MIN = 8


def _elapsed_text(task: Task) -> str:
    """H:MM:SS elapsed for the task (shared by _plan_table and the column)."""
    elapsed = task.finished_time if task.finished else task.elapsed
    if elapsed is None:
        return "-:--:--"
    return str(timedelta(seconds=max(0, int(elapsed))))


def _rate_text(task: Task) -> str | None:
    """Throughput text, or None when the warmup/zero guards hide it."""
    if task.completed <= 0:
        return None
    elapsed = task.elapsed
    if elapsed is None or elapsed < 0.25:
        return None
    unit = task.fields.get("unit", "")
    return f"{_human_rate(task.completed / elapsed)}{unit}/s"


def _count_text(task: Task, *, compact: bool) -> str:
    """Count cell text for a task: full precision or compact SI-style."""
    if task.completed <= 0:
        return ""
    n = int(task.completed)
    unit = task.fields.get("unit", "")
    count = f"{n:,}" if not compact else _compact_count(n)
    return f"{count} {unit}".rstrip()


def _plan_table(tasks: list[Task], width: int) -> tuple[list[str], list[str], int]:
    """Pick the table layout that fits the terminal width (issue #102).

    Rich's Table.grid sizes every column to the max content across the
    visible rows, so the plan sizes the whole table, not one row. Column
    widths are fixed by the plan and every column renders no_wrap, so
    nothing truncates or wraps. The bar is elastic (0..24) within each
    tier; tiers, highest first:

    1. full-precision counts + rate, with a bar of at least 8;
    2. full-precision counts, rate dropped, bar of at least 8;
    3. compact counts (1.7M / 365K), rate dropped (the last tier accepts
       any bar width down to 0).

    A bar below 8 is noise, so the tier drops the rate (and, if needed,
    count precision) to keep it usable. Wide terminals (>150 cols) keep
    the pre-#102 full columns (full count, rate, 24-char bar) for every
    realistic phase line.

    Returns:
        (counts, rates, bar_width) lists aligned with ``tasks``.

    """
    descs = [t.description for t in tasks]
    pcts = [f"{t.percentage:.0f}%" if t.total else "" for t in tasks]
    els = [_elapsed_text(t) for t in tasks]
    counts_full = [_count_text(t, compact=False) for t in tasks]
    counts_compact = [_count_text(t, compact=True) for t in tasks]
    rates: list[str | None] = [_rate_text(t) for t in tasks]

    desc_w = max((len(d) for d in descs), default=0)
    pct_w = max((len(p) for p in pcts), default=0)
    el_w = max((len(e) for e in els), default=0)

    def table_width(counts: list[str], rate_texts: list[str], bar: int) -> int:
        c_w = max((len(c) for c in counts), default=0)
        r_w = max((len(r) for r in rate_texts), default=0)
        w = 1 + desc_w + c_w + bar + pct_w + el_w + r_w + (_N_PROGRESS_COLS - 1)
        if any(not r for r in rate_texts):
            w += 1  # trailing space on rows whose last cell is empty
        return w

    rate_texts = [r or "" for r in rates]
    if any(rates):
        base = table_width(counts_full, rate_texts, 0)
        bar = min(24, width - base)
        if bar >= _BAR_MIN:
            return (counts_full, rate_texts, bar)
    rate_blank = [""] * len(tasks)
    base = table_width(counts_full, rate_blank, 0)
    bar = min(24, width - base)
    if bar >= _BAR_MIN:
        return (counts_full, rate_blank, bar)
    base = table_width(counts_compact, rate_blank, 0)
    return (counts_compact, rate_blank, max(0, min(24, width - base)))


def _summary_lines(title: str, rows: list[tuple[str, str] | str]) -> list[str]:
    """Build the aligned plain-text run summary block (issue #102).

    ``label: value`` rows are padded to the longest label; bare strings
    (output paths) sit at column 0, one per line, copyable.
    """
    label_w = max((len(row[0]) for row in rows if isinstance(row, tuple)), default=0)
    out = [title]
    for row in rows:
        if isinstance(row, tuple):
            label, value = row
            out.append(f"  {label}:".ljust(label_w + 3) + value)
        else:
            out.append(row)
    return out


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
    """Completed count + unit, e.g. ``12,345 rows`` / ``1,692,256 cells``.

    Full precision when the table fits the terminal; compact (``1.7M rows``)
    on narrow terminals so nothing truncates (issue #102).
    """

    def __init__(
        self,
        width_fn: Callable[[], int] | None = None,
        tasks_fn: Callable[[], list[Task]] | None = None,
    ) -> None:
        """Bind the terminal-width and visible-task probes (default: wide, self-only)."""
        super().__init__()
        self._width_fn = width_fn if width_fn is not None else (lambda: _WIDE_DEFAULT)
        self._tasks_fn = tasks_fn if tasks_fn is not None else list

    @staticmethod
    def get_table_column() -> Column:
        return Column(no_wrap=True)

    def render(self, task: Task) -> Text:
        if task.completed <= 0:
            return Text("")
        tasks = self._tasks_fn() or [task]
        counts, _, _ = _plan_table(tasks, self._width_fn())
        for i, t in enumerate(tasks):
            if t is task:
                return Text(counts[i])
        return Text(_count_text(task, compact=False))


class _BarColumn(BarColumn):
    """Green (ok) bar; red (error) once the phase ended with a failure.

    Bar width comes from the table plan: 24 when the full table fits,
    elastic (shrinking to 0) on narrow terminals (issue #102).
    """

    def __init__(
        self,
        width_fn: Callable[[], int] | None = None,
        tasks_fn: Callable[[], list[Task]] | None = None,
    ) -> None:
        """Bind the terminal-width and visible-task probes (default: wide, self-only)."""
        super().__init__()
        self._width_fn = width_fn if width_fn is not None else (lambda: _WIDE_DEFAULT)
        self._tasks_fn = tasks_fn if tasks_fn is not None else list

    @staticmethod
    def get_table_column() -> Column:
        return Column(no_wrap=True)

    def render(self, task: Task) -> Text:
        tasks = self._tasks_fn() or [task]
        bar = _plan_table(tasks, self._width_fn())[2]
        if bar <= 0:
            return Text("")
        style = "red" if task.fields.get("failed") else "green"
        # D-4: the unfinished track (and the indeterminate pulse tail) is
        # DIM, not the terminal default color: an unstyled track rendered
        # brighter than the green fill and read as inverted. No-color
        # consoles drop the dim entirely (rich then leaves the track as
        # spaces, as before).
        return BarColumn(
            bar_width=bar,
            style="dim",
            complete_style=style,
            finished_style=style,
            pulse_style=style,
        ).render(task)


class _PercentColumn(ProgressColumn):
    """Determinate percentage.

    Indeterminate tasks render blank: an invented percent is never shown
    (issue #101).
    """

    @staticmethod
    def get_table_column() -> Column:
        return Column(no_wrap=True)

    @staticmethod
    def render(task: Task) -> Text:
        if not task.total:
            return Text("")
        return Text(f"{task.percentage:.0f}%")


class _ElapsedColumn(ProgressColumn):
    """H:MM:SS elapsed for the task. Unstyled: semantic color only."""

    @staticmethod
    def get_table_column() -> Column:
        return Column(no_wrap=True)

    @staticmethod
    def render(task: Task) -> Text:
        return Text(_elapsed_text(task))


class _RateColumn(ProgressColumn):
    """Throughput, e.g. ``8.3Kcells/s``.

    Shown once the task has both counts and a little history (the first
    samples are too noisy); dropped by the table plan when the full table
    no longer fits the terminal (issue #102).
    """

    def __init__(
        self,
        width_fn: Callable[[], int] | None = None,
        tasks_fn: Callable[[], list[Task]] | None = None,
    ) -> None:
        """Bind the terminal-width and visible-task probes (default: wide, self-only)."""
        super().__init__()
        self._width_fn = width_fn if width_fn is not None else (lambda: _WIDE_DEFAULT)
        self._tasks_fn = tasks_fn if tasks_fn is not None else list

    @staticmethod
    def get_table_column() -> Column:
        return Column(no_wrap=True)

    def render(self, task: Task) -> Text:
        tasks = self._tasks_fn() or [task]
        _, rates, _ = _plan_table(tasks, self._width_fn())
        for i, t in enumerate(tasks):
            if t is task:
                return Text(rates[i])
        return Text(_rate_text(task) or "")


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
    semantic-color line above the table (D-7 re-adds the ``[warn]`` /
    ``error:`` display markers in ``UI._emit``; the record message itself
    carries no logging prefix).
    """

    def __init__(self, ui: UI) -> None:
        super().__init__()
        self._ui = ui

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
            self._ui._emit(kind, msg)
        except Exception:  # ruff: ignore[blind-except] - logging must never raise
            self.handleError(record)


class _RuleProgress(Progress):
    """Progress with a thin dim Rule rendered directly above the tasks table.

    Concept X1: live notes print into the live region, and the divider
    (titled ``ppgrid``) sits immediately above the table. rich's Progress
    feeds its Live a ``get_renderable`` built from ``get_renderables()``,
    so the divider must come from that iterator — an instance-level
    ``live.update(...)`` would be ignored.
    """

    def __init__(self, *columns: Any, rule: Rule | None = None, **kwargs: Any) -> None:
        # Set before super().__init__: Live renders once at construction,
        # which already calls get_renderables().
        self._rule = rule
        super().__init__(*columns, **kwargs)

    def get_renderables(self) -> Iterable[RenderableType]:
        if self._rule is not None:
            yield self._rule
        try:
            yield from super().get_renderables()
        except RuntimeError:
            # Auto-refresh tick raced a phase start (main thread inserted
            # into the tasks dict mid-iteration). Skip the table for this
            # tick; the next refresh (100 ms) renders it. Without this
            # guard the exception would kill the refresh thread (D-3).
            return


class _PlainLogFormatter(logging.Formatter):
    """The unified plain-mode line format: ``L ppgrid.name: msg`` (D-7).

    WARNING and ERROR records carry semantic SGR (yellow/red) when color
    is on; everything else is unstyled. No other colors are emitted.
    """

    def __init__(self, *, color: bool) -> None:
        super().__init__(_LOG_FORMAT)
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if not self._color:
            return text
        if record.levelno >= logging.ERROR:
            code = _SGR_ERROR
        elif record.levelno >= logging.WARNING:
            code = _SGR_WARN
        else:
            return text
        return f"\x1b[{code}m{text}\x1b[{_SGR_RESET}m"


def plain_log_formatter(*, color: bool) -> logging.Formatter:
    """Build the unified plain-mode formatter (D-7) for the CLI handler."""
    return _PlainLogFormatter(color=color)


def _has_ppgrid_handler() -> bool:
    """Check for an installed CLI run handler.

    True when a CLI run installed its ppgrid stderr handler (or the live
    UI handler). Plain-mode events then go through the unified logging
    format; without one (library mode) they keep the legacy direct-print
    surface.

    """
    return any(
        getattr(h, "ppgrid_handler", False) or isinstance(h, _UIStreamHandler)
        for h in logging.getLogger("ppgrid").handlers
    )


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
        quiet: bool = False,
        width: int | None = None,
    ) -> None:
        """Store mode flags; build the rich Console/Progress when live.

        ``quiet`` suppresses the final run summary (#102); ``width`` pins
        the console width (tests, and terminals that report no size).
        """
        self._stream = stream if stream is not None else sys.stderr
        self._live = live
        self._color = color
        self._unicode = unicode
        self._spinner_min_s = spinner_min_s
        self._refresh_interval = refresh_interval
        self._last_refresh = 0.0
        self._quiet = quiet
        self._console: Console | None = None
        self._stdout_console: Console | None = None
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
                width=width,
            )

            def _layout_width() -> int:
                """Live terminal width for the table layout plan (issue #102)."""
                return self._console.size.width if self._console is not None else _WIDE_DEFAULT

            def _visible_tasks() -> list[Task]:
                """Return the visible tasks; the table plan sizes to all of them (#102)."""
                if self._progress is None:
                    return []
                return [t for t in self._progress.tasks if t.visible]

            # Concept X1: one thin dim divider, titled `ppgrid`, directly
            # above the tasks table. `dim` (rule and title) only when color
            # is on: a no-color console drops it, and the point is "dimmer
            # than the green fill", not a specific gray. This rich has no
            # Rule(title_style=...): a Text title carries the style.
            rule = Rule(
                Text("ppgrid", style="dim") if color else "ppgrid",
                characters="─" if unicode else "-",
                style="dim" if color else None,
            )
            # stdout console for live-only stdout artifacts (the --plan
            # box, X2); the stderr console keeps the run surface.
            self._stdout_console = Console(
                file=sys.stdout,
                no_color=not color,
                force_terminal=True,
                highlight=False,
                soft_wrap=False,
                emoji=False,
            )
            # D-3: rich's own refresh loop (10 fps) so the spinner and bar
            # animate between the pipeline's throttled manual refreshes.
            self._progress = _RuleProgress(
                _SpinnerColumn(unicode=unicode, min_s=spinner_min_s),
                TextColumn("{task.description}"),
                _CountColumn(width_fn=_layout_width, tasks_fn=_visible_tasks),
                _BarColumn(width_fn=_layout_width, tasks_fn=_visible_tasks),
                _PercentColumn(),
                _ElapsedColumn(),
                _RateColumn(width_fn=_layout_width, tasks_fn=_visible_tasks),
                rule=rule,
                console=self._console,
                auto_refresh=True,
                refresh_per_second=10,
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
            quiet=quiet,
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

    @property
    def quiet(self) -> bool:
        """True when --quiet suppresses the final run summary (issue #102)."""
        return self._quiet

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
        """Success line (green; shown with an ``ok: `` prefix in live mode — D-8)."""
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

    def summary(self, title: str, rows: list[tuple[str, str] | str]) -> None:
        """Render the final run summary block (issue #102).

        Aligned ``label: value`` rows (padded to the longest label) plus
        bare path lines at column 0 (copyable). The block is built to fit
        80 columns, so it never wraps or truncates at 80/120. The title is
        the success marker (green when colored); the values are neutral.

        Live mode (concept X3): the block goes into a rounded panel titled
        ``summary`` — the plain ``run summary`` header line becomes the
        panel title; the rows keep their plain-text form.

        Args:
            title: The block header (e.g. ``run summary``).
            rows: ``(label, value)`` pairs and/or bare path strings, in order.

        """
        lines = _summary_lines(title, rows)
        if self._live and self._console is not None:
            panel = Panel(
                Text("\n".join(lines[1:]), no_wrap=True),
                title=Text("summary", style="green" if self._color else ""),
                box=box.ROUNDED if self._unicode else box.ASCII,
                expand=False,
            )
            self._console.print(panel)
            self._refresh()
            return
        for i, text in enumerate(lines):
            if i == 0 and self._color:
                print(f"\x1b[{_SGR_OK}m{text}\x1b[{_SGR_RESET}m", file=self._stream)
            else:
                print(text, file=self._stream)

    def print_plan(self, header: str, kv: list[str]) -> None:
        r"""Emit the ``--plan`` printout on stdout (issue #53).

        Plain: ``header + '\n' + kv`` exactly as the pre-#102 ``print``
        (byte-identical). Live (concept X2): the header stays a plain
        line; the key/value block goes into a square box (dim border when
        color is on), hug-width, with the kv lines' own indent kept
        (padding (0, 0)). The env report and progress rows around it are
        unaffected (stderr, unboxed).

        """
        if self._live and self._stdout_console is not None:
            print(header, file=sys.stdout)  # ruff: ignore[print]
            panel = Panel(
                Text("\n".join(kv), no_wrap=True),
                box=box.SQUARE if self._unicode else box.ASCII,
                border_style="dim" if self._color else None,
                padding=(0, 0),
                expand=False,
            )
            self._stdout_console.print(panel)
            return
        print(header + "\n" + "\n".join(kv), file=sys.stdout)  # ruff: ignore[print]

    def _emit(self, kind: str, text: str) -> None:
        """Emit one status line in the active mode (D-7).

        Live: the semantic marker (``ok: `` / ``[warn] `` / ``error: ``)
        is re-added here — call sites carry the message only once — and a
        line longer than the terminal wraps with a hanging indent (D-6).
        Plain with an installed ppgrid handler: one
        ``L ppgrid.pipeline: msg`` record (WARNING/ERROR carry semantic
        SGR when color is on). Plain without one (library mode): legacy
        direct print, byte-identical.
        """
        if self._live and self._console is not None:
            shown = _LIVE_MARKERS[kind] + text
            width = self._console.width
            if len(shown) > width:
                shown = textwrap.fill(shown, width=width, subsequent_indent="    ")
            t = Text(shown)
            style = _LIVE_STYLES.get(kind)
            if style and self._color:
                t.stylize(style)
            self._console.print(t, soft_wrap=True)
            self._refresh()
            return
        if _has_ppgrid_handler():
            pipe_log = logging.getLogger("ppgrid.pipeline")
            if kind == "error":
                pipe_log.error("%s", text)
            elif kind == "warn":
                pipe_log.warning("%s", text)
            else:
                pipe_log.info("%s", text)
            return
        code = _PLAIN_SGR.get(kind) if self._color else None
        if code:
            print(f"\x1b[{code}m{text}\x1b[{_SGR_RESET}m", file=self._stream)
        else:
            print(text, file=self._stream)
