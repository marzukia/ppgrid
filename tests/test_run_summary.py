"""Final run summary + adaptive narrow-width progress layout (issue #102).

Covers: the run-summary block (live + plain, width-invariant, no escapes),
quiet suppression, the JSON status field, fallback recording in the
summary, and the adaptive progress-table layout at 80 / 120 / 150 cols
(nothing truncates or wraps at 80/120; full columns at wide).
"""

import io
import json
import re
import sys
from pathlib import Path

import pytest
from rich.console import Console

import ppgrid.ui as _ui
from ppgrid import pipeline as _pl
from tests.test_ui import FakeTTY

_SUMMARY_ROWS: list[tuple] = [
    ("rows read", "13,580"),
    ("rows accepted", "13,580"),
    ("rows dropped", "0"),
    ("grid", "11x8 (88 cells)"),
    ("strategy", "standard, workers 4"),
    ("elapsed", "12.34s"),
    ("peak RAM", "14.8 GB (measured)"),
    ("value.tif", "5.2 MB"),
    ("support_km.tif", "3.1 MB"),
    "/out/value.tif",
    "/out/support_km.tif",
]

_EXPECTED_SUMMARY = """run summary
  rows read:     13,580
  rows accepted: 13,580
  rows dropped:  0
  grid:          11x8 (88 cells)
  strategy:      standard, workers 4
  elapsed:       12.34s
  peak RAM:      14.8 GB (measured)
  value.tif:     5.2 MB
  support_km.tif:3.1 MB
/out/value.tif
/out/support_km.tif
"""

_BAR = "\u2501"  # full bar block
_PART = "\u2578"  # partial bar block

# Wide-tier final frame for the 16.9M-cells task (120/150 snapshots share it).
_WIDE_INTERP = (
    f"  interpolate                                    16,922,560 cells {_BAR * 22}{_PART}  94% 0:00:10 1.7Mcells/s"
)


def _drive_snapshot_tasks(ui: _ui.UI) -> None:
    """Two finished tasks with deterministic state.

    One long indeterminate write row (desc 46, no count) and one large
    completed interpolate row (16,922,560 of 18,000,000 cells, 10 s
    elapsed -> rate active).
    """
    p = ui._progress
    a = p.add_task("write: value.tif 0.0 MB, support_km.tif 0.0 MB", total=None, completed=0)
    b = p.add_task("interpolate", total=18_000_000, completed=16_922_560, unit="cells")
    for tid in (a, b):
        t = p.tasks[tid]
        t.start_time = p.get_time() - 10.0
        t.finished_time = 10.0
        p.stop_task(tid)


def _render_table(ui: _ui.UI, width: int) -> list[str]:
    """Render the final tasks table to a clean console at the fixed width."""
    buf = io.StringIO()
    Console(file=buf, width=width, force_terminal=False).print(ui._progress.make_tasks_table(ui._progress.tasks))
    return buf.getvalue().splitlines()


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def test_compact_count() -> None:
    """Compact counts use K/M suffixes; small numbers stay exact."""
    assert _ui._compact_count(999) == "999"
    assert _ui._compact_count(13580) == "13.6K"
    assert _ui._compact_count(365000) == "365K"
    assert _ui._compact_count(1_000_000) == "1M"
    assert _ui._compact_count(1_692_256) == "1.7M"
    assert _ui._compact_count(0) == "0"


def test_human_size() -> None:
    """Output sizes use decimal units (matching the --ram-gb convention)."""
    assert _pl._human_size(999) == "999 B"
    assert _pl._human_size(856_000) == "856 KB"
    assert _pl._human_size(1_234_567) == "1.2 MB"
    assert _pl._human_size(5_200_000) == "5.2 MB"
    assert _pl._human_size(1_480_000_000) == "1.5 GB"
    assert _pl._human_size(0) == "0 B"


# ---------------------------------------------------------------------------
# run summary block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [80, 120])
def test_summary_snapshot_live_width_invariant(width: int) -> None:
    """The live-mode summary block is byte-identical at 80 and 120 cols."""
    fake = io.StringIO()
    ui = _ui.UI(live=True, color=False, unicode=True, stream=fake, width=width, spinner_min_s=2.0)
    ui.summary("run summary", _SUMMARY_ROWS)
    text = fake.getvalue()
    assert "\x1b" not in text
    i = text.find("run summary")
    assert i >= 0
    block = text[i : i + len(_EXPECTED_SUMMARY)]
    assert block == _EXPECTED_SUMMARY
    # every line comfortably inside the terminal (well under 80).
    assert max(len(line) for line in _EXPECTED_SUMMARY.splitlines()) <= 78
    # paths are copyable: one per line, bare, no decoration.
    assert "/out/value.tif\n/out/support_km.tif" in text


def test_summary_plain_mode_no_escapes() -> None:
    """Plain mode (non-TTY) prints the same block with zero escape codes."""
    fake = io.StringIO()
    ui = _ui.UI(live=False, color=False, unicode=True, stream=fake, width=80, spinner_min_s=2.0)
    ui.summary("run summary", _SUMMARY_ROWS)
    text = fake.getvalue()
    assert "\x1b" not in text
    i = text.find("run summary")
    assert i >= 0
    assert text[i : i + len(_EXPECTED_SUMMARY)] == _EXPECTED_SUMMARY


def test_summary_plain_mode_colour_title() -> None:
    """Colour-on plain mode marks only the title green."""
    fake = io.StringIO()
    ui = _ui.UI(live=False, color=True, unicode=True, stream=fake, width=80, spinner_min_s=2.0)
    ui.summary("run summary", _SUMMARY_ROWS)
    lines = fake.getvalue().splitlines()
    assert lines[0] == "\x1b[32mrun summary\x1b[0m"
    for line in lines[1:]:
        assert "\x1b" not in line, f"non-title line carries escapes: {line!r}"


# ---------------------------------------------------------------------------
# adaptive progress table layout (issue #102 narrow-width acceptance)
# ---------------------------------------------------------------------------


def test_progress_snapshot_80_compact_counts_bar_no_rate() -> None:
    """At 80 cols: compact counts, rate dropped, bar kept, nothing wraps."""
    fake = io.StringIO()
    ui = _ui.UI(live=True, color=False, unicode=True, stream=fake, width=80, spinner_min_s=2.0)
    _drive_snapshot_tasks(ui)
    lines = _render_table(ui, 80)
    assert lines == [
        f"  write: value.tif 0.0 MB, support_km.tif 0.0 MB             {_BAR * 5}     0:00:10  ",
        f"  interpolate                                    16.9M cells {_BAR * 4}{_PART} 94% 0:00:10  ",
    ]
    assert all(len(line) <= 80 for line in lines)
    joined = "".join(lines)
    assert "16,922,560" not in joined  # compact tier: no full-precision counts
    assert "cells/s" not in joined  # rate dropped below the width threshold
    assert "0:00:10" in joined  # elapsed kept
    assert "94%" in joined  # percent kept


def test_progress_snapshot_120_full_columns() -> None:
    """At 120 cols: full-precision counts + rate + 24-char bar all fit."""
    fake = io.StringIO()
    ui = _ui.UI(live=True, color=False, unicode=True, stream=fake, width=120, spinner_min_s=2.0)
    _drive_snapshot_tasks(ui)
    lines = _render_table(ui, 120)
    assert lines == [
        f"  write: value.tif 0.0 MB, support_km.tif 0.0 MB                  {_BAR * 24}     0:00:10            ",
        _WIDE_INTERP,
    ]
    assert all(len(line) <= 120 for line in lines)
    assert "16,922,560 cells" in lines[1]
    assert "1.7Mcells/s" in lines[1]


def test_progress_snapshot_150_full_columns() -> None:
    """Wide (>150) keeps the pre-#102 full columns unchanged."""
    fake = io.StringIO()
    ui = _ui.UI(live=True, color=False, unicode=True, stream=fake, width=150, spinner_min_s=2.0)
    _drive_snapshot_tasks(ui)
    lines = _render_table(ui, 150)
    assert lines == [
        f"  write: value.tif 0.0 MB, support_km.tif 0.0 MB                  {_BAR * 24}     0:00:10            ",
        _WIDE_INTERP,
    ]
    assert all(len(line) <= 150 for line in lines)


def test_progress_no_wrap_at_80_short_desc() -> None:
    """Short phase descs at 80 keep the full columns (wide tier fits)."""
    fake = io.StringIO()
    ui = _ui.UI(live=True, color=False, unicode=True, stream=fake, width=80, spinner_min_s=2.0)
    p = ui._progress
    b = p.add_task("interpolate", total=1_800_000, completed=1_692_256, unit="cells")
    t = p.tasks[b]
    t.start_time = p.get_time() - 10.0
    t.finished_time = 10.0
    p.stop_task(b)
    lines = _render_table(ui, 80)
    assert lines == [f"  interpolate 1,692,256 cells {_BAR * 22}{_PART}  94% 0:00:10 169Kcells/s"]
    assert len(lines[0]) <= 80


# ---------------------------------------------------------------------------
# pipeline-level: summary in stderr, quiet, json, fallbacks
# ---------------------------------------------------------------------------


def _drops_csv(tmp_path: Path) -> str:
    """5 rows: 2 valid, 1 non-finite value, 2 out-of-domain coords."""
    csv = tmp_path / "drops.csv"
    csv.write_text(
        "value,lng,lat\n1.0,149.0,-35.0\n2.0,149.1,-35.1\nnan,149.2,-35.0\n3.0,149.3,95.0\n4.0,200.0,-35.0\n"
    )
    return str(csv)


def _run_argv(csv: str, out: Path, *extra: str) -> list[str]:
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


def test_run_summary_in_stderr_default_verbosity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-TTY default run prints the summary block with drop reasons."""
    fake = FakeTTY()
    monkeypatch.setattr(sys, "stderr", fake)
    argv = _run_argv(_drops_csv(tmp_path), tmp_path / "out")
    _pl.main(argv)
    text = fake.getvalue()
    assert "run summary" in text
    assert "  rows read:     5" in text
    assert "  rows accepted: 2" in text
    assert "  rows dropped:  3 (1 non-finite value; 2 out-of-domain)" in text
    assert re.search(r"grid:\s+\d+x\d+ \([\d,]+ cells\)", text)
    assert "strategy:      standard, workers 2" in text
    assert re.search(r"elapsed:\s+\d+\.\d\ds", text)
    assert re.search(r"peak RAM:\s+(?:\d+\.\d GB \(measured\)|~\d+\.\d GB \(estimated\))", text)
    assert re.search(r"value\.tif:\s+\d+(\.\d+)? (B|KB|MB|GB)", text)
    # copyable bare paths, one per line
    assert f"\n{tmp_path / 'out' / 'value.tif'}\n" in text
    assert f"\n{tmp_path / 'out' / 'support_km.tif'}\n" in text
    # only the colour title carries escapes (no cursor/movement codes)
    block = text[text.find("run summary") :]
    assert "\x1b" not in block.replace("\x1b[32m", "").replace("\x1b[0m", "")


def test_quiet_suppresses_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--quiet keeps the pre-#101 zero-noise surface: no summary block."""
    fake = FakeTTY()
    monkeypatch.setattr(sys, "stderr", fake)
    argv = _run_argv(_drops_csv(tmp_path), tmp_path / "out", "--quiet")
    _pl.main(argv)
    assert "run summary" not in fake.getvalue()


def test_json_gains_status_ok(tmp_path: Path) -> None:
    """--json run_summary.json gains the additive 'status' field."""
    out = tmp_path / "out"
    _pl.main(_run_argv(_drops_csv(tmp_path), out, "--json"))
    s = json.loads((out / "run_summary.json").read_text())
    assert s["status"] == "ok"
    assert s["ingest"]["rows_read"] == 5
    assert s["ingest"]["points"] == 2


def test_fallback_recorded_in_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A zstd gate fallback is listed in the summary (fallbacks visible)."""
    fake = FakeTTY()
    monkeypatch.setattr(sys, "stderr", fake)

    def _zstd_ok(self: _pl.Pipeline) -> bool:
        self._zstd_gate_reason = "oracle mismatch"
        return False

    monkeypatch.setattr(_pl.Pipeline, "_turbo_zstd_ok", _zstd_ok)
    argv = _run_argv(_drops_csv(tmp_path), tmp_path / "out", "--turbo", "16")
    _pl.main(argv)
    text = fake.getvalue()
    assert "  fallbacks:     zstd parallel write -> stock parallel (oracle mismatch)" in text
    # Cap is clamped to detected RAM (macos runner: 7.5 GB -> floored to 8 GB);
    # assert the structure, not the values (CI 38400658211, 2026-10-13).
    assert re.search(r"strategy:\s+\S.* turbo, cap [\d.]+ GB, budget [\d.]+ GB, workers \d+", text)


def test_summary_printed_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Plain mode: exactly one summary block (not once plain + once live)."""
    fake = FakeTTY()
    monkeypatch.setattr(sys, "stderr", fake)
    argv = _run_argv(_drops_csv(tmp_path), tmp_path / "out")
    _pl.main(argv)
    assert fake.getvalue().count("run summary") == 1
