"""Tests for ppgrid.turbop: pre-check estimator + --max-ram regime sizing.

Issue #30; design /home/monky/reports/ppgrid-turbo-design.md sections 2.2,
3.5, 3.6. The 3.6.2 worked table (full-AU anchor geometry) is the oracle:
all four box-size rows (16/32/64/128 GB) are reproduced by plan().
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pytest

from ppgrid import turbop
from ppgrid.turbop import (
    ANCHOR_WALL_S,
    B0_BYTES,
    GB,
    M_IN_B_PER_CELL,
    SIZING_TABLE,
    RssBackstop,
    _read_cgroup_max_gb,
    _read_physical_gb,
    box_chunk_bytes,
    descent_band_bytes,
    detect_cap_gb,
    est_peak,
    per_box_wall_s,
    phase_peak_bytes,
    plan,
    precheck,
    precheck_budget_bytes,
    resolve_cap,
    sizing_budget_bytes,
    table_row,
    tile_cache_tiles,
    wall_model_s,
)

# Full-AU anchor geometry (design 2.2 / 3.6.2): 4100 x 3819 km @ 100 m,
# padded cells N, padded columns Wc (512-padded), cap radius r (25 km).
N_FULL_AU = 1_580_000_000
WC_FULL_AU = 41472
R_FULL_AU = 250
N_PTS_FULL_AU = 16_000_000  # point count for the 2.2 phase estimates
CPU = 48  # hydrogen-class box


def _plan_full_au(cap_gb: float) -> turbop.TurboPlan:
    """Plan the full-AU anchor geometry at a cap, on a 48-thread box."""
    return plan(cap_gb, N_FULL_AU, WC_FULL_AU, R_FULL_AU, cpu=CPU)


# ---------------------------------------------------------------------------
# 2.2 phase estimates
# ---------------------------------------------------------------------------


def test_est_peak_is_phase_max_not_sum() -> None:
    """est_peak is the MAX of phases A-D (phases do not run at once)."""
    est = est_peak(N_FULL_AU, N_PTS_FULL_AU)
    phase_a = phase_peak_bytes("A", N_FULL_AU, N_PTS_FULL_AU)
    total = sum(phase_peak_bytes(p, N_FULL_AU, N_PTS_FULL_AU) for p in ("A", "B", "C", "D"))
    assert est == phase_a  # phase A dominates at ~24.5 B/cell
    assert est < total  # max, not sum
    # 2.2 full-AU phase A: ~40 GB model / 38-40 GB RSS sum (no B0).
    assert est / GB == pytest.approx(39.35, rel=0.01)


def test_phase_peak_prices_match_design_2_2() -> None:
    """Phase prices reproduce the 2.2 full-AU phase totals (16M pts)."""
    expected_gb = {
        "A": 39.35,  # 24.5 * 1.58e9 + 16e6 * 40
        "B": 19.07,  # pyramid 10.67 + near 1, plus points; 2.2: ~19 GB
        "C": 31.71,  # pyramid + val/sup + near ~19.7, plus points; 2.2: ~32 GB
        "D": 18.02,  # val/sup + near + int16 DN, plus points; 2.2: ~18 GB
    }
    for phase, gb in expected_gb.items():
        assert phase_peak_bytes(phase, N_FULL_AU, N_PTS_FULL_AU) / GB == pytest.approx(gb, rel=0.02)


def test_phase_peak_rejects_unknown_phase() -> None:
    """Unknown phase letter raises ValueError."""
    with pytest.raises(ValueError, match="unknown phase"):
        phase_peak_bytes("E", 10, 10)


# ---------------------------------------------------------------------------
# Budget model
# ---------------------------------------------------------------------------


def test_budget_reconciliation() -> None:
    """Bt + B0 == 0.85*C exactly: the pre-check counts B0 inside (3.6.2)."""
    for cap in (16.0, 20.0, 32.0, 64.0, 128.0):
        assert (sizing_budget_bytes(cap) + B0_BYTES) / GB == pytest.approx(precheck_budget_bytes(cap) / GB, abs=1e-9)


def test_tile_cache_clamp() -> None:
    """T = clamp(floor(0.1 * Bt / M_tile), 8, 256)."""
    assert tile_cache_tiles(5.0e7) == 8  # 5 tiles -> floor to min
    assert tile_cache_tiles(1.5e8) == 15  # mid-range, no clamp
    assert tile_cache_tiles(2.32e10) == 256  # C=32 Bt -> ceiling
    assert tile_cache_tiles(1.0e11) == 256  # C=128 Bt -> ceiling


# ---------------------------------------------------------------------------
# The 3.6.2 worked table: all four box-size rows (the test oracle)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cap_gb", "regime", "c", "b", "workers", "t", "val_memmap", "peak_gb", "wall_range"),
    [
        (16.0, "per_box", 0, 0, 4, 0, True, None, (290.0, 380.0)),
        (32.0, "B", 3, 4, 3, 256, True, 27.2, (240.0, 330.0)),
        (64.0, "A", 0, 0, 8, 256, False, 42.7, (115.0, 160.0)),
        (128.0, "A", 0, 0, 8, 256, False, 42.7, (115.0, 160.0)),
    ],
)
def test_worked_table_rows(
    cap_gb: float,
    regime: str,
    c: int,
    b: int,
    workers: int,
    t: int,
    *,
    val_memmap: bool,
    peak_gb: float | None,
    wall_range: tuple[float, float],
) -> None:
    """plan() reproduces every row of the 3.6.2 worked table."""
    pl = _plan_full_au(cap_gb)
    assert pl.regime == regime
    assert pl.box_chunks == c
    assert pl.descent_bands == b
    assert pl.workers == workers
    assert pl.tile_cache_tiles == t
    assert pl.val_sup_memmap is val_memmap
    assert pl.zstd_ctx_bytes == workers * 2_000_000
    # Wall: central estimate inside the hand-widened row range.
    assert pl.wall_range_s == wall_range
    assert wall_range[0] <= pl.wall_s <= wall_range[1]
    # Est peak: table range (per-box row) or the derived single value.
    row = table_row(cap_gb)
    assert row is not None
    assert row.regime == pl.regime
    assert row.box_chunks == pl.box_chunks
    assert row.descent_bands == pl.descent_bands
    assert row.workers == pl.workers
    lo, hi = row.est_peak_range_gb
    assert lo <= pl.est_peak_bytes / GB <= hi
    if peak_gb is not None:
        assert pl.est_peak_bytes / GB == pytest.approx(peak_gb, abs=0.05)


def test_worked_table_anchor_ratios() -> None:
    """Wall ranges divide to the table's vs-anchor ratios (479.6 s)."""
    expected = {16.0: (0.60, 0.79), 32.0: (0.50, 0.69), 64.0: (0.24, 0.33), 128.0: (0.24, 0.33)}
    assert {row.cap_gb for row in SIZING_TABLE} == set(expected)
    for row in SIZING_TABLE:
        lo = round(row.wall_range_s[0] / ANCHOR_WALL_S, 2)
        hi = round(row.wall_range_s[1] / ANCHOR_WALL_S, 2)
        assert (lo, hi) == expected[row.cap_gb]


def test_wall_model_central() -> None:
    """Wall model: 30 s serial floor + 740 s / workers; per-box from the anchor."""
    assert wall_model_s(4) == pytest.approx(215.0)  # A4 target at 4 workers
    assert wall_model_s(8) == pytest.approx(122.5)
    assert per_box_wall_s() == pytest.approx(303.2, abs=0.1)  # 479.6 - 65.2 - 21.9 - 89.3


def test_in_flight_sizing_formulas() -> None:
    """M_box / M_band prices and the regime B in-flight derivation (C=32)."""
    m_box = box_chunk_bytes(WC_FULL_AU, R_FULL_AU)
    m_band = descent_band_bytes(WC_FULL_AU)
    assert m_box / GB == pytest.approx(2.287, rel=0.001)  # 12 B/cell x 4596 rows x 41472
    assert m_band / GB == pytest.approx(1.869, rel=0.001)  # (40*1024 + 8*512) x 41472
    bt = sizing_budget_bytes(32.0)
    left = bt - M_IN_B_PER_CELL * N_FULL_AU - 256 * 1_000_000
    assert left // m_box == 3  # c = floor(left / M_box)
    assert left // m_band == 4  # b = floor(left / M_band)


def test_bc_knee_boundary() -> None:
    """Regime B/C knee at C* = 24.13 GB (design 3.6.2 'Reading the table')."""
    m_box = box_chunk_bytes(WC_FULL_AU, R_FULL_AU)
    m_band = descent_band_bytes(WC_FULL_AU)
    c_star = (M_IN_B_PER_CELL * N_FULL_AU + max(m_box, m_band) + B0_BYTES) / (0.85 * GB)
    assert c_star == pytest.approx(24.13, abs=0.01)
    assert _plan_full_au(c_star + 0.5).regime == "B"
    assert _plan_full_au(c_star - 0.5).regime == "C"


@pytest.mark.parametrize("cap_gb", [24.13, 24.2, 24.42])
def test_b_knife_edge_falls_through_to_c(cap_gb: float) -> None:
    """B admits at the knee but the tile cache leaves c = 0: no crash.

    The B candidate is degenerate (0 workers) on full-AU for C in
    [24.13, 24.42) GB, so plan() falls through to regime C, which yields
    c = 3, b = 4, 3 workers, inside the 0.85*C budget.
    """
    pl = _plan_full_au(cap_gb)
    assert pl.regime == "C"
    assert pl.box_chunks == 3
    assert pl.descent_bands == 4
    assert pl.workers == 3
    assert pl.val_sup_memmap is True
    assert pl.est_peak_bytes <= pl.budget_bytes
    dec = precheck(pl, strict=True, cap_source="auto")
    assert dec.path == "shared"
    assert dec.exit_code == 0


def test_b_knife_edge_regime_b_one_unit() -> None:
    """Just past the hole (24.43 GB): regime B with c = 1, b = 1, 1 worker."""
    pl = _plan_full_au(24.43)
    assert pl.regime == "B"
    assert pl.box_chunks == 1
    assert pl.descent_bands == 1
    assert pl.workers == 1
    assert pl.wall_s == pytest.approx(770.0)  # 30 + 740 / 1


def test_regime_c_large_radius_falls_to_per_box() -> None:
    """Same guard class in regime C at large r: c = 0 goes per-box, no crash.

    r = 2000 cells makes M_box ~4.03 GB; at 18.7 GB the 8 GB floor leaves
    c = 0, b = 2, which the old max(c, b) >= 2 gate admitted with 0
    workers (wall_model_s division by zero).
    """
    pl = plan(18.7, N_FULL_AU, WC_FULL_AU, 2000, cpu=CPU)
    assert pl.regime == "per_box"
    assert pl.box_chunks == 0
    assert pl.descent_bands == 0
    assert pl.workers == 4


def test_regime_b_48gb() -> None:
    """48 GB (not a preset): regime B, c = 9, 8 workers (3.6.2 reading)."""
    pl = _plan_full_au(48.0)
    assert pl.regime == "B"
    assert pl.box_chunks == 9
    assert pl.workers == 8


def test_laptop_grid_regime_a_at_floor() -> None:
    """fine16m (67.1M cells) reaches regime A even at the C = 8 floor."""
    pl = plan(8.0, 67_100_000, 8192, 6400, cpu=CPU)
    assert pl.regime == "A"
    assert pl.workers == 8
    assert pl.est_peak_bytes == pytest.approx(67_100_000 * 24.5 + B0_BYTES, rel=1e-9)


def test_plan_rejects_bad_geometry() -> None:
    """Non-positive cells/columns or negative radius raise ValueError."""
    with pytest.raises(ValueError, match="need n_cells"):
        plan(32.0, 0, WC_FULL_AU, R_FULL_AU, cpu=CPU)
    with pytest.raises(ValueError, match="need n_cells"):
        plan(32.0, N_FULL_AU, 0, R_FULL_AU, cpu=CPU)
    with pytest.raises(ValueError, match="need n_cells"):
        plan(32.0, N_FULL_AU, WC_FULL_AU, -1, cpu=CPU)


# ---------------------------------------------------------------------------
# Pre-check decision path (A9)
# ---------------------------------------------------------------------------


def test_a9_regime_c_at_20gb() -> None:
    """--max-ram 20 on full-AU: regime C (c=2, b=2), warn, exit 0; strict fits."""
    pl = _plan_full_au(20.0)
    assert pl.regime == "C"
    assert pl.box_chunks == 2  # floor((Bt - 8 GB) / M_box)
    assert pl.descent_bands == 2  # floor((Bt - 8 GB) / M_band)
    assert pl.workers == 2
    assert pl.val_sup_memmap is True
    assert pl.est_peak_bytes / GB == pytest.approx(16.57, abs=0.05)  # 3.6.2: 16.6 GB
    assert pl.est_peak_bytes <= pl.budget_bytes  # fits 0.85*C = 17.0 GB

    dec = precheck(pl, strict=False, cap_source="preset 20 GB")
    assert dec.path == "shared"
    assert dec.regime == "C"
    assert dec.exit_code == 0
    assert dec.budget_gb == pytest.approx(17.0, abs=1e-9)
    assert dec.message is not None
    assert dec.message.startswith("[warn]")
    assert "38.7" in dec.message  # regime-A field estimate, A9
    assert "regime C" in dec.message

    strict = precheck(pl, strict=True, cap_source="preset 20 GB")
    assert strict.exit_code == 0  # strict at 20 GB passes: regime C fits
    assert strict.message is None or strict.message.startswith("[warn]")


def test_exit3_pinned_at_16gb() -> None:
    """--max-ram 16 on full-AU: per-box; non-strict warns, strict exits 3."""
    pl = _plan_full_au(16.0)
    assert pl.regime == "per_box"
    assert pl.box_chunks == 0
    assert pl.descent_bands == 0  # 0 in-flight after the 8 GB floor
    assert pl.workers == 4  # per-box pool

    dec = precheck(pl, strict=False, cap_source="preset 16 GB")
    assert dec.path == "per_box"
    assert dec.exit_code == 0
    assert dec.message is not None
    assert dec.message.startswith("[warn]")
    assert "per-box" in dec.message

    strict = precheck(pl, strict=True, cap_source="preset 16 GB")
    assert strict.exit_code == 3
    assert strict.message is not None
    assert strict.message.startswith("error:")


def test_precheck_clean_at_64gb() -> None:
    """Regime A at 64 GB: no warn, exit 0, printable summary (deploy visibility)."""
    pl = _plan_full_au(64.0)
    _cap, source = resolve_cap(64.0, physical_gb=128.0)
    dec = precheck(pl, strict=True, cap_source=source)
    assert dec.path == "shared"
    assert dec.regime == "A"
    assert dec.exit_code == 0
    assert dec.message is None
    assert "cap 64 GB" in dec.summary
    assert "preset 64 GB" in dec.summary
    assert "budget 0.85*C = 54.4 GB" in dec.summary
    assert "Bt = 50.4 GB" in dec.summary
    assert "regime A" in dec.summary


def test_budget_is_printable() -> None:
    """The value used (cap + source) is printable per A9 deploy visibility."""
    cap, source = resolve_cap(None, physical_gb=125.0, cgroup_gb=64.0)
    assert cap == 56.0
    assert source == "auto min(physical 125.0 GB, cgroup 64.0 GB) - 8 GB headroom"
    pl = _plan_full_au(cap)
    dec = precheck(pl, strict=False, cap_source=source)
    assert "cap 56 GB" in dec.summary
    assert "64.0 GB" in dec.summary


# ---------------------------------------------------------------------------
# Auto-detect and cap resolution (mocked cgroup/physical values)
# ---------------------------------------------------------------------------


def test_detect_cap_auto() -> None:
    """Auto cap: min(physical, cgroup) - 8 GB headroom, floor 8 GB."""
    cap, source = detect_cap_gb(physical_gb=125.0, cgroup_gb=64.0)
    assert cap == 56.0
    assert source.startswith("auto")
    # cgroup smaller than physical (the A.5 incident box, 12 GB cap)
    cap, _ = detect_cap_gb(physical_gb=125.0, cgroup_gb=12.0)
    assert cap == 8.0  # 12 - 8 = 4, floored to 8
    # unlimited cgroup
    cap, source = detect_cap_gb(physical_gb=125.0, cgroup_gb=math.inf)
    assert cap == 117.0
    assert "unlimited" in source


def test_resolve_cap_clamp_and_floor() -> None:
    """Preset >= physical clamps to physical - 8; sub-floor presets hit 8."""
    cap, source = resolve_cap(32.0, physical_gb=64.0)
    assert cap == 32.0
    assert source == "preset 32 GB"
    cap, _ = resolve_cap(128.0, physical_gb=64.0)
    assert cap == 56.0  # clamped to physical - 8
    cap, _ = resolve_cap(64.0, physical_gb=64.0)
    assert cap == 56.0  # >= physical: whole box
    cap, _ = resolve_cap(4.0, physical_gb=64.0)
    assert cap == 8.0  # floored
    cap, source = resolve_cap(None, physical_gb=64.0, cgroup_gb=64.0)
    assert cap == 56.0
    assert source.startswith("auto")


def test_cgroup_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_read_cgroup_max_gb: walk-up min over the process hierarchy, legacy fallback."""
    # Walk-up: service cgroup is 'max', parent slice is finite -> the slice binds.
    cg = tmp_path / "cg"
    service = cg / "user.slice" / "user-1000.slice" / "user@1000.service"
    service.mkdir(parents=True)
    (service / "memory.max").write_text("max\n")
    (cg / "user.slice" / "user-1000.slice" / "memory.max").write_text(f"{64 * 10**9}\n")
    (cg / "memory.max").write_text("max\n")
    proc = tmp_path / "self_cgroup"
    proc.write_text("0::/user.slice/user-1000.slice/user@1000.service\n")
    monkeypatch.setattr(turbop, "CGROUP_ROOT", str(cg))
    monkeypatch.setattr(turbop, "PROC_SELF_CGROUP", str(proc))
    monkeypatch.setattr(turbop, "CGROUP_MEMORY_MAX", str(tmp_path / "missing"))
    monkeypatch.setattr(turbop, "CGROUP_V1_LIMIT", str(tmp_path / "missing_v1"))
    assert _read_cgroup_max_gb() == pytest.approx(64.0, abs=1e-6)

    # Deeper hierarchy: nearest finite limit wins when nested values differ.
    (service / "memory.max").write_text(f"{16 * 10**9}\n")
    assert _read_cgroup_max_gb() == pytest.approx(16.0, abs=1e-6)

    # All 'max' -> inf (unlimited).
    (service / "memory.max").write_text("max\n")
    (cg / "user.slice" / "user-1000.slice" / "memory.max").write_text("max\n")
    assert math.isinf(_read_cgroup_max_gb())

    # Legacy fallback: unreadable /proc/self/cgroup -> fixed v2/v1 paths.
    monkeypatch.setattr(turbop, "PROC_SELF_CGROUP", str(tmp_path / "nope"))
    f = tmp_path / "memory.max"
    f.write_text("max\n")
    f1 = tmp_path / "limit"
    f1.write_text(str(16 * 10**9))
    monkeypatch.setattr(turbop, "CGROUP_MEMORY_MAX", str(f))
    monkeypatch.setattr(turbop, "CGROUP_V1_LIMIT", str(f1))
    assert _read_cgroup_max_gb() == pytest.approx(16.0, abs=1e-6)


def test_physical_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    """_read_physical_gb parses MemTotal kB from /proc/meminfo."""
    monkeypatch.setattr(turbop, "MEMINFO_PATH", "/dev/null")
    with pytest.raises(ValueError, match="MemTotal"):
        _read_physical_gb()


# ---------------------------------------------------------------------------
# Runtime RSS backstop
# ---------------------------------------------------------------------------


def test_rss_backstop_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    """First cross of budget x 0.95 logs a single [warn]; later checks are quiet."""
    bs = RssBackstop(budget_gb=1.0e-5)  # 10 MB budget: this process is over it
    with caplog.at_level(logging.WARNING, logger="ppgrid.turbop"):
        assert bs.check() is True
        assert bs.check() is False  # single [warn]
        assert bs.check() is False
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    assert warns[0].getMessage().startswith("[warn] turbo: rss")


def test_rss_backstop_quiet_under_limit(caplog: pytest.LogCaptureFixture) -> None:
    """Under the limit: no warn, check returns False."""
    bs = RssBackstop(budget_gb=1.0e6)  # 1 PB budget: never crossed
    with caplog.at_level(logging.WARNING, logger="ppgrid.turbop"):
        assert bs.check() is False
    assert caplog.records == []
