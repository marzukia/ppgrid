"""Tests for ppgrid.calibrate."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from ppgrid.calibrate import (
    PercentileTransform,
    blocked_cv_skill,
    choose_transform,
    make_transform,
    transforms,
    validate_calibration,
)


def test_percentile_transform_round_trip() -> None:
    """fwd(inv(v)) ~ v and inv(fwd(v)) ~ v after fitting."""
    v = np.random.default_rng(0).exponential(scale=10, size=1000).astype(np.float64)
    t = PercentileTransform().fit(v)

    # fwd then inv should recover original values approximately
    p = t.fwd(v)
    recovered = t.inv(p)
    np.testing.assert_allclose(recovered, v, rtol=0.02)

    # inv then fwd should recover percentiles approximately
    p2 = t.fwd(v)
    recovered_p = t.fwd(t.inv(p2))
    np.testing.assert_allclose(recovered_p, p2, rtol=0.05)


def test_choose_transform() -> None:
    """choose_transform returns a valid transform."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1e5, size=500)
    y = rng.normal(0, 1e5, size=500)
    values = rng.exponential(10, size=500).astype(np.float64)

    best, results = choose_transform(x, y, values)
    assert best is not None
    assert len(results) > 0
    assert best.name in {"identity", "log10", "sqrt", "percentile"}


def test_make_transform_error() -> None:
    """make_transform raises ValueError on unknown name."""
    with pytest.raises(ValueError, match="Unknown transform"):
        make_transform({"name": "foobar"})


def test_transforms_factory() -> None:
    """transforms() returns fresh instances each call."""
    list1 = transforms()
    list2 = transforms()
    assert len(list1) == len(list2)
    # Instances should be different objects
    assert list1 is not list2
    assert list1[0] is not list2[0]


def test_blocked_cv_skill_clamps_grid_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Large extents must not bin into an unbounded 1 km grid (OOM, issue #2).

    A ~40,000 km extent at res=1 km is a 40,960 x 40,960 cell grid. The clamp
    keeps the grid at <= 4096 cells per axis, so the resolution used by the
    CV binning must be >= extent / 4096.
    """
    import ppgrid.calibrate as cal

    rng = np.random.default_rng(0)
    n = 2000
    x = rng.uniform(0.0, 4.0e7, n)
    y = rng.uniform(0.0, 4.0e7, n)
    tv = rng.normal(0.0, 1.0, n)

    seen: list[float] = []
    sats: list[float] = []
    orig = cal._fit_predict  # ruff: ignore[private-member-access]

    def spy(
        xx: np.ndarray,
        yy: np.ndarray,
        t: np.ndarray,
        train: np.ndarray,
        res: float,
        levels: int,
        saturation: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        seen.append(res)
        sats.append(saturation)
        return orig(xx, yy, t, train, res, levels, saturation=saturation)

    monkeypatch.setattr(cal, "_fit_predict", spy)
    cal.blocked_cv_skill(x, y, tv, res=1000.0, block_km=100.0)

    assert seen, "CV did not run any folds"
    expected = max(1000.0, max(float(np.ptp(x)), float(np.ptp(y))) / 4096.0)
    np.testing.assert_allclose(seen, expected)
    assert min(seen) > 1000.0, f"grid resolution not clamped: {seen[:3]}"


def test_percentile_flat_input_maps_to_50th_percentile() -> None:
    """#24: a constant column is degenerate for the staircase (q[0] == c, so fwd left-clamps to 0).

    It must map to the 50th percentile and round-trip.
    """
    for c in (42.0, 0.0, -3.5, 1e9):
        t = PercentileTransform().fit(np.full(100, c))
        p = t.fwd(np.array([c]))
        np.testing.assert_allclose(p, [50.0], atol=1e-9)
        v = t.inv(p)
        np.testing.assert_allclose(v, [c], rtol=1e-12, atol=1e-9)


def test_percentile_single_point_maps_to_50th_percentile() -> None:
    """#24: a one-point dataset is flat too."""
    t = PercentileTransform().fit(np.array([7.0]))
    p = t.fwd(np.array([7.0]))
    np.testing.assert_allclose(p, [50.0], atol=1e-9)
    np.testing.assert_allclose(t.inv(p), [7.0], rtol=1e-12, atol=1e-9)


def test_blocked_cv_skill_degenerate_targets() -> None:
    """#14: empty folds and zero-variance targets report 0, not NaN/-inf.

    No RuntimeWarnings are emitted.
    """
    y = np.zeros(6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # Zero-variance target with valid predictions (multi-block extent).
        x = np.array([0.0, 10000.0, 20000.0, 60000.0, 70000.0, 80000.0])
        skill, rows = blocked_cv_skill(x, y, np.full(6, 42.0), block_km=50.0, res=1000.0)
        assert skill == 0.0
        assert rows == []
        # Extent smaller than one CV block: every fold is skipped -> empty act.
        x2 = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
        skill2, rows2 = blocked_cv_skill(x2, y, np.full(6, 42.0), block_km=50.0, res=1000.0)
        assert skill2 == 0.0
        assert rows2 == []


# ---------------------------------------------------------------------------
# Issue #79: choose_transform guards, calibration.json validation, bootstrap 0/0
# ---------------------------------------------------------------------------


def test_choose_transform_empty_values() -> None:
    """#79: empty values raise a friendly ValueError (was a raw zero-size minimum error)."""
    with pytest.raises(ValueError, match="empty values"):
        choose_transform(np.empty(0), np.empty(0), np.empty(0))


def test_choose_transform_all_nan_values() -> None:
    """#79: every candidate rejected (non-finite input) -> ValueError naming the cause (was IndexError)."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1e5, 500)
    y = rng.normal(0, 1e5, 500)
    with pytest.raises(ValueError, match="non-finite input"):
        choose_transform(x, y, np.full(500, np.nan))


def test_validate_calibration_accepts_well_formed() -> None:
    """#79: a file the pipeline itself wrote must pass validation."""
    q = np.linspace(0.0, 10.0, PercentileTransform.NQ).tolist()
    validate_calibration(
        {
            "transform": "percentile",
            "transform_state": {"name": "percentile", "quantiles": q},
            "percentile_quantiles": q,
            "cap_km": 20.0,
        },
        "cal.json",
    )
    validate_calibration({"transform": "identity", "cap_km": 20.0}, "cal.json")
    validate_calibration({"transform_state": {"name": "identity"}, "cap_km": 20.0}, "cal.json")


@pytest.mark.parametrize(
    ("cal", "match"),
    [
        ({"transform": "percentile", "transform_state": {"name": "percentile"}}, "transform_state.quantiles"),
        ({"transform": "bogus"}, "bogus"),
        ({"transform_state": {"name": "percentile", "quantiles": [0.0, 1.0]}}, "transform_state.quantiles"),
        ({"percentile_quantiles": [0.0, 1.0]}, "percentile_quantiles"),
        ({"percentile_quantiles": "nope"}, "percentile_quantiles"),
        ({"percentile_quantiles": [0.0] * 1000 + ["x"]}, "not a finite number"),
        ({"transform": "identity", "transform_state": "nope"}, "transform_state must be a JSON object"),
        ([1, 2, 3], "top level must be a JSON object"),
    ],
)
def test_validate_calibration_rejects(tmp_path: Path, cal: object, match: str) -> None:
    """#79: each malformed field is one friendly ValueError naming file and field."""
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(cal))
    with pytest.raises(ValueError, match=match):
        validate_calibration(cal, str(path))
    with pytest.raises(ValueError, match=r"bad\.json"):
        validate_calibration(cal, str(path))


def test_bootstrap_zero_baseline_draws_keep_finite_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    """#79: controlled stub repro (1,600 pts, dominant mode exactly at the global mean).

    A bootstrap draw where the baseline is exactly constant (0/0) is skipped,
    not NaN: no RuntimeWarning, finite CI, and the CI keeps the point skill.
    """
    import ppgrid.calibrate as cal

    n = 1_600
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 4.0e5, n)
    y = rng.uniform(0.0, 4.0e5, n)
    tv = np.full(n, 10.0)
    # Four balanced outliers: the global mean stays exactly the dominant mode in float64.
    tv[10], tv[20] = 11.0, 9.0  # in the 1 km support half
    tv[1010], tv[1020] = 12.0, 8.0  # in the 5 km support half
    assert tv.mean() == 10.0

    def stub(
        _xx: np.ndarray,
        _yy: np.ndarray,
        t: np.ndarray,
        _train: np.ndarray,
        _res: float,
        _levels: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        sup = np.where(np.arange(len(t)) < n // 2, 1.0, 5.0)
        return t, sup  # perfect prediction: e_m == 0 everywhere

    monkeypatch.setattr(cal, "_fit_predict", stub)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        overall, rows = cal.blocked_cv_skill(x, y, tv)

    assert overall == 1.0
    assert {r["hi_km"] for r in rows} >= {2.0, 8.0}  # both populated bins survive
    for r in rows:
        assert np.isfinite(r["skill"])
        assert np.isfinite(r["ci_lo"])
        assert np.isfinite(r["ci_hi"])
        assert r["ci_lo"] == 1.0
        assert r["ci_hi"] == 1.0


def test_fill_cap_nonfinite_ci_uses_point_skill(monkeypatch: pytest.MonkeyPatch) -> None:
    """#79: a NaN ci in the first bin must not break the cap prefix loop.

    Non-finite CI falls back to the point skill, so the later healthy bins
    decide the cap (old behaviour: silent 25.0 default).
    """
    import ppgrid.calibrate as cal

    def fake_blocked(
        _xx: np.ndarray,
        _yy: np.ndarray,
        _tv: np.ndarray,
        block_km: float,  # ruff: ignore[unused-function-argument] — bound by keyword from calibrate_fill_cap
        **_kw: object,
    ) -> tuple[float, list]:
        rows = [
            {"lo_km": 0.0, "hi_km": 2.0, "n": 500, "skill": 0.7, "ci_lo": float("nan"), "ci_hi": float("nan")},
            {"lo_km": 2.0, "hi_km": 4.0, "n": 500, "skill": 0.7, "ci_lo": 0.7, "ci_hi": 0.9},
            {"lo_km": 4.0, "hi_km": 8.0, "n": 500, "skill": 0.6, "ci_lo": 0.6, "ci_hi": 0.85},
        ]
        return 0.8, rows

    monkeypatch.setattr(cal, "blocked_cv_skill", fake_blocked)
    cap, detail = cal.calibrate_fill_cap(np.empty(0), np.empty(0), np.empty(0), block_km=(100.0,))
    assert cap == 8.0  # not 25.0 (FILL_CAP_DEFAULT_KM) and not broken at the first bin
    assert detail[100.0]["cap_km"] == 8.0
# Audit #83: saturation plumbing + CV edge derivation
# ---------------------------------------------------------------------------


def test_cv_max_edge_derived_from_unresolved() -> None:
    """Top CV edge is derived from _UNRESOLVED_M (was a literal 1e9 km)."""
    from itertools import pairwise

    from ppgrid import calibrate
    from ppgrid.pullpush import _UNRESOLVED_M

    edges = calibrate._CV_EDGES  # ruff: ignore[private-member-access]
    assert edges[-1] == pytest.approx(_UNRESOLVED_M / 1000.0)  # 1e6 km
    assert all(b > a for a, b in pairwise(edges))


def test_saturation_plumbed_to_pull_push(monkeypatch: pytest.MonkeyPatch) -> None:
    """blocked_cv_skill forwards saturation to pull_push (audit #83)."""
    from collections.abc import Callable

    import ppgrid.calibrate as cal

    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(0.0, 1.0e6, n)
    y = rng.uniform(0.0, 1.0e6, n)
    tv = rng.normal(0.0, 1.0, n)

    seen: list[float] = []
    orig_pp = cal.pull_push

    def spy_pp(
        s: np.ndarray,
        c: np.ndarray,
        res: float,
        levels: int,
        upsample: Callable[[np.ndarray], np.ndarray] | None = None,
        saturation: float = 1.0,
        unresolved_m: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del upsample, unresolved_m
        seen.append(saturation)
        return orig_pp(s, c, res, levels, saturation=saturation)

    monkeypatch.setattr(cal, "pull_push", spy_pp)
    cal.blocked_cv_skill(x, y, tv, res=1000.0, block_km=100.0, saturation=1.7)

    assert len(seen) > 0, "CV did not run any folds"
    assert all(s == pytest.approx(1.7) for s in seen), f"saturation not forwarded: {seen}"
