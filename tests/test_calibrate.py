"""Tests for ppgrid.calibrate."""

import numpy as np
import pytest

from ppgrid.calibrate import (
    PercentileTransform,
    choose_transform,
    make_transform,
    transforms,
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
    orig = cal._fit_predict  # ruff: ignore[private-member-access]

    def spy(
        xx: np.ndarray,
        yy: np.ndarray,
        t: np.ndarray,
        train: np.ndarray,
        res: float,
        levels: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        seen.append(res)
        return orig(xx, yy, t, train, res, levels)

    monkeypatch.setattr(cal, "_fit_predict", spy)
    cal.blocked_cv_skill(x, y, tv, res=1000.0, block_km=100.0)

    assert seen, "CV did not run any folds"
    expected = max(1000.0, max(float(np.ptp(x)), float(np.ptp(y))) / 4096.0)
    np.testing.assert_allclose(seen, expected)
    assert min(seen) > 1000.0, f"grid resolution not clamped: {seen[:3]}"
