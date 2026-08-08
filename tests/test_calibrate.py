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
    assert best.name in ("identity", "log10", "sqrt", "percentile")


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
