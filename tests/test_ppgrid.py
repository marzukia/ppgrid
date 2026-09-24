"""Tests for ppgrid.pullpush."""

import numpy as np
import pytest

from ppgrid.pullpush import (
    bin_points,
    box_count,
    downsample_sum,
    pad_to_pyramid,
    pull_push,
    upsample_bilinear,
    upsample_nearest,
)


def test_downsample_sum() -> None:
    """2x2 block sum on a known 4x4 array."""
    a = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    result = downsample_sum(a)
    expected = np.array(
        [
            [1 + 2 + 5 + 6, 3 + 4 + 7 + 8],
            [9 + 10 + 13 + 14, 11 + 12 + 15 + 16],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(result, expected)


def test_upsample_nearest() -> None:
    """2x nearest neighbor upsample doubles dimensions."""
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    result = upsample_nearest(a)
    assert result.shape == (4, 4)
    expected = np.array(
        [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(result, expected)


def test_upsample_bilinear() -> None:
    """2x bilinear upsample removes blocking."""
    a = np.array([[0, 100], [0, 100]], dtype=np.float32)
    result = upsample_bilinear(a)
    assert result.shape == (4, 4)
    # Result should be smooth (no hard blocking), values between 0 and 100
    assert np.all(result >= 0)
    assert np.all(result <= 100)
    # Interior values should differ from nearest-neighbor blocks
    nearest = upsample_nearest(a)
    assert not np.allclose(result, nearest)


def test_box_count() -> None:
    """box_count returns correct counts within a radius."""
    counts = np.zeros((5, 5), dtype=np.float32)
    counts[2, 2] = 10
    result = box_count(counts, radius=1)
    # radius=1 means a 3x3 window centered at (2,2)
    assert result[2, 2] == 10
    assert result[1, 1] == 10
    assert result[1, 2] == 10
    assert result[0, 0] == 0  # outside radius


def test_box_count_float32_boundary() -> None:
    """The float32 SAT guard flips at exactly 2**24 total points.

    (1<<24)-1 total counts -> float32 fast path (returns float32).
    1<<24 total counts    -> int64 fallback (returns int64).
    Both must be bit-exact against the int64 SAT reference. The dtype
    asserts pin the threshold: moving it either way fails CI here.
    """
    from ppgrid.pullpush import _box_count_int64, box_count_banded

    rng = np.random.default_rng(7)
    n0, n1, r = 64, 64, 5
    c = rng.integers(0, 100, (n0, n1), dtype=np.int64)

    c_below = c.copy()
    c_below[0, 0] += (1 << 24) - 1 - c.sum()
    assert c_below.sum() == (1 << 24) - 1

    c_at = c_below.copy()
    c_at[0, 0] += 1
    assert c_at.sum() == 1 << 24

    ref_below = _box_count_int64(c_below, r)
    ref_at = _box_count_int64(c_at, r)
    assert ref_below[0, 0] > 1_000_000  # non-trivial window structure

    below = box_count(c_below, r)
    at = box_count(c_at, r)
    # Pins which path each total takes (threshold regression guard).
    assert below.dtype == np.float32, "below 2**24 must take the float32 fast path"
    assert at.dtype == np.int64, "at 2**24 must take the int64 fallback"
    # The two paths give identical counts, and both match the reference.
    np.testing.assert_array_equal(below, ref_below)
    np.testing.assert_array_equal(at, ref_at)

    # box_count_banded shares the threshold; value-check it at both sides.
    out = np.empty((n0, n1), dtype=bool)
    box_count_banded(c_below, r, out)
    np.testing.assert_array_equal(out, ref_below > 0)
    box_count_banded(c_at, r, out)
    np.testing.assert_array_equal(out, ref_at > 0)


def test_bin_points() -> None:
    """bin_points scatters points correctly into sum/count grids."""
    ix = np.array([0, 0, 1, 2], dtype=np.int64)
    iy = np.array([0, 0, 1, 0], dtype=np.int64)
    values = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    s, c = bin_points(ix, iy, values, nx=3, ny=2)
    # Cell (0,0) has two points: 10+20=30 sum, count=2
    assert s[0, 0] == 30.0
    assert c[0, 0] == 2
    # Cell (1,1) has one point: 30 sum, count=1
    assert s[1, 1] == 30.0
    assert c[1, 1] == 1
    # Cell (2,0) has one point: 40 sum, count=1
    assert s[2, 0] == 40.0
    assert c[2, 0] == 1


def test_pad_to_pyramid() -> None:
    """pad_to_pyramid returns a multiple of 2**levels."""
    assert pad_to_pyramid(10, levels=3) == 16  # 2**3 = 8, ceil(10/8)*8 = 16
    assert pad_to_pyramid(16, levels=3) == 16  # already a multiple
    assert pad_to_pyramid(17, levels=3) == 24  # ceil(17/8)*8 = 24
    assert pad_to_pyramid(1, levels=2) == 4  # 2**2 = 4


def test_pull_push_basic() -> None:
    """pull_push returns finite values where counts exist."""
    s = np.zeros((8, 8), dtype=np.float32)
    c = np.zeros((8, 8), dtype=np.float32)
    s[4, 4] = 100.0
    c[4, 4] = 1.0
    val, sup = pull_push(s, c, res=500.0, levels=2)
    assert val.shape == (8, 8)
    assert sup.shape == (8, 8)
    # The cell with data should have a finite value
    assert np.isfinite(val[4, 4])
    assert np.isfinite(sup[4, 4])


def test_pull_push_dimension_error() -> None:
    """pull_push raises ValueError on odd dimensions."""
    s = np.zeros((7, 7), dtype=np.float32)
    c = np.zeros((7, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="divisible"):
        pull_push(s, c, res=500.0, levels=2)


def test_smooth3_size_1_guard() -> None:
    """_smooth3 handles size-1 axes without error."""
    from ppgrid.pullpush import _smooth3

    # Single row
    a1 = np.array([[1.0, 2.0, 3.0]])
    r1 = _smooth3(a1)
    assert r1.shape == a1.shape
    assert np.isfinite(r1).all()

    # Single column
    a2 = np.array([[1.0], [2.0], [3.0]])
    r2 = _smooth3(a2)
    assert r2.shape == a2.shape
    assert np.isfinite(r2).all()

    # Single element
    a3 = np.array([[5.0]])
    r3 = _smooth3(a3)
    assert r3.shape == a3.shape
    assert r3[0, 0] == 5.0
