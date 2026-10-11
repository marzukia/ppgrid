"""Bit-identity regression tests for the box_count_mt buffer-reuse passes (perf wave 04).

Pass 2 of ``box_count_mt`` used to be transpose -> contiguous cumsum ->
transpose back into a fresh (n0+1, n1) array: two full strided copies plus
four extra fresh-buffer first-touches per run. The window sums and the
exclusive row prefix now share one (n0+1, n1) workspace, and numpy scans it
in place along axis 0. Both forms compute the identical addition sequence per
element (q[i+1, j] = q[i, j] + w[i, j]), so the outputs must be bit-equal to
the serial reference ``box_count`` in every regime and branch.
"""

from __future__ import annotations

import numpy as np
import pytest

from ppgrid.pullpush import box_count, box_count_mt


@pytest.mark.parametrize("n_threads", [1, 2, 5, 8])
def test_box_count_mt_wide_bit_equal(n_threads: int) -> None:
    """2r+1 <= n1 branch (the full-AU shape class)."""
    rng = np.random.default_rng(11)
    c = rng.random((300, 1200), dtype=np.float32)
    r = 50
    assert np.array_equal(box_count_mt(c, r, n_threads=n_threads), box_count(c, r))


@pytest.mark.parametrize("n_threads", [1, 2, 5, 8])
def test_box_count_mt_narrow_bit_equal(n_threads: int) -> None:
    """2r+1 > n1 branch (fancy-index window gather)."""
    rng = np.random.default_rng(12)
    c = rng.random((400, 60), dtype=np.float32)
    r = 50
    assert np.array_equal(box_count_mt(c, r, n_threads=n_threads), box_count(c, r))


@pytest.mark.parametrize("n_threads", [1, 4])
def test_box_count_mt_int64_regime_bit_equal(n_threads: int) -> None:
    """Total >= 2**24 forces the exact int64 regime."""
    rng = np.random.default_rng(13)
    c = rng.random((200, 400), dtype=np.float32) * (2.0**20)
    r = 25
    assert np.array_equal(box_count_mt(c, r, n_threads=n_threads), box_count(c, r))


def test_box_count_mt_radius_gt_n0_bit_equal() -> None:
    """Window rows clamped past the grid edges."""
    rng = np.random.default_rng(14)
    c = rng.random((128, 300), dtype=np.float32)
    r = 200
    assert np.array_equal(box_count_mt(c, r, n_threads=4), box_count(c, r))


def test_box_count_mt_fullau_proportions_bit_equal() -> None:
    """Radius 250 cells matches the full-AU 100 m / 25 km cap (wide branch)."""
    rng = np.random.default_rng(15)
    c = rng.random((1024, 2048), dtype=np.float32)
    r = 250
    assert np.array_equal(box_count_mt(c, r, n_threads=8), box_count(c, r))


def test_box_count_mt_int64_input_bit_equal() -> None:
    """Already-int64 counts (the c == counts aliasing path)."""
    rng = np.random.default_rng(16)
    c = rng.integers(0, 3, (160, 500), dtype=np.int64)
    r = 40
    assert np.array_equal(box_count_mt(c, r, n_threads=4), box_count(c, r))
