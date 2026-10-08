"""Bit-identity guards for the threaded pullpush primitives (issue #31).

Every primitive defaults to n_threads=1 and must be bit-identical to the
serial implementation (A8). The per-element math is unchanged; only
independent work is split across a thread pool. The float32 box_count regime
is order-independent because every partial sum stays below 2**24 (exact in
float32); the int64 regime is exact in any order. The descent row-chunk pass
reuses the banded descent math, whose one-row overlap makes chunks
independent.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np

from ppgrid.pullpush import (
    _descent_banded,
    _pull_push_descent,
    bin_points,
    box_count,
    box_count_mt,
    downsample_sum,
    grid_index_mt,
    upsample_bilinear,
    upsample_nearest,
)

_THREADS = 4


def _pyramid(s0: np.ndarray, c0: np.ndarray, levels: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build the pull-push pyramid (finest first), like the pipeline."""
    sums = [s0]
    counts = [c0]
    for _ in range(levels):
        sums.append(downsample_sum(sums[-1]))
        counts.append(downsample_sum(counts[-1]))
    return sums, counts


def test_box_count_mt_f32_regime() -> None:
    """16M points (< 2**24) float32 regime: mt == serial at 1 and 4 threads."""
    rng = np.random.default_rng(31)
    n = 16_000_000
    ix = rng.integers(0, 4096, n)
    iy = rng.integers(0, 4096, n)
    _, c = bin_points(ix, iy, np.ones(n), 4096, 4096)
    assert c.sum() < (1 << 24)
    for r in (33, 2048):
        ref = box_count(c, r)
        assert ref.dtype == np.float32
        for n_threads in (1, _THREADS):
            mt = box_count_mt(c, r, n_threads=n_threads)
            assert mt.dtype == np.float32
            assert np.array_equal(ref, mt), f"r={r} n_threads={n_threads}"


def test_box_count_mt_int64_regime() -> None:
    """17M points (>= 2**24) int64 regime: mt == serial at 1 and 4 threads."""
    rng = np.random.default_rng(32)
    n = 17_000_000
    ix = rng.integers(0, 4096, n)
    iy = rng.integers(0, 4096, n)
    _, c = bin_points(ix, iy, np.ones(n), 4096, 4096)
    assert c.sum() >= (1 << 24)
    for r in (33, 2048):
        ref = box_count(c, r)
        assert ref.dtype == np.int64
        for n_threads in (1, _THREADS):
            mt = box_count_mt(c, r, n_threads=n_threads)
            assert mt.dtype == np.int64
            assert np.array_equal(ref, mt), f"r={r} n_threads={n_threads}"


def test_box_count_mt_regime_flip() -> None:
    """The 2**24 total-count boundary picks the same regime as box_count."""
    rng = np.random.default_rng(33)
    n0 = n1 = r = 64
    c = rng.integers(0, 100, (n0, n1), dtype=np.int64)
    c_below = c.copy()
    c_below[0, 0] += (1 << 24) - 1 - c.sum()
    c_at = c_below.copy()
    c_at[0, 0] += 1
    for cc, dtype in ((c_below, np.float32), (c_at, np.int64)):
        ref = box_count(cc, r)
        assert ref.dtype == dtype
        for n_threads in (1, _THREADS):
            mt = box_count_mt(cc, r, n_threads=n_threads)
            assert mt.dtype == dtype
            assert np.array_equal(ref, mt), f"dtype={dtype} n_threads={n_threads}"


def test_box_count_mt_small_shapes() -> None:
    """Edge shapes, r=0, clamped radii: mt == serial at 1 and 4 threads."""
    rng = np.random.default_rng(34)
    cases = [
        (7, 3, 50, 2),
        (1, 1, 1, 0),
        (1, 1, 1, 3),
        (16, 32, 200, 40),  # r > both dims
        (300, 77, 20_000, 90),
        (128, 1, 64, 3),
    ]
    for n0, n1, npts, r in cases:
        ix = rng.integers(0, n0, npts)
        iy = rng.integers(0, n1, npts)
        _, c = bin_points(ix, iy, np.ones(npts), n0, n1)
        ref = box_count(c, r)
        for n_threads in (1, _THREADS):
            mt = box_count_mt(c, r, n_threads=n_threads)
            assert np.array_equal(ref, mt), f"shape=({n0}, {n1}) r={r} n_threads={n_threads}"


def test_descent_banded_mt_1e8_cells(tmp_path: Path) -> None:
    """1e8-cell banded descent (8 levels): threaded == serial, bit-exact.

    Pre-descends to level 2 in RAM (as the pipeline banded tier does), then
    bands levels 2 -> 0 over 10240^2 = 1.05e8 cells.
    """
    rng = np.random.default_rng(35)
    n0 = n1 = 10240
    levels = 8
    res, sat = 1000.0, 4.0
    n = 2_000_000
    ix = rng.integers(0, n0, n)
    iy = rng.integers(0, n1, n)
    w = rng.lognormal(0.0, 1.0, n)
    s0, c0 = bin_points(ix, iy, w, n0, n1)
    sums, counts = _pyramid(s0, c0, levels)
    val_in, sup_in = _pull_push_descent(sums, counts, res, levels, saturation=sat, stop_level=2)

    band = 512
    val_ref = np.zeros((n0, n1), np.float32)
    sup_ref = np.zeros((n0, n1), np.float32)
    _descent_banded(
        sums,
        counts,
        res,
        sat,
        start_level=2,
        val_in=val_in,
        sup_in=sup_in,
        out_val=val_ref,
        out_sup=sup_ref,
        level_dir=tmp_path,
        band_rows=band,
    )
    val_mt = np.zeros((n0, n1), np.float32)
    sup_mt = np.zeros((n0, n1), np.float32)
    _descent_banded(
        sums,
        counts,
        res,
        sat,
        start_level=2,
        val_in=val_in,
        sup_in=sup_in,
        out_val=val_mt,
        out_sup=sup_mt,
        level_dir=tmp_path,
        band_rows=band,
        n_threads=_THREADS,
    )
    assert np.array_equal(val_ref, val_mt)
    assert np.array_equal(sup_ref, sup_mt)


def test_descent_banded_mt_odd_bands(tmp_path: Path) -> None:
    """band_rows=3 (odd): the one-row overlap at odd r0 stays bit-exact.

    Small multi-level banded descent (4 banded levels, odd r0 bands at every
    level plus edge-clamped final bands): threaded == serial.
    """
    rng = np.random.default_rng(40)
    n0 = n1 = 128
    levels = 5
    res, sat = 50.0, 2.0
    n = 5_000
    ix = rng.integers(0, n0, n)
    iy = rng.integers(0, n1, n)
    w = rng.lognormal(0.0, 1.0, n)
    s0, c0 = bin_points(ix, iy, w, n0, n1)
    sums, counts = _pyramid(s0, c0, levels)
    val_in, sup_in = _pull_push_descent(sums, counts, res, levels, saturation=sat, stop_level=3)
    (tmp_path / "ref").mkdir()
    (tmp_path / "mt").mkdir()

    val_ref = np.zeros((n0, n1), np.float32)
    sup_ref = np.zeros((n0, n1), np.float32)
    _descent_banded(
        sums,
        counts,
        res,
        sat,
        start_level=3,
        val_in=val_in,
        sup_in=sup_in,
        out_val=val_ref,
        out_sup=sup_ref,
        level_dir=tmp_path / "ref",
        band_rows=3,
    )
    val_mt = np.zeros((n0, n1), np.float32)
    sup_mt = np.zeros((n0, n1), np.float32)
    _descent_banded(
        sums,
        counts,
        res,
        sat,
        start_level=3,
        val_in=val_in,
        sup_in=sup_in,
        out_val=val_mt,
        out_sup=sup_mt,
        level_dir=tmp_path / "mt",
        band_rows=3,
        n_threads=_THREADS,
    )
    assert np.array_equal(val_ref, val_mt)
    assert np.array_equal(sup_ref, sup_mt)


def test_pull_push_descent_mt_odd_band_rows() -> None:
    """band_rows=63 (odd): reviewer repro n0=128, full band at odd r0 (issue #41).

    A full band starting at odd r0 needs (band_rows + 1)//2 + 1 = 34 parent
    rows, i.e. 2h = band_rows + 5 upsampled rows; the buffer was sized
    band_rows + 4 and the bilinear/nearest copies hit a broadcast error on
    band [63, 126) of the 128-row level. Threaded must be bit-exact vs
    serial (n_threads=1 = full-array pass).
    """
    rng = np.random.default_rng(41)
    n0 = n1 = 128
    levels = 2
    res, sat = 50.0, 2.0
    n = 5_000
    ix = rng.integers(0, n0, n)
    iy = rng.integers(0, n1, n)
    w = rng.lognormal(0.0, 1.0, n)
    s0, c0 = bin_points(ix, iy, w, n0, n1)
    sums, counts = _pyramid(s0, c0, levels)

    val_ref, sup_ref = _pull_push_descent(sums, counts, res, levels, saturation=sat)
    for nt in (_THREADS,):
        val_mt, sup_mt = _pull_push_descent(sums, counts, res, levels, saturation=sat, n_threads=nt, band_rows=63)
        assert np.array_equal(val_ref, val_mt)
        assert np.array_equal(sup_ref, sup_mt)


def test_pull_push_descent_mt() -> None:
    """In-RAM descent: threaded row chunks == full-array pass, bit-exact."""
    rng = np.random.default_rng(36)
    n0 = n1 = 4096
    levels = 6
    res, sat = 100.0, 4.0
    n = 1_000_000
    ix = rng.integers(0, n0, n)
    iy = rng.integers(0, n1, n)
    w = rng.lognormal(0.0, 1.0, n)
    s0, c0 = bin_points(ix, iy, w, n0, n1)
    sums, counts = _pyramid(s0, c0, levels)

    val_ref, sup_ref = _pull_push_descent(sums, counts, res, levels, saturation=sat)
    val_mt, sup_mt = _pull_push_descent(sums, counts, res, levels, saturation=sat, n_threads=_THREADS, band_rows=1024)
    assert np.array_equal(val_ref, val_mt)
    assert np.array_equal(sup_ref, sup_mt)

    # n_threads=1 must run the exact serial pass (A8).
    val_1, sup_1 = _pull_push_descent(sums, counts, res, levels, saturation=sat, n_threads=1)
    assert np.array_equal(val_ref, val_1)
    assert np.array_equal(sup_ref, sup_1)


def test_pull_push_descent_mt_nearest() -> None:
    """upsample_nearest (1-row tap) also admits the threaded path."""
    rng = np.random.default_rng(38)
    n0 = n1 = 2048
    levels = 5
    n = 200_000
    ix = rng.integers(0, n0, n)
    iy = rng.integers(0, n1, n)
    w = rng.lognormal(0.0, 1.0, n)
    s0, c0 = bin_points(ix, iy, w, n0, n1)
    sums, counts = _pyramid(s0, c0, levels)

    val_ref, sup_ref = _pull_push_descent(sums, counts, 10.0, levels, upsample=upsample_nearest)
    val_mt, sup_mt = _pull_push_descent(
        sums, counts, 10.0, levels, upsample=upsample_nearest, n_threads=_THREADS, band_rows=512
    )
    assert np.array_equal(val_ref, val_mt)
    assert np.array_equal(sup_ref, sup_mt)


def test_pull_push_descent_mt_custom_upsample_fallback() -> None:
    """A custom upsample takes the full-array pass at n_threads>1, bit-exact.

    Neither nearest nor bilinear, so the threaded banded pass must not run.
    The shape recorder pins the fallback: the banded pass would call
    upsample per band slice, the full-array pass once per level on the whole
    parent array (band_rows=128 < level sizes, so a stray banded pass would
    show different shapes).
    """
    rng = np.random.default_rng(41)
    n0 = n1 = 512
    levels = 4
    res, sat = 10.0, 2.0
    n = 50_000
    ix = rng.integers(0, n0, n)
    iy = rng.integers(0, n1, n)
    w = rng.lognormal(0.0, 1.0, n)
    s0, c0 = bin_points(ix, iy, w, n0, n1)
    sums, counts = _pyramid(s0, c0, levels)

    def recording(shapes: list[tuple[int, int]]) -> Callable[[np.ndarray], np.ndarray]:
        def up(x: np.ndarray) -> np.ndarray:
            shapes.append(x.shape)
            return upsample_bilinear(x) + np.float32(1.5)

        return up

    shapes_ref: list[tuple[int, int]] = []
    val_ref, sup_ref = _pull_push_descent(sums, counts, res, levels, upsample=recording(shapes_ref), saturation=sat)

    shapes_mt: list[tuple[int, int]] = []
    val_mt, sup_mt = _pull_push_descent(
        sums,
        counts,
        res,
        levels,
        upsample=recording(shapes_mt),
        saturation=sat,
        n_threads=_THREADS,
        band_rows=128,
    )

    assert np.array_equal(val_ref, val_mt)
    assert np.array_equal(sup_ref, sup_mt)

    # Fallback pin: whole parent array, twice per level (val + sup), no band slices.
    expected: list[tuple[int, int]] = []
    for k in range(levels - 1, -1, -1):
        parent = n0 >> (k + 1)
        expected += [(parent, parent), (parent, parent)]
    assert shapes_mt == expected
    assert shapes_mt == shapes_ref


def test_grid_index_mt() -> None:
    """Cell-divide + scatter: threaded == serial pipeline code, bit-exact."""
    rng = np.random.default_rng(37)
    n = 2_000_000
    nbx, nby, bsize, res = 16, 12, 512, 10.0
    x0, y0 = 1000.0, 2000.0
    x = x0 + rng.uniform(0.0, nbx * bsize * res, n)
    y = y0 + rng.uniform(0.0, nby * bsize * res, n)
    tv = rng.lognormal(0.0, 1.0, n)

    # Serial reference: today's pipeline.grid() inline code.
    bx_ref = np.clip(((x - x0) // res // bsize).astype(np.int64), 0, nbx - 1)
    by_ref = np.clip(((y - y0) // res // bsize).astype(np.int64), 0, nby - 1)
    bid_ref = bx_ref * nby + by_ref
    order_ref = np.argsort(bid_ref, kind="stable")

    for n_threads in (1, _THREADS):
        px = np.empty(n, np.float64)
        py = np.empty(n, np.float64)
        ptv = np.empty(n, np.float64)
        bid, order = grid_index_mt(x, y, tv, x0, y0, res, bsize, nbx, nby, px, py, ptv, n_threads=n_threads)
        assert np.array_equal(bid, bid_ref)
        assert np.array_equal(order, order_ref)
        assert np.array_equal(px, x[order_ref])
        assert np.array_equal(py, y[order_ref])
        assert np.array_equal(ptv, tv[order_ref])


def test_grid_index_mt_memmap(tmp_path: Path) -> None:
    """Scatter writes through memmap band views (the pipeline's pts bands)."""
    rng = np.random.default_rng(39)
    n = 100_000
    nbx, nby, bsize, res = 8, 6, 128, 10.0
    x0, y0 = 0.0, 0.0
    x = x0 + rng.uniform(0.0, nbx * bsize * res, n)
    y = y0 + rng.uniform(0.0, nby * bsize * res, n)
    tv = rng.lognormal(0.0, 1.0, n)
    pts = np.lib.format.open_memmap(tmp_path / "pts.npy", mode="w+", dtype=np.float64, shape=(3, n))
    bid, order = grid_index_mt(x, y, tv, x0, y0, res, bsize, nbx, nby, pts[0], pts[1], pts[2], n_threads=_THREADS)
    pts.flush()
    bid_ref = np.clip(
        (np.clip(((x - x0) // res // bsize).astype(np.int64), 0, nbx - 1)) * nby
        + np.clip(((y - y0) // res // bsize).astype(np.int64), 0, nby - 1),
        0,
        nbx * nby - 1,
    )
    assert np.array_equal(bid, bid_ref)
    assert np.array_equal(pts[0], x[order])
    assert np.array_equal(pts[1], y[order])
    assert np.array_equal(pts[2], tv[order])
