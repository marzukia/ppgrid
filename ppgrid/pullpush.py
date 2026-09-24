"""Pull-push (mipmap) scattered-data interpolation for large point sets.

Replaces IDW: exact IDW is O(N*M) and intractable at continent scale.
Pull-push is O(M) and independent of N.

Core identity: IDW over grid-snapped points is a normalised convolution,
    z = (s * k) / (c * k)
where s is the per-cell value sum and c the per-cell count. Pull-push
evaluates the normalisation across a mipmap pyramid so cost is independent
of fill radius.

Emits a value band and a *support* band (effective spatial scale of the
estimate, in metres) which drives honest opacity / masking downstream.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path

import numpy as np


def downsample_sum(a: np.ndarray) -> np.ndarray:
    """2x2 block sum. Sums (not means) so s and c stay consistent.

    Returns:
        Downsampled array with half the dimensions.

    """
    return a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]


def upsample_nearest(a: np.ndarray) -> np.ndarray:
    """2x nearest-neighbour upsample.

    Fast, but leaves hard square edges at every pyramid boundary -- as
    vector tiles those become real polygon edges.

    Returns:
        Upsampled array with double the dimensions.

    """
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)


def _smooth3(a: np.ndarray) -> np.ndarray:
    """Separable [1,2,1]/4 filter, vectorised. Replicate boundary.

    Returns:
        Smoothed array with same shape as input.

    """
    if a.shape[0] == 1:
        return a.copy()
    if a.shape[1] == 1:
        return a.copy()

    b = np.empty_like(a)
    b[1:-1] = 0.25 * (a[:-2] + 2.0 * a[1:-1] + a[2:])
    b[0] = 0.25 * (3.0 * a[0] + a[1])
    b[-1] = 0.25 * (a[-2] + 3.0 * a[-1])
    c = np.empty_like(b)
    c[:, 1:-1] = 0.25 * (b[:, :-2] + 2.0 * b[:, 1:-1] + b[:, 2:])
    c[:, 0] = 0.25 * (3.0 * b[:, 0] + b[:, 1])
    c[:, -1] = 0.25 * (b[:, -2] + 3.0 * b[:, -1])
    return c


def upsample_bilinear(a: np.ndarray) -> np.ndarray:
    """2x nearest followed by a tent filter == bilinear. Removes the blocking.

    Returns:
        Upsampled and smoothed array with double the dimensions.

    """
    return _smooth3(upsample_nearest(a).astype(np.float32))


def _box_count_int64(counts: np.ndarray, radius: int) -> np.ndarray:
    """Exact int64 summed-area-table box count (reference / fallback path).

    Returns:
        Box-count array with same shape as input, dtype int64.

    """
    n0, n1 = counts.shape
    cs = np.pad(np.cumsum(counts, axis=0, dtype=np.int64), ((1, 0), (0, 0)))
    lo = np.clip(np.arange(n0) - radius, 0, n0)
    hi = np.clip(np.arange(n0) + radius + 1, 0, n0)
    a = cs[hi] - cs[lo]
    cs = np.pad(np.cumsum(a, axis=1, dtype=np.int64), ((0, 0), (1, 0)))
    lo = np.clip(np.arange(n1) - radius, 0, n1)
    hi = np.clip(np.arange(n1) + radius + 1, 0, n1)
    return cs[:, hi] - cs[:, lo]


def box_count(counts: np.ndarray, radius: int) -> np.ndarray:
    """Count points within a (2r+1)^2 window.

    Cost is independent of r -- a 100km radius costs the same as 1km. This is
    the exact answer to "is there data within `cap` of this cell?", which is a
    different question from "what spatial scale supports this estimate?".
    Deriving the mask from the pyramid conflates the two and makes coverage
    depend on pyramid depth.

    Uses a 4-pass separable box sum (2 cumsum + 2 window-subtract) in float32.
    Counts are non-negative integers, so every partial sum is at most the
    total point count; while that stays below 2**24 (16.7M) the float32 sums
    and differences are exact, making this bit-identical to the int64 SAT.
    At 2**24 or more total points it falls back to the exact int64 path.

    Returns:
        Box-count array with same shape as input. float32 (exact integer
        values) on the fast path, int64 on the fallback.

    """
    if radius == 0:
        return counts.astype(np.int64)
    if counts.sum() < (1 << 24):
        n0, n1 = counts.shape
        c = counts.astype(np.float32, copy=False)
        # Exclusive row prefix: p[i, j] = sum(c[i, :j]), one cumsum + one copy
        p = np.empty((n0, n1 + 1), np.float32)
        p[:, 0] = 0.0
        p[:, 1:] = np.cumsum(c, axis=1)
        lo = np.clip(np.arange(n1) - radius, 0, n1)
        hi = np.clip(np.arange(n1) + radius + 1, 0, n1)
        w = p[:, hi] - p[:, lo]
        # Exclusive column prefix over the row-window sums
        q = np.empty((n0 + 1, n1), np.float32)
        q[0] = 0.0
        q[1:] = np.cumsum(w, axis=0)
        lo = np.clip(np.arange(n0) - radius, 0, n0)
        hi = np.clip(np.arange(n0) + radius + 1, 0, n0)
        return q[hi] - q[lo]
    return _box_count_int64(counts, radius)


def box_count_banded(counts: np.ndarray, radius: int, out: np.ndarray, band_rows: int = 4096) -> None:
    """Row-banded box count, writing a boolean mask into `out` in place.

    Same window as box_count; each band re-derives the row prefixes over
    [b0-radius, b1+radius) so interior rows get exactly the same window sums.
    On the float32 fast path every partial sum is below 2**24, so the sums
    are exact and the banding is bit-identical to the full pass. At 2**24 or
    more total points it falls back to the exact int64 SAT (full arrays).

    The window subtractions use contiguous slices (the index offsets are
    linear in the row/column position except at the clamped edges), avoiding
    the per-element gathers of the full pass.

    """
    n0, n1 = counts.shape
    if radius == 0:
        out[:] = counts > 0
        return
    if counts.sum() >= (1 << 24):
        out[:] = _box_count_int64(counts, radius) > 0
        return
    small_cols = n1 <= 2 * radius
    if small_cols:
        lo_idx = np.clip(np.arange(n1) - radius, 0, n1)
        hi_idx = np.clip(np.arange(n1) + radius + 1, 0, n1)
    for b0 in range(0, n0, band_rows):
        b1 = min(n0, b0 + band_rows)
        e0 = max(0, b0 - radius)
        e1 = min(n0, b1 + radius)
        seg = counts[e0:e1]
        p = np.empty((e1 - e0, n1 + 1), np.float32)
        p[:, 0] = 0.0
        p[:, 1:] = np.cumsum(seg, axis=1)
        if small_cols:
            w = p[:, hi_idx] - p[:, lo_idx]
        else:
            w = np.empty((e1 - e0, n1), np.float32)
            w[:, 0:radius] = p[:, radius + 1 : 2 * radius + 1]
            w[:, radius : n1 - radius] = p[:, 2 * radius + 1 : n1 + 1] - p[:, 0 : n1 - 2 * radius]
            w[:, n1 - radius :] = p[:, n1 : n1 + 1] - p[:, n1 - 2 * radius : n1 - radius]
        q = np.empty((e1 - e0 + 1, n1), np.float32)
        q[0] = 0.0
        q[1:] = np.cumsum(w, axis=0)
        # Row-window subtraction. alo(i) = clip(i-r, 0, n0) is constant for
        # i < r and linear for i >= r; ahi(i) = clip(i+r+1, 0, n0) is linear
        # for i < n0-r and constant for i >= n0-r. Cutting at those points
        # leaves contiguous slices (or one broadcast row) per segment.
        cuts = sorted({b0, b1, radius, n0 - radius})
        for a, b in itertools.pairwise(cuts):
            if b <= a or a < b0 or b > b1:
                continue
            alo = q[a - radius - e0 : b - radius - e0] if a >= radius else q[0]
            ahi = q[a + radius + 1 - e0 : b + radius + 1 - e0] if b <= n0 - radius else q[n0 - e0]
            out[a:b] = (ahi - alo) > 0


def _pull_push_descent(
    sums: list[np.ndarray],
    counts: list[np.ndarray],
    res: float,
    levels: int,
    upsample: Callable[[np.ndarray], np.ndarray] = upsample_bilinear,
    saturation: float = 1.0,
    unresolved_m: float = 1e9,
    stop_level: int = 0,
    *,
    free_levels: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Descend a prebuilt pull-push pyramid (coarsest -> finest).

    Args:
        sums: Pyramid of per-cell value sums, sums[0] finest, sums[levels] coarsest.
        counts: Pyramid of per-cell point counts, same layout as sums.
        res: Cell size in metres.
        levels: Pyramid depth; len(sums) must be levels + 1.
        upsample: Upsampling function. Defaults to bilinear.
        saturation: Counts needed for a cell to be fully self-trusting.
            >1 shrinks thin cells toward the coarser (more reliable) estimate.
        unresolved_m: Support assigned to cells with no data anywhere in their
            pyramid ancestry.
        stop_level: Descend down to (but not including) this level. 0 = full
            descent to the finest grid.
        free_levels: Release pyramid levels once consumed (saves memory on
            large grids). Mutates the input lists (sets entries to None).

    Returns:
        Tuple of interpolated value grid and support grid in metres, at
        level stop_level.

    """
    # Seed at the coarsest level
    val = sums[-1] / np.maximum(counts[-1], 1e-9)
    sup = np.where(
        counts[-1] > 0,
        np.float32(res * (1 << levels)),
        np.float32(unresolved_m),
    ).astype(np.float32)

    # Push: descend, blending local estimate against upsampled parent.
    # NaN-safe: saturated cells (a >= 1.0) take `local` directly, so a NaN
    # parent multiplied by zero (0*NaN = NaN) cannot corrupt them.
    for k in range(levels - 1, stop_level - 1, -1):
        c = counts[k]
        a = np.minimum(c / saturation, 1.0).astype(np.float32)
        local = sums[k] / np.maximum(c, 1e-9)
        parent = upsample(val)
        val = np.where(a >= 1.0, local, a * local + (1.0 - a) * parent)
        sup = a * np.float32(res * (1 << k)) + (1.0 - a) * upsample(sup)
        if free_levels:
            sums[k + 1] = None
            counts[k + 1] = None

    return val, sup


def _descent_banded(
    sums: list[np.ndarray],
    counts: list[np.ndarray],
    res: float,
    saturation: float,
    start_level: int,
    val_in: np.ndarray,
    sup_in: np.ndarray,
    out_val: np.ndarray,
    out_sup: np.ndarray,
    level_dir: Path,
    band_rows: int = 1024,
) -> None:
    """Compute the descent from level start_level down to level 0, row-banded.

    Bounds memory on large grids: each band reads a one-row overlap of the
    coarser level, so every tent-filter tap inside the band sees the same
    elements as the full-array pass (band edge rows are never read) and the
    result is bit-identical to the un-banded descent. Intermediate levels
    (start_level-1 .. 1) are written to memmap files in level_dir; level 0
    goes to out_val / out_sup (memmaps or arrays).

    """
    val = val_in
    sup = sup_in
    for k in range(start_level - 1, -1, -1):
        if k > 0:
            out = np.lib.format.open_memmap(
                level_dir / f"_val_lvl{k}.npy", mode="w+", dtype=np.float32, shape=sums[k].shape
            )
            outs = np.lib.format.open_memmap(
                level_dir / f"_sup_lvl{k}.npy", mode="w+", dtype=np.float32, shape=counts[k].shape
            )
        else:
            out, outs = out_val, out_sup
        n0 = sums[k].shape[0]
        for r0 in range(0, n0, band_rows):
            r1 = min(n0, r0 + band_rows)
            # Coarser window: level-k rows [r0, r1) read upsample taps at
            # repeated rows [r0-1, r1] -> level-(k+1) rows [r0//2-1, r1//2].
            e0 = max(0, r0 // 2 - 1)
            e1 = min(val.shape[0], r1 // 2 + 1)
            p0 = 2 * e0
            a = np.minimum(counts[k][r0:r1] / saturation, 1.0).astype(np.float32)
            local = sums[k][r0:r1] / np.maximum(counts[k][r0:r1], 1e-9)
            pv = upsample_bilinear(val[e0:e1])[r0 - p0 : r1 - p0]
            ps = upsample_bilinear(sup[e0:e1])[r0 - p0 : r1 - p0]
            out[r0:r1] = np.where(a >= 1.0, local, a * local + (1.0 - a) * pv)
            outs[r0:r1] = a * np.float32(res * (1 << k)) + (1.0 - a) * ps
        if k > 0:
            val, sup = out, outs


def pull_push(
    sum_grid: np.ndarray,
    count_grid: np.ndarray,
    res: float,
    levels: int,
    upsample: Callable[[np.ndarray], np.ndarray] = upsample_bilinear,
    saturation: float = 1.0,
    unresolved_m: float = 1e9,
) -> tuple[np.ndarray, np.ndarray]:
    """Pull-push mipmap interpolation.

    Args:
        sum_grid: Per-cell sum of (transformed) values.
        count_grid: Per-cell point count.
        res: Cell size in metres.
        levels: Pyramid depth; max fill reach is res * 2**levels.
        upsample: Upsampling function. Defaults to bilinear.
        saturation: Counts needed for a cell to be fully self-trusting.
            >1 shrinks thin cells toward the coarser (more reliable) estimate.
        unresolved_m: Support assigned to cells with no data anywhere in their
            pyramid ancestry. Must be a large constant, NOT res*2**levels --
            otherwise the support band (and therefore the mask) depends on the
            pyramid depth rather than on the data, and the same cap yields
            different coverage at different resolutions.

    Returns:
        Tuple of interpolated value grid and support grid in metres.

    Raises:
        ValueError: If grid shapes mismatch or dimensions are not divisible
            by 2**levels.

    """
    if sum_grid.shape != count_grid.shape:
        msg = f"sum_grid shape {sum_grid.shape} != count_grid shape {count_grid.shape}"
        raise ValueError(msg)
    step = 1 << levels
    for dim in sum_grid.shape:
        if dim % step != 0:
            msg = f"Grid dimensions must be divisible by 2**levels ({step}): got shape {sum_grid.shape}"
            raise ValueError(msg)

    sums: list[np.ndarray] = [sum_grid]
    counts: list[np.ndarray] = [count_grid]
    for _ in range(levels):
        sums.append(downsample_sum(sums[-1]))
        counts.append(downsample_sum(counts[-1]))

    return _pull_push_descent(sums, counts, res, levels, upsample, saturation, unresolved_m)


def bin_points(
    ix: np.ndarray,
    iy: np.ndarray,
    values: np.ndarray,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter points into sum and count grids indexed [easting, northing].

    Args:
        ix: X indices, must be in [0, nx).
        iy: Y indices, must be in [0, ny).
        values: Values to scatter.
        nx: Grid width.
        ny: Grid height.

    Returns:
        Tuple of (sum_grid, count_grid) of shape (nx, ny).

    Raises:
        ValueError: If indices are out of range.

    """
    if ix.min() < 0 or ix.max() >= nx:
        msg = f"ix indices out of range [0, {nx}): min={ix.min()}, max={ix.max()}"
        raise ValueError(msg)
    if iy.min() < 0 or iy.max() >= ny:
        msg = f"iy indices out of range [0, {ny}): min={iy.min()}, max={iy.max()}"
        raise ValueError(msg)
    key = ix.astype(np.int64) * ny + iy.astype(np.int64)
    s = np.bincount(key, weights=values, minlength=nx * ny).reshape(nx, ny).astype(np.float32)
    c = np.bincount(key, minlength=nx * ny).reshape(nx, ny).astype(np.float32)
    return s, c


def bin_points_banded(
    s_out: np.ndarray,
    c_out: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    values: np.ndarray,
    ny: int,
) -> None:
    """Scatter points into preallocated float32 sum/count grids, row-banded.

    Fills s_out / c_out (shape (nx, ny), e.g. memmaps) in place. Each grid row
    is binned separately with np.bincount over that row's points in input
    order, so per-cell float64 accumulation order matches bin_points and the
    result is bit-identical, without materialising the full flat bincount.

    """
    w = values if values.dtype == np.float64 else values.astype(np.float64)
    order = np.argsort(ix, kind="stable")
    xs = ix[order]
    ys = iy[order]
    ws = w[order]
    bounds = np.concatenate((np.zeros(1, xs.dtype), np.flatnonzero(np.diff(xs)) + 1, np.array([xs.size], xs.dtype)))
    if bounds.size < 2 or bounds[1] == 0:
        return
    rows = xs[bounds[:-1]]
    for k in range(bounds.size - 1):
        a, b = int(bounds[k]), int(bounds[k + 1])
        r = int(rows[k])
        s_out[r] = np.bincount(ys[a:b], weights=ws[a:b], minlength=ny).astype(np.float32)
        c_out[r] = np.bincount(ys[a:b], minlength=ny).astype(np.float32)


def pad_to_pyramid(n: int, levels: int) -> int:
    """Pad dimension to be divisible by 2**levels.

    Returns:
        Padded dimension.

    """
    step = 1 << levels
    return ((n + step - 1) // step) * step
