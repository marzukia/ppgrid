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

from collections.abc import Callable

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


def box_count(counts: np.ndarray, radius: int) -> np.ndarray:
    """Count points within a (2r+1)^2 window via a summed-area table.

    Cost is independent of r -- a 100km radius costs the same as 1km. This is
    the exact answer to "is there data within `cap` of this cell?", which is a
    different question from "what spatial scale supports this estimate?".
    Deriving the mask from the pyramid conflates the two and makes coverage
    depend on pyramid depth.

    Returns:
        Box-count array with same shape as input.

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

    # Seed at the coarsest level
    val = sums[-1] / np.maximum(counts[-1], 1e-9)
    sup = np.where(
        counts[-1] > 0,
        np.float32(res * (1 << levels)),
        np.float32(unresolved_m),
    ).astype(np.float32)

    # Push: descend, blending local estimate against upsampled parent
    for k in range(levels - 1, -1, -1):
        c = counts[k]
        a = np.minimum(c / saturation, 1.0).astype(np.float32)
        local = sums[k] / np.maximum(c, 1e-9)
        val = a * local + (1.0 - a) * upsample(val)
        sup = a * np.float32(res * (1 << k)) + (1.0 - a) * upsample(sup)

    return val, sup


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


def pad_to_pyramid(n: int, levels: int) -> int:
    """Pad dimension to be divisible by 2**levels.

    Returns:
        Padded dimension.

    """
    step = 1 << levels
    return ((n + step - 1) // step) * step
