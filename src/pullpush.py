"""
Pull-push (mipmap) scattered-data interpolation for large point sets.

Replaces IDW: exact IDW is O(N*M) and intractable at continent scale.
Pull-push is O(M) and independent of N.

Core identity: IDW over grid-snapped points is a normalised convolution,
    z = (S * K) / (C * K)
where S is the per-cell value sum and C the per-cell count. Pull-push
evaluates the normalisation across a mipmap pyramid so cost is independent
of fill radius.

Emits a value band and a *support* band (effective spatial scale of the
estimate, in metres) which drives honest opacity / masking downstream.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def downsample_sum(a: np.ndarray) -> np.ndarray:
    """2x2 block sum. Sums (not means) so S and C stay consistent."""
    return a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]


def upsample_nearest(a: np.ndarray) -> np.ndarray:
    """2x nearest-neighbour. Fast, but leaves hard square edges at every
    pyramid boundary -- as vector tiles those become real polygon edges."""
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)


def _smooth3(a: np.ndarray) -> np.ndarray:
    """Separable [1,2,1]/4 filter, vectorised. Replicate boundary."""
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
    """2x nearest followed by a tent filter == bilinear. Removes the blocking."""
    return _smooth3(upsample_nearest(a).astype(np.float32))


def box_count(C: np.ndarray, r: int) -> np.ndarray:
    """Number of points within a (2r+1)^2 window, via a summed-area table.

    Cost is independent of r -- a 100km radius costs the same as 1km. This is
    the exact answer to "is there data within `cap` of this cell?", which is a
    different question from "what spatial scale supports this estimate?".
    Deriving the mask from the pyramid conflates the two and makes coverage
    depend on pyramid depth.
    """
    n0, n1 = C.shape
    cs = np.pad(np.cumsum(C, axis=0, dtype=np.float32), ((1, 0), (0, 0)))
    lo = np.clip(np.arange(n0) - r, 0, n0)
    hi = np.clip(np.arange(n0) + r + 1, 0, n0)
    a = cs[hi] - cs[lo]
    cs = np.pad(np.cumsum(a, axis=1, dtype=np.float32), ((0, 0), (1, 0)))
    lo = np.clip(np.arange(n1) - r, 0, n1)
    hi = np.clip(np.arange(n1) + r + 1, 0, n1)
    return cs[:, hi] - cs[:, lo]


def pull_push(
    S: np.ndarray,
    C: np.ndarray,
    res: float,
    levels: int,
    upsample: Callable[[np.ndarray], np.ndarray] = upsample_bilinear,
    saturation: float = 1.0,
    unresolved_m: float = 1e9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    S : per-cell sum of (transformed) values
    C : per-cell point count
    res : cell size in metres
    levels : pyramid depth; max fill reach is res * 2**levels
    saturation : counts needed for a cell to be fully self-trusting.
        >1 shrinks thin cells toward the coarser (more reliable) estimate.
    unresolved_m : support assigned to cells with no data anywhere in their
        pyramid ancestry. Must be a large constant, NOT res*2**levels --
        otherwise the support band (and therefore the mask) depends on the
        pyramid depth rather than on the data, and the same cap yields
        different coverage at different resolutions.

    Returns (value, support_m).
    """
    Ss: list[np.ndarray] = [S]
    Cs: list[np.ndarray] = [C]
    for _ in range(levels):
        Ss.append(downsample_sum(Ss[-1]))
        Cs.append(downsample_sum(Cs[-1]))

    # Seed at the coarsest level
    V = Ss[-1] / np.maximum(Cs[-1], 1e-9)
    R = np.where(Cs[-1] > 0, np.float32(res * (1 << levels)),
                 np.float32(unresolved_m)).astype(np.float32)

    # Push: descend, blending local estimate against upsampled parent
    for k in range(levels - 1, -1, -1):
        c = Cs[k]
        a = np.minimum(c / saturation, 1.0).astype(np.float32)
        local = Ss[k] / np.maximum(c, 1e-9)
        V = a * local + (1.0 - a) * upsample(V)
        R = a * np.float32(res * (1 << k)) + (1.0 - a) * upsample(R)

    return V, R


def bin_points(
    ix: np.ndarray,
    iy: np.ndarray,
    values: np.ndarray,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter points into sum and count grids indexed [easting, northing]."""
    key = ix.astype(np.int64) * ny + iy.astype(np.int64)
    S = np.bincount(key, weights=values, minlength=nx * ny)
    C = np.bincount(key, minlength=nx * ny)
    return (S.reshape(nx, ny).astype(np.float32),
            C.reshape(nx, ny).astype(np.float32))


def pad_to_pyramid(n: int, levels: int) -> int:
    step = 1 << levels
    return ((n + step - 1) // step) * step
