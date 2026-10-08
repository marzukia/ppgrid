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

import ctypes
import itertools
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from ._tempfiles import _open_fresh_memmap

# float32 is exact for integers up to 2**24; beyond that, box counts can no
# longer be represented and the float32 fast path must fall back to int64.
_FLOAT32_EXACT_LIMIT = 1 << 24
# Float tolerance for "count > 0" comparisons (a mass below this is noise).
_COUNT_EPS = 1e-9
# Sentinel support (metres) for "no data anywhere in the pyramid ancestry".
_UNRESOLVED_M = 1e9

_LIBC = ctypes.CDLL(None, use_errno=True)
_MADV_NOHUGEPAGE = 15
_PAGE = os.sysconf("SC_PAGE_SIZE")
# Below this, glibc serves chunks from arena heap VMAs where a mid-VMA
# MADV_NOHUGEPAGE split has tripped glibc heap asserts (non-deterministic
# abort in the test suite). >= 16 MiB arrays are direct mmaps, where the
# advice is clean, and they are the ones whose fault cost matters.
_NOHUGE_MIN_BYTES = 16 * 1024 * 1024


def _nohuge(*arrays: np.ndarray | None) -> None:
    """Request plain 4KB pages (no THP) for fresh large arrays.

    On this box the kernel runs THP=always with synchronous defrag=always:
    the first-touch fault of a fresh multi-GB mapping tries to build a 2MB
    huge page and stalls in page compaction while the rest of the process is
    allocating (issue #39 full-AU verify: 20-40 us/page in-run vs ~1 us for
    4KB pages). Calling this right after allocation (before any first touch)
    keeps the later fills fast. Values are unaffected - only the page size.
    No-op on None/empty; safe on memmaps (shmem honours the advice too).
    """
    for a in arrays:
        if a is None or a.size == 0 or a.nbytes < _NOHUGE_MIN_BYTES:
            continue
        # numpy's buffer can start mid-page (observed +16 into its own 4KB
        # VMA, with the rest in the next VMA), and madvise wants both ends
        # page-aligned: snap the start down, round the length up.
        start = a.ctypes.data & ~(_PAGE - 1)
        length = (a.nbytes + (a.ctypes.data - start) + _PAGE - 1) & ~(_PAGE - 1)
        _LIBC.madvise(ctypes.c_void_p(start), ctypes.c_size_t(length), _MADV_NOHUGEPAGE)


def downsample_sum(a: np.ndarray) -> np.ndarray:
    """2x2 block sum. Sums (not means) so s and c stay consistent.

    Args:
        a: Input array.

    Returns:
        Downsampled array with half the dimensions.

    """
    out = np.empty((a.shape[0] // 2, a.shape[1] // 2), dtype=a.dtype)
    _nohuge(out)
    np.add(a[0::2, 0::2], a[1::2, 0::2], out=out)
    out += a[0::2, 1::2]
    out += a[1::2, 1::2]
    return out


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
    _nohuge(b)
    b[1:-1] = 0.25 * (a[:-2] + 2.0 * a[1:-1] + a[2:])
    b[0] = 0.25 * (3.0 * a[0] + a[1])
    b[-1] = 0.25 * (a[-2] + 3.0 * a[-1])
    c = np.empty_like(b)
    _nohuge(c)
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


def _smooth3_into(a: np.ndarray, out: np.ndarray, t: np.ndarray) -> None:
    """In-place separable [1,2,1]/4 filter, same formulas as _smooth3.

    All reads of `a` complete before any write to `out` (row pass, then
    column pass), so `out` may alias `a` and `t` is scratch only. Bit-
    identical to _smooth3: same per-element formulas and association; the
    only reassociation is commutative addend order (x + y == y + x exactly
    in IEEE-754).
    """
    if a.shape[0] <= 1 or a.shape[1] <= 1:
        out[: a.shape[0], : a.shape[1]] = a
        return
    # Row pass: accumulate the (1,2,1) row convolution into t, then scale
    # into out - every read of a lands in t first.
    np.multiply(2.0, a[1:-1], out=t[1:-1])
    t[1:-1] += a[:-2]
    t[1:-1] += a[2:]
    np.multiply(3.0, a[0], out=t[0])
    t[0] += a[1]
    np.multiply(3.0, a[-1], out=t[-1])
    t[-1] += a[-2]
    np.multiply(0.25, t[1:-1], out=out[1:-1])
    np.multiply(0.25, t[0], out=out[0])
    np.multiply(0.25, t[-1], out=out[-1])
    # Column pass on the row-smoothed array (now `out`): same pattern.
    np.multiply(2.0, out[:, 1:-1], out=t[:, 1:-1])
    t[:, 1:-1] += out[:, :-2]
    t[:, 1:-1] += out[:, 2:]
    np.multiply(3.0, out[:, 0], out=t[:, 0])
    t[:, 0] += out[:, 1]
    np.multiply(3.0, out[:, -1], out=t[:, -1])
    t[:, -1] += out[:, -2]
    np.multiply(0.25, t[:, 1:-1], out=out[:, 1:-1])
    np.multiply(0.25, t[:, 0], out=out[:, 0])
    np.multiply(0.25, t[:, -1], out=out[:, -1])


def _upsample_bilinear_into(src: np.ndarray, dst: np.ndarray, t: np.ndarray) -> None:
    """In-place upsample_bilinear: 2x nearest + tent, into preallocated buffers.

    Bit-identical to upsample_bilinear(src) (same per-element formulas; the
    nearest step is four strided copies instead of two np.repeat passes).
    dst and t must have shape at least (2 * src.shape[0], 2 * src.shape[1]);
    only that prefix is touched.
    """
    h, w = src.shape
    hh, ww = 2 * h, 2 * w
    up = dst[:hh, :ww]
    tt = t[:hh, :ww]
    up[0::2, 0::2] = src
    up[1::2, 0::2] = src
    up[0::2, 1::2] = src
    up[1::2, 1::2] = src
    _smooth3_into(up, up, tt)


def _band_bufs(band_rows: int, n1: int) -> dict[str, np.ndarray]:
    """Per-thread preallocated scratch for _descent_band_body (issue #39).

    Sized for the widest band (band_rows rows) at a level with n1 columns;
    a thread reuses its set across every band of the level, which cuts the
    ~9 full-band-sized per-call allocations to zero first-touch pages. The
    up/t buffers hold band_rows + 5 rows: an odd-start full band's parent
    slice is (band_rows + 1) // 2 + 1 rows, i.e. 2 * ((band_rows + 1) // 2
    + 1) = band_rows + 5 upsampled rows for odd band_rows (issue #41).
    """
    f = np.float32
    bufs = {
        "a": np.empty((band_rows, n1), f),
        "local": np.empty((band_rows, n1), f),
        "om": np.empty((band_rows, n1), f),
        "pv": np.empty((band_rows, n1), f),
        "ps": np.empty((band_rows, n1), f),
        "up": np.empty((band_rows + 5, n1), f),
        "t": np.empty((band_rows + 5, n1), f),
    }
    _nohuge(*bufs.values())
    return bufs


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
    if counts.sum() < _FLOAT32_EXACT_LIMIT:
        n0, n1 = counts.shape
        c = counts.astype(np.float32, copy=False)
        # Exclusive row prefix: p[i, j] = sum(c[i, :j]), one cumsum + one copy
        p = np.empty((n0, n1 + 1), np.float32)
        _nohuge(p)
        p[:, 0] = 0.0
        p[:, 1:] = np.cumsum(c, axis=1)
        lo = np.clip(np.arange(n1) - radius, 0, n1)
        hi = np.clip(np.arange(n1) + radius + 1, 0, n1)
        w = np.empty((n0, n1), np.float32)
        _nohuge(w)
        w[:] = p[:, hi] - p[:, lo]
        # Exclusive column prefix over the row-window sums
        q = np.empty((n0 + 1, n1), np.float32)
        _nohuge(q)
        q[0] = 0.0
        q[1:] = np.cumsum(w, axis=0)
        lo = np.clip(np.arange(n0) - radius, 0, n0)
        hi = np.clip(np.arange(n0) + radius + 1, 0, n0)
        return q[hi] - q[lo]
    return _box_count_int64(counts, radius)


def _split_bands(n: int, n_threads: int) -> list[tuple[int, int]]:
    """Split [0, n) into at most n_threads contiguous, roughly equal bands.

    Returns an empty list for n == 0 and a single band for n_threads <= 1.

    """
    if n_threads <= 1:
        return [(0, n)] if n else []
    size = -(-n // n_threads)
    return [(r0, min(n, r0 + size)) for r0 in range(0, n, size)]


def _thread_bands(bands: list[tuple[int, int]], fn: Callable[[int, int], None], n_threads: int) -> None:
    """Run fn(r0, r1) over bands, at most n_threads concurrently.

    Serial loop when n_threads <= 1 or there is a single band. numpy releases
    the GIL during the heavy ops inside fn, so the threads scale on
    arithmetic. Bands must be independent: disjoint output rows, read-only
    shared inputs.

    """
    if n_threads <= 1 or len(bands) <= 1:
        for r0, r1 in bands:
            fn(r0, r1)
        return
    with ThreadPoolExecutor(max_workers=min(n_threads, len(bands))) as ex:
        list(ex.map(lambda b: fn(b[0], b[1]), bands))


def _row_prefix_rows(p: np.ndarray, c: np.ndarray, r0: int, r1: int) -> None:
    """box_count_mt pass 1 chunk: exclusive column prefix for rows [r0, r1)."""
    # out= skips the per-chunk (rows, n1+1) cumsum temporary (issue #39);
    # the accumulation sequence per row is unchanged (bit-identical).
    np.cumsum(c[r0:r1], axis=1, dtype=p.dtype, out=p[r0:r1, 1:])


def _window_sub_rows(q: np.ndarray, out: np.ndarray, lo: np.ndarray, hi: np.ndarray, r0: int, r1: int) -> None:
    """box_count_mt pass 3 chunk: window subtraction for rows [r0, r1)."""
    out[r0:r1] = q[hi[r0:r1]] - q[lo[r0:r1]]


def box_count_mt(
    counts: np.ndarray,
    radius: int,
    n_threads: int = 1,
) -> np.ndarray:
    """Threaded box count (S3.1). Same window and dtype regimes as box_count.

    A chunked 2D scan, three passes:

    1. Row-window pass: the exclusive column prefix per row. Rows are
       independent, so the pass runs a thread pool over row chunks. Bit-exact:
       each row's cumsum is the identical sequential op, and while the total
       stays below 2**24 the float32 partial sums are exactly representable
       (the int64 regime is exact in any order).
    2. Column-window pass: the row prefix of the window sums, computed as a
       contiguous cumsum over the transposed array instead of the strided
       axis-0 cumsum (same addition sequence per element; the layout fix is
       most of the speedup).
    3. Final window subtract: elementwise, parallel over row chunks.

    n_threads=1 is bit-identical to box_count (A8). Peak transient memory,
    tracemalloc-measured: float32 regime ~2 full arrays vs box_count's ~4
    (each buffer is freed as soon as the next pass owns the data); int64
    regime on float32 input (the pipeline case) ~5 vs ~5, parity, because
    the int64 cast is held across pass 1 and the fancy-index copies set the
    peak.

    Returns:
        Box-count array with the same shape as input. float32 (exact integer
        values) below 2**24 total points, int64 at or above.

    """
    if radius == 0:
        return counts.astype(np.int64)
    dtype = np.float32 if counts.sum() < _FLOAT32_EXACT_LIMIT else np.int64
    n0, n1 = counts.shape
    if counts.dtype == dtype:
        c = counts
    else:
        c = np.empty(counts.shape, dtype)
        _nohuge(c)
        c[:] = counts

    # Pass 1: exclusive column prefix per row, threaded over row chunks.
    p = np.empty((n0, n1 + 1), dtype)
    _nohuge(p)
    p[:, 0] = 0
    _thread_bands(_split_bands(n0, n_threads), partial(_row_prefix_rows, p, c), n_threads)

    lo = np.clip(np.arange(n1) - radius, 0, n1)
    hi = np.clip(np.arange(n1) + radius + 1, 0, n1)
    # Window sums into one preallocated buffer (issue #39). The index offsets
    # are linear in the column position except at the clamped edges, so the
    # interior and both edges are contiguous slice subtractions - no
    # fancy-index gather temporaries. Bit-identical to p[:, hi] - p[:, lo]:
    # same elementwise subtractions, same order.
    w = np.empty((n0, n1), dtype)
    _nohuge(w)
    r = radius
    if 2 * r + 1 <= n1:
        np.subtract(p[:, r + 1 : 2 * r + 1], p[:, :1], out=w[:, :r])
        np.subtract(p[:, 2 * r + 1 :], p[:, : n1 - 2 * r], out=w[:, r : n1 - r])
        np.subtract(p[:, n1 : n1 + 1], p[:, n1 - 2 * r : n1 - r], out=w[:, n1 - r :])
    else:
        w = p[:, hi] - p[:, lo]
    del p

    # Pass 2: row prefix of the window sums, contiguous via the transpose.
    wt = np.empty((n1, n0), dtype)
    _nohuge(wt)
    wt[:] = w.T
    del w
    qt = np.empty((n1, n0 + 1), dtype)
    _nohuge(qt)
    qt[:, 0] = 0
    qt[:, 1:] = np.cumsum(wt, axis=1, dtype=dtype)
    del wt
    q = np.empty((n0 + 1, n1), dtype)
    _nohuge(q)
    q[:] = qt.T
    del qt

    # Pass 3: window subtract, elementwise over row chunks.
    lo = np.clip(np.arange(n0) - radius, 0, n0)
    hi = np.clip(np.arange(n0) + radius + 1, 0, n0)
    out = np.empty((n0, n1), dtype)
    _nohuge(out)
    _thread_bands(_split_bands(n0, n_threads), partial(_window_sub_rows, q, out, lo, hi), n_threads)
    return out


def box_count_banded(
    counts: np.ndarray, radius: int, out: np.ndarray, band_rows: int = 4096, n_threads: int = 1
) -> None:
    """Row-banded box count, writing a boolean mask into `out` in place.

    Same window as box_count; each band re-derives the row prefixes over
    [b0-radius, b1+radius) so interior rows get exactly the same window sums.
    On the float32 fast path every partial sum is below 2**24, so the sums
    are exact and the banding is bit-identical to the full pass. At 2**24 or
    more total points it falls back to the exact int64 SAT (full arrays).

    The window subtractions use contiguous slices (the index offsets are
    linear in the row/column position except at the clamped edges), avoiding
    the per-element gathers of the full pass.

    n_threads > 1 (S3.1) runs the band loop in a thread pool: each band
    writes disjoint rows of `out` and re-derives its prefixes from the
    read-only `counts`, so the bands are independent and the result stays
    bit-identical to the serial loop (n_threads=1).

    """
    n0, n1 = counts.shape
    if radius == 0:
        out[:] = counts > 0
        return
    if counts.sum() >= _FLOAT32_EXACT_LIMIT:
        out[:] = _box_count_int64(counts, radius) > 0
        return
    small_cols = n1 <= 2 * radius
    if small_cols:
        lo_idx = np.clip(np.arange(n1) - radius, 0, n1)
        hi_idx = np.clip(np.arange(n1) + radius + 1, 0, n1)

    # Per-thread p/w/q buffers (issue #39 pass 3): the bands of one run
    # used to allocate ~2 GB fresh per band (4096-row bands: 3 x ~700 MB),
    # ~28 GB of churn at full-AU scale. Each band overwrites every row and
    # column of its slice, so a per-thread buffer (max band size) reused
    # across that thread's bands is bit-identical with far fewer faults.
    tls = threading.local()
    max_rows = band_rows + 2 * radius + 2

    def _bufs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        b = tls.__dict__.get("boxcount")
        if b is None:
            p = np.empty((max_rows, n1 + 1), np.float32)
            w = np.empty((max_rows, n1), np.float32)
            q = np.empty((max_rows + 1, n1), np.float32)
            _nohuge(p, w, q)
            b = tls.boxcount = (p, w, q)
        return b

    def band(b0: int) -> None:
        b1 = min(n0, b0 + band_rows)
        e0 = max(0, b0 - radius)
        e1 = min(n0, b1 + radius)
        seg = counts[e0:e1]
        p, w, q = _bufs()
        p = p[: e1 - e0]
        w = w[: e1 - e0]
        p[:, 0] = 0.0
        np.cumsum(seg, axis=1, out=p[:, 1:])
        if small_cols:
            np.subtract(p[:, hi_idx], p[:, lo_idx], out=w)
        else:
            w[:, 0:radius] = p[:, radius + 1 : 2 * radius + 1]
            w[:, radius : n1 - radius] = p[:, 2 * radius + 1 : n1 + 1] - p[:, 0 : n1 - 2 * radius]
            w[:, n1 - radius :] = p[:, n1 : n1 + 1] - p[:, n1 - 2 * radius : n1 - radius]
        q = q[: e1 - e0 + 1]
        q[0] = 0.0
        np.cumsum(w, axis=0, out=q[1:])
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

    band_starts = [(b0, min(n0, b0 + band_rows)) for b0 in range(0, n0, band_rows)]
    if n_threads > 1:
        _thread_bands(band_starts, lambda b0, _b1: band(b0), n_threads)
    else:
        for b0, _b1 in band_starts:
            band(b0)


def _descent_band_body(
    sums_k: np.ndarray,
    counts_k: np.ndarray,
    val: np.ndarray,
    sup: np.ndarray,
    res: float,
    saturation: float,
    k: int,
    r0: int,
    r1: int,
    out: np.ndarray,
    outs: np.ndarray,
    upsample: Callable[[np.ndarray], np.ndarray] = upsample_bilinear,
    bufs: dict[str, np.ndarray] | None = None,
) -> None:
    """One row band of the pull-push descent: level k -> output rows [r0, r1).

    Output row r reads upsample taps at parent rows (r-1)//2 .. (r+1)//2, so
    the band needs parent rows [(r0-1)//2, r1//2]; the slice [e0, e1) with
    e0 = r0//2 - 1 covers it (at most one extra top row when r0 is odd).
    The slice's own upsampled boundary rows (sub-rows 0 and len-1, global
    rows 2*e0 and 2*e1 - 1) fall below r0 or at/above r1, so band rows never
    read them. Only when e0/e1 clamp at the array edges can a boundary row
    coincide with a band row, and then it is a true edge row where the
    full-array pass applies the same boundary formula. Bands write disjoint
    output rows and read the parent level read-only, so they are independent
    and the result is bit-identical to the full-array descent.

    """
    e0 = max(0, r0 // 2 - 1)
    e1 = min(val.shape[0], r1 // 2 + 1)
    p0 = 2 * e0
    if bufs is None:
        a = np.minimum(counts_k[r0:r1] / saturation, 1.0).astype(np.float32)
        local = sums_k[r0:r1] / np.maximum(counts_k[r0:r1], _COUNT_EPS)
        pv = upsample(val[e0:e1])[r0 - p0 : r1 - p0]
        ps = upsample(sup[e0:e1])[r0 - p0 : r1 - p0]
        out[r0:r1] = np.where(a >= 1.0, local, a * local + (1.0 - a) * pv)
        outs[r0:r1] = a * np.float32(res * (1 << k)) + (1.0 - a) * ps
        return
    # Preallocated path (issue #39): same formulas and IEEE-754 association
    # as the alloc path, but every scratch array is a per-thread buffer, so
    # a band touches zero fresh pages. (1.0 - a) below is bit-equal to the
    # alloc path's: x - y == x + (-y) exactly in IEEE-754.
    bh = r1 - r0
    a = bufs["a"][:bh]
    local = bufs["local"][:bh]
    om = bufs["om"][:bh]
    pv = bufs["pv"][:bh]
    ps = bufs["ps"][:bh]
    np.divide(counts_k[r0:r1], saturation, out=a)
    np.minimum(a, 1.0, out=a)
    np.maximum(counts_k[r0:r1], _COUNT_EPS, out=om)
    np.divide(sums_k[r0:r1], om, out=local)
    up = bufs["up"]
    if upsample is upsample_nearest:
        # 2x2 block repeat into the same buffer the bilinear path uses
        # (pure copies - bit-identical to upsample_nearest).
        h = e1 - e0
        w = val[e0:e1].shape[1]
        vs = val[e0:e1]
        up[0 : 2 * h : 2, 0 : 2 * w : 2] = vs
        up[0 : 2 * h : 2, 1 : 2 * w : 2] = vs
        up[1 : 2 * h : 2, 0 : 2 * w : 2] = vs
        up[1 : 2 * h : 2, 1 : 2 * w : 2] = vs
        pv[:] = up[r0 - p0 : r1 - p0, :]
        ss = sup[e0:e1]
        up[0 : 2 * h : 2, 0 : 2 * w : 2] = ss
        up[0 : 2 * h : 2, 1 : 2 * w : 2] = ss
        up[1 : 2 * h : 2, 0 : 2 * w : 2] = ss
        up[1 : 2 * h : 2, 1 : 2 * w : 2] = ss
        ps[:] = up[r0 - p0 : r1 - p0, :]
    else:  # upsample_bilinear
        tt = bufs["t"]
        _upsample_bilinear_into(val[e0:e1], up, tt)
        pv[:] = up[r0 - p0 : r1 - p0]
        _upsample_bilinear_into(sup[e0:e1], up, tt)
        ps[:] = up[r0 - p0 : r1 - p0]
    o = out[r0:r1]
    np.multiply(a, local, out=o)
    np.subtract(1.0, a, out=om)
    np.multiply(om, pv, out=om)
    o += om
    np.copyto(o, local, where=a >= 1.0)
    os = outs[r0:r1]
    np.multiply(a, np.float32(res * (1 << k)), out=os)
    # om holds (1-a)*pv after the value blend; recompute 1-a so the
    # support blend is a*res2k + (1-a)*ps, not (1-a)*pv*ps.
    np.subtract(1.0, a, out=om)
    np.multiply(om, ps, out=om)
    os += om


def _pull_push_descent(
    sums: list[np.ndarray],
    counts: list[np.ndarray],
    res: float,
    levels: int,
    upsample: Callable[[np.ndarray], np.ndarray] = upsample_bilinear,
    saturation: float = 1.0,
    unresolved_m: float = _UNRESOLVED_M,
    stop_level: int = 0,
    *,
    free_levels: bool = False,
    n_threads: int = 1,
    band_rows: int = 512,
    out_val: np.ndarray | None = None,
    out_sup: np.ndarray | None = None,
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
        n_threads: Thread pool size for the per-level row chunks (S3.2).
            1 (default) runs the exact full-array pass. >1 runs the banded
            row-chunk pass, bit-identical for upsample_nearest /
            upsample_bilinear (see _descent_band_body); any other upsample
            falls back to the full-array pass.
        band_rows: Row-chunk height for the threaded pass.
        out_val: Optional preallocated array (same shape as the level
            stop_level output, e.g. a tmpfs memmap) that the FINAL descent
            level writes into instead of allocating. Intermediate levels
            still allocate. Used by regime A (issue #39 pass 3) to back the
            output fields with tmpfs, whose pages reclaim as clean file
            cache instead of anon swap I/O under external pressure.
        out_sup: Same as out_val, for the support field.

    Returns:
        Tuple of interpolated value grid and support grid in metres, at
        level stop_level.

    """
    # Seed at the coarsest level
    val = sums[-1] / np.maximum(counts[-1], _COUNT_EPS)
    sup = np.where(
        counts[-1] > 0,
        np.float32(res * (1 << levels)),
        np.float32(unresolved_m),
    ).astype(np.float32)

    # Push: descend, blending local estimate against upsampled parent.
    # NaN-safe: saturated cells (a >= 1.0) take `local` directly, so a NaN
    # parent multiplied by zero (0*NaN = NaN) cannot corrupt them.
    threaded = n_threads > 1 and upsample in (upsample_nearest, upsample_bilinear)
    for k in range(levels - 1, stop_level - 1, -1):
        if threaded:
            n0 = sums[k].shape[0]
            n1 = sums[k].shape[1]
            bands = [(r0, min(n0, r0 + band_rows)) for r0 in range(0, n0, band_rows)]
            if k == stop_level and out_val is not None:
                out, outs = out_val, out_sup
            else:
                out = np.empty_like(sums[k])
                outs = np.empty_like(counts[k])
                _nohuge(out, outs)
            # Per-thread buffer sets (issue #39): a thread reuses one set
            # across every band of the level; the set is sized for this
            # level's column count and dropped when the level ends.
            bufs: dict[int, dict[str, np.ndarray]] = {}

            def _band(
                r0: int,
                r1: int,
                _k: int = k,
                _out: np.ndarray = out,
                _outs: np.ndarray = outs,
                _val: np.ndarray = val,
                _sup: np.ndarray = sup,
                _sums_k: np.ndarray = sums[k],
                _counts_k: np.ndarray = counts[k],
                _bufs: dict[int, dict[str, np.ndarray]] = bufs,
                _n1: int = n1,
                _n0: int = n0,
            ) -> None:
                tid = threading.get_ident()
                b = _bufs.get(tid)
                if b is None:
                    b = _bufs[tid] = _band_bufs(min(band_rows, _n0), _n1)
                _descent_band_body(
                    _sums_k, _counts_k, _val, _sup, res, saturation, _k, r0, r1, _out, _outs, upsample, bufs=b
                )

            _thread_bands(bands, _band, n_threads)
            val, sup = out, outs
            bufs.clear()
        else:
            c = counts[k]
            a = np.minimum(c / saturation, 1.0).astype(np.float32)
            local = sums[k] / np.maximum(c, _COUNT_EPS)
            if k == stop_level and out_val is not None:
                # Final level into preallocated arrays (regime A tmpfs
                # fields). Same formula and IEEE association as the fresh
                # path below; saturated cells go through the same 0*parent
                # -> NaN -> restore-from-`local` dance as the threaded band
                # body, so the bits match.
                sat = a >= 1.0
                om = np.empty_like(a)
                _nohuge(om)
                np.multiply(a, local, out=out_val)
                np.subtract(1.0, a, out=om)
                np.multiply(om, upsample(val), out=om)
                out_val += om
                np.copyto(out_val, local, where=sat)
                np.multiply(a, np.float32(res * (1 << k)), out=out_sup)
                np.subtract(1.0, a, out=om)
                np.multiply(om, upsample(sup), out=om)
                out_sup += om
                val, sup = out_val, out_sup
            else:
                parent = upsample(val)
                val = np.where(a >= 1.0, local, a * local + (1.0 - a) * parent)
                sup = a * np.float32(res * (1 << k)) + (1.0 - a) * upsample(sup)
        if free_levels:
            sums[k + 1] = None
            counts[k + 1] = None

    return val, sup


def _mem_available_bytes() -> int | None:
    """Return available system RAM from /proc/meminfo in bytes, or None if unknown."""
    try:
        with Path("/proc/meminfo").open(encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _peak_rss_bytes() -> int:
    """Return the current peak RSS of this process in bytes, 0 if unavailable."""
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except (ImportError, OSError, ValueError):
        return 0


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
    n_threads: int = 1,
    nworkers: int | None = None,
) -> None:
    """Compute the descent from level start_level down to level 0, row-banded.

    Bounds memory on large grids: each band reads a one-row overlap of the
    coarser level, so every tent-filter tap inside the band sees the same
    elements as the full-array pass (band edge rows are never read) and the
    result is bit-identical to the un-banded descent. Intermediate levels
    (start_level-1 .. 1) are written to memmap files in level_dir; level 0
    goes to out_val / out_sup (memmaps or arrays).

    n_threads > 1 runs the band loop in a thread pool (S3.2): bands write
    disjoint output rows and read the parent level read-only, so they are
    independent and the result stays bit-identical to the serial loop
    (n_threads=1). ``nworkers`` is an alias for ``n_threads`` (PR #27 naming,
    kept for its tests); when given it overrides ``n_threads``.

    Low-RAM guard (PR #27): the per-level pool is clamped to
    min(n_threads, nbands, 8), and each concurrent band holds its own
    intermediate buffers (a, local, pv, ps). When /proc/meminfo reports
    less available memory than the current peak RSS plus two bands' worth of
    float32 buffers, the level falls back to single-threaded (nw=1), where
    peak RSS cannot exceed the sequential path. The reserve covers only two
    bands against up to eight concurrent, so the gate is a headroom
    heuristic, not a memory bound: it trades wall time under memory
    pressure and never changes the output.
    """
    if nworkers is not None:
        n_threads = nworkers
    val = val_in
    sup = sup_in
    for k in range(start_level - 1, -1, -1):
        if k > 0:
            # Fresh 0600 regular files (issue #45): a pre-planted symlink is
            # replaced, never followed, and the O_EXCL race is guarded.
            out = _open_fresh_memmap(level_dir / f"_val_lvl{k}.npy", np.float32, sums[k].shape)
            outs = _open_fresh_memmap(level_dir / f"_sup_lvl{k}.npy", np.float32, counts[k].shape)
            _nohuge(out, outs)
        else:
            out, outs = out_val, out_sup
        n0 = sums[k].shape[0]
        bands = [(r0, min(n0, r0 + band_rows)) for r0 in range(0, n0, band_rows)]

        def _band(
            r0: int,
            r1: int,
            _k: int = k,
            _out: np.ndarray = out,
            _outs: np.ndarray = outs,
            _val: np.ndarray = val,
            _sup: np.ndarray = sup,
            _sums_k: np.ndarray = sums[k],
            _counts_k: np.ndarray = counts[k],
        ) -> None:
            _descent_band_body(_sums_k, _counts_k, _val, _sup, res, saturation, _k, r0, r1, _out, _outs)

        # Per-level pool bound + RAM gate (see docstring).
        nw = max(1, min(n_threads, len(bands), 8))
        if nw > 1:
            avail = _mem_available_bytes()
            need = _peak_rss_bytes() + 2 * band_rows * n0 * 4 * 2
            if avail is not None and avail < need:
                nw = 1
        _thread_bands(bands, _band, nw)
        if k > 0:
            val, sup = out, outs


def pull_push(
    sum_grid: np.ndarray,
    count_grid: np.ndarray,
    res: float,
    levels: int,
    upsample: Callable[[np.ndarray], np.ndarray] = upsample_bilinear,
    saturation: float = 1.0,
    unresolved_m: float = _UNRESOLVED_M,
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
    s64 = np.bincount(key, weights=values, minlength=nx * ny)
    _nohuge(s64)
    s = s64.reshape(nx, ny).astype(np.float32)
    c64 = np.bincount(key, minlength=nx * ny)
    _nohuge(c64)
    c = c64.reshape(nx, ny).astype(np.float32)
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
    _nohuge(order, xs, ys, ws)
    bounds = np.concatenate((np.zeros(1, xs.dtype), np.flatnonzero(np.diff(xs)) + 1, np.array([xs.size], xs.dtype)))
    if bounds.size < 2 or bounds[1] == 0:
        return
    rows = xs[bounds[:-1]]
    for k in range(bounds.size - 1):
        a, b = int(bounds[k]), int(bounds[k + 1])
        r = int(rows[k])
        s_out[r] = np.bincount(ys[a:b], weights=ws[a:b], minlength=ny).astype(np.float32)
        c_out[r] = np.bincount(ys[a:b], minlength=ny).astype(np.float32)


def grid_index_mt(
    x: np.ndarray,
    y: np.ndarray,
    tv: np.ndarray,
    x0: float,
    y0: float,
    res: float,
    bsize: int,
    nbx: int,
    nby: int,
    px: np.ndarray,
    py: np.ndarray,
    ptv: np.ndarray,
    n_threads: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Block index + point-memmap scatter, thread-chunked (S3.6).

    The cell-divide half (per-point block ids) and the memmap-write half
    (gathering points into px / py / ptv in block order) run a thread pool
    over point ranges. The stable argsort stays sequential: it must, to keep
    the per-block point order, which the float64 bincount accumulation in
    bin_points depends on for bit-exactness.

    n_threads=1 is bit-identical to the serial pipeline code (A8).

    Args:
        x: Point coordinates (metres, working CRS), length n.
        y: Point coordinates (metres, working CRS), length n.
        tv: Transformed values, length n.
        x0: Grid origin x.
        y0: Grid origin y.
        res: Cell size in metres.
        bsize: Block size in cells.
        nbx: Block count along x.
        nby: Block count along y.
        px: Writable float64 output, length n (e.g. memmap band).
            Filled with x in stable block order.
        py: Writable float64 output, length n. Filled with y in stable block order.
        ptv: Writable float64 output, length n. Filled with tv in stable block order.
        n_threads: Thread pool size over point ranges. Values <= 1 run the
            serial path, bit-identical to the pipeline code (A8).

    Returns:
        Tuple of (bid, order): per-point block ids and the stable sort
        permutation. The caller derives block starts via
        searchsorted(bid[order], arange(nbx * nby + 1)).

    """
    n = x.shape[0]
    bid = np.empty(n, np.int64)

    def _cell_div(r0: int, r1: int) -> None:
        bx = np.clip(((x[r0:r1] - x0) // res // bsize).astype(np.int64), 0, nbx - 1)
        by = np.clip(((y[r0:r1] - y0) // res // bsize).astype(np.int64), 0, nby - 1)
        bid[r0:r1] = bx * nby + by

    _thread_bands(_split_bands(n, n_threads), _cell_div, n_threads)

    order = np.argsort(bid, kind="stable")

    def _scatter(r0: int, r1: int) -> None:
        o = order[r0:r1]
        px[r0:r1] = x[o]
        py[r0:r1] = y[o]
        ptv[r0:r1] = tv[o]

    _thread_bands(_split_bands(n, n_threads), _scatter, n_threads)
    return bid, order


def pad_to_pyramid(n: int, levels: int) -> int:
    """Pad dimension to be divisible by 2**levels.

    Returns:
        Padded dimension.

    """
    step = 1 << levels
    return ((n + step - 1) // step) * step
