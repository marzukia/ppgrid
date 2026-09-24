"""Bit-identity guards for the tier 2 performance changes.

- banded primitives (bin_points_banded / box_count_banded / _descent_banded)
  match their full-array twins exactly
- the shared full-field path (Pipeline._prepare_shared + _block_shared)
  matches the per-box reference algorithm for every block
- the committed Melbourne 10m example is reproduced bit-for-bit
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.windows import Window

from ppgrid.pipeline import M_PER_KM, NODATA, PTS_TV, PTS_X, PTS_Y, Pipeline, _block_points, _quantize
from ppgrid.pullpush import (
    _descent_banded,
    _pull_push_descent,
    bin_points,
    bin_points_banded,
    box_count,
    box_count_banded,
    downsample_sum,
    pull_push,
)


def _pyramid(s0: np.ndarray, c0: np.ndarray, levels: int) -> tuple[list, list]:
    sums = [s0]
    counts = [c0]
    for _ in range(levels):
        sums.append(downsample_sum(sums[-1]))
        counts.append(downsample_sum(counts[-1]))
    return sums, counts


def test_bin_points_banded_bit_equal() -> None:
    """Row-banded binning must equal the flat bincount exactly."""
    rng = np.random.default_rng(42)
    for n, nx, ny in ((200_000, 300, 200), (10_000, 4096, 2048), (1, 5, 5)):
        ix = rng.integers(0, nx, n)
        iy = rng.integers(0, ny, n)
        w = rng.lognormal(0, 1, n)
        s_ref, c_ref = bin_points(ix, iy, w, nx, ny)
        s = np.zeros((nx, ny), np.float32)
        c = np.zeros((nx, ny), np.float32)
        bin_points_banded(s, c, ix, iy, w, ny)
        assert np.array_equal(s_ref, s)
        assert np.array_equal(c_ref, c)


def test_box_count_banded_bit_equal() -> None:
    """Row-banded box counting must equal the full SAT pass exactly."""
    rng = np.random.default_rng(7)
    for n0, n1, npts, r, band in (
        (512, 256, 50_000, 0, 128),
        (512, 256, 50_000, 1, 7),
        (512, 256, 50_000, 7, 1),
        (512, 256, 50_000, 33, 1024),
        (512, 256, 50_000, 128, 63),
        (512, 256, 50_000, 4096, 1024),
        (32, 16, 500, 100, 5),
        (64, 64, 10_000, 63, 1),
        (64, 64, 10_000, 65, 4096),
        (128, 300, 20_000, 90, 25),
        (7, 3, 50, 2, 3),
    ):
        ix = rng.integers(0, n0, npts)
        iy = rng.integers(0, n1, npts)
        c = bin_points(ix, iy, rng.uniform(0, 1, npts), n0, n1)[1]
        ref = box_count(c, r) > 0
        out = np.zeros((n0, n1), bool)
        box_count_banded(c, r, out, band_rows=band)
        assert np.array_equal(ref, out), f"n0={n0} n1={n1} r={r} band={band}"


def test_descent_banded_bit_equal(tmp_path: Path) -> None:
    """_descent_banded (from level 2 and 1) must equal the full descent."""
    rng = np.random.default_rng(11)
    nx, ny, levels, res, sat = 2048, 1536, 4, 1000.0, 4.0
    ix = rng.integers(0, nx, 300_000)
    iy = rng.integers(0, ny, 300_000)
    w = rng.lognormal(0, 1, 300_000)
    s0, c0 = bin_points(ix, iy, w, nx, ny)

    sums, counts = _pyramid(s0.copy(), c0.copy(), levels)
    val_ref, sup_ref = _pull_push_descent(sums, counts, res, levels, saturation=sat)

    for start_level in (1, 2):
        sums, counts = _pyramid(s0.copy(), c0.copy(), levels)
        val_in, sup_in = _pull_push_descent(
            sums, counts, res, levels, saturation=sat, stop_level=start_level, free_levels=True
        )
        val_out = np.zeros((nx, ny), np.float32)
        sup_out = np.zeros((nx, ny), np.float32)
        for band in (1, 7):
            val_out[:] = 0
            sup_out[:] = 0
            _descent_banded(
                sums,
                counts,
                res,
                sat,
                start_level=start_level,
                val_in=val_in,
                sup_in=sup_in,
                out_val=val_out,
                out_sup=sup_out,
                level_dir=tmp_path,
                band_rows=band,
            )
            assert np.array_equal(val_ref, val_out), f"start_level={start_level} band={band}"
            assert np.array_equal(sup_ref, sup_out), f"start_level={start_level} band={band}"


def _perbox_reference(p: Pipeline, bx: int, by: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Baseline per-box algorithm (pull-push over the snapped halo box).

    Reference implementation of the pre-tier2 _process_block body, kept here
    as the bit-identity oracle for the shared full-field path.
    """
    res, halo, bsize = p.res, p.cfg.halo, p.cfg.bsize
    step = 1 << p.cfg.levels
    i0, j0 = bx * bsize, by * bsize
    i1, j1 = min(i0 + bsize, p.cfg.nx), min(j0 + bsize, p.cfg.ny)
    hi0 = max(0, ((i0 - halo) // step) * step)
    hj0 = max(0, ((j0 - halo) // step) * step)
    hi1 = min(p.cfg.nx_padded, -(-(i1 + halo) // step) * step)
    hj1 = min(p.cfg.ny_padded, -(-(j1 + halo) // step) * step)

    sel = _block_points(p.cfg, bx, by)
    if sel.shape[1] == 0:
        return None
    ix = ((sel[PTS_X] - p.cfg.x0) // res).astype(np.int64)
    iy = ((sel[PTS_Y] - p.cfg.y0) // res).astype(np.int64)
    m = (ix >= hi0) & (ix < hi1) & (iy >= hj0) & (iy < hj1)
    if not m.any():
        return None
    s, c_grid = bin_points(ix[m] - hi0, iy[m] - hj0, sel[PTS_TV][m], hi1 - hi0, hj1 - hj0)
    val, sup = pull_push(s, c_grid, res, p.cfg.levels, saturation=p.cfg.sat)
    cap_cells = round(p.cfg.cap_km * M_PER_KM / res)
    near = box_count(c_grid, cap_cells) > 0
    a0, b0 = i0 - hi0, j0 - hj0
    sl = (slice(a0, a0 + (i1 - i0)), slice(b0, b0 + (j1 - j0)))
    return _quantize(
        p.cfg,
        val[sl],
        sup[sl] / M_PER_KM,
        near[sl],
    )


def test_shared_field_matches_perbox(tmp_path: Path) -> None:
    """Full-field shared path vs per-box reference: bit-equal on every block.

    ~2560x2560 cells at res 10 -> 2x2 blocks (bsize 2048), padded grid
    4096x4096 < 1e8 cells so the in-RAM shared descent is exercised.
    """
    rng = np.random.default_rng(3)
    n = 20000
    lons = 144.9 + rng.uniform(0, 0.29, n)
    lats = -37.6 + rng.uniform(0, 0.231, n)
    csv = tmp_path / "data.csv"
    pd.DataFrame({"lon": lons, "lat": lats, "val": rng.uniform(10, 100, n)}).to_csv(csv, index=False)

    out_dir = tmp_path / "out"
    # out_crs == work_crs skips the final reprojection, so the written
    # raster aligns 1:1 with the working grid the reference computes on.
    p = Pipeline(
        str(csv),
        "val",
        "lon",
        "lat",
        str(out_dir),
        res=10.0,
        cap_km=2.0,
        workers=2,
        skip_calibration=True,
        out_crs=6933,
    )
    p.run()
    assert p.cfg.nbx == 2
    assert p.cfg.nby == 2

    with rasterio.open(out_dir / "value.tif") as vd, rasterio.open(out_dir / "support_km.tif") as sd:
        for bx in range(p.cfg.nbx):
            for by in range(p.cfg.nby):
                i0, j0 = bx * p.cfg.bsize, by * p.cfg.bsize
                i1, j1 = min(i0 + p.cfg.bsize, p.cfg.nx), min(j0 + p.cfg.bsize, p.cfg.ny)
                w = Window(i0, p.cfg.ny - j1, i1 - i0, j1 - j0)
                v_got = vd.read(1, window=w)
                s_got = sd.read(1, window=w)
                ref = _perbox_reference(p, bx, by)
                if ref is None:
                    assert (v_got == NODATA).all(), f"value block {bx},{by} not all nodata"
                    assert (s_got == NODATA).all(), f"support block {bx},{by} not all nodata"
                else:
                    # The pipeline writes blocks as vq.T[::-1] (grid [x, y] ->
                    # raster [row, col]); apply the same transform to the ref.
                    v_ref = ref[0].T[::-1, :]
                    s_ref = ref[1].T[::-1, :]
                    assert np.array_equal(v_got, v_ref), f"value block {bx},{by}: {int((v_got != v_ref).sum())} px"
                    assert np.array_equal(s_got, s_ref), f"support block {bx},{by}: {int((s_got != s_ref).sum())} px"


def test_melb10_anchor_bit_equal(tmp_path: Path) -> None:
    """The committed 10m Melbourne example must be reproduced bit-for-bit."""
    repo = Path(__file__).resolve().parent.parent
    csv = repo / "data" / "melb_houses.csv"
    anchor = repo / "examples" / "melb" / "10m"
    out_dir = tmp_path / "melb10"

    p = Pipeline(str(csv), "price", "longitude", "latitude", str(out_dir), res=10.0, cap_km=10.0, skip_calibration=True)
    p.run()

    for name in ("value.tif", "support_km.tif"):
        with rasterio.open(anchor / name) as a, rasterio.open(out_dir / name) as o:
            assert a.width == o.width
            assert a.height == o.height
            assert a.nodata == o.nodata
            assert np.array_equal(a.read(1), o.read(1)), name


def test_perbox_after_shared_no_stale_ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A per-box run after a shared run in one process must not reuse stale _CTX."""
    import ppgrid.pipeline as ig

    # Force the per-box fallback for grid B while grid A stays on the shared
    # path, regardless of the production threshold.
    monkeypatch.setattr(ig, "_SHARED_MAX_CELLS", 1_000_000)

    def run(csv: Path, out: Path) -> np.ndarray:
        p = Pipeline(
            str(csv),
            "val",
            "lon",
            "lat",
            str(out),
            res=10.0,
            cap_km=10.0,
            workers=2,
            skip_calibration=True,
            out_crs=6933,
        )
        p.run()
        with rasterio.open(out / "value.tif") as v:
            assert v.width * v.height > 0
            return v.read(1)

    rng = np.random.default_rng(11)
    small = tmp_path / "small.csv"
    pd.DataFrame(
        {
            "lon": 144.9 + rng.uniform(0, 0.02, 3000),
            "lat": -37.6 + rng.uniform(0, 0.02, 3000),
            "val": rng.uniform(1, 100, 3000),
        }
    ).to_csv(small, index=False)
    run(small, tmp_path / "out_small")  # shared path (padded ~256^2 < 1e6)

    big = tmp_path / "big.csv"
    pd.DataFrame(
        {
            "lon": 144.9 + rng.uniform(0, 0.6, 20000),
            "lat": -37.6 + rng.uniform(0, 0.46, 20000),
            "val": rng.uniform(1, 100, 20000),
        }
    ).to_csv(big, index=False)
    vb_after_small = run(big, tmp_path / "out_big1")  # per-box path
    vb_alone = run(big, tmp_path / "out_big2")  # per-box path again

    assert vb_after_small.shape == vb_alone.shape
    # A stale shared field from the small run would shrink/shift the slices
    # and corrupt this raster; per-box must be identical in both orders.
    assert np.array_equal(vb_after_small, vb_alone)
    assert (vb_alone != NODATA).sum() > 1000, "expected real content, got empty raster"
