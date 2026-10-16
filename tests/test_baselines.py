"""Tests for ppgrid.baselines (issue #107).

IDW vs hand-computed closed form, NN vs brute force, neighbour-policy
semantics, config validation, the identical-domain helper against a real
pipeline run, and a sanity-scale run. Every scipy-dependent test is gated
with ``pytest.importorskip("scipy")`` so the core suite passes without the
``[baselines]`` extra.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from ppgrid import baselines
from ppgrid.baselines import BaselineConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# IDW: hand-computed closed form + cross-checks
# ---------------------------------------------------------------------------


def test_idw_three_point_closed_form() -> None:
    """idw() matches the hand-computed IDW sum for a 3-point fixture."""
    pytest.importorskip("scipy")
    points = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    values = np.array([10.0, 20.0, 30.0])
    query = np.array([[4.0, 3.0]])

    # Closed form: d = (5, sqrt(45), sqrt(65)); w = d^-power; val = sum(wv)/sum(w).
    d = [5.0, math.sqrt(45.0), math.sqrt(65.0)]
    for power in (2.0, 1.0):
        w = [x ** (-power) for x in d]
        expected = sum(wi * vi for wi, vi in zip(w, values, strict=True)) / sum(w)
        r = baselines.idw(points, values, query, power=power, k=3)
        np.testing.assert_allclose(r.value, [expected], rtol=1e-12)
        np.testing.assert_allclose(r.dist, [5.0], rtol=1e-12)
    assert r.kind == "idw"


def test_idw_k1_equals_nearest_value() -> None:
    """idw(k=1) is the nearest source value, whatever power is."""
    pytest.importorskip("scipy")
    points = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    values = np.array([10.0, 20.0, 30.0])
    query = np.array([[4.0, 3.0], [9.0, 1.0]])
    r = baselines.idw(points, values, query, power=2.0, k=1)
    np.testing.assert_allclose(r.value, [10.0, 20.0], rtol=1e-12)


def test_idw_zero_distance_takes_coincident_value() -> None:
    """d=0 limit: the coincident point's value (mean over coincident points)."""
    pytest.importorskip("scipy")
    points = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 0.0]])
    values = np.array([10.0, 20.0, 20.0])
    query = np.array([[0.0, 0.0], [1.0, 1.0]])
    r = baselines.idw(points, values, query, power=2.0, k=3)
    # Query 0 coincides with two points (10, 20) -> the mean, 15.
    np.testing.assert_allclose(r.value[0], 15.0, rtol=1e-12)
    # Query 1: d = (sqrt(2), sqrt(82), sqrt(2)); weights (1/2, 1/82, 1/2).
    w = [1.0 / 2.0, 1.0 / 82.0, 1.0 / 2.0]
    expected = (w[0] * 10.0 + w[1] * 20.0 + w[2] * 20.0) / sum(w)
    np.testing.assert_allclose(r.value[1], expected, rtol=1e-12)


def test_idw_min_neighbours_policy() -> None:
    """Policy 'k' needs the full k; '1' and int q need that many."""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 100, size=(5, 2))
    values = rng.uniform(0, 10, size=5)
    query = rng.uniform(0, 100, size=(4, 2))

    r_k = baselines.idw(points, values, query, k=8, min_neighbours="k")
    assert np.isnan(r_k.value).all()  # 5 < 8: no value under the 'k' policy
    r_one = baselines.idw(points, values, query, k=8, min_neighbours="1")
    assert np.isfinite(r_one.value).all()
    r3 = baselines.idw(points, values, query, k=8, min_neighbours=3)
    assert np.isfinite(r3.value).all()
    np.testing.assert_allclose(r3.value, r_one.value, rtol=1e-15)  # both use all 5
    r6 = baselines.idw(points, values, query, k=8, min_neighbours=6)
    assert np.isnan(r6.value).all()  # 5 < 6


def test_idw_matches_direct_computation_with_chunking() -> None:
    """Blocked search (query_chunk=7) matches the direct per-query IDW sum."""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(1)
    points = rng.uniform(0, 1000, size=(500, 2))
    values = rng.uniform(1, 100, size=500)
    query = rng.uniform(0, 1000, size=(30, 2))
    r = baselines.idw(points, values, query, power=2.5, k=4, query_chunk=7)
    for qi in range(query.shape[0]):
        d = np.sqrt(((points - query[qi]) ** 2).sum(axis=1))
        j = np.argsort(d)[:4]
        w = d[j] ** (-2.5)
        expected = (w * values[j]).sum() / w.sum()
        np.testing.assert_allclose(r.value[qi], expected, rtol=1e-12)
    # k=4 of 500 points: dist sorted ascending within the neighbourhood.
    assert (r.dist >= 0).all()


# ---------------------------------------------------------------------------
# Nearest: brute force on a 10k-point fixture
# ---------------------------------------------------------------------------


def test_nearest_vs_brute_force_10k() -> None:
    """nearest() picks exactly the brute-force nearest of 10k points."""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(42)
    points = rng.uniform(0, 1000, size=(10_000, 2))
    query = rng.uniform(0, 1000, size=(200, 2))
    r = baselines.nearest(points, query)
    d2 = ((points[None, :, :] - query[:, None, :]) ** 2).sum(axis=2)
    expected_idx = np.argmin(d2, axis=1)
    np.testing.assert_array_equal(r.idx, expected_idx)
    np.testing.assert_allclose(r.dist, np.sqrt(d2[np.arange(query.shape[0]), expected_idx]), rtol=1e-12)


def test_nearest_values_transfer() -> None:
    """nearest(values=...) transfers the nearest source value; idx is stable."""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(7)
    points = rng.uniform(0, 100, size=(1000, 2))
    values = rng.uniform(0, 50, size=1000)
    query = rng.uniform(0, 100, size=(50, 2))
    r = baselines.nearest(points, query, values)
    np.testing.assert_array_equal(r.value, values[r.idx])
    r_bare = baselines.nearest(points, query)
    assert r_bare.value is None
    np.testing.assert_array_equal(r_bare.idx, r.idx)
    np.testing.assert_allclose(r_bare.dist, r.dist)


# ---------------------------------------------------------------------------
# Result object: config carried + methods section
# ---------------------------------------------------------------------------


def test_result_carries_config_and_methods() -> None:
    """The result records power/k/policy/search; methods() is pasteable."""
    pytest.importorskip("scipy")
    points = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    values = np.array([1.0, 2.0, 3.0])
    query = np.array([[4.0, 3.0]])
    cfg = BaselineConfig(n_source=3, n_queries=1, wall_limit_s=60.0, rss_limit_gb=4.0, needs_andryo_confirmation=False)
    r = baselines.idw(points, values, query, power=2.0, k=3, min_neighbours="k", limits=cfg, points_crs=6933)

    assert r.config["power"] == 2.0
    assert r.config["k"] == 3
    assert r.config["min_neighbours"] == "k"
    assert r.config["min_neighbours_required"] == 3
    assert "cKDTree" in r.config["search"]
    assert r.config["points_crs"] == 6933
    assert json.dumps(r.to_dict())  # JSON-safe
    m = r.methods()
    for token in ("p=2", "k=3", "'k'", "cKDTree", "n_points=3", "6933", "wall<=60 s", "4 GiB"):
        assert token in m, f"{token!r} missing from methods(): {m}"


# ---------------------------------------------------------------------------
# Config: claimed regime, validation, round-trip
# ---------------------------------------------------------------------------


def test_config_defaults_are_the_claimed_regime() -> None:
    """Default BaselineConfig == the v1 author-claimed regime (issue #107)."""
    c = BaselineConfig()
    assert c.n_source == 20_000_000
    assert c.n_queries == 1_000_000
    assert c.wall_limit_s == 600.0
    assert c.rss_limit_gb == 32.0
    assert c.needs_andryo_confirmation is True
    assert c.power == 2.0
    assert c.k == 8
    assert c.min_neighbours == "k"


def test_load_config_committed_file_matches_defaults() -> None:
    """bench/config/baselines.json pins the claimed regime (round-trips)."""
    path = REPO_ROOT / "bench" / "config" / "baselines.json"
    c = load_config(path)
    assert c.to_dict() == {
        "n_source": 20_000_000,
        "n_queries": 1_000_000,
        "power": 2.0,
        "k": 8,
        "min_neighbours": "k",
        "wall_limit_s": 600.0,
        "rss_limit_gb": 32.0,
        "needs_andryo_confirmation": True,
        "note": c.note,
        "source": str(path),
    }
    json.dumps(c.to_dict())


def test_config_unknown_key_rejected() -> None:
    """A typo'd config key is rejected (strict schema)."""
    with pytest.raises(ValueError, match="unknown baseline config key"):
        BaselineConfig.from_dict({"k": 8, "bogus": 1})


def test_config_field_validation() -> None:
    """Type/range violations are named errors; bad JSON files are named too."""
    bad = [
        {"k": 0},
        {"power": -1.0},
        {"n_source": 0},
        {"n_queries": -3},
        {"wall_limit_s": 0.0},
        {"rss_limit_gb": float("nan")},
        {"min_neighbours": "x"},
        {"min_neighbours": True},
        {"needs_andryo_confirmation": "yes"},
        {"note": 5},
    ]
    for d in bad:
        with pytest.raises(ValueError, match="baseline config"):
            BaselineConfig.from_dict(d)
    with pytest.raises(ValueError, match="not valid JSON"):
        load_config(REPO_ROOT / "pyproject.toml")


def test_load_config_missing_file() -> None:
    """A missing config file is a named ValueError."""
    with pytest.raises(ValueError, match="cannot read baseline config"):
        load_config(REPO_ROOT / "does_not_exist.json")


# ---------------------------------------------------------------------------
# Input validation + scipy-free import
# ---------------------------------------------------------------------------


def test_input_validation_errors() -> None:
    """Shape/finiteness errors are raised before the scipy search."""
    with pytest.raises(ValueError, match=r"points must be an \(N, 2\)"):
        baselines.idw(np.zeros((5, 3)), np.zeros(5), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="non-finite"):
        baselines.idw(np.array([[0.0, 0.0], [float("nan"), 1.0]]), np.ones(2), np.zeros((1, 2)))
    with pytest.raises(ValueError, match=r"values must be shape \(2,\)"):
        baselines.idw(np.zeros((2, 2)), np.ones(3), np.zeros((1, 2)))
    with pytest.raises(ValueError, match="power"):
        baselines.idw(np.zeros((2, 2)), np.ones(2), np.zeros((1, 2)), power=0.0)
    with pytest.raises(ValueError, match="min_neighbours"):
        baselines.idw(np.zeros((2, 2)), np.ones(2), np.zeros((1, 2)), min_neighbours="x")
    with pytest.raises(ValueError, match="non-empty"):
        baselines.nearest(np.zeros((0, 2)), np.zeros((1, 2)))


def test_module_imports_without_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    """`import ppgrid.baselines` works on the lean core (scipy is optional).

    The module is re-executed fresh with scipy masked, then the original
    module object is restored so class identity can't leak into other tests.
    """
    import importlib

    monkeypatch.setitem(sys.modules, "scipy", None)
    saved = sys.modules.pop("ppgrid.baselines", None)
    try:
        fresh = importlib.import_module("ppgrid.baselines")
        assert callable(fresh.idw)
        assert callable(fresh.nearest)
        assert callable(fresh.load_config)
    finally:
        if saved is not None:
            sys.modules["ppgrid.baselines"] = saved


# ---------------------------------------------------------------------------
# Synthetic workload
# ---------------------------------------------------------------------------


def test_synthetic_workload_deterministic_and_sized() -> None:
    """Same seed = identical arrays; different seed differs; size/cap rules hold."""
    a = baselines.synthetic_workload(1000, 500, seed=3, box_m=(1000.0, 800.0), res=50.0)
    b = baselines.synthetic_workload(1000, 500, seed=3, box_m=(1000.0, 800.0), res=50.0)
    c = baselines.synthetic_workload(1000, 500, seed=4, box_m=(1000.0, 800.0), res=50.0)
    for arr, ref in zip(a, b, strict=True):
        np.testing.assert_array_equal(arr, ref)
    assert not np.array_equal(a[0], c[0])
    assert not np.array_equal(a[1], c[1])

    points, values, query = a
    assert points.shape == (1000, 2)
    assert (points >= 0).all()
    assert (points[:, 0] < 1000).all()
    assert (points[:, 1] < 800).all()
    assert values.shape == (1000,)
    assert np.isfinite(values).all()
    assert (values > 0).all()
    assert query.shape[1] == 2
    # Grid over the extent at res=50 is ~20x16 = <= 500 cells: full grid.
    assert query.shape[0] > 0
    assert query.shape[0] <= 500
    # The n_queries cap is honoured exactly when the grid is larger.
    _, _, q_cap = baselines.synthetic_workload(1000, 100, seed=3, box_m=(1000.0, 800.0), res=50.0)
    assert q_cap.shape[0] == min(100, query.shape[0])
    with pytest.raises(ValueError, match="n_source"):
        baselines.synthetic_workload(0, 10)


# ---------------------------------------------------------------------------
# Domain: the identical domain for a finished ppgrid run
# ---------------------------------------------------------------------------


def test_domain_from_pipeline_matches_run(tmp_path: Path) -> None:
    """domain_from_pipeline reproduces a run's points, CRS, res, grid, mask."""
    pytest.importorskip("scipy")
    from ppgrid.pipeline import Pipeline
    from ppgrid.pullpush import box_count

    p = Pipeline(
        str(REPO_ROOT / "data" / "melb_houses.csv"),
        "price",
        "longitude",
        "latitude",
        str(tmp_path),
        res=500.0,
        cap_km=10.0,
        skip_calibration=True,
        dry_run=True,
    )
    p.ingest()
    p.calibrate()
    p.grid()
    d = baselines.domain_from_pipeline(p, holdout_frac=0.2, seed=7)

    assert d.crs == p.work_crs
    assert d.res == p.res
    assert d.x.shape[0] == p.n
    np.testing.assert_array_equal(d.x, p.x)
    np.testing.assert_array_equal(d.y, p.y)
    np.testing.assert_array_equal(d.values, p.v)
    assert d.x0 == p.x0
    assert d.y0 == p.y0
    assert d.nx == p.nx
    assert d.ny == p.ny
    assert d.cap_km == p.cap_km_val

    # Holdout: exact size, disjoint, complete, deterministic in seed.
    assert d.n_holdout == math.ceil(0.2 * p.n)
    h, tr = d.holdout_idx, d.train_idx
    assert h.shape[0] + tr.shape[0] == p.n
    assert np.intersect1d(h, tr).size == 0
    d_same = baselines.domain_from_pipeline(p, holdout_frac=0.2, seed=7)
    np.testing.assert_array_equal(h, d_same.holdout_idx)
    d_none = baselines.domain_from_pipeline(p)
    assert d_none.holdout_idx is None
    assert d_none.n_holdout == 0
    assert d_none.train_idx.shape[0] == p.n
    with pytest.raises(ValueError, match="holdout_frac"):
        baselines.domain_from_pipeline(p, holdout_frac=1.5)

    # Grid query: every pixel centre, raster convention.
    g = d.grid_query()
    assert g.shape == (p.nx * p.ny, 2)
    np.testing.assert_allclose(g[0], [d.x0 + 0.5 * d.res, d.y0 + 0.5 * d.res])

    # Coverage mask == the pipeline's own box-count near-mask definition.
    r = round(p.cap_km_val * 1000.0 / d.res)
    counts = np.zeros((d.ny, d.nx), dtype=np.int64)
    ix = ((d.x - d.x0) // d.res).astype(np.intp)
    iy = ((d.y - d.y0) // d.res).astype(np.intp)
    np.add.at(counts, (iy, ix), 1)
    np.testing.assert_array_equal(d.coverage_mask(), box_count(counts, r) > 0)
    assert json.dumps(d.to_dict())


def test_coverage_mask_exact_small_domain() -> None:
    """Dilation twin of box_count>0 on a 5x5 grid with a brute-force check."""
    pytest.importorskip("scipy")
    res = 10.0
    d = baselines.Domain(
        x=np.array([1.5 * res, 3.5 * res]),
        y=np.array([1.5 * res, 3.5 * res]),
        values=np.array([1.0, 2.0]),
        crs=None,
        res=res,
        x0=0.0,
        y0=0.0,
        nx=5,
        ny=5,
    )
    # cap 0.02 km -> r = round(20/10) = 2
    mask = d.coverage_mask(cap_km=0.02)
    expected = np.zeros((5, 5), dtype=bool)
    for j in range(5):
        for i in range(5):
            expected[j, i] = any(max(abs(i - pi), abs(j - pj)) <= 2 for pi, pj in ((1, 1), (3, 3)))
    np.testing.assert_array_equal(mask, expected)
    # cap 0.001 km -> r = 0: only the source cells.
    mask0 = d.coverage_mask(cap_km=0.001)
    assert mask0.sum() == 2
    assert mask0[1, 1]
    assert mask0[3, 3]
    with pytest.raises(ValueError, match="no cap_km"):
        d.coverage_mask()


# ---------------------------------------------------------------------------
# Sanity scale (issue #107 item 6): 100k points / 10k queries, CI-safe guard
# ---------------------------------------------------------------------------


def test_sanity_scale_100k_points_10k_queries() -> None:
    """Modest-scale run: exact, deterministic, well inside the CI wall guard."""
    pytest.importorskip("scipy")
    points, values, query = baselines.synthetic_workload(
        100_000, 10_000, seed=1, box_m=(300_000.0, 200_000.0), res=100.0
    )
    assert points.shape == (100_000, 2)
    assert query.shape == (10_000, 2)  # grid > 10k cells: subsampled to the cap

    cfg = BaselineConfig(
        n_source=100_000,
        n_queries=10_000,
        wall_limit_s=120.0,
        rss_limit_gb=16.0,
        needs_andryo_confirmation=False,
    )
    t0 = time.perf_counter()
    r_idw = baselines.idw(points, values, query, limits=cfg)
    r_nn = baselines.nearest(points, query, values, limits=cfg)
    wall = time.perf_counter() - t0
    # Loose CI guard (ubuntu + macOS runners): 100k/10k exact k-NN is seconds.
    assert wall < 120.0
    assert r_idw.wall_ok is True
    assert r_nn.wall_ok is True
    assert np.isfinite(r_idw.value).all()
    assert np.isfinite(r_nn.value).all()
    # Deterministic: re-run is bit-identical.
    r2 = baselines.idw(points, values, query)
    np.testing.assert_array_equal(r_idw.value, r2.value)
    np.testing.assert_array_equal(r_idw.dist, r2.dist)
    # Spot-check 5 random queries against the direct IDW sum.
    rng = np.random.default_rng(99)
    for qi in rng.choice(query.shape[0], size=5, replace=False):
        d = np.sqrt(((points - query[qi]) ** 2).sum(axis=1))
        j = np.argsort(d)[:8]
        w = d[j] ** (-2.0)
        np.testing.assert_allclose(r_idw.value[qi], (w * values[j]).sum() / w.sum(), rtol=1e-12)
