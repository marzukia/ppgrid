"""bench/regimes.py — seeded generators, descriptors, gates (issue #108).

Fast by construction: generator-level tests are array-only, and the one
end-to-end test runs the full pipeline on a 100-cell grid (~100 points),
twice, to pin the results-record schema and the rerun bit-identity that the
anchor gate relies on.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))

import regimes

# Small domain for generator tests: 1 km x 1 km @ 10 m = 100x100 cells.
_BASE_GEN: dict = {
    "type": "synthetic",
    "seed": 42,
    "center": [145.0, -37.5],
    "width_m": 1000.0,
    "height_m": 1000.0,
    "res_m": 10.0,
    "points_per_cell": 4.0,
    "occupied_fraction": 0.25,
    "density": {"mode": "uniform"},
}


def _gen(**over: object) -> dict:
    """Return a base generator spec with one or more keys overridden."""
    g = json.loads(json.dumps(_BASE_GEN))
    g.update(over)
    return g


def _gen_csv(gen: dict, d: Path) -> Path:
    """Generate one CSV under d and return its path."""
    p = d / "gen.csv"
    regimes.generate_synthetic(gen, p)
    return p


def _cfg(cfg_id: str, gen: dict, pipeline: dict | None = None) -> dict:
    """Build a minimal valid config spec entry for cfg_id."""
    if pipeline is None:
        pipeline = {"res": 10.0, "cap_km": 2.0, "transform": "identity", "skip_calibration": True, "workers": 2}
    cfg = {
        "id": cfg_id,
        "ci": True,
        "generator": gen,
        "pipeline": pipeline,
    }
    regimes._validate_cfg(cfg)
    return cfg


# --- generator determinism ---------------------------------------------------


def test_generator_deterministic(tmp_path: Path) -> None:
    """Same spec twice = byte-identical CSV."""
    g = _gen()
    a = _gen_csv(g, tmp_path / "a").read_bytes()
    b = _gen_csv(g, tmp_path / "b").read_bytes()
    assert a == b


def test_generator_seed_changes_bytes(tmp_path: Path) -> None:
    """Different seed = different CSV."""
    a = _gen_csv(_gen(seed=1), tmp_path / "a").read_bytes()
    b = _gen_csv(_gen(seed=2), tmp_path / "b").read_bytes()
    assert a != b


# --- controlled sparsity descriptors ------------------------------------------


def test_occupied_fraction_exact(tmp_path: Path) -> None:
    """Occupied cell count hits the target exactly (choice without replacement)."""
    achieved = regimes.generate_synthetic(_gen(occupied_fraction=0.31), tmp_path / "a.csv")
    assert achieved["n_occupied"] == 3100
    assert achieved["occupied_fraction"] == pytest.approx(0.31, abs=1e-9)


def test_points_per_cell_targeted(tmp_path: Path) -> None:
    """Poisson(points_per_cell) clamped at 1 lands near the target mean."""
    achieved = regimes.generate_synthetic(_gen(points_per_cell=8.0), tmp_path / "a.csv")
    assert achieved["points_per_occupied_cell"] == pytest.approx(8.0, rel=0.12)
    assert achieved["n_points"] > 0


def test_ppc_zero_rejected(tmp_path: Path) -> None:
    """points_per_cell <= 0 is a spec error."""
    with pytest.raises(ValueError, match="points_per_cell"):
        regimes.generate_synthetic(_gen(points_per_cell=0.0), tmp_path / "a.csv")


def test_extent_must_be_multiple_of_res(tmp_path: Path) -> None:
    """A domain that is not a whole number of cells is a spec error."""
    with pytest.raises(ValueError, match="whole multiples"):
        regimes.generate_synthetic(_gen(width_m=1005.0), tmp_path / "a.csv")


# --- density families ----------------------------------------------------------


def _occupancy(gen: dict, d: Path) -> tuple[np.ndarray, dict]:
    """Boolean (ny, nx) map of occupied cells, from the generated CSV."""
    import pandas as pd
    from pyproj import Transformer

    achieved = regimes.generate_synthetic(gen, d / "a.csv")
    df = pd.read_csv(d / "a.csv")
    tr = Transformer.from_crs(4326, 6933, always_xy=True)
    x, y = tr.transform(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    x0, _y0, _x1, y1, _tr, _tri = regimes._projected_domain(gen["center"], gen["width_m"], gen["height_m"])
    nx = int(gen["width_m"] // gen["res_m"])
    ny = int(gen["height_m"] // gen["res_m"])
    res = gen["res_m"]
    ii = np.clip(((x - x0) // res).astype(int), 0, nx - 1)
    jj = np.clip(((y1 - y) // res).astype(int), 0, ny - 1)
    occupied = np.zeros((ny, nx), dtype=bool)
    occupied[jj, ii] = True
    return occupied, achieved


def test_clustered_concentrates_occupied_cells(tmp_path: Path) -> None:
    """Clustered occupancy beats the uniform null in the cluster capture region."""
    g = _gen(density={"mode": "clustered", "n_clusters": 4, "boost": 50.0, "sigma_frac": 0.08})
    occ, achieved = _occupancy(g, tmp_path / "c")
    occ_u, _ach_u = _occupancy(_gen(density={"mode": "uniform"}), tmp_path / "u")
    x0, _y0, _x1, y1, _tr, _tri = regimes._projected_domain(g["center"], g["width_m"], g["height_m"])
    w, h = g["width_m"], g["height_m"]
    res = g["res_m"]
    rng = np.random.default_rng(g["seed"])
    margin = 0.3
    cu = rng.uniform(margin, 1.0 - margin, (4, 2))
    centers = np.column_stack((x0 + cu[:, 0] * w, y1 - cu[:, 1] * h))
    sigma = g["density"]["sigma_frac"] * np.hypot(w, h)
    ny, nx = occ.shape
    gx, gy = np.meshgrid(x0 + (np.arange(nx) + 0.5) * res, y1 - (np.arange(ny) + 0.5) * res)
    in_cap = np.zeros_like(occ, dtype=bool)
    for a, b in centers:
        in_cap |= (gx - a) ** 2 + (gy - b) ** 2 <= sigma**2
    rate_c = occ[in_cap].sum() / in_cap.sum()
    rate_u = occ_u[in_cap].sum() / in_cap.sum()
    assert rate_c > 2.0 * rate_u
    assert achieved["n_occupied"] == int(0.25 * occ.size)


def test_edge_mode_concentrates_on_rim(tmp_path: Path) -> None:
    """Edge weighting puts more occupied cells per cell on the rim than the core."""
    g = _gen(density={"mode": "edge", "gamma": 3.0, "boost": 4.0}, occupied_fraction=0.3)
    occ, _ach = _occupancy(g, tmp_path / "e")
    ny, nx = occ.shape
    cy = np.arange(ny)[:, None] + 0.5
    cx = np.arange(nx)[None, :] + 0.5
    d = np.hypot(cx - nx / 2, cy - ny / 2)
    dmax = 0.5 * np.hypot(nx, ny)
    rim = d / dmax > 0.75
    core = d / dmax < 0.4
    rate_rim = occ[rim].sum() / rim.sum()
    rate_core = occ[core].sum() / core.sum()
    assert rate_rim > rate_core


def test_unknown_density_mode_rejected() -> None:
    """Density modes outside the allow-list fail validation."""
    with pytest.raises(ValueError, match="density mode"):
        regimes._validate_cfg(_cfg("x", _gen(density={"mode": "spiral"})))


def test_unknown_pipeline_key_rejected() -> None:
    """Pipeline keys outside the allow-list fail validation."""
    with pytest.raises(ValueError, match="pipeline key"):
        regimes._validate_cfg(_cfg("x", _gen(), {"res": 10.0, "zoom": 2}))


# --- wall gate logic -------------------------------------------------------------


def _fake_run(cid: str, total: float, cfg: dict | None = None) -> dict:
    """Return a minimal results record for wall-gate unit tests."""
    timings = {"ingest": 1.0, "calibrate": 0.0, "grid": total - 3.0, "write": 2.0, "total": total}
    return {"config_id": cid, "config": cfg or {"res": 10.0}, "timings_s": timings}


def _write_baseline(tmp_path: Path, runs: list[dict], n_cpus: int | None = 4) -> Path:
    """Write a baseline results doc and return its path."""
    p = tmp_path / "base.json"
    doc = {"schema_version": 1, "env": {"n_cpus": n_cpus}, "runs": runs}
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def test_wall_gate_pass_and_fail(tmp_path: Path) -> None:
    """Ratios under the threshold pass; over it fails the run."""
    n = os.cpu_count()
    base = _write_baseline(tmp_path, [_fake_run("a", 10.0), _fake_run("b", 10.0)], n_cpus=n)
    ok = regimes.check_wall([_fake_run("a", 15.0), _fake_run("b", 15.0)], base, 2.0)
    assert ok["status"] == "pass"
    assert ok["per_config"]["a"]["wall_ratio"] == pytest.approx(1.5)
    bad = regimes.check_wall([_fake_run("a", 21.0), _fake_run("b", 15.0)], base, 2.0)
    assert bad["status"] == "fail"
    assert bad["per_config"]["a"]["status"] == "fail"
    assert bad["per_config"]["b"]["status"] == "pass"


def test_wall_gate_skips_on_machine_or_spec_change(tmp_path: Path) -> None:
    """n_cpus drift or config drift skips (no fail) for that config."""
    base = _write_baseline(tmp_path, [_fake_run("a", 10.0)], n_cpus=os.cpu_count() + 1)
    sk = regimes.check_wall([_fake_run("a", 99.0)], base, 2.0)
    assert sk["status"] == "skipped"
    assert any("n_cpus" in r for r in sk["per_config"]["a"]["reasons"])
    base2 = _write_baseline(tmp_path, [_fake_run("a", 10.0)], n_cpus=os.cpu_count())
    ch = regimes.check_wall([_fake_run("a", 99.0, {"res": 50.0})], base2, 2.0)
    assert ch["status"] == "skipped"
    assert ch["per_config"]["a"]["status"] == "config-changed"


def test_wall_gate_missing_baseline_file(tmp_path: Path) -> None:
    """No baseline file (or no baseline) = gate off, never a fail."""
    off = regimes.check_wall([_fake_run("a", 99.0)], tmp_path / "nope.json", 2.0)
    assert off["status"] == "off"
    assert "missing" in off["note"]
    none = regimes.check_wall([_fake_run("a", 99.0)], None, 2.0)
    assert none["status"] == "off"


# --- end-to-end: record schema + rerun bit-identity -------------------------------

_GEN_E2E = _gen(width_m=1000.0, height_m=1000.0, res_m=100.0, points_per_cell=2.0, occupied_fraction=0.5, seed=7)


@pytest.fixture(scope="module")
def e2e_runs(tmp_path_factory: pytest.TempPathFactory) -> list[dict]:
    """Two full pipeline runs of the same 100-cell config (rerun identity)."""
    out = tmp_path_factory.mktemp("regimes-e2e")
    e2e_pipeline = {"res": 100.0, "cap_km": 2.0, "transform": "identity", "skip_calibration": True, "workers": 2}
    cfg = _cfg("e2e", _GEN_E2E, e2e_pipeline)
    recs = []
    for i in range(2):
        d = out / f"run{i}"
        d.mkdir()
        recs.append(regimes.run_worker(cfg, d))
    return recs


def test_e2e_rerun_bit_identical(e2e_runs: list[dict]) -> None:
    """Same config + machine reruns byte-identical tifs (anchor gate premise)."""
    a, b = e2e_runs
    assert a["shas"]["value"] == b["shas"]["value"]
    assert a["shas"]["support"] == b["shas"]["support"]
    assert a["n_points"] == b["n_points"]


def test_e2e_record_schema(e2e_runs: list[dict]) -> None:
    """The results record carries every field the issue's metrics list names."""
    rec = e2e_runs[0]
    for k in (
        "config_id",
        "config",
        "generator",
        "n_points",
        "grid",
        "timings_s",
        "subphases_s",
        "wall_process_s",
        "peak_rss_mib",
        "temp_disk_hwm_mb",
        "outputs_bytes",
        "shas",
    ):
        assert k in rec, k
    for k in ("ingest", "calibrate", "grid", "write", "total"):
        assert k in rec["timings_s"], k
    assert rec["timings_s"]["total"] > 0
    assert rec["peak_rss_mib"] > 0
    assert rec["outputs_bytes"]["value.tif"] > 0
    assert rec["outputs_bytes"]["support_km.tif"] > 0
    # 1 km x 1 km @ 100 m domain; the occupied subset cannot exceed it.
    assert 1 <= rec["grid"]["nx"] <= 10
    assert 1 <= rec["grid"]["ny"] <= 10
    assert rec["generator"]["n_occupied"] == 50  # 0.5 x 100 cells, exact
    assert rec["n_points"] == rec["generator"]["n_points"]  # ingest lost nothing
