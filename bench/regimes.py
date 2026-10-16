"""Versioned regime bench harness (issue #108).

Runs controlled-sparsity regimes through the full pipeline and records, per
run: per-phase + total wall, peak RSS, temp-disk high-water, final output
sizes. Every run carries a correctness check so a fast invalid output cannot
win, and the raw results + config + commit hash are written to versioned
JSON under bench/results/ (see bench/results/README.md).

Config spec
-----------
The regime table is a versioned config spec (``SCHEMA_VERSION``). The three
density families shipped in CONFIGS (uniform / clustered-with-gaps /
edge-heavy) are SEEDS for Andryo to extend, not fixed code paths: the
generator is driven purely by numeric descriptors and a per-family parameter
block, and ``--config-file`` replaces the built-in table wholesale.

Sparsity descriptors (synthetic generators):
    points_per_cell    target mean points per OCCUPIED cell (Poisson lam,
                       clamped to >= 1; achieved value is reported)
    occupied_fraction  target fraction of domain cells holding >= 1 point
                       (the exact count is sampled, so it is achieved exactly)

Synthetic generator spec (``generator.type == "synthetic"``):
    seed                int, np.random.default_rng seed
    center              [lon, lat] domain centre (EPSG:4326)
    width_m, height_m   domain extent in metres (working-CRS metres; must be
                        whole multiples of res_m)
    res_m               cell size in metres
    points_per_cell     float > 0
    occupied_fraction   float in (0, 1]
    density             {"mode": "uniform" | "clustered" | "edge", ...}
                        clustered: n_clusters (8), boost (8.0), sigma_frac (0.15)
                        edge:      gamma (3.0), boost (4.0)
    value               {"base" (5), "amp" (40), "noise_sigma" (0.5),
                         "sigma_frac" (0.2)} — smooth blob field + lognormal
                        noise, so the surface is positive and structured

Draw order is fixed (cluster centres -> occupied cells -> per-cell counts ->
in-cell offsets -> value blob centres -> value noise), so a spec regenerates
byte-identical CSVs on every platform numpy runs on.

Dataset generator spec (``generator.type == "dataset"``):
    path        repo-relative CSV under data/ (the committed fixtures)
    value_col   column name (default "value")

Pipeline spec (``pipeline``): plain Pipeline() kwargs (allow-listed below);
``workers`` is pinned in the table so runs are comparable.

Anchors / correctness
---------------------
Every config in the table carries an ``anchor`` pointing at committed GeoTIFFs
(dataset configs reuse examples/; synthetic configs use examples/regimes/).
On the canonical platform (x86_64 Linux, where the anchors were cut) the
gate is byte-exact sha256; elsewhere it falls back to functional equivalence
(same grid/metadata, near-identical decoded pixels — the check_anchors.py
rules). A config with no anchor is recorded as ``no-anchor`` (warning, not a
failure) — the wall gate result for it is then unverified.

Wall gate
---------
With ``--baseline FILE`` (default: bench/results/baseline/<set>.json when it
exists) the harness compares each run's total wall against the baseline's and
fails (exit 1) on any ratio above ``--wall-threshold`` (default 2.0). The
gate is skipped (not failed) when the baseline was cut on a different CPU
count or the config spec changed — comparing walls across machines or specs
would gate noise. Per-phase ratios are always recorded (no per-phase gate).
The per-run ``vs_baseline`` block is where #107's baseline-comparison column
lands; it exists now so that work adds values, not schema.

Usage
-----
    # default table, ci set, correctness + wall gate vs the committed baseline
    uv run python bench/regimes.py --outdir /tmp/regimes

    # full table (hydrogen analysis set)
    uv run python bench/regimes.py --set all --outdir /tmp/regimes

    # Andryo's extended spec, explicit ids, no wall gate
    uv run python bench/regimes.py --config-file my-spec.json --ids a,b \
        --outdir /tmp/regimes --no-wall-gate

    # worker mode (internal, spawned per config):
    uv run python bench/regimes.py --worker --config-json '{...}' --outdir /tmp/regimes

Exit codes: 0 = correctness + wall gate pass, 1 = gate failure or run crash,
2 = config/spec error.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SCRIPT = Path(__file__).resolve()

SCHEMA_VERSION = 1
WALL_GATE_THRESHOLD = 2.0
_TEMP_SAMPLE_S = 0.05

# Pipeline() kwargs the config spec may set; anything else is a spec error.
_PIPELINE_KEYS = frozenset(
    {
        "res",
        "cap_km",
        "transform",
        "saturation",
        "block_size",
        "workers",
        "calib_path",
        "scale",
        "compress",
        "percentile_step",
        "calib_max_points",
        "src_crs",
        "work_crs",
        "out_crs",
        "skip_calibration",
        "max_band_parallel",
        "n_threads",
        "turbo",
        "turbo_cap_gb",
        "turbo_strict",
        "seed",
        "dry_run",
    }
)

_SYN_KEYS = frozenset(
    {
        "type",
        "seed",
        "center",
        "width_m",
        "height_m",
        "res_m",
        "points_per_cell",
        "occupied_fraction",
        "density",
        "value",
    }
)
_DS_KEYS = frozenset({"type", "path", "value_col"})
_CFG_KEYS = frozenset({"id", "ci", "generator", "pipeline", "anchor", "note"})
_DENSITY_KEYS = {
    "uniform": frozenset({"mode"}),
    "clustered": frozenset({"mode", "n_clusters", "boost", "sigma_frac"}),
    "edge": frozenset({"mode", "gamma", "boost"}),
}
_VALUE_KEYS = frozenset({"base", "amp", "noise_sigma", "sigma_frac"})


def _validate_cfg(cfg: dict) -> None:
    """Validate one config spec entry.

    Raises:
        ValueError: On any spec drift (unknown keys, out-of-range values).
    """
    for k in cfg:
        if k not in _CFG_KEYS:
            msg = f"unknown config key {k!r} in {cfg.get('id', '?')!r}"
            raise ValueError(msg)
    if not cfg.get("id"):
        msg = "config missing 'id'"
        raise ValueError(msg)
    gen = cfg.get("generator")
    if not isinstance(gen, dict):
        msg = f"{cfg['id']}: missing 'generator'"
        raise ValueError(msg)
    allowed = _SYN_KEYS if gen.get("type") == "synthetic" else _DS_KEYS
    for k in gen:
        if k not in allowed:
            msg = f"{cfg['id']}: unknown generator key {k!r}"
            raise ValueError(msg)
    if gen.get("type") == "synthetic":
        d = gen.get("density") or {"mode": "uniform"}
        if d.get("mode", "uniform") not in _DENSITY_KEYS:
            msg = f"{cfg['id']}: unknown density mode {d.get('mode')!r}"
            raise ValueError(msg)
        for k in d:
            if k not in _DENSITY_KEYS[d["mode"]]:
                msg = f"{cfg['id']}: unknown density key {k!r} for mode {d['mode']!r}"
                raise ValueError(msg)
        v = gen.get("value") or {}
        for k in v:
            if k not in _VALUE_KEYS:
                msg = f"{cfg['id']}: unknown value key {k!r}"
                raise ValueError(msg)
        if not (gen["points_per_cell"] > 0):
            msg = f"{cfg['id']}: points_per_cell must be > 0"
            raise ValueError(msg)
        if not (0 < gen["occupied_fraction"] <= 1):
            msg = f"{cfg['id']}: occupied_fraction must be in (0, 1]"
            raise ValueError(msg)
    elif gen.get("type") != "dataset":
        msg = f"{cfg['id']}: unknown generator type {gen.get('type')!r}"
        raise ValueError(msg)
    pw = cfg.get("pipeline") or {}
    for k in pw:
        if k not in _PIPELINE_KEYS:
            msg = f"{cfg['id']}: unknown pipeline key {k!r}"
            raise ValueError(msg)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def _projected_domain(center: list[float], w_m: float, h_m: float) -> tuple[float, float, float, float, object, object]:
    """Fit the projected (EPSG:6933 metres) rectangle centred at [lon, lat].

    Wagner VII is cylindrical, so projecting the four corners yields the exact
    rectangle. The iterative lon/lat box fit mirrors bench/bench_tier2.py.

    Returns:
        (x0, y0, x1, y1, transformer, inverse_transformer) with x0 < x1,
        y0 < y1; both transformers are always_xy.

    """
    from pyproj import Transformer

    tr = Transformer.from_crs(4326, 6933, always_xy=True)
    tr_inv = Transformer.from_crs(6933, 4326, always_xy=True)
    clon, clat = center
    half_lon = w_m / 87000.0 / 2
    half_lat = h_m / 110000.0 / 2
    for _ in range(8):
        lons = np.array([clon - half_lon, clon, clon + half_lon])
        lats = np.array([clat - half_lat, clat, clat + half_lat])
        xx, yy = np.meshgrid(lons, lats)
        x, y = tr.transform(xx.ravel(), yy.ravel())
        dx, dy = x.max() - x.min(), y.max() - y.min()
        half_lon *= w_m / dx
        half_lat *= h_m / dy
    xa, ya = tr.transform(clon - half_lon, clat - half_lat)
    xb, yb = tr.transform(clon + half_lon, clat + half_lat)
    return min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb), tr, tr_inv


def generate_synthetic(gen: dict, path: Path) -> dict:
    """Write the seeded synthetic CSV for a ``synthetic`` generator spec.

    Deterministic: for a fixed spec the numpy PCG64 stream is consumed in a
    fixed draw order (see the module docstring), so the CSV bytes are stable
    across platforms and numpy versions within the pinned lock.

    Returns:
        Achieved descriptor dict (n_points, n_cells, n_occupied,
        occupied_fraction, points_per_occupied_cell) for the results record.

    Raises:
        ValueError: On descriptor violations (extent not a whole multiple of
            res, occupied count out of range).

    """
    import pandas as pd

    res = float(gen["res_m"])
    w_m = float(gen["width_m"])
    h_m = float(gen["height_m"])
    ppc = float(gen["points_per_cell"])
    occ_frac = float(gen["occupied_fraction"])
    if ppc <= 0:
        msg = f"points_per_cell must be > 0 (got {ppc})"
        raise ValueError(msg)
    nx = round(w_m / res)
    ny = round(h_m / res)
    if abs(nx * res - w_m) > 1e-6 * w_m or abs(ny * res - h_m) > 1e-6 * h_m:
        msg = f"width_m/height_m must be whole multiples of res_m ({w_m}x{h_m} @ {res})"
        raise ValueError(msg)
    n_cells = nx * ny
    n_occ = round(occ_frac * n_cells)
    if not (1 <= n_occ <= n_cells):
        msg = f"occupied_fraction {occ_frac} of {n_cells} cells is out of [1, {n_cells}]"
        raise ValueError(msg)

    rng = np.random.default_rng(int(gen["seed"]))
    x0, _y0, _x1, y1, _tr, tr_inv = _projected_domain(gen["center"], w_m, h_m)

    d = gen.get("density") or {}
    mode = d.get("mode", "uniform")
    # Cell centres: row 0 = north edge (y1), column 0 = west edge (x0).
    cx = x0 + (np.arange(nx) + 0.5) * res  # (nx,)
    cy = y1 - (np.arange(ny) + 0.5) * res  # (ny,)
    cxy = np.stack(np.meshgrid(cx, cy, indexing="ij"), axis=-1).reshape(-1, 2)  # (n_cells, 2)

    if mode == "clustered":
        k = int(d.get("n_clusters", 8))
        beta = float(d.get("boost", 8.0))
        sigma = float(d.get("sigma_frac", 0.15)) * math.hypot(w_m, h_m)
        margin = 0.3
        # Draw 1: cluster centres (unit square, scaled into the domain).
        cu = rng.uniform(margin, 1.0 - margin, (k, 2))
        cc = np.column_stack((x0 + cu[:, 0] * w_m, y1 - cu[:, 1] * h_m))
        w = np.ones(n_cells)
        for a, b in cc:
            w += beta * np.exp(-((cxy[:, 0] - a) ** 2 + (cxy[:, 1] - b) ** 2) / (2.0 * sigma * sigma))
        blob_centers = cc
    elif mode == "edge":
        gamma = float(d.get("gamma", 3.0))
        beta = float(d.get("boost", 4.0))
        r = np.hypot(cxy[:, 0] - (x0 + w_m / 2), cxy[:, 1] - (y1 - h_m / 2)) / (0.5 * math.hypot(w_m, h_m))
        w = 1.0 + beta * r**gamma
        # Draw 1 (this branch): two value-blob centres.
        cu = rng.uniform(0.2, 0.8, (2, 2))
        blob_centers = np.column_stack((x0 + cu[:, 0] * w_m, y1 - cu[:, 1] * h_m))
    elif mode == "uniform":
        w = np.ones(n_cells)
        cu = rng.uniform(0.2, 0.8, (2, 2))
        blob_centers = np.column_stack((x0 + cu[:, 0] * w_m, y1 - cu[:, 1] * h_m))
    else:  # validated by _validate_cfg; keep the raise for direct callers
        msg = f"unknown density mode {mode!r}"
        raise ValueError(msg)

    p = w / w.sum()
    # Draw 2: occupied cells (exact count, weighted without replacement).
    occ_idx = rng.choice(n_cells, size=n_occ, replace=False, p=p)
    # Draw 3: points per occupied cell (Poisson, clamped to >= 1).
    counts = np.maximum(1, rng.poisson(ppc, size=n_occ))
    cell_of_pt = np.repeat(occ_idx, counts)
    tot = int(counts.sum())
    # Draw 4: in-cell offsets.
    u = rng.random(tot)
    v = rng.random(tot)
    ii = cell_of_pt % nx
    jj = cell_of_pt // nx
    px = x0 + (ii + u) * res
    py = y1 - (jj + v) * res

    # Value surface: base + smooth blobs (cluster centres when clustered) + noise.
    val = gen.get("value") or {}
    base = float(val.get("base", 5.0))
    amp = float(val.get("amp", 40.0))
    vsigma = float(val.get("sigma_frac", 0.2)) * math.hypot(w_m, h_m)
    nsig = float(val.get("noise_sigma", 0.5))
    field = np.full(tot, base)
    # Draw 5/6: (blob centres already drawn for uniform/edge) then noise.
    for a, b in blob_centers:
        field += amp * np.exp(-((px - a) ** 2 + (py - b) ** 2) / (2.0 * vsigma * vsigma))
    field += rng.lognormal(0.0, nsig, tot)

    lon, lat = tr_inv.transform(px, py)
    df = pd.DataFrame({"value": field, "longitude": lon, "latitude": lat})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return {
        "n_points": tot,
        "n_cells": int(n_cells),
        "n_occupied": int(n_occ),
        "occupied_fraction": round(n_occ / n_cells, 6),
        "points_per_occupied_cell": round(float(counts.mean()), 4),
    }


def make_input(cfg: dict, data_dir: Path) -> tuple[Path, dict]:
    """Materialise a config's input CSV. Returns (csv_path, achieved dict).

    Raises:
        ValueError: When a dataset generator's file is missing.
    """
    gen = cfg["generator"]
    if gen["type"] == "dataset":
        csv = REPO / gen["path"]
        if not csv.is_file():
            msg = f"{cfg['id']}: dataset path missing: {gen['path']}"
            raise ValueError(msg)
        return csv, {"dataset": gen["path"]}
    csv = data_dir / f"{cfg['id']}.csv"
    return csv, generate_synthetic(gen, csv)


# ---------------------------------------------------------------------------
# Default table (seeds, not fixed: replace via --config-file)
# ---------------------------------------------------------------------------


def _syn(
    seed: int,
    center: list[float],
    w_m: float,
    h_m: float,
    res_m: float,
    ppc: float,
    occ: float,
    density: dict,
) -> dict:
    return {
        "type": "synthetic",
        "seed": seed,
        "center": center,
        "width_m": w_m,
        "height_m": h_m,
        "res_m": res_m,
        "points_per_cell": ppc,
        "occupied_fraction": occ,
        "density": density,
    }


# Domain: 10 km x 10 km around Melbourne at 10 m (1000x1000 cells) for the CI
# set — sized so the whole ci set stays under ~15 min on a 4 vCPU / 16 GB
# ubuntu-latest runner (2026-10-16, measured on hydrogen, ~4-6x slower there).
_MELB_CENTRE = [144.995, -37.75]
_W4 = {"workers": 4}  # pinned worker count: comparable runs


def _ci(pw: dict) -> dict:
    return dict(_W4, **pw)


def _syn_anchor(cfg_id: str) -> dict:
    return {
        "value": f"examples/regimes/{cfg_id}/value.tif",
        "support": f"examples/regimes/{cfg_id}/support_km.tif",
    }


def default_configs() -> list[dict]:
    """The built-in regime table (issue #108 seeds)."""
    table: list[dict] = [
        # -- real datasets: two materially different anchors (s29.2).
        # Pipeline kwargs mirror bench/bench_tier2.py's anchored configs, so
        # the committed examples/ outputs ARE the correctness anchors.
        {
            "id": "melb10",
            "ci": True,
            "generator": {"type": "dataset", "path": "data/melb_houses.csv", "value_col": "price"},
            "pipeline": _ci(
                {
                    "res": 10.0,
                    "cap_km": 10.0,
                    "skip_calibration": True,
                }
            ),
            "anchor": {"value": "examples/melb/10m/value.tif", "support": "examples/melb/10m/support_km.tif"},
            "note": "urban, 13.5k pts, 10 m — the canonical CI-safe anchor config",
        },
        {
            "id": "melb500",
            "ci": True,
            "generator": {"type": "dataset", "path": "data/melb_houses.csv", "value_col": "price"},
            "pipeline": _ci(
                {
                    "res": 500.0,
                    "cap_km": 10.0,
                    "skip_calibration": True,
                }
            ),
            "anchor": {"value": "examples/melb/500m/value.tif", "support": "examples/melb/500m/support_km.tif"},
            "note": "same data, coarse res — resolution axis",
        },
        {
            "id": "equakes500",
            "ci": False,
            "generator": {"type": "dataset", "path": "data/all_equakes.csv", "value_col": "mag"},
            "pipeline": _ci(
                {
                    "res": 500.0,
                    "cap_km": 25.0,
                    "transform": "identity",
                    "skip_calibration": True,
                    "out_crs": 6933,
                }
            ),
            "anchor": {"value": "examples/equakes/value.tif", "support": "examples/equakes/support_km.tif"},
            "note": (
                "seismic, 44k sparse pts, 500 m, out_crs 6933 — the second real anchor. "
                "ci: false: 8.1-8.4 GB peak RSS exceeds the 16 GB ubuntu-latest headroom "
                "the anchor gate already reserves (hydrogen-only)."
            ),
        },
        # -- synthetic sparsity sweep (10 km x 10 km @ 10 m, 1M cells).
        # ppc = points per OCCUPIED cell; occ = occupied-cell fraction.
        {
            "id": "syn-unif-p1-o25",
            "ci": True,
            "generator": _syn(20261, _MELB_CENTRE, 10_000, 10_000, 10.0, 1.0, 0.25, {"mode": "uniform"}),
            "pipeline": _ci({"res": 10.0, "cap_km": 8.0, "transform": "identity", "skip_calibration": True}),
            "anchor": _syn_anchor("syn-unif-p1-o25"),
            "note": "sparse + gappy: 1 pt/cell in 25% of cells (~340k pts)",
        },
        {
            "id": "syn-unif-p8-o50",
            "ci": True,
            "generator": _syn(20262, _MELB_CENTRE, 10_000, 10_000, 10.0, 8.0, 0.5, {"mode": "uniform"}),
            "pipeline": _ci({"res": 10.0, "cap_km": 8.0, "transform": "identity", "skip_calibration": True}),
            "anchor": _syn_anchor("syn-unif-p8-o50"),
            "note": "dense uniform: 8 pts/cell in 50% of cells (~4M pts)",
        },
        {
            "id": "syn-cluster-p4-o30",
            "ci": True,
            "generator": _syn(
                20263,
                _MELB_CENTRE,
                10_000,
                10_000,
                10.0,
                4.0,
                0.30,
                {"mode": "clustered", "n_clusters": 8, "boost": 8.0, "sigma_frac": 0.15},
            ),
            "pipeline": _ci({"res": 10.0, "cap_km": 8.0, "transform": "identity", "skip_calibration": True}),
            "anchor": _syn_anchor("syn-cluster-p4-o30"),
            "note": "clustered-with-gaps: 8 hot clusters, 4 pts/cell in 30% of cells (~1.2M pts)",
        },
        {
            "id": "syn-edge-p4-o40",
            "ci": True,
            "generator": _syn(
                20264,
                _MELB_CENTRE,
                10_000,
                10_000,
                10.0,
                4.0,
                0.40,
                {"mode": "edge", "gamma": 3.0, "boost": 4.0},
            ),
            "pipeline": _ci({"res": 10.0, "cap_km": 8.0, "transform": "identity", "skip_calibration": True}),
            "anchor": _syn_anchor("syn-edge-p4-o40"),
            "note": "edge-heavy: density rises toward the domain rim, 4 pts/cell in 40% of cells (~1.6M pts)",
        },
        # -- hydrogen-only extension (ci: false): denser/sparser corners of the
        # sparsity surface for Andryo's cost curves.
        {
            "id": "syn-unif-p1-o100",
            "ci": False,
            "generator": _syn(20265, _MELB_CENTRE, 10_000, 10_000, 10.0, 1.0, 1.0, {"mode": "uniform"}),
            "pipeline": _ci({"res": 10.0, "cap_km": 8.0, "transform": "identity", "skip_calibration": True}),
            "anchor": _syn_anchor("syn-unif-p1-o100"),
            "note": "fully occupied, 1 pt/cell (~1.37M pts)",
        },
        {
            "id": "syn-unif-p16-o10",
            "ci": False,
            "generator": _syn(20266, _MELB_CENTRE, 10_000, 10_000, 10.0, 16.0, 0.10, {"mode": "uniform"}),
            "pipeline": _ci({"res": 10.0, "cap_km": 8.0, "transform": "identity", "skip_calibration": True}),
            "anchor": _syn_anchor("syn-unif-p16-o10"),
            "note": "dense clumps in 10% of cells (~1.6M pts)",
        },
        {
            "id": "syn-cluster-p16-o10",
            "ci": False,
            "generator": _syn(
                20267,
                _MELB_CENTRE,
                10_000,
                10_000,
                10.0,
                16.0,
                0.10,
                {"mode": "clustered", "n_clusters": 16, "boost": 16.0, "sigma_frac": 0.10},
            ),
            "pipeline": _ci({"res": 10.0, "cap_km": 8.0, "transform": "identity", "skip_calibration": True}),
            "anchor": _syn_anchor("syn-cluster-p16-o10"),
            "note": "16 tight clusters, 16 pts/cell where occupied (~1.6M pts)",
        },
        {
            "id": "equakes100",
            "ci": False,
            "generator": {"type": "dataset", "path": "data/all_equakes.csv", "value_col": "mag"},
            "pipeline": _ci(
                {
                    "res": 100.0,
                    "cap_km": 25.0,
                    "transform": "identity",
                    "skip_calibration": True,
                    "out_crs": 6933,
                }
            ),
            "anchor": None,
            "note": "5x finer than the equakes anchor; no committed anchor yet (correctness = no-anchor warn)",
        },
    ]
    for c in table:
        _validate_cfg(c)
    return table


def load_configs(args: argparse.Namespace) -> list[dict]:
    """Resolve the selected config list from CLI + spec file.

    Raises:
        SystemExit: 2 on unreadable/invalid spec file or empty selection.
    """
    if args.config_file:
        spec = json.loads(Path(args.config_file).read_text(encoding="utf-8"))
        if spec.get("schema_version") != SCHEMA_VERSION:
            msg = f"config file schema_version {spec.get('schema_version')!r} != {SCHEMA_VERSION}"
            raise SystemExit(msg)
        table = spec["configs"]
    else:
        table = default_configs()
    for c in table:
        _validate_cfg(c)
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
        known = {c["id"] for c in table}
        missing = wanted - known
        if missing:
            msg = f"unknown config id(s): {', '.join(sorted(missing))} (known: {', '.join(sorted(known))})"
            raise SystemExit(msg)
        table = [c for c in table if c["id"] in wanted]
    elif args.set == "ci":
        table = [c for c in table if c.get("ci")]
    if not table:
        msg = "empty config selection"
        raise SystemExit(msg)
    return table


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _peak_rss_mib() -> float:
    """Process peak RSS in MiB (0 where the platform cannot report it)."""
    try:
        import resource
    except ImportError:  # pragma: no cover - non-Unix
        return 0.0
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(v / 1_048_576, 1)
    return round(v / 1024, 1)


def _mem_total_gb() -> float | None:
    try:
        with Path("/proc/meminfo").open(encoding="ascii") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1_048_576, 1)
    except (OSError, ValueError):
        pass
    return None
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_temp_name(name: str) -> bool:
    """Pipeline temp files all start with _ or end .tmp; finals never do."""
    return name.startswith("_") or name.endswith(".tmp")


class _TempSampler(threading.Thread):
    """Sample the temp-file footprint in the run dir; hwm = max summed size.

    The pipeline keeps every per-run temp inside the output dir (see
    ppgrid.pipeline._run_temp_names plus the _stock_scratch_*.tif frames);
    the final outputs (value.tif / support_km.tif / calibration.json) match
    neither the ``_`` prefix nor the ``.tmp`` suffix, so the sum tracks temp
    disk usage only. A 50 ms sample is coarse against sub-second phases but
    the temps grow over seconds — the high-water is the point, not the curve.

    """

    def __init__(self, d: Path) -> None:
        super().__init__(daemon=True)
        self.d = d
        self.hwm = 0
        self.samples = 0
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            total = 0
            try:
                for f in self.d.iterdir():
                    if f.is_file() and _is_temp_name(f.name):
                        with contextlib.suppress(OSError):  # temp removed mid-scan
                            total += f.stat().st_size
            except OSError:
                pass
            self.hwm = max(self.hwm, total)
            self.samples += 1
            self._stop.wait(_TEMP_SAMPLE_S)

    def stop(self) -> None:
        self._stop.set()
        self.join()


def _env_block() -> dict:
    """Machine + toolchain fingerprint for the results record."""
    import pyproj
    import rasterio

    from ppgrid import __version__, _cinterp

    return {
        "os": sys.platform,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "n_cpus": os.cpu_count(),
        "mem_total_gb": _mem_total_gb(),
        "python": platform.python_version(),
        "ppgrid": __version__,
        "numpy": np.__version__,
        "pyproj": pyproj.__version__,
        "rasterio": rasterio.__version__,
        "gdal": ".".join(str(p) for p in rasterio.gdal_version()),
        "cinterp_backend": _cinterp.backend(),
    }


# ---------------------------------------------------------------------------
# Worker (one subprocess per config: honest per-run RSS + crash isolation)
# ---------------------------------------------------------------------------


def run_worker(cfg: dict, outdir: Path) -> dict:
    """Run one config end-to-end and return its results record."""
    from ppgrid import Pipeline

    gen = cfg["generator"]
    csv, achieved = make_input(cfg, outdir / "data")
    cdir = outdir / cfg["id"]
    cdir.mkdir(parents=True, exist_ok=True)
    p = Pipeline(
        str(csv),
        gen.get("value_col", "value"),
        "longitude",
        "latitude",
        str(cdir),
        **cfg.get("pipeline", {}),
    )
    sampler = _TempSampler(cdir)
    sampler.start()
    t0 = time.perf_counter()
    try:
        p.run()
    finally:
        sampler.stop()
    wall_proc = time.perf_counter() - t0
    s = p.run_summary()
    tm = s["timings_ms"]
    cal = cdir / "calibration.json"
    rec = {
        "config_id": cfg["id"],
        "config": cfg,
        "generator": achieved,
        "n_points": s["ingest"]["points"],
        "grid": s["grid"],
        "timings_s": {k: round(tm[k] / 1000.0, 3) for k in ("ingest", "calibrate", "grid", "write", "total")},
        "subphases_s": {k: round(v / 1000.0, 3) for k, v in tm.get("subphases", {}).items()},
        "wall_process_s": round(wall_proc, 3),
        "peak_rss_mib": _peak_rss_mib(),
        "temp_disk_hwm_mb": round(sampler.hwm / 1e6, 2),
        "temp_samples": sampler.samples,
        "outputs_bytes": {
            "value.tif": (cdir / "value.tif").stat().st_size,
            "support_km.tif": (cdir / "support_km.tif").stat().st_size,
            "calibration.json": cal.stat().st_size if cal.is_file() else 0,
        },
        "shas": {
            "value": _sha256(cdir / "value.tif"),
            "support": _sha256(cdir / "support_km.tif"),
        },
    }
    return rec


def _worker_main(config_json: str, outdir: Path) -> None:
    cfg = json.loads(config_json)
    _validate_cfg(cfg)
    rec = run_worker(cfg, outdir)
    out = outdir / cfg["id"] / "result.json"
    out.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def check_correctness(cfg: dict, outdir: Path) -> dict:
    """Anchor gate for one run: bit-identity (canonical) or functional equivalence."""
    from ppgrid._platform import is_canonical_platform

    anchor = cfg.get("anchor")
    if not anchor:
        return {"status": "no-anchor", "mode": None, "anchor": None, "bands": {}}
    canonical = is_canonical_platform()
    bands: dict[str, dict] = {}
    ok = True
    for band, fname in (("value", "value.tif"), ("support", "support_km.tif")):
        want = REPO / anchor[band]
        got = outdir / cfg["id"] / fname
        gw = _sha256(got) if got.is_file() else None
        aw = _sha256(want) if want.is_file() else None
        if aw is None:
            bands[band] = {"status": "missing-anchor", "path": anchor[band]}
            ok = False
        elif gw == aw:
            bands[band] = {"status": "bit-identical", "sha256": gw}
        elif not canonical and _functional_equivalent(got, want):
            bands[band] = {"status": "functional", "sha256": gw}
        else:
            bands[band] = {"status": "mismatch", "sha256": gw, "anchor_sha256": aw}
            ok = False
    mode = "bit-identity" if canonical else "functional-equivalence"
    return {"status": "pass" if ok else "fail", "mode": mode, "anchor": anchor, "bands": bands}


def _functional_equivalent(got: Path, want: Path) -> bool:
    """Late import: check_anchors pulls the rasterio compare rules in one place."""
    sys.path.insert(0, str(SCRIPT.parent))
    from check_anchors import _functional_equivalent as fe

    return fe(got, want)


def check_wall(runs: list[dict], baseline_path: Path | None, threshold: float) -> dict:
    """Compare total walls against the committed baseline; fail above threshold.

    Skips (without failing) when the baseline is absent, was cut on a
    different CPU count, or a config's spec changed since — walls only mean
    something machine-to-machine and spec-to-spec.

    """
    gate: dict = {
        "status": "off",
        "baseline": str(baseline_path) if baseline_path else None,
        "threshold": threshold,
        "per_config": {},
    }
    if baseline_path is None:
        return gate
    if not Path(baseline_path).is_file():
        gate["status"] = "off"
        gate["note"] = f"baseline file missing: {baseline_path}"
        return gate
    bl = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    bl_runs = {r["config_id"]: r for r in bl.get("runs", [])}
    skip_reasons: list[str] = []
    if bl.get("env", {}).get("n_cpus") != os.cpu_count():
        skip_reasons.append(f"n_cpus mismatch: baseline {bl.get('env', {}).get('n_cpus')} vs current {os.cpu_count()}")
    n_gated = 0
    for r in runs:
        cid = r["config_id"]
        b = bl_runs.get(cid)
        if b is None:
            gate["per_config"][cid] = {"status": "no-baseline"}
            continue
        if r.get("status") == "crashed":
            gate["per_config"][cid] = {"status": "skipped", "reasons": ["run crashed"]}
            continue
        if skip_reasons:
            gate["per_config"][cid] = {"status": "skipped", "reasons": skip_reasons}
            continue
        if json.dumps(r["config"], sort_keys=True) != json.dumps(b.get("config", {}), sort_keys=True):
            gate["per_config"][cid] = {"status": "config-changed"}
            continue
        bt = b["timings_s"]["total"]
        ct = r["timings_s"]["total"]
        ratio = ct / bt if bt > 0 else None
        status = "pass" if (ratio is not None and ratio <= threshold) else "fail"
        n_gated += 1
        if status == "fail":
            gate["status"] = "fail"
        phases = {}
        for ph in ("ingest", "calibrate", "grid", "write"):
            bph = b["timings_s"].get(ph)
            cph = r["timings_s"].get(ph)
            phases[ph] = round(cph / bph, 3) if bph else None
        gate["per_config"][cid] = {
            "baseline_total_s": bt,
            "total_s": ct,
            "wall_ratio": round(ratio, 3) if ratio is not None else None,
            "status": status,
            "phase_ratios": phases,
        }
    if gate["status"] != "fail":
        gate["status"] = "pass" if n_gated else "skipped"
        if skip_reasons:
            gate["note"] = "; ".join(skip_reasons)
    return gate


# ---------------------------------------------------------------------------
# Parent
# ---------------------------------------------------------------------------


def _git_env() -> dict:
    """Env for git calls: the worktree may be owned by another user (andryo runs)."""
    env = dict(os.environ)
    env["GIT_CONFIG_COUNT"] = "1"
    env["GIT_CONFIG_KEY_0"] = "safe.directory"
    env["GIT_CONFIG_VALUE_0"] = str(REPO)
    return env


def _commit_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            env=_git_env(),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)  # internal
    ap.add_argument("--config-json", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--set", choices=("ci", "all"), default="ci", help="config set (default: ci)")
    ap.add_argument("--ids", default=None, help="comma-separated config ids (overrides --set)")
    ap.add_argument("--config-file", default=None, help="JSON spec replacing the built-in table")
    ap.add_argument("--tag", default=None, help="results tag (default: <set>-<commit7>)")
    ap.add_argument("--outdir", required=True, help="run outputs dir (temp CSVs, tifs, worker results)")
    ap.add_argument(
        "--results-out", default=None, help="where to write the results JSON (default: <outdir>/<tag>.json)"
    )
    ap.add_argument(
        "--baseline",
        default=None,
        help="baseline results JSON for the wall gate (default: bench/results/baseline/<set>.json if present)",
    )
    ap.add_argument(
        "--wall-threshold",
        type=float,
        default=WALL_GATE_THRESHOLD,
        help=f"fail above this wall ratio (default {WALL_GATE_THRESHOLD})",
    )
    ap.add_argument("--no-wall-gate", action="store_true", help="skip the wall gate (baseline cutting)")
    ap.add_argument("--skip-correctness", action="store_true", help="skip the anchor gate (dev)")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir).resolve()
    if args.worker:
        if not args.config_json:
            ap.error("--worker requires --config-json")
        _worker_main(args.config_json, outdir)
        return 0

    try:
        configs = load_configs(args)
    except SystemExit as e:
        if e.code:
            print(f"[config] {e}", file=sys.stderr)
        raise
    except ValueError as e:
        print(f"[config] {e}", file=sys.stderr)
        return 2

    commit = _commit_hash()
    tag = args.tag or f"{args.set}-{commit[:7]}"
    if args.no_wall_gate:
        baseline_path = None
    elif args.baseline:
        baseline_path = Path(args.baseline)
    else:
        baseline_path = REPO / "bench" / "results" / "baseline" / f"{args.set}.json"
    results_path = Path(args.results_out) if args.results_out else outdir / f"{tag}.json"

    outdir.mkdir(parents=True, exist_ok=True)
    print(f"regime bench: set={args.set} tag={tag} commit={commit[:12]} outdir={outdir}", flush=True)
    print(f"wall gate: {'off' if baseline_path is None else baseline_path}", flush=True)

    runs: list[dict] = []
    ok = True
    for cfg in configs:
        cdir = outdir / cfg["id"]
        cdir.mkdir(parents=True, exist_ok=True)
        print(f"== {cfg['id']} ==", flush=True)
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--worker", "--config-json", json.dumps(cfg), "--outdir", str(outdir)],
            cwd=REPO,
        )
        if proc.returncode != 0:
            print(f"[FAIL] {cfg['id']}: worker exited {proc.returncode}", file=sys.stderr)
            ok = False
            runs.append({"config_id": cfg["id"], "config": cfg, "status": "crashed", "exit_code": proc.returncode})
            continue
        rec = json.loads((cdir / "result.json").read_text(encoding="utf-8"))
        if not args.skip_correctness:
            cor = check_correctness(cfg, outdir)
            rec["correctness"] = cor
            if cor["status"] == "no-anchor":
                print(f"  [warn] {cfg['id']}: no committed anchor — wall result unverified", flush=True)
            elif cor["status"] != "pass":
                ok = False
                for band, b in cor["bands"].items():
                    if b["status"] == "mismatch":
                        extra = f" got={b.get('sha256')} anchor={b.get('anchor_sha256')}"
                    else:
                        extra = f" ({b.get('path')})"
                    print(f"[FAIL] {cfg['id']}/{band}: {b['status']}{extra}", file=sys.stderr)
            else:
                how = "bit-identical" if cor["mode"] == "bit-identity" else "functional"
                print(f"  [ok] correctness {how}", flush=True)
        rec["status"] = "ok"
        runs.append(rec)
        t = rec.get("timings_s", {})
        print(
            f"  n={rec.get('n_points')} grid={rec.get('grid', {}).get('nx')}x{rec.get('grid', {}).get('ny')} "
            f"ingest {t.get('ingest')}s calib {t.get('calibrate')}s grid {t.get('grid')}s "
            f"write {t.get('write')}s TOTAL {t.get('total')}s "
            f"peakRSS {rec.get('peak_rss_mib')}MiB tmpHWM {rec.get('temp_disk_hwm_mb')}MB "
            f"out {rec.get('outputs_bytes')}",
            flush=True,
        )

    wall = check_wall(runs, baseline_path, args.wall_threshold)
    if wall["status"] == "fail":
        ok = False
        for cid, e in wall["per_config"].items():
            if e.get("status") == "fail":
                print(
                    f"[FAIL] wall gate {cid}: {e['wall_ratio']}x baseline (> {args.wall_threshold}x)", file=sys.stderr
                )
    elif wall["status"] == "skipped":
        print(f"[warn] wall gate skipped: {wall.get('per_config', {})}", flush=True)
    else:
        for cid, e in wall["per_config"].items():
            if e.get("wall_ratio") is not None:
                print(f"  [ok] wall {cid}: {e['wall_ratio']}x baseline", flush=True)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "kind": "ppgrid-regime-results",
        "tag": tag,
        "commit": commit,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": _env_block(),
        "set": args.set,
        "wall_gate": wall,
        "runs": runs,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"results -> {results_path}")
    print(f"regime bench: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
