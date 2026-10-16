"""Independent IDW and nearest-neighbour baselines for ppgrid comparison.

Issue #107 (beta plan B-10): the benchmark's "IDW baseline" must not be
ppgrid compared to itself. This module is deliberately independent of
``ppgrid.pipeline``: the two search functions (``idw``, ``nearest``) need
only numpy, plus scipy when the exact k-NN search runs (the default and
only supported search; scipy is the ``[baselines]`` optional extra, so the
core install stays lean).

- ``idw(points, values, query, power=2.0, k=8, min_neighbours="k")`` —
  classical inverse-distance weighting with exact k-NN
  (``scipy.spatial.cKDTree``).
- ``nearest(points, query, values=None)`` — nearest-neighbour value
  transfer (exact 1-NN).
- ``BaselineConfig`` / ``load_config`` — the benchmark workload
  (source-point count, query count) and limits (wall, RSS) are read from a
  JSON spec (``bench/config/baselines.json``). The v1 default is the
  author-claimed regime (20M points / 1M queries / <10 min / <32 GB) with
  ``needs_andryo_confirmation: true`` — a claim to confirm, not a baked-in
  threshold. Limits are *reported against*, never enforced.
- ``domain_from_pipeline(p, holdout_frac, seed)`` — builds the identical
  domain (same source points, work CRS, resolution, grid extent,
  box-count coverage mask, optional seeded holdout) for a finished ppgrid
  run, so baselines score exactly the pixels a ppgrid run scored.
- ``synthetic_workload(...)`` — a seeded points/values/query triple at the
  configured scale (used by ``bench/baselines_bench.py`` and the tests).

Every ``BaselineResult`` carries the exact search configuration used
(power, k, neighbour policy, search method, scipy version) plus measured
wall/RSS against the configured limits; ``methods()`` renders it as a
single paragraph for a paper methods section (issue #107, s29.1 box 2).

Install the extra: ``pip install "ppgrid[baselines]"``.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ppgrid.pipeline import Pipeline

__all__ = [
    "BaselineConfig",
    "BaselineResult",
    "Domain",
    "domain_from_pipeline",
    "idw",
    "load_config",
    "nearest",
    "synthetic_workload",
]

#: The exact k-NN search method string recorded in every result config.
SEARCH_METHOD = "scipy.spatial.cKDTree.query (exact k-NN)"

#: Default query block size (bounds transient query memory at 20M+ points).
DEFAULT_QUERY_CHUNK = 100_000

_M_PER_KM = 1000.0


def _peak_rss_mib() -> float:
    """Process peak RSS in MiB (0.0 where the platform cannot report it)."""
    if sys.platform == "win32":  # POSIX-only stdlib module
        return 0.0
    import resource

    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(v / 1_048_576, 1)  # macOS reports bytes
    return round(v / 1024, 1)  # Linux reports KiB


def _as_points2d(name: str, arr: Any) -> np.ndarray:
    """Validate an (N, 2) metres array; return a float64 C-contiguous copy.

    Raises:
        ValueError: If the shape is not (N, 2) or any coordinate is non-finite.

    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2:
        msg = f"{name} must be an (N, 2) array of (x, y) coordinates, got shape {a.shape}"
        raise ValueError(msg)
    if not np.isfinite(a).all():
        bad = int(np.count_nonzero(~np.isfinite(a)))
        msg = f"{name} has {bad} non-finite coordinate(s); the k-NN search requires finite input"
        raise ValueError(msg)
    return np.ascontiguousarray(a)


def _resolve_min_neighbours(policy: str | int, k: int) -> int:
    """Resolve the min-neighbours policy to the required neighbour count.

    "k" requires the full k (a value is NaN when fewer than k points exist),
    "1" requires at least one, an int q requires at least q.

    Raises:
        ValueError: If the policy is not "k", "1", or an int >= 1.

    """
    if policy == "k":
        return k
    if policy == "1":
        return 1
    if isinstance(policy, int) and not isinstance(policy, bool) and policy >= 1:
        return policy
    msg = f"min_neighbours must be 'k', '1', or an int >= 1, got {policy!r}"
    raise ValueError(msg)


def _build_tree(points: np.ndarray) -> tuple[Any, str]:
    """Build the exact k-NN tree; return (tree, scipy version)."""
    import scipy
    from scipy.spatial import cKDTree

    return cKDTree(points), scipy.__version__


def _query_knn(tree: Any, query: np.ndarray, k: int, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact k-NN over blocked queries; returns (m, k) distances + int64 indices."""
    m = query.shape[0]
    d = np.empty((m, k), dtype=np.float64)
    i = np.empty((m, k), dtype=np.int64)
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        dd, ii = tree.query(query[s:e], k=k)
        if k == 1:
            dd = dd[:, None]
            ii = ii[:, None]
        d[s:e] = dd
        i[s:e] = ii
    return d, i


def _limits_dict(limits: BaselineConfig | dict[str, Any] | None) -> dict[str, float] | None:
    """Normalise limits (config, dict, or None) to the two limit keys.

    Raises:
        TypeError: If limits is not a BaselineConfig, a dict, or None.

    """
    if limits is None:
        return None
    if isinstance(limits, BaselineConfig):
        src: dict[str, Any] = limits.to_dict()
    elif isinstance(limits, dict):
        src = limits
    else:
        msg = "limits must be a BaselineConfig, a dict with wall_limit_s/rss_limit_gb, or None"
        raise TypeError(msg)
    out: dict[str, float] = {}
    for key in ("wall_limit_s", "rss_limit_gb"):
        if src.get(key) is not None:
            out[key] = float(src[key])
    return out or None


def _limit_flags(wall_s: float, rss_mib: float, limits: dict[str, float] | None) -> tuple[bool | None, bool | None]:
    if limits is None:
        return None, None
    wall_ok = wall_s <= limits["wall_limit_s"] if "wall_limit_s" in limits else None
    rss_ok = (rss_mib / 1024.0 <= limits["rss_limit_gb"]) if "rss_limit_gb" in limits and rss_mib > 0.0 else None
    return wall_ok, rss_ok


# ---------------------------------------------------------------------------
# Config: workload + limits for a baseline benchmark run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineConfig:
    """Workload + limits for a baseline benchmark run (issue #107).

    The defaults are the author-claimed v1 regime (s29.7): 20M source
    points, 1M queries, <10 min wall, <32 GB RSS. That regime is a claim
    pending Andryo's confirmation — ``needs_andryo_confirmation`` stays
    true until confirmed, and the limits are reported against, never
    enforced as thresholds.

    Attributes:
        n_source: Source-point count of the benchmark workload.
        n_queries: Query count of the benchmark workload.
        power: IDW distance exponent (p in w = d^-p).
        k: IDW neighbourhood size (exact k-NN).
        min_neighbours: Neighbour policy: "k" (full k required), "1"
            (at least one), or an int q (at least q).
        wall_limit_s: Claimed wall-time limit for one baseline run.
        rss_limit_gb: Claimed process-peak RSS limit for one baseline run.
        needs_andryo_confirmation: True while the regime is author-claimed.
        note: Free-text provenance note (carried into results/reports).
        source: Path of the config file this instance was loaded from
            (None for constructed defaults).

    """

    n_source: int = 20_000_000
    n_queries: int = 1_000_000
    power: float = 2.0
    k: int = 8
    min_neighbours: str | int = "k"
    wall_limit_s: float = 600.0
    rss_limit_gb: float = 32.0
    needs_andryo_confirmation: bool = True
    note: str | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict of the full config (the methods-section record)."""
        return {
            "n_source": self.n_source,
            "n_queries": self.n_queries,
            "power": self.power,
            "k": self.k,
            "min_neighbours": self.min_neighbours,
            "wall_limit_s": self.wall_limit_s,
            "rss_limit_gb": self.rss_limit_gb,
            "needs_andryo_confirmation": self.needs_andryo_confirmation,
            "note": self.note,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, source: str | None = None) -> BaselineConfig:
        """Build a validated config from a JSON object; missing keys default.

        Args:
            d: The decoded JSON object (a dict of scalars).
            source: Path recorded as the config provenance (loader sets it).

        Raises:
            ValueError: If d is not a dict, has unknown keys, or a field
                fails type/range validation.

        """
        if not _is_dict(d):
            msg = "baseline config must be a JSON object"
            raise ValueError(msg)
        unknown = sorted(set(d) - set(_VALIDATORS))
        if unknown:
            msg = f"unknown baseline config key(s): {', '.join(unknown)}"
            raise ValueError(msg)
        vals: dict[str, Any] = {name: _VALIDATORS[name](name, raw) for name, raw in d.items()}
        for name, default in _DEFAULTS.items():
            vals.setdefault(name, default)
        vals["source"] = source
        return cls(**vals)


def _is_dict(raw: Any) -> bool:
    """Report whether raw is a dict."""
    return isinstance(raw, dict)


def _is_count(raw: Any) -> bool:
    """Report whether raw is a positive int (bool excluded)."""
    return isinstance(raw, int) and not isinstance(raw, bool) and raw >= 1


def _is_number(raw: Any) -> bool:
    """Report whether raw is an int/float (bool excluded)."""
    return isinstance(raw, (int, float)) and not isinstance(raw, bool)


def _positive_int(key: str, raw: Any) -> int:
    if not _is_count(raw):
        msg = f"baseline config {key!r} must be an int >= 1, got {raw!r}"
        raise ValueError(msg)
    return raw


def _positive_finite_float(key: str, raw: Any) -> float:
    if not _is_number(raw):
        msg = f"baseline config {key!r} must be a finite number > 0, got {raw!r}"
        raise ValueError(msg)
    v = float(raw)
    if not math.isfinite(v) or v <= 0:
        msg = f"baseline config {key!r} must be a finite number > 0, got {raw!r}"
        raise ValueError(msg)
    return v


def _min_neighbours_field(key: str, raw: Any) -> str | int:
    try:
        _resolve_min_neighbours(raw, 1)
    except ValueError:
        msg = f"baseline config {key!r} must be 'k', '1', or an int >= 1, got {raw!r}"
        raise ValueError(msg) from None
    return raw


def _bool_field(key: str, raw: Any) -> bool:
    if raw is True or raw is False:
        return raw
    msg = f"baseline config {key!r} must be a bool, got {raw!r}"
    raise ValueError(msg)


def _note_field(key: str, raw: Any) -> str | None:
    if raw is not None and not isinstance(raw, str):
        msg = f"baseline config {key!r} must be a string or null, got {raw!r}"
        raise ValueError(msg)
    return raw


_VALIDATORS = {
    "n_source": _positive_int,
    "n_queries": _positive_int,
    "power": _positive_finite_float,
    "k": _positive_int,
    "min_neighbours": _min_neighbours_field,
    "wall_limit_s": _positive_finite_float,
    "rss_limit_gb": _positive_finite_float,
    "needs_andryo_confirmation": _bool_field,
    "note": _note_field,
}
_DEFAULTS = {
    "n_source": 20_000_000,
    "n_queries": 1_000_000,
    "power": 2.0,
    "k": 8,
    "min_neighbours": "k",
    "wall_limit_s": 600.0,
    "rss_limit_gb": 32.0,
    "needs_andryo_confirmation": True,
    "note": None,
}


def load_config(path: str | Path) -> BaselineConfig:
    """Read and validate a baseline bench config JSON file.

    Args:
        path: Path to the config (e.g. bench/config/baselines.json).

    Raises:
        ValueError: If the file is missing/unreadable, not valid JSON, or
            fails BaselineConfig validation.

    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        msg = f"cannot read baseline config {p}: {e.strerror or e}"
        raise ValueError(msg) from e
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        msg = f"baseline config {p} is not valid JSON: {e}"
        raise ValueError(msg) from e
    return BaselineConfig.from_dict(raw, source=str(p))


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class BaselineResult:
    """One baseline run: outputs + the exact config used + measured limits.

    Attributes:
        kind: "idw" or "nearest".
        value: (m,) float64 baseline values (NaN where the neighbour
            policy has no value; None for nearest() without values).
        dist: (m,) float64 distance to the nearest source point.
        idx: (m,) int64 nearest-point index (nearest only; None for idw).
        config: JSON-safe search config (power, k, policy, search method,
            scipy version, scale) — the methods-section record.
        wall_s: Wall seconds for the search run (tree build + queries).
        peak_rss_mib: Process peak RSS in MiB (0.0 on non-POSIX).
        limits: The wall/RSS limits reported against (or None).
        wall_ok: Wall limit met (None when no limit / not measurable).
        rss_ok: RSS limit met (None when no limit / not measurable).

    """

    kind: str
    value: np.ndarray | None
    dist: np.ndarray
    idx: np.ndarray | None
    config: dict[str, Any]
    wall_s: float
    peak_rss_mib: float
    limits: dict[str, float] | None
    wall_ok: bool | None
    rss_ok: bool | None

    def methods(self) -> str:
        """Render the run as one paragraph for a paper methods section."""
        c = self.config
        search = f"scipy {c['scipy_version']}, {SEARCH_METHOD}"
        if self.kind == "idw":
            head = (
                f"Independent IDW baseline: power p={c['power']:g}, k={c['k']} nearest neighbours "
                f"(k_used={c['k_used']}, exact k-NN via {search}), min-neighbour policy "
                f"{c['min_neighbours']!r} (value only when {c['min_neighbours_required']}+ neighbours "
                f"are available, NaN otherwise)"
            )
        else:
            head = (
                f"Independent nearest-neighbour baseline: nearest source point (exact 1-NN via "
                f"{search}), value = source value at the nearest point"
            )
        s = head + (f"; n_points={c['n_points']}, n_queries={c['n_queries']}, query_chunk={c['query_chunk']}")
        if c.get("points_crs") is not None:
            s += f", points CRS={c['points_crs']}"
        s += f". Measured wall {self.wall_s:.2f} s, process peak RSS {self.peak_rss_mib:.0f} MiB"
        if self.limits is not None:
            s += (
                f" against limits wall<={self.limits['wall_limit_s']:g} s (met={self.wall_ok}), "
                f"RSS<={self.limits['rss_limit_gb']:g} GiB (met={self.rss_ok})"
            )
        return s + "."

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary (array shapes/stats, no raw array data)."""
        out: dict[str, Any] = {
            "kind": self.kind,
            "config": self.config,
            "wall_s": self.wall_s,
            "peak_rss_mib": self.peak_rss_mib,
            "limits": self.limits,
            "wall_ok": self.wall_ok,
            "rss_ok": self.rss_ok,
            "dist": {
                "shape": list(self.dist.shape),
                "min": float(self.dist.min()) if self.dist.size else None,
                "max": float(self.dist.max()) if self.dist.size else None,
            },
        }
        if self.value is not None:
            out["value"] = {
                "shape": list(self.value.shape),
                "finite": int(np.count_nonzero(np.isfinite(self.value))),
                "nan": int(np.count_nonzero(np.isnan(self.value))),
            }
        if self.idx is not None:
            out["idx"] = {"shape": list(self.idx.shape)}
        return out


# ---------------------------------------------------------------------------
# The two baselines
# ---------------------------------------------------------------------------


def idw(
    points: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
    power: float = 2.0,
    k: int = 8,
    min_neighbours: str | int = "k",
    limits: BaselineConfig | dict[str, Any] | None = None,
    points_crs: int | str | None = None,
    query_chunk: int = DEFAULT_QUERY_CHUNK,
) -> BaselineResult:
    """Independent inverse-distance-weighting grid (exact k-NN).

    value(q) = sum(w_i * v_i) / sum(w_i) over the k nearest source points,
    w_i = d_i^-power. A query coincident with source point(s) (d = 0) takes
    the mean of the coincident points' values (the d -> 0 limit is
    dominated by them). Queries with fewer available neighbours than the
    policy requires get NaN.

    Args:
        points: (n, 2) source coordinates in metres (any CRS; label via
            points_crs). Finite required.
        values: (n,) source values (float64; NaN/inf propagate).
        query: (m, 2) query coordinates in the same units. Finite required.
        power: Distance exponent p (w = d^-p); must be finite and > 0.
        k: Neighbourhood size (exact k-NN); clamped to n internally.
        min_neighbours: Neighbour policy: "k" (the full k required — NaN
            when n < k), "1" (at least one), or an int q (at least q).
        limits: Optional BaselineConfig (or dict with wall_limit_s /
            rss_limit_gb); the run is reported against them, never cut off.
        points_crs: Optional CRS label recorded in the result config.
        query_chunk: Query block size for the blocked k-NN search.

    Raises:
        ValueError: On bad shapes, non-finite coordinates, or invalid
            power / k / min_neighbours / query_chunk.

    """
    pts = _as_points2d("points", points)
    n = pts.shape[0]
    if n == 0:
        msg = "idw: points must be non-empty"
        raise ValueError(msg)
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim != 1 or vals.shape[0] != n:
        msg = f"idw: values must be shape ({n},), got {vals.shape}"
        raise ValueError(msg)
    qry = _as_points2d("query", query)
    m = qry.shape[0]
    if (
        not isinstance(power, (int, float))
        or isinstance(power, bool)
        or not math.isfinite(float(power))
        or float(power) <= 0
    ):
        msg = f"idw: power must be a finite number > 0, got {power!r}"
        raise ValueError(msg)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        msg = f"idw: k must be an int >= 1, got {k!r}"
        raise ValueError(msg)
    if isinstance(query_chunk, bool) or not isinstance(query_chunk, int) or query_chunk < 1:
        msg = f"idw: query_chunk must be an int >= 1, got {query_chunk!r}"
        raise ValueError(msg)
    power = float(power)
    required = _resolve_min_neighbours(min_neighbours, k)
    k_eff = min(k, n)

    lim = _limits_dict(limits)
    t0 = time.perf_counter()
    tree, scipy_version = _build_tree(pts)
    d, i = _query_knn(tree, qry, k_eff, query_chunk)
    wall = time.perf_counter() - t0

    val = np.full(m, np.nan, dtype=np.float64)
    if k_eff >= required:
        d0 = d[:, 0]
        zero = d0 == 0.0
        if np.any(zero):
            dz = d[zero]
            vz = vals[i[zero]]
            w0 = dz == 0.0
            val[zero] = np.where(w0, vz, 0.0).sum(axis=1) / w0.sum(axis=1)
        rest = ~zero
        if np.any(rest):
            w = d[rest] ** (-power)
            w /= w.sum(axis=1, keepdims=True)
            val[rest] = (w * vals[i[rest]]).sum(axis=1)

    config: dict[str, Any] = {
        "kind": "idw",
        "power": power,
        "k": k,
        "k_used": k_eff,
        "min_neighbours": min_neighbours,
        "min_neighbours_required": required,
        "search": SEARCH_METHOD,
        "scipy_version": scipy_version,
        "n_points": n,
        "n_queries": m,
        "query_chunk": query_chunk,
        "points_crs": points_crs,
    }
    wall_ok, rss_ok = _limit_flags(wall, _peak_rss_mib(), lim)
    return BaselineResult(
        kind="idw",
        value=val,
        dist=d[:, 0],
        idx=None,
        config=config,
        wall_s=wall,
        peak_rss_mib=_peak_rss_mib(),
        limits=lim,
        wall_ok=wall_ok,
        rss_ok=rss_ok,
    )


def nearest(
    points: np.ndarray,
    query: np.ndarray,
    values: np.ndarray | None = None,
    limits: BaselineConfig | dict[str, Any] | None = None,
    points_crs: int | str | None = None,
    query_chunk: int = DEFAULT_QUERY_CHUNK,
) -> BaselineResult:
    """Independent nearest-neighbour grid (exact 1-NN).

    Args:
        points: (n, 2) source coordinates in metres. Finite required.
        query: (m, 2) query coordinates in the same units. Finite required.
        values: Optional (n,) source values; when given, result.value is
            the value transferred from each query's nearest point.
        limits: Optional BaselineConfig (or dict); reported against, never
            enforced.
        points_crs: Optional CRS label recorded in the result config.
        query_chunk: Query block size for the blocked k-NN search.

    Raises:
        ValueError: On bad shapes, non-finite coordinates, or a values /
            query_chunk mismatch.

    """
    pts = _as_points2d("points", points)
    n = pts.shape[0]
    if n == 0:
        msg = "nearest: points must be non-empty"
        raise ValueError(msg)
    qry = _as_points2d("query", query)
    m = qry.shape[0]
    vals: np.ndarray | None = None
    if values is not None:
        vals = np.asarray(values, dtype=np.float64)
        if vals.ndim != 1 or vals.shape[0] != n:
            msg = f"nearest: values must be shape ({n},), got {vals.shape}"
            raise ValueError(msg)
    if isinstance(query_chunk, bool) or not isinstance(query_chunk, int) or query_chunk < 1:
        msg = f"nearest: query_chunk must be an int >= 1, got {query_chunk!r}"
        raise ValueError(msg)

    lim = _limits_dict(limits)
    t0 = time.perf_counter()
    tree, scipy_version = _build_tree(pts)
    d, i = _query_knn(tree, qry, 1, query_chunk)
    wall = time.perf_counter() - t0
    val: np.ndarray | None = None if vals is None else vals[i[:, 0]]

    config: dict[str, Any] = {
        "kind": "nearest",
        "power": None,
        "k": 1,
        "k_used": 1,
        "min_neighbours": "1",
        "min_neighbours_required": 1,
        "search": SEARCH_METHOD,
        "scipy_version": scipy_version,
        "n_points": n,
        "n_queries": m,
        "query_chunk": query_chunk,
        "points_crs": points_crs,
    }
    wall_ok, rss_ok = _limit_flags(wall, _peak_rss_mib(), lim)
    return BaselineResult(
        kind="nearest",
        value=val,
        dist=d[:, 0],
        idx=i[:, 0].astype(np.int64, copy=False),
        config=config,
        wall_s=wall,
        peak_rss_mib=_peak_rss_mib(),
        limits=lim,
        wall_ok=wall_ok,
        rss_ok=rss_ok,
    )


# ---------------------------------------------------------------------------
# Domain: the identical scoring domain for a finished ppgrid run
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Domain:
    """The scoring domain of a ppgrid run: points, values, grid, CRS, mask.

    Attributes:
        x: (n,) source x coordinates, work-CRS metres (pipeline-ingested).
        y: (n,) source y coordinates, work-CRS metres.
        values: (n,) raw source values (pipeline's ``p.v``, untransformed).
        crs: Work-CRS EPSG code (or a label; None for CRS-agnostic metres).
        res: Cell size in metres (the pipeline's ``--res``).
        x0: Grid origin x (work CRS, metres).
        y0: Grid origin y (work CRS, metres).
        nx: Grid width in cells.
        ny: Grid height in cells.
        cap_km: The pipeline's effective fill cap (None when uncalibrated).
        seed: Holdout split seed (0 = no split requested).
        holdout_idx: Sorted holdout point indices (None for no holdout).

    """

    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    crs: int | str | None
    res: float
    x0: float
    y0: float
    nx: int
    ny: int
    cap_km: float | None = None
    seed: int = 0
    holdout_idx: np.ndarray | None = None

    @property
    def n_holdout(self) -> int:
        """Holdout point count (0 for no holdout)."""
        return int(self.holdout_idx.size) if self.holdout_idx is not None else 0

    @property
    def train_idx(self) -> np.ndarray:
        """Complement of the holdout (all indices when there is no holdout)."""
        if self.holdout_idx is None:
            return np.arange(self.x.size, dtype=np.intp)
        mask = np.ones(self.x.size, dtype=bool)
        mask[self.holdout_idx] = False
        return np.nonzero(mask)[0]

    def grid_query(self) -> np.ndarray:
        """Query array: every grid pixel centre, (ny*nx, 2), row-major.

        The same pixel-centre convention the pipeline's rasters use
        (centre of each work-CRS cell, y increasing with row).
        """
        ix = (np.arange(self.nx, dtype=np.float64) + 0.5) * self.res + self.x0
        iy = (np.arange(self.ny, dtype=np.float64) + 0.5) * self.res + self.y0
        xx, yy = np.meshgrid(ix, iy, indexing="xy")
        return np.column_stack((xx.ravel(), yy.ravel()))

    def coverage_mask(self, cap_km: float | None = None) -> np.ndarray:
        """(ny, nx) bool: True where the ppgrid fill would emit a value.

        The independent twin of the pipeline's box-count near-mask: a cell
        is covered iff at least one source point falls in its (2r+1)^2 cell
        window, with r = round(cap_km * 1000 / res) — the same rounding the
        pipeline applies (box_count(counts, r) > 0).

        Args:
            cap_km: Fill cap override; defaults to the domain's cap_km.

        Raises:
            ValueError: If no cap_km is available (uncalibrated domain).

        """
        cap = self.cap_km if cap_km is None else cap_km
        if cap is None:
            msg = "coverage_mask: no cap_km (pass cap_km= or use a calibrated pipeline run)"
            raise ValueError(msg)
        from scipy.ndimage import binary_dilation  # [baselines] extra

        r = round(cap * _M_PER_KM / self.res)
        grid = np.zeros((self.ny, self.nx), dtype=bool)
        ix = ((self.x - self.x0) // self.res).astype(np.intp)
        iy = ((self.y - self.y0) // self.res).astype(np.intp)
        grid[iy, ix] = True
        if r > 0:
            grid = binary_dilation(grid, structure=np.ones((2 * r + 1, 2 * r + 1), dtype=bool))
        return grid

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe domain summary (for bench reports / methods sections)."""
        return {
            "n_points": int(self.x.size),
            "crs": self.crs,
            "res_m": self.res,
            "extent_m": [self.x0, self.y0, self.x0 + self.nx * self.res, self.y0 + self.ny * self.res],
            "grid": f"{self.nx}x{self.ny}",
            "cap_km": self.cap_km,
            "seed": self.seed,
            "n_holdout": self.n_holdout,
        }


def domain_from_pipeline(
    p: Pipeline,
    holdout_frac: float = 0.0,
    seed: int = 0,
) -> Domain:
    """Build the identical domain for a finished ppgrid run (issue #107).

    "Identical" means: the same source points the pipeline ingested
    (work-CRS metres, post-drop), the same work CRS and resolution, the
    same grid extent/origin (the min/max rule ``Pipeline.grid`` applies),
    and the same coverage mask definition (box-count, ``coverage_mask``).
    ``values`` are the raw pipeline values (``p.v``).

    Value-transform note: the pipeline interpolates its *transformed*
    values (``p.tv``) and decodes them through ``calibration.json``; the
    baselines score the raw ``p.v``. Compare in raw-value space: decode the
    pipeline raster via its calibration LUT, then compare against the
    baseline output. Geometry, points, and masks are identical either way.

    Holdout note: ppgrid has no built-in train/holdout split. With
    ``holdout_frac > 0`` the points are split by a seeded permutation
    (``np.random.default_rng(seed).permutation(n)``; the first
    ``ceil(frac * n)`` are the holdout, returned sorted). Apply the same
    split to both sides: re-run ppgrid on the train subset (or score the
    baseline on the holdout) — the split is deterministic in ``seed``.

    Args:
        p: A Pipeline with ``ingest()`` run (``x``/``y``/``v`` set).
            ``calibrate()``/``grid()`` are not required; ``cap_km`` is
            recorded when available (post-``calibrate()``).
        holdout_frac: Holdout fraction in [0, 1] (0 = no holdout).
        seed: Holdout split seed.

    Raises:
        ValueError: If holdout_frac is not in [0, 1].

    """
    if not math.isfinite(holdout_frac) or not 0.0 <= holdout_frac <= 1.0:
        msg = f"holdout_frac must be in [0, 1], got {holdout_frac!r}"
        raise ValueError(msg)
    x = np.asarray(p.x, dtype=np.float64)
    y = np.asarray(p.y, dtype=np.float64)
    v = np.asarray(p.v, dtype=np.float64)
    if x.shape != y.shape or x.shape != v.shape:
        msg = f"pipeline point arrays disagree: x{x.shape} y{y.shape} v{v.shape}"
        raise ValueError(msg)
    res = float(p.res)
    x0 = float(x.min())
    y0 = float(y.min())
    nx = int((float(x.max()) - x0) // res) + 1
    ny = int((float(y.max()) - y0) // res) + 1
    n = x.size
    holdout: np.ndarray | None = None
    n_holdout = math.ceil(holdout_frac * n)
    if n_holdout > 0:
        perm = np.random.default_rng(seed).permutation(n)
        holdout = np.sort(perm[:n_holdout]).astype(np.intp, copy=False)
    cap_raw = getattr(p, "cap_km_val", None)
    cap = float(cap_raw) if cap_raw is not None and math.isfinite(float(cap_raw)) else None
    return Domain(
        x=x,
        y=y,
        values=v,
        crs=p.work_crs,
        res=res,
        x0=x0,
        y0=y0,
        nx=nx,
        ny=ny,
        cap_km=cap,
        seed=seed,
        holdout_idx=holdout,
    )


# ---------------------------------------------------------------------------
# Synthetic workload at the configured scale
# ---------------------------------------------------------------------------


def synthetic_workload(
    n_source: int,
    n_queries: int,
    seed: int = 0,
    box_m: tuple[float, float] = (4_000_000.0, 3_000_000.0),
    res: float = 500.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seeded source points + values + query grid at the configured scale.

    Points: uniform in [0, box_m) (work-CRS-like metres; the default box is
    an AU-like extent). Values: a smooth two-Gaussian field plus lognormal
    noise — positive, the same spirit as ``bench/bench_tier2.py``,
    implemented independently. Query: the pixel centres of the regular
    ``res`` grid over the source extent (the shape a ppgrid run's grid
    takes for these points), seeded-subsampled to exactly ``n_queries``
    cells when the grid is larger (``n_queries`` is a cap).

    Args:
        n_source: Source-point count (>= 1).
        n_queries: Query cap (>= 0); the grid is subsampled to this count
            when it has more cells.
        seed: RNG seed (deterministic output).
        box_m: (width, height) of the source box in metres; both > 0.
        res: Query grid cell size in metres; > 0.

    Returns:
        (points (n_source, 2), values (n_source,), query (<= n_queries, 2)),
        all float64.

    Raises:
        ValueError: On invalid n_source / n_queries / box_m / res.

    """
    if not _is_count(n_source):
        msg = f"n_source must be an int >= 1, got {n_source!r}"
        raise ValueError(msg)
    if isinstance(n_queries, bool) or not isinstance(n_queries, int) or n_queries < 0:
        msg = f"n_queries must be an int >= 0, got {n_queries!r}"
        raise ValueError(msg)
    if isinstance(res, bool) or not isinstance(res, (int, float)) or res <= 0:
        msg = f"res must be a number > 0, got {res!r}"
        raise ValueError(msg)
    bx, by = float(box_m[0]), float(box_m[1])
    if bx <= 0 or by <= 0:
        msg = f"box_m must be (width, height) with both > 0, got {box_m!r}"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, bx, n_source)
    y = rng.uniform(0.0, by, n_source)
    gx, gy = x - x.min(), y - y.min()
    ex = max(float(gx.max()), 1.0)
    ey = max(float(gy.max()), 1.0)
    g1 = np.exp(-((gx - 0.3 * ex) ** 2 + (gy - 0.4 * ey) ** 2) / (2.0 * (0.15 * max(ex, ey)) ** 2))
    g2 = np.exp(-((gx - 0.8 * ex) ** 2 + (gy - 0.7 * ey) ** 2) / (2.0 * (0.25 * max(ex, ey)) ** 2))
    values = 5.0 + 40.0 * (g1 + g2) + rng.lognormal(0.0, 0.5, n_source)
    points = np.column_stack((x, y))

    x0 = float(x.min())
    y0 = float(y.min())
    nx = int((float(x.max()) - x0) // res) + 1
    ny = int((float(y.max()) - y0) // res) + 1
    grid = Domain(
        x=points[:, 0],
        y=points[:, 1],
        values=values,
        crs=None,
        res=res,
        x0=x0,
        y0=y0,
        nx=nx,
        ny=ny,
    ).grid_query()
    if grid.shape[0] > n_queries:
        sel = np.sort(rng.choice(grid.shape[0], size=n_queries, replace=False))
        grid = grid[sel]
    return points, values, grid
