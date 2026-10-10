"""Per-layer calibration: pick the value transform, and derive the fill cap.

Derive the fill cap from the data rather than hardcoding it.

1. Which transform makes the field spatially predictable?
   Scored by intraclass correlation (between-cell variance / total variance)
   at coarse scales.

2. How far may we interpolate before the result is worthless?
   Answered by *spatially blocked* cross-validation. Random k-fold is
   useless here: with clustered address data it puts 92% of held-out
   points within 500m of a training point and reports ~4x the real skill.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from .pullpush import _UNRESOLVED_M, bin_points, pad_to_pyramid, pull_push

# Shared numeric constants (imported by the pipeline).
PERCENTILE_MAX: float = 100.0  # largest output percentile (value band DN = percentile * scale)
M_PER_KM: float = 1000.0
# Fill cap returned by calibrate_fill_cap when no bin clears the skill bar.
# (The pipeline's FILL_FALLBACK_CAP_KM is a different fallback: it applies
# when there is no calibrated cap at all.)
FILL_CAP_DEFAULT_KM: float = 25.0

# Largest CV support-scale bin edge: the unresolved-support sentinel from
# pullpush, expressed in km. A bin this wide is "no neighbor within the cap",
# not a literal 1e9 km scale - the edge must track the sentinel so the two
# cannot drift (audit #83).
_CV_MAX_EDGE_KM: float = _UNRESOLVED_M / M_PER_KM
_CV_EDGES: tuple[float, ...] = (0, 2, 4, 8, 16, 32, 64, 128, 256, _CV_MAX_EDGE_KM)

# ---------------------------------------------------------------- transforms


class Transform:
    """Base class for value transforms."""

    def __init__(
        self,
        name: str,
        fwd: Callable[[np.ndarray], np.ndarray],
        inv: Callable[[np.ndarray], np.ndarray],
        valid: Callable[[np.ndarray], bool],
    ) -> None:
        """Initialise a transform with forward, inverse, and validity functions."""
        self.name: str = name
        self._fwd: Callable[[np.ndarray], np.ndarray] = fwd
        self._inv: Callable[[np.ndarray], np.ndarray] = inv
        self.valid: Callable[[np.ndarray], bool] = valid

    def fit(self, _v: np.ndarray) -> Transform:
        """No-op for stateless transforms.

        Returns:
            self.

        """
        return self

    def fwd(self, v: np.ndarray) -> np.ndarray:
        """Transform values to the interpolation space.

        Returns:
            Transformed values.

        """
        return self._fwd(v)

    def inv(self, v: np.ndarray) -> np.ndarray:
        """Map values back to the original scale.

        Returns:
            Inverse-transformed values.

        """
        return self._inv(v)

    def state(self) -> dict[str, Any]:
        """Serialise transform parameters.

        Returns:
            Dict with serialisable transform state.

        """
        return {"name": self.name}


class PercentileTransform(Transform):
    """Map values onto their empirical percentile, 0-100."""

    NQ = 1001

    def __init__(self, q: np.ndarray | None = None) -> None:
        """Initialise with optional pre-fitted quantiles."""
        self.name: str = "percentile"
        self.valid: Callable[[np.ndarray], bool] = lambda _v: True
        self.q: np.ndarray | None = None if q is None else np.asarray(q, np.float64)

    def fit(self, v: np.ndarray) -> PercentileTransform:
        """Fit quantiles from data.

        Flat input (all values equal, or a single point) is degenerate for the
        usual staircase: q[0] stays the constant, so fwd left-clamps the
        constant to percentile 0 — a constant column quantized to 0 instead
        of its true position (issue #24). Map the constant to the 50th
        percentile on a staircase wide enough that float32 interpolation
        noise around the constant still decodes to 50 (and a constant DN,
        at the default scale). Decode round-trips: inv(fwd(c)) == c.

        Args:
            v: 1-D values.

        Returns:
            self.

        """
        v = v.astype(np.float64)
        if v.size > 0 and np.ptp(v) == 0.0:
            c = float(v[0])
            sp = max(np.spacing(abs(c) + 1.0), abs(c) * 2e-2)
            self.q = c + (np.arange(self.NQ) - self.NQ // 2) * sp
            return self
        q = np.quantile(v, np.linspace(0, 1, self.NQ))
        self.q = np.maximum.accumulate(q)
        eps = np.arange(self.NQ) * np.spacing(np.abs(self.q).max() + 1.0)
        self.q += eps
        return self

    @property
    def _p(self) -> np.ndarray:
        return np.linspace(0.0, PERCENTILE_MAX, self.NQ)

    def fwd(self, v: np.ndarray) -> np.ndarray:
        """Map values to percentiles.

        Returns:
            Percentile values in range 0-100.

        """
        return np.interp(v, self.q, self._p)

    def inv(self, p: np.ndarray) -> np.ndarray:
        """Map percentiles back to values.

        Returns:
            Original-scale values.

        """
        return np.interp(p, self._p, self.q)

    def state(self) -> dict[str, Any]:
        """Return serialisable state with quantiles.

        Returns:
            Dict with name and quantiles.

        """
        return {"name": "percentile", "quantiles": self.q.tolist()}


def transforms() -> list[Transform]:
    """Fresh candidate instances. MUST be a factory, not a module-level list.

    Returns:
        List of unfitted transform candidates.

    """
    return [
        Transform("identity", lambda v: v, lambda t: t, lambda _v: True),
        Transform("log10", np.log10, lambda t: 10.0**t, lambda v: v.min() > 0),
        Transform("sqrt", np.sqrt, lambda t: t**2, lambda v: v.min() >= 0),
        PercentileTransform(),
    ]


def make_transform(state: dict[str, Any]) -> Transform:
    """Rebuild a fitted transform from its serialised state.

    Returns:
        Reconstructed transform instance.

    Raises:
        ValueError: If the transform name is not recognised.

    """
    if state.get("name") == "percentile":
        return PercentileTransform(state.get("quantiles"))
    name = state.get("name")
    for t in transforms():
        if t.name == name:
            return t
    msg = f"Unknown transform: {name!r}"
    raise ValueError(msg)


def _check_quantiles_list(q: Any, path: str, field: str) -> None:
    """Assert a calibration quantiles field is a list of NQ finite numbers (issue #79).

    Args:
        q: The parsed JSON field value.
        path: The calibration file path, for the error message.
        field: The field name, for the error message.

    Raises:
        ValueError: Naming the file and the bad field.

    """
    nq = PercentileTransform.NQ
    if not isinstance(q, list):
        msg = f"calibration file {path}: {field} must be a list of {nq} finite numbers (got {type(q).__name__})"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] — ValueError is the CLI-mapped error type (exit 2)
    if len(q) != nq:
        msg = f"calibration file {path}: {field} must be a list of {nq} finite numbers (got length {len(q)})"
        raise ValueError(msg)
    for i, x in enumerate(q):
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            msg = f"calibration file {path}: {field}[{i}] is not a finite number ({x!r})"
            raise ValueError(msg)


def validate_calibration(cal: Any, path: str) -> None:
    """Schema-validate a user-supplied calibration.json at load time (issue #79).

    The file is user-controlled input (the documented reuse mechanism). Without
    this guard a bad file dies mid-run with a raw traceback: an unfitted
    percentile state hits AttributeError in PercentileTransform.state(), an
    unknown transform name hits StopIteration in the refit branch, and a
    wrong-length quantile list survives until the write phase (np.interp
    fp/xp mismatch) after all the grid work.

    Args:
        cal: The parsed JSON.
        path: The file path, for error messages.

    Raises:
        ValueError: One friendly message naming the file and the bad field.

    """
    if not isinstance(cal, dict):
        msg = f"calibration file {path}: top level must be a JSON object (got {type(cal).__name__})"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] — ValueError is the CLI-mapped error type (exit 2)
    known = sorted(t.name for t in transforms())

    def _known_name(value: Any, field: str) -> str:
        if not isinstance(value, str) or value not in known:
            msg = f"calibration file {path}: {field} has unknown transform name {value!r} (expected one of {known})"
            raise ValueError(msg)
        return value

    _known_name(cal.get("transform", "identity"), "transform")
    ts = cal.get("transform_state")
    if ts is not None:
        if not isinstance(ts, dict):
            msg = f"calibration file {path}: transform_state must be a JSON object (got {type(ts).__name__})"
            raise ValueError(msg)
        if _known_name(ts.get("name"), "transform_state.name") == "percentile":
            _check_quantiles_list(ts.get("quantiles"), path, "transform_state.quantiles")
    if "percentile_quantiles" in cal:
        _check_quantiles_list(cal["percentile_quantiles"], path, "percentile_quantiles")


def _cell_key(x: np.ndarray, y: np.ndarray, res: float) -> np.ndarray:
    """Hash (x, y) coordinates to a unique integer per cell at the given resolution.

    Returns:
        Integer array of cell keys.

    """
    ix = ((x - x.min()) // res).astype(np.int64)
    iy = ((y - y.min()) // res).astype(np.int64)
    return ix * (int(iy.max()) + 1) + iy


def icc_from_inv(inv: np.ndarray, values: np.ndarray) -> float:
    """ICC from a precomputed cell-key inverse mapping (see _cell_key + np.unique).

    Args:
        inv: Per-point index into the sorted unique cells.
        values: Values to score.

    Returns:
        ICC value between 0 and 1.

    """
    n = int(inv.max()) + 1
    csum = np.bincount(inv, weights=values, minlength=n)
    ccnt = np.bincount(inv, minlength=n)
    cmean = csum / np.maximum(ccnt, 1)
    within = np.mean((values - cmean[inv]) ** 2)
    total = values.var()
    return float(1.0 - within / total) if total > 0 else 0.0


def icc(values: np.ndarray, x: np.ndarray, y: np.ndarray, res: float) -> float:
    """Intraclass correlation at cell size `res`.

    The share of total variance explained by location. ~0 means the value
    is not a spatial field at all.

    Returns:
        ICC value between 0 and 1.

    """
    key = _cell_key(x, y, res)
    _, inv = np.unique(key, return_inverse=True)
    return icc_from_inv(inv.ravel(), values)


def choose_transform(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    scales: tuple[float, ...] = (5000.0, 25000.0, 100000.0),
) -> tuple[Transform, list[tuple[Transform, float, dict[float, float]]]]:
    """Pick the transform maximising mean ICC across coarse scales.

    The per-scale cell key + np.unique depends only on x, y, res, so it is
    computed once per scale and shared across candidate transforms (was
    rebuilt on every icc call: ~12x for 4 transforms x 3 scales).

    Returns:
        Tuple of best transform and all scored results.

    Raises:
        ValueError: If values is empty, or every candidate is rejected
            (non-finite input).

    """
    if values.size == 0:
        msg = "choose_transform: empty values"
        raise ValueError(msg)
    inv_by_scale: dict[float, np.ndarray] = {}
    for s in scales:
        key = _cell_key(x, y, s)
        _, inv = np.unique(key, return_inverse=True)
        inv_by_scale[s] = inv.ravel()

    results: list[tuple[Transform, float, dict[float, float]]] = []
    for candidate in transforms():
        if not candidate.valid(values):
            continue
        fitted = candidate.fit(values)
        tv = fitted.fwd(values)
        if not np.all(np.isfinite(tv)):
            continue
        per_scale: dict[float, float] = {float(s): icc_from_inv(inv_by_scale[s], tv) for s in scales}
        results.append((fitted, float(np.mean(list(per_scale.values()))), per_scale))
    if not results:
        n_nan = int(np.isnan(values).sum())
        n_inf = int(np.isinf(values).sum())
        msg = (
            f"choose_transform: all candidate transforms rejected (non-finite input: "
            f"{n_nan} NaN, {n_inf} Inf of {values.size} values)"
        )
        raise ValueError(msg)
    results.sort(key=lambda r: -r[1])
    return results[0][0], results


def _fit_predict(
    x: np.ndarray,
    y: np.ndarray,
    tv: np.ndarray,
    train: np.ndarray,
    res: float,
    levels: int,
    saturation: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Train on the given indices, predict at all (x, y) positions.

    Args:
        x: Working-CRS x coordinates of all points (m).
        y: Working-CRS y coordinates of all points (m).
        tv: Transformed values of all points.
        train: Boolean mask of training points (held-out block excluded).
        res: Grid resolution in m.
        levels: Pyramid depth.
        saturation: Cell counts for full self-trust, forwarded to
            pull_push. Must match the run's own saturation or the CV skill
            estimates a different model than the one the pipeline trains
            (audit #83: the run's --saturation was dropped here).

    Returns:
        Tuple of predicted values and support in km.

    """
    x0, y0 = x.min(), y.min()
    ix = ((x - x0) // res).astype(np.int64)
    iy = ((y - y0) // res).astype(np.int64)
    nx = pad_to_pyramid(int(ix.max()) + 1, levels)
    ny = pad_to_pyramid(int(iy.max()) + 1, levels)
    s, c = bin_points(ix[train], iy[train], tv[train], nx, ny)
    val, sup = pull_push(s, c, res, levels, saturation=saturation)
    return val[ix, iy], sup[ix, iy] / M_PER_KM


class CVDetail(dict):
    """Single CV bin result."""

    lo_km: float
    hi_km: float
    n: int
    skill: float
    ci_lo: float
    ci_hi: float


class CVCurve(dict):
    """Per-block-size CV results."""

    overall_skill: float
    rows: list[CVDetail]


def blocked_cv_skill(
    x: np.ndarray,
    y: np.ndarray,
    tv: np.ndarray,
    res: float = 1000.0,
    levels: int = 9,
    block_km: float = 100.0,
    n_folds: int = 4,
    seed: int = 0,
    edges: tuple[float, ...] = _CV_EDGES,
    saturation: float = 1.0,
    n_boot: int = 200,
    min_n: int = 150,
    boot_max_n: int = 200_000,
) -> tuple[float, list[CVDetail]]:
    """Hold out whole spatial blocks, predict them, and report skill.

    Report skill (1 - RMSE/RMSE_baseline) per support-scale bin with a
    bootstrap CI.

    Returns:
        Tuple of overall skill score and per-bin CV details.

    """
    rng = np.random.default_rng(seed)
    block_size_m = block_km * M_PER_KM
    # Clamp grid resolution: full-bounding-box binning at 1 km OOMs on large
    # extents (e.g. ~40,000 km global -> ~1.6B cells). Cap the grid at 4096
    # cells per axis.
    extent = max(float(np.ptp(x)), float(np.ptp(y)))
    res = max(res, extent / 4096.0)
    bkey = _cell_key(x, y, block_size_m)
    blocks = np.unique(bkey)
    perm = rng.permutation(len(blocks))
    folds = np.array_split(perm, n_folds)

    preds = np.full(len(tv), np.nan)
    sups = np.full(len(tv), np.nan)
    for f in folds:
        held = np.isin(bkey, blocks[f])
        if held.sum() == 0 or (~held).sum() == 0:
            continue
        p, s = _fit_predict(x, y, tv, ~held, res, levels, saturation=saturation)
        preds[held] = p[held]
        sups[held] = s[held]

    ok = np.isfinite(preds) & np.isfinite(sups)
    pred, act, sup = preds[ok], tv[ok], sups[ok]
    if act.size == 0:
        # No valid predictions (e.g. the data extent is smaller than one CV
        # block, so every fold is skipped). Skill is undefined: report 0
        # rather than NaN + 'Mean of empty slice' warnings (issue #14).
        return 0.0, []
    if np.ptp(act) == 0.0:
        # Zero-variance target (constant values): 1 - RMSE/RMSE_baseline
        # divides by zero. Report 0 rather than -inf/NaN (issue #14).
        return 0.0, []
    base = float(np.mean(act))

    rows: list[CVDetail] = []
    for lo, hi in itertools.pairwise(edges):
        m = (sup >= lo) & (sup < hi)
        n = int(m.sum())
        if n < min_n:
            continue
        e_m, e_b = act[m] - pred[m], act[m] - base
        base_rms2 = float(np.mean(e_b**2))
        if base_rms2 == 0.0:
            continue  # baseline constant in this bin: skill undefined
        skill = 1 - np.sqrt(np.mean(e_m**2)) / np.sqrt(base_rms2)
        nb = min(n, boot_max_n)
        if nb < n:
            sub = rng.choice(n, nb, replace=False)
            e_m, e_b = e_m[sub], e_b[sub]
        # A bootstrap draw where e_b is exactly 0 (act == global mean in
        # float64, realistic on data with a dominant mode) is a 0/0 skill:
        # skip the draw instead of letting NaN pollute the CI (issue #79).
        # If every draw is skipped the CI is undefined: keep the point skill.
        bs = np.full(n_boot, np.nan, dtype=np.float32)
        nb_ok = 0
        for _ in range(n_boot):
            s = rng.integers(0, nb, size=nb)
            b_rms2 = np.mean(e_b[s] ** 2)
            if b_rms2 == 0.0:
                continue
            bs[nb_ok] = 1 - np.sqrt(np.mean(e_m[s] ** 2)) / np.sqrt(b_rms2)
            nb_ok += 1
        if nb_ok == 0:
            ci_lo = ci_hi = float(skill)
        else:
            ci_lo = float(np.percentile(bs[:nb_ok], 5))
            ci_hi = float(np.percentile(bs[:nb_ok], 95))
        rows.append(
            {
                "lo_km": float(lo),
                "hi_km": float(hi),
                "n": n,
                "skill": float(skill),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            },
        )

    overall = 1 - (np.sqrt(np.mean((act - pred) ** 2)) / np.sqrt(np.mean((act - base) ** 2)))
    return float(overall), rows


def calibrate_fill_cap(
    x: np.ndarray,
    y: np.ndarray,
    tv: np.ndarray,
    block_km: tuple[float, ...] = (50.0, 100.0, 200.0, 400.0),
    min_skill: float = 0.05,
    default_km: float = FILL_CAP_DEFAULT_KM,
    saturation: float = 1.0,
    **kw: Any,
) -> tuple[float, dict[float, CVCurve]]:
    """Derive the fill cap from blocked CV across several held-out block sizes.

    RULE 1 (admissibility): a support-scale bin is only admissible for a given
    block size if `hi_km <= block_km / 2`. Otherwise the held-out points at
    that support still had training data closer than the scale being tested.

    RULE 2 (smallest admissible block wins, not the worst): a bin far below
    its block size is populated only by points near the block boundary.

    Returns:
        Tuple of fill cap in km and per-block-size CV curves.

    """
    detail: dict[float, CVCurve] = {}
    curve: dict[float, tuple[float, float, int, int]] = {}
    for bk in sorted(block_km):
        overall, rows = blocked_cv_skill(x, y, tv, block_km=bk, saturation=saturation, **kw)
        detail[bk] = {"overall_skill": overall, "rows": rows}
        for r in rows:
            if r["hi_km"] <= bk / 2.0 and r["hi_km"] not in curve:
                curve[r["hi_km"]] = (r["ci_lo"], r["skill"], bk, r["n"])

    cap: float | None = None
    for hi in sorted(curve):
        ci, skill, _bk, _n = curve[hi]
        if not math.isfinite(ci):
            # Non-finite CI (0/0 bootstrap): use the point skill instead of
            # breaking the prefix loop on NaN (issue #79).
            ci = skill
        if ci > min_skill:
            cap = hi
        else:
            break
    for curve_data in detail.values():
        curve_data["cap_km"] = cap
    return (float(cap) if cap else default_km), detail
