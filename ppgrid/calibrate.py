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
from collections.abc import Callable
from typing import Any

import numpy as np

from .pullpush import bin_points, pad_to_pyramid, pull_push

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

        Returns:
            self.

        """
        q = np.quantile(v.astype(np.float64), np.linspace(0, 1, self.NQ))
        self.q = np.maximum.accumulate(q)
        eps = np.arange(self.NQ) * np.spacing(np.abs(self.q).max() + 1.0)
        self.q += eps
        return self

    @property
    def _p(self) -> np.ndarray:
        return np.linspace(0.0, 100.0, self.NQ)

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


def _cell_key(x: np.ndarray, y: np.ndarray, res: float) -> np.ndarray:
    """Hash (x, y) coordinates to a unique integer per cell at the given resolution.

    Returns:
        Integer array of cell keys.

    """
    ix = ((x - x.min()) // res).astype(np.int64)
    iy = ((y - y.min()) // res).astype(np.int64)
    return ix * (int(iy.max()) + 1) + iy


def icc(values: np.ndarray, x: np.ndarray, y: np.ndarray, res: float) -> float:
    """Intraclass correlation at cell size `res`.

    The share of total variance explained by location. ~0 means the value
    is not a spatial field at all.

    Returns:
        ICC value between 0 and 1.

    """
    key = _cell_key(x, y, res)
    _, inv = np.unique(key, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1
    csum = np.bincount(inv, weights=values, minlength=n)
    ccnt = np.bincount(inv, minlength=n)
    cmean = csum / np.maximum(ccnt, 1)
    within = np.mean((values - cmean[inv]) ** 2)
    total = values.var()
    return float(1.0 - within / total) if total > 0 else 0.0


def choose_transform(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    scales: tuple[float, ...] = (5000.0, 25000.0, 100000.0),
) -> tuple[Transform, list[tuple[Transform, float, dict[float, float]]]]:
    """Pick the transform maximising mean ICC across coarse scales.

    Returns:
        Tuple of best transform and all scored results.

    """
    results: list[tuple[Transform, float, dict[float, float]]] = []
    for candidate in transforms():
        if not candidate.valid(values):
            continue
        fitted = candidate.fit(values)
        tv = fitted.fwd(values)
        if not np.all(np.isfinite(tv)):
            continue
        per_scale: dict[float, float] = {float(s): icc(tv, x, y, s) for s in scales}
        results.append((fitted, float(np.mean(list(per_scale.values()))), per_scale))
    results.sort(key=lambda r: -r[1])
    return results[0][0], results


def _fit_predict(
    x: np.ndarray,
    y: np.ndarray,
    tv: np.ndarray,
    train: np.ndarray,
    res: float,
    levels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train on the given indices, predict at all (x, y) positions.

    Returns:
        Tuple of predicted values and support in km.

    """
    x0, y0 = x.min(), y.min()
    ix = ((x - x0) // res).astype(np.int64)
    iy = ((y - y0) // res).astype(np.int64)
    nx = pad_to_pyramid(int(ix.max()) + 1, levels)
    ny = pad_to_pyramid(int(iy.max()) + 1, levels)
    s, c = bin_points(ix[train], iy[train], tv[train], nx, ny)
    val, sup = pull_push(s, c, res, levels)
    return val[ix, iy], sup[ix, iy] / 1000.0


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
    edges: tuple[float, ...] = (0, 2, 4, 8, 16, 32, 64, 128, 256, 1e9),
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
    block_size_m = block_km * 1000.0
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
        p, s = _fit_predict(x, y, tv, ~held, res, levels)
        preds[held] = p[held]
        sups[held] = s[held]

    ok = np.isfinite(preds) & np.isfinite(sups)
    pred, act, sup = preds[ok], tv[ok], sups[ok]
    base = float(np.mean(act))

    rows: list[CVDetail] = []
    for lo, hi in itertools.pairwise(edges):
        m = (sup >= lo) & (sup < hi)
        n = int(m.sum())
        if n < min_n:
            continue
        e_m, e_b = act[m] - pred[m], act[m] - base
        skill = 1 - np.sqrt(np.mean(e_m**2)) / np.sqrt(np.mean(e_b**2))
        nb = min(n, boot_max_n)
        if nb < n:
            sub = rng.choice(n, nb, replace=False)
            e_m, e_b = e_m[sub], e_b[sub]
        bs = np.empty(n_boot, dtype=np.float32)
        for k in range(n_boot):
            s = rng.integers(0, nb, size=nb)
            bs[k] = 1 - np.sqrt(np.mean(e_m[s] ** 2)) / np.sqrt(np.mean(e_b[s] ** 2))
        rows.append(
            {
                "lo_km": float(lo),
                "hi_km": float(hi),
                "n": n,
                "skill": float(skill),
                "ci_lo": float(np.percentile(bs, 5)),
                "ci_hi": float(np.percentile(bs, 95)),
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
    default_km: float = 25.0,
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
    curve: dict[float, tuple[float, float, int]] = {}
    for bk in sorted(block_km):
        overall, rows = blocked_cv_skill(x, y, tv, block_km=bk, **kw)
        detail[bk] = {"overall_skill": overall, "rows": rows}
        for r in rows:
            if r["hi_km"] <= bk / 2.0 and r["hi_km"] not in curve:
                curve[r["hi_km"]] = (r["ci_lo"], bk, r["n"])

    cap: float | None = None
    for hi in sorted(curve):
        if curve[hi][0] > min_skill:
            cap = hi
        else:
            break
    for curve_data in detail.values():
        curve_data["cap_km"] = cap
    return (float(cap) if cap else default_km), detail
