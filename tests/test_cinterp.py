"""Tests for the C-interp hot path (ppgrid._cinterp + PercentileTransform wiring).

The C kernel must be bit-identical to ``np.interp`` (with default
left/right) for every input; the np.interp fallback must be bit-identical
by construction. These tests run against whichever backend is active and,
where the C module is required, skip cleanly when no compiler is present.
"""

from __future__ import annotations

import os
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import ppgrid._cinterp as ci
from ppgrid.calibrate import PercentileTransform

C = pytest.mark.skipif(not ci.available(), reason="C kernel unavailable (np.interp fallback active)")

SEED = 20261011


def _bits(a: np.ndarray) -> np.ndarray:
    """uint64 view for exact bit comparison (float64 arrays)."""
    return np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)


def _fitted_q(n: int = 8_000_000) -> np.ndarray:
    """Build a realistic fitted quantile grid (strictly increasing, as fit() produces)."""
    """A realistic fitted quantile grid (strictly increasing, like fit())."""
    rng = np.random.default_rng(SEED)
    q = np.maximum.accumulate(np.quantile(rng.lognormal(17.0, 0.4, n), np.linspace(0.0, 1.0, 1001)))
    q += np.arange(1001) * np.spacing(abs(q).max() + 1.0)
    return q


def _slopes(xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Compute the per-node slope table numpy's interp uses internally."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])


def check_bits(x: np.ndarray, xp: np.ndarray, fp: np.ndarray, slopes: np.ndarray | None = None) -> None:
    """Check that ci.interp matches np.interp bit for bit."""
    ref = np.interp(x, xp, fp)
    if slopes is None:
        slopes = _slopes(xp, fp)
    out = ci.interp(x, xp, fp, slopes)
    assert out.dtype == np.float64
    assert out.shape == x.shape
    assert np.array_equal(_bits(ref), _bits(out))


@pytest.mark.parametrize("nxp", [1, 2, 3, 4, 5, 8, 12, 64, 1001])
def test_bit_identity_length_sweep(nxp: int) -> None:
    """Bit-identity across node counts 1..1001, incl. out-of-range keys."""
    rng = np.random.default_rng(SEED + nxp)
    q = np.sort(rng.uniform(1e4, 1e5, nxp))
    p = np.linspace(0.0, 100.0, nxp)
    x = rng.uniform(1e4 - 100, 1e5 + 100, 200_000)
    check_bits(x, q, p)


def test_bit_identity_fitted_q_block() -> None:
    """Bit-identity on a full 2048x2048 fwd block (fitted lognormal q)."""
    rng = np.random.default_rng(SEED + 1)
    q = _fitted_q()
    p = np.linspace(0.0, 100.0, 1001)
    x = rng.lognormal(17.0, 0.4, (2048, 2048))
    check_bits(x, q, p)


def test_bit_identity_inv_block() -> None:
    """Bit-identity on a full 2048x2048 inv block, incl. out-of-range percentiles."""
    rng = np.random.default_rng(SEED + 2)
    q = _fitted_q()
    p = np.linspace(0.0, 100.0, 1001)
    x = rng.uniform(-5.0, 105.0, (2048, 2048))
    check_bits(x, p, q)


def test_bit_identity_tied_and_nonmonotonic_xp() -> None:
    """Bit-identity with tied and non-monotonic node grids."""
    rng = np.random.default_rng(SEED + 3)
    xp_tied = np.array([0.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 5.0])
    fp = np.arange(len(xp_tied), dtype=np.float64)
    x = rng.uniform(0.0, 5.0, 100_000)
    check_bits(x, xp_tied, fp)
    xp_bad = rng.uniform(0.0, 10.0, 128)  # non-monotonic: numpy's search is data-dependent
    check_bits(rng.uniform(0.0, 10.0, 50_000), xp_bad, rng.uniform(0.0, 10.0, 128))


def test_bit_identity_nan_xp() -> None:
    """NaN in xp (lenxp > 1) routes to the verbatim clone; bits match np.interp."""
    # M2 repro: np.interp -> 15.0, old chunked kernel -> nan.
    check_bits(
        np.array([1.5, 2.5, 4.0, 6.0, 100.0]),
        np.array([1.0, 2.0, np.nan, 5.0, 9.0]),
        np.array([10.0, 20.0, 30.0, 40.0, 80.0]),
    )
    rng = np.random.default_rng(SEED + 11)
    for i in range(5):
        xpn = np.sort(rng.uniform(0.0, 10.0, 64))
        xpn[0 if i == 4 else 10 + 12 * i] = np.nan
        check_bits(rng.uniform(0.0, 10.0, 50_000), xpn, rng.uniform(0.0, 10.0, 64))


def test_bit_identity_special_values() -> None:
    """Bit-identity with NaN/inf/exact-node/out-of-range/0.0/tiny/huge keys."""
    rng = np.random.default_rng(SEED + 4)
    q = _fitted_q()
    p = np.linspace(0.0, 100.0, 1001)
    base = rng.lognormal(17.0, 0.4, 100_000)
    x = np.concatenate(
        [
            base,
            np.array([np.nan, np.inf, -np.inf]),
            np.array([q[0], q[-1], q[500]]),  # exact node hits
            np.array([q[0] - 1e-9, q[-1] + 1e-9]),  # just outside both ends
            np.array([0.0, 1e-300, 1e300]),
        ]
    )
    check_bits(x, q, p)
    check_bits(x, p, q)  # inv direction too (q as fp)


def test_bit_identity_lenxp1() -> None:
    """Bit-identity on the single-node branch (numpy maps NaN to fp[0])."""
    x = np.array([1.0, 5.0, 100.0, np.nan, np.inf, -np.inf, 5.0])
    for xp in [np.array([5.0]), np.array([np.nan]), np.array([-1.0])]:
        check_bits(x, xp, xp * 10.0)


def test_bit_identity_empty() -> None:
    """Bit-identity on empty inputs (incl. empty 2-D)."""
    q = _fitted_q(1_000_000)
    p = np.linspace(0.0, 100.0, 1001)
    check_bits(np.empty(0), q, p)
    check_bits(np.empty((0, 4096)), q, p)


def test_bit_identity_f32() -> None:
    """Bit-identity on f32 input (exact promotion, no intermediate cast)."""
    rng = np.random.default_rng(SEED + 5)
    q = _fitted_q()
    p = np.linspace(0.0, 100.0, 1001)
    x = rng.lognormal(17.0, 0.4, 200_000).astype(np.float32)
    x = np.concatenate([x, np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)])
    check_bits(x, q, p)


def test_interp_f32_tables_fall_back() -> None:
    """f32 xp/fp/slopes fall back to np.interp instead of C-level TypeErrors."""
    rng = np.random.default_rng(SEED + 12)
    x = rng.uniform(0.0, 10.0, 50_000)
    for which in ("xp", "fp", "slopes"):
        xp = np.sort(rng.uniform(0.0, 10.0, 128))
        fp = rng.uniform(0.0, 10.0, 128)
        slopes = _slopes(xp, fp)
        if which == "xp":
            xp = xp.astype(np.float32)
        elif which == "fp":
            fp = fp.astype(np.float32)
        else:
            slopes = slopes.astype(np.float32)
        check_bits(x, xp, fp, slopes)


def test_bit_identity_strided_view() -> None:
    """Bit-identity on a non-contiguous strided view of a larger field."""
    rng = np.random.default_rng(SEED + 6)
    q = _fitted_q(1_000_000)
    p = np.linspace(0.0, 100.0, 1001)
    big = rng.lognormal(17.0, 0.4, (4096, 4096))
    x = big[::3, ::7]  # non-contiguous in both dims
    check_bits(x, q, p)


def test_error_cases() -> None:
    """Length/size mismatch errors raise ValueError with the expected messages."""
    q = np.linspace(0.0, 1.0, 5)
    p = np.linspace(0.0, 100.0, 5)
    x = np.ones(10)
    with pytest.raises(ValueError, match="fp and xp"):
        ci.interp(x, q, p[:-1], _slopes(q, p))
    if not ci.available():
        pytest.skip("C-specific error messages")
    mod = ci._mod
    assert mod is not None
    out = np.empty(10)
    with pytest.raises(ValueError, match="slopes"):
        mod.interp_f64(x, q, p, _slopes(q, p)[:-1], out)
    with pytest.raises(ValueError, match="same size"):
        mod.interp_f64(x, q, p, _slopes(q, p), np.empty(5))
    with pytest.raises(ValueError, match="empty"):
        mod.interp_f64(x, np.empty(0), np.empty(0), np.empty(0), out)


def test_fallback_when_c_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the C module disabled, interp is np.interp itself (bits included)."""
    monkeypatch.setattr(ci, "_mod", None)
    assert not ci.available()
    rng = np.random.default_rng(SEED + 7)
    q = _fitted_q(1_000_000)
    p = np.linspace(0.0, 100.0, 1001)
    x = rng.lognormal(17.0, 0.4, 50_000)
    out = ci.interp(x, q, p, _slopes(q, p))
    assert np.array_equal(_bits(np.interp(x, q, p)), _bits(out))
    # non-float dtypes always fall back to np.interp
    xi = np.arange(100, dtype=np.int64)
    out_i = ci.interp(xi, np.arange(100.0), np.arange(100.0), _slopes(np.arange(100.0), np.arange(100.0)))
    assert np.array_equal(_bits(np.interp(xi, np.arange(100.0), np.arange(100.0))), _bits(out_i))


def test_tmpdir_owner_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPath) -> None:
    """A shared tmpdir cache dir owned by another uid is treated as absent."""
    root = tmp_path / "ppgrid-cinterp"
    root.mkdir()
    foe = root / f"_cinterp_c-{'0' * 12}{ci._SO_SUFFIX}"
    foe.write_bytes(b"placeholder")
    imports: list = []
    real_stat = os.stat

    def fake_stat(path: object, *a: object, **k: object) -> object:
        st = real_stat(path, *a, **k)
        if str(path) == str(root):
            return SimpleNamespace(st_uid=(os.geteuid() + 1) % 2**32)
        return st

    monkeypatch.setattr(ci, "_TMPDIR_ROOT", root)
    monkeypatch.setattr(ci, "_candidate_paths", lambda: [foe])
    monkeypatch.setattr(ci, "_mod", None)
    monkeypatch.setattr(ci, "_reason", None)
    monkeypatch.setattr(ci, "_build_err", None)
    monkeypatch.setattr(ci, "_try_import", lambda path: imports.append(path) or True)
    monkeypatch.setattr(os, "stat", fake_stat)
    ci._load()
    assert imports == []  # the other-user .so was never imported
    assert ci._mod is None
    assert ci._reason is not None


def _pct() -> PercentileTransform:
    """Fit a PercentileTransform on a realistic 2M-point lognormal sample."""
    rng = np.random.default_rng(SEED + 8)
    return PercentileTransform().fit(rng.lognormal(17.0, 0.4, 2_000_000))


def test_percentile_transform_fwd_inv_bits() -> None:
    """fwd/inv via the C path match np.interp bit for bit."""
    t = _pct()
    rng = np.random.default_rng(SEED + 9)
    q = t.q
    assert q is not None
    p = t._p
    x = rng.lognormal(17.0, 0.4, (1024, 1024))
    x = np.concatenate([x.ravel(), np.array([np.nan, np.inf, -np.inf, q[0], q[-1]])])
    assert np.array_equal(_bits(np.interp(x, q, p)), _bits(t.fwd(x)))
    y = rng.uniform(-2.0, 102.0, 200_000)
    y = np.concatenate([y, np.array([np.nan, np.inf, -np.inf, 0.0, 100.0])])
    assert np.array_equal(_bits(np.interp(y, p, q)), _bits(t.inv(y)))


def test_percentile_transform_flat_fit() -> None:
    """Flat (constant) fit: fwd/inv round-trip stays bit-exact."""
    t = PercentileTransform().fit(np.full(1000, 123.456))
    x = np.array([123.456, 123.455, 123.457, np.nan])
    assert np.array_equal(_bits(np.interp(x, t.q, t._p)), _bits(t.fwd(x)))
    assert np.array_equal(_bits(np.interp(t.fwd(x), t._p, t.q)), _bits(t.inv(t.fwd(x))))


def test_percentile_transform_state_roundtrip() -> None:
    """State round-trip rebuild: C-path outputs unchanged."""
    t = _pct()
    t2 = PercentileTransform(t.state()["quantiles"])
    x = np.linspace(1e4, 1e5, 10_000)
    assert np.array_equal(_bits(t.fwd(x)), _bits(t2.fwd(x)))
    assert np.array_equal(_bits(t.inv(x / 1e4)), _bits(t2.inv(x / 1e4)))


def test_percentile_transform_refit_resets_cprep() -> None:
    """Fit -> fwd -> re-fit -> fwd: slope cache reset, bits match np.interp on new q."""
    rng = np.random.default_rng(SEED + 13)
    v1 = rng.lognormal(2.0, 1.0, 500_000)
    t = PercentileTransform().fit(v1)
    q1 = t.q
    assert q1 is not None
    x1 = np.array([float(np.median(v1))])
    assert np.array_equal(_bits(np.interp(x1, q1, t._p)), _bits(t.fwd(x1)))
    if ci.available():
        assert t._c_ready  # C tables built from the first fit
    v2 = rng.lognormal(5.0, 1.0, 500_000)
    t.fit(v2)
    q2 = t.q
    assert q2 is not None
    if ci.available():
        assert not t._c_ready  # re-fit invalidated the old slope tables
    x2 = np.array([float(np.median(v2))])
    got = t.fwd(x2)
    want = np.interp(x2, q2, t._p)
    assert np.array_equal(_bits(want), _bits(got))
    # A stale cache evaluates the new median on the OLD grid (reviewer repro:
    # ~51.83); the correct value is the ~50 class on the new grid.
    assert float(got[0]) == pytest.approx(50.0, abs=1.5)


def test_percentile_transform_no_c(monkeypatch: pytest.MonkeyPatch) -> None:
    """fwd/inv with the C module disabled still match np.interp bits."""
    monkeypatch.setattr(ci, "_mod", None)
    t = _pct()
    x = np.linspace(1e4, 1e5, 10_000)
    assert np.array_equal(_bits(np.interp(x, t.q, t._p)), _bits(t.fwd(x)))
    y = np.linspace(-1.0, 101.0, 10_000)
    assert np.array_equal(_bits(np.interp(y, t._p, t.q)), _bits(t.inv(y)))


def test_percentile_transform_q_none_errors_as_before() -> None:
    """Unfitted transform raises the same ValueError as before the C path."""
    t = PercentileTransform()
    with pytest.raises(ValueError, match="too small depth"):
        t.fwd(np.ones(4))
    with pytest.raises(ValueError, match="too small depth"):
        t.inv(np.ones(4))


@C
def test_threaded_fwd_bit_identity() -> None:
    """Eight threads sharing one transform stay bit-exact (GIL released in C)."""
    """Check that eight workers sharing one transform stay bit-exact."""
    t = _pct()
    rng = np.random.default_rng(SEED + 10)
    q = t.q
    assert q is not None
    blocks = [rng.lognormal(17.0, 0.4, (2048, 2048)) for _ in range(4)]
    refs = [np.interp(b.ravel(), q, t._p) for b in blocks]
    errors: list[BaseException] = []

    def work(chunk: int) -> None:
        try:
            for i in range(chunk):
                b = blocks[i % len(blocks)].ravel()
                out = t.fwd(b)
                assert np.array_equal(_bits(refs[i % len(blocks)]), _bits(out))
        except BaseException as exc:  # ruff: ignore[blind-except] - propagate to the test
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(16,)) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors


def test_backend_string() -> None:
    """backend() reports the active kernel and reason() is set when unavailable."""
    tag = ci.backend()
    assert isinstance(tag, str)
    assert tag
    if ci.available():
        assert "C" in tag
    else:
        assert ci.reason() is not None
