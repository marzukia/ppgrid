"""Regression tests for the turbo dnfill worker-side DN write (perf wave 04).

The turbo write path used to copy each block's int16 DN grids into the tmpfs
DN fields from the main thread as results arrived; the copies now run in the
worker threads via ``_CTX["dn"]`` (disjoint block windows -> thread-safe,
byte-identical fields). These e2e tests pin that both turbo regimes (shared
in-RAM and per_box) still produce bytes identical to a non-turbo run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from ppgrid import pipeline as pipeline_mod
from ppgrid.pipeline import Pipeline

if TYPE_CHECKING:
    import pytest

DATA_CSV = Path(pipeline_mod.__file__).resolve().parent.parent / "data" / "melb_houses.csv"


def _sha(p: str | Path) -> str:
    """sha256 hex digest of a file."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _identity_cap(preset: float | None, *_args: float | None, **_kwargs: float | None) -> tuple[float | None, str]:
    """resolve_cap stub: the explicit preset stands, source labelled 'explicit'."""
    return preset, "explicit"


def test_turbo_e2e_perbox_worker_dn_write_matches_nonturbo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """per_box regime: worker-side DN writes must stay byte-identical (A6)."""
    monkeypatch.setattr(pipeline_mod.turbop, "resolve_cap", _identity_cap)
    p0 = Pipeline(str(DATA_CSV), "price", "longitude", "latitude", str(tmp_path / "ref"), res=50.0, workers=4)
    v0, s0 = p0.run()
    p1 = Pipeline(
        str(DATA_CSV),
        "price",
        "longitude",
        "latitude",
        str(tmp_path / "turbo"),
        res=50.0,
        workers=4,
        turbo=True,
        turbo_cap_gb=4.0,
    )
    v1, s1 = p1.run()
    assert p1._turbo_decision is not None  # ruff: ignore[private-member-access]
    assert p1._turbo_decision.path == "per_box"  # ruff: ignore[private-member-access]
    assert _sha(v0) == _sha(v1)
    assert _sha(s0) == _sha(s1)


def test_ctx_dn_cleared_after_block_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_CTX must not leak the DN pair into a subsequent serial (per-box) run."""
    import ppgrid.pipeline as pm

    monkeypatch.setattr(pm.turbop, "resolve_cap", _identity_cap)
    p1 = Pipeline(
        str(DATA_CSV),
        "price",
        "longitude",
        "latitude",
        str(tmp_path / "turbo"),
        res=50.0,
        workers=2,
        turbo=True,
        turbo_cap_gb=64.0,
    )
    p1.run()
    assert "dn" not in pm._CTX  # ruff: ignore[private-member-access]
