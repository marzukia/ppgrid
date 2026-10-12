"""OS-portability simulations: reproduce the Windows/macOS failure modes on Linux.

These two simulations are the red/green proof for the PR #97 CI-fix wave. Each
patches the platform primitives that differ on the target OS, re-runs the code
path that was failing, and asserts it now succeeds. With the fix reverted they
fail (FileNotFoundError / TypeError / ZstdmtError); with the fix they pass.

- Windows: CPython's ctypes.CDLL(None) raises TypeError on os.name == 'nt'
  (the '/' in name fullpath check runs before the None special case). The fix
  guards the CDLL call so `import ppgrid` succeeds with _MADVISE None.

- macOS: no /proc/meminfo (physical RAM unknown) and no /proc/self/maps (zstdmt
  unavailable). The fix makes _read_physical_gb return None (unknown RAM,
  conservative cap floor), the turbo precheck skip the ceiling note, and the
  parallel ZSTD writer return False (stock fallback) instead of raising.

"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rasterio.transform import from_origin

from ppgrid import pipeline as pipeline_mod
from ppgrid import pullpush as pp
from ppgrid import turbop, zstdmt
from ppgrid.pipeline import OUT_CRS, WORK_CRS, Pipeline, _turbo_write_parallel

if TYPE_CHECKING:
    import pytest

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Windows: ctypes.CDLL(None) raises TypeError at import time
# ---------------------------------------------------------------------------


def test_windows_cdll_typeerror_sim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Import ppgrid succeeds when CDLL(None) raises TypeError (Windows nt).

    Runs in a fresh interpreter (the import-time CDLL call cannot be undone
    in-process). ctypes.CDLL is patched to raise the exact Windows TypeError
    before ppgrid is imported. The fix degrades _LIBC/_MADVISE to None so the
    import completes; without it the import crashes with the TypeError.

    """
    code = (
        "import ctypes\n"
        "def _raise(*a, **k):\n"
        "    raise TypeError('argument of type ' + repr(None) + ' is not iterable')\n"
        "ctypes.CDLL = _raise\n"
        "import ppgrid.pullpush as pp\n"
        "print('LIBC_NONE' if pp._LIBC is None else 'LIBC_SET')\n"
        "print('MADVISE_NONE' if pp._MADVISE is None else 'MADVISE_SET')\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
    )
    assert proc.returncode == 0, f"import crashed (the Windows failure): {proc.stderr}"
    assert "LIBC_NONE" in proc.stdout
    assert "MADVISE_NONE" in proc.stdout
    # _nohuge must be a clean no-op with _MADVISE None (the degraded state the
    # guarded import leaves on Windows); a >16 MiB array would madvise otherwise.
    monkeypatch.setattr(pp, "_MADVISE", None)
    pp._nohuge(np.ones(1 << 20, dtype=np.float64))  # must not raise


# ---------------------------------------------------------------------------
# macOS: no /proc/meminfo -> physical RAM unknown
# ---------------------------------------------------------------------------


def test_macos_no_proc_meminfo_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_read_physical_gb returns None (not FileNotFoundError) when /proc is absent."""
    monkeypatch.setattr(turbop, "MEMINFO_PATH", str(tmp_path / "missing-meminfo"))
    assert turbop._read_physical_gb() is None


def test_macos_no_proc_cap_floor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown physical RAM plans conservatively: floor, or the cgroup bound.

    Without the fix detect_cap_gb(physical_gb=None) re-reads /proc/meminfo and
    raises FileNotFoundError here; with the fix it resolves to the smaller
    footprint (a finite cgroup bound when present, else the 8 GB floor).

    """
    monkeypatch.setattr(turbop, "MEMINFO_PATH", str(tmp_path / "missing-meminfo"))
    # unlimited cgroup: the conservative floor, never an unbounded cap.
    cap, source = turbop.detect_cap_gb(physical_gb=None, cgroup_gb=math.inf)
    assert cap == 8.0
    assert "conservative floor 8 GB" in source
    # finite cgroup: the slice bound binds, exactly as on Linux.
    cap, source = turbop.detect_cap_gb(physical_gb=None, cgroup_gb=24.0)
    assert cap == 16.0
    assert "physical RAM unknown" in source
    # a preset with unknown physical is floored but not clamped (legacy rule).
    cap, source = turbop.resolve_cap(16.0, physical_gb=None, cgroup_gb=math.inf)
    assert cap == 16.0
    assert source == "preset 16 GB"


def test_macos_no_proc_turbo_precheck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The turbo precheck survives a missing /proc (no TypeError from min()).

    Emulates the macOS auto-turbo path: /proc/meminfo absent (physical None) and
    no cgroup bound (inf). Without the fix available_gb = min(None, inf) raises
    TypeError; with the fix it is None and precheck skips the ceiling note.

    """
    monkeypatch.setattr(turbop, "MEMINFO_PATH", str(tmp_path / "missing-meminfo"))
    monkeypatch.setattr(turbop, "_read_cgroup_max_gb", lambda: math.inf)
    monkeypatch.setattr(pipeline_mod.turbop, "resolve_cap", lambda preset, **_k: (preset or 8.0, "explicit"))
    p = Pipeline(
        str(REPO / "data" / "melb_houses.csv"),
        "price",
        "longitude",
        "latitude",
        str(tmp_path / "o"),
        res=100.0,
        cap_km=25.0,
        turbo=True,
        turbo_cap_gb=16.0,
    )
    p.n, p.nx_padded, p.ny_padded, p.res, p.cap_km_val = (400_000, 41472, 38400, 100.0, 25)
    p._turbo_precheck()
    assert p._turbo_decision is not None
    # per_box at 16 GB full-AU; available_gb None means no ceiling warning.
    assert p._turbo_decision.path == "per_box"


def test_macos_no_proc_write_parallel_returns_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_turbo_write_parallel is total: False when zstdmt is unavailable.

    Faithful macOS sim: /proc/self/maps is unreadable, so the CPL pfn cannot be
    resolved. Without the fix the direct call reaches zstdmt.compress_tiles and
    raises ZstdmtError (cannot read /proc/self/maps); with the fix the early
    `available()` gate returns False so the caller falls back to stock codec.

    """

    def _no_maps() -> str:
        msg = "cannot read /proc/self/maps; zstdmt needs a Linux /proc"
        raise zstdmt.ZstdmtError(msg)

    monkeypatch.setattr(zstdmt, "_gdal_path", _no_maps)
    monkeypatch.setattr(zstdmt._STATE, "lib", None)  # force re-resolution
    monkeypatch.setattr(zstdmt._STATE, "pfn", None)  # (cache would short-circuit)
    assert zstdmt.available() is False  # the macOS state, through the real code path
    a = np.zeros((64, 64), dtype=np.int16)
    xform = from_origin(0.0, 64 * 100.0, 100.0, 100.0)
    crs = f"EPSG:{WORK_CRS}"
    profile = {
        "driver": "GTiff",
        "dtype": "int16",
        "nodata": -32768,
        "compress": "ZSTD",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "predictor": 2,
        "count": 1,
    }
    ok = _turbo_write_parallel(
        a,
        xform,
        crs,
        str(tmp_path / "x.tif"),
        profile,
        f"EPSG:{OUT_CRS}",
        xform,
        64,
        64,
        tags={},
        scales=(1.0,),
        offsets=(0.0,),
        n_threads=1,
    )
    assert ok is False
