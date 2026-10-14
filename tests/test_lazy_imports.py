"""Verify rasterio/pyproj/rich are lazy-loaded (not imported by `import ppgrid`).

The CLI and library only need the native stack (rasterio+pyproj+GDAL) and
the terminal-render stack (ui -> rich) at first use (ingest/warp/write/
CLI), so importing ppgrid must not pay those costs. Regression guard for
the PEP 562 lazy-import layer in pipeline.py / zstdmt.py (rich + ui since
issue #109).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_NOT_LOADED = (
    "import sys, ppgrid, ppgrid.pipeline, ppgrid.zstdmt\n"
    "assert 'rasterio' not in sys.modules, 'rasterio loaded at import time'\n"
    "assert 'pyproj' not in sys.modules, 'pyproj loaded at import time'\n"
    "assert 'rich' not in sys.modules, 'rich loaded at import time'\n"
    "assert 'ppgrid.ui' not in sys.modules, 'ppgrid.ui loaded at import time'\n"
)

# Bare `import ppgrid` must be cheap (issue #109): nothing beyond the
# stdlib, and the beta public surface resolves lazily on first access.
_PKG_LAZY = (
    "import sys, ppgrid\n"
    "for mod in ('rasterio', 'pyproj', 'rich', 'numpy', 'pandas', 'ppgrid.pipeline', 'ppgrid.ui'):\n"
    "    assert mod not in sys.modules, f'{mod} loaded by bare `import ppgrid`'\n"
    "assert ppgrid.__all__ == ['Pipeline', 'main', 'version']\n"
    "assert ppgrid.version() == ppgrid.__version__\n"
    "assert ppgrid.Pipeline.__name__ == 'Pipeline'  # triggers the lazy import\n"
    "assert 'ppgrid.pipeline' in sys.modules\n"
    "assert ppgrid.main.__name__ == 'main'\n"
    "assert issubclass(ppgrid.TurboInfeasibleError, Exception)\n"
    "for mod in ('rasterio', 'pyproj', 'rich'):\n"
    "    assert mod not in sys.modules, f'{mod} loaded by attribute access'\n"
)

_HELP_CHECK = (
    "import sys\n"
    "from ppgrid.pipeline import main\n"
    "try:\n"
    "    main(['--help'])\n"
    "except SystemExit as e:\n"
    "    assert e.code in (0, None), f'--help exited {e.code}'\n"
    "assert 'rasterio' not in sys.modules, 'rasterio loaded by --help'\n"
    "assert 'pyproj' not in sys.modules, 'pyproj loaded by --help'\n"
)


def _fresh_python(code: str) -> subprocess.CompletedProcess[str]:
    """Run code in a fresh interpreter (no rasterio/pyproj in sys.modules)."""
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_import_ppgrid_defers_native_modules() -> None:
    """Importing ppgrid leaves rasterio, pyproj and rich out of sys.modules."""
    result = _fresh_python(_NOT_LOADED)
    assert result.returncode == 0, result.stderr


def test_package_import_is_lazy() -> None:
    """Bare `import ppgrid` loads nothing heavy; the public surface resolves lazily (#109)."""
    result = _fresh_python(_PKG_LAZY)
    assert result.returncode == 0, result.stderr


def test_cli_help_defers_native_modules() -> None:
    """Running `ppgrid --help` exits 0 without loading rasterio or pyproj."""
    result = _fresh_python(_HELP_CHECK)
    assert result.returncode == 0, result.stderr


def test_lazy_names_resolve_and_cache() -> None:
    """Lazy names resolve on first attribute access and cache in the module dict."""
    import ppgrid.pipeline as pgrid

    # rasterio-side names resolve to the canonical objects
    rasterio = pgrid.rasterio
    assert "rasterio" in sys.modules
    window = pgrid.Window
    from_origin = pgrid.from_origin
    reproject = pgrid.reproject
    resampling = pgrid.Resampling
    memory_dataset = pgrid.MemoryDataset
    not_georeferenced = pgrid.NotGeoreferencedWarning
    affine = pgrid.Affine
    assert rasterio.windows.Window is window
    assert rasterio.transform.from_origin is from_origin
    assert rasterio.warp.reproject is reproject
    assert rasterio.warp.Resampling is resampling
    assert rasterio.errors.NotGeoreferencedWarning is not_georeferenced
    assert rasterio.transform.Affine is affine
    assert memory_dataset.__module__.startswith("rasterio")
    # resolution is cached in the module dict (second access is a plain lookup)
    assert pgrid.__dict__["reproject"] is reproject

    # pyproj-side name
    tr = pgrid.Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    x, y = tr.transform([144.96], [-37.81])
    # Melbourne in Wagner VII: x ~ 13.99e6 m, y ~ -4.2e6 m (order-of-magnitude check)
    assert 1e7 < float(x[0]) < 2e7
    assert -1e7 < float(y[0]) < 0
    assert "pyproj" in sys.modules
    # pyproj.exceptions name still resolves and is catchable
    crs_error = pgrid.CRSError
    with pytest.raises(crs_error):
        pgrid.Transformer.from_crs("EPSG:4326", "not-a-crs", always_xy=True)


def test_unknown_name_raises_attribute_error() -> None:
    """AttributeError for names that are not lazy imports."""
    import ppgrid.pipeline as pgrid

    missing = "no_such_ppgrid_name"
    with pytest.raises(AttributeError, match="no attribute"):
        getattr(pgrid, missing)


def test_lazy_coexists_with_monkeypatch() -> None:
    """A patched module attribute shadows the lazy __getattr__ (test contract)."""
    import ppgrid.pipeline as pgrid

    original = pgrid.reproject  # resolved + cached first
    assert original is not None
    hits: list[int] = []

    def fake(*_args: object, **_kwargs: object) -> None:
        hits.append(1)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pgrid, "reproject", fake)
        assert pgrid.reproject is fake
        pgrid.reproject()
    assert len(hits) == 1
    assert pgrid.reproject is original  # restored


def test_deleted_lazy_name_self_heals_via_load_native() -> None:
    """A deleted lazy name re-resolves on next access (review F1, PEP 562).

    Pre-fix, the second getattr raised KeyError (globals()[name] after the
    _load_native fast-return), not the lazy value.

    """
    import ppgrid.pipeline as pgrid

    original = pgrid.reproject  # resolved + cached first
    with pytest.MonkeyPatch.context() as mp:
        mp.delattr(pgrid, "reproject")
        assert "reproject" not in vars(pgrid)
        assert pgrid.reproject is original  # _load_native re-publishes
    assert pgrid.reproject is original


def test_zstdmt_lazy_names_resolve() -> None:
    """Lazy names in zstdmt resolve on first use."""
    from ppgrid import zstdmt

    window = zstdmt.Window
    rasterio = zstdmt.rasterio
    assert rasterio.windows.Window is window
    missing = "no_such_zstdmt_name"
    with pytest.raises(AttributeError, match="no attribute"):
        getattr(zstdmt, missing)


def test_tiny_reproject_runs_with_lazy_stack(tmp_path: Path) -> None:
    """A real 2x2 warp works once the lazy stack loads (no eager-import reliance)."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from ppgrid.pipeline import NODATA, _reproject_band

    arr = np.array([[1, 2], [3, 4]], dtype=np.int16)
    t = from_origin(0.0, 2.0, 1.0, 1.0)
    profile = {
        "driver": "GTiff",
        "dtype": "int16",
        "width": 2,
        "height": 2,
        "count": 1,
        "nodata": NODATA,
        "crs": "EPSG:6933",
        "transform": t,
        "compress": "LZW",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }
    src = tmp_path / "src.tif"
    dst = tmp_path / "dst.tif"
    with rasterio.open(src, "w", **profile) as ds:
        ds.write(arr, 1)
    _reproject_band(str(src), str(dst), profile, "EPSG:6933", t, 2, 2)
    with rasterio.open(dst) as ds:
        assert ds.width == 2
        assert ds.height == 2
        assert ds.read(1).tolist() == arr.tolist()
