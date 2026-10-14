"""ppgrid — pull-push scattered data interpolation.

Turns N geolocated points into a pair of int16 GeoTIFFs (value
percentile + support-km) at continent scale in minutes, no GPU.

Public API (beta, issue #109):

- ``Pipeline`` — the interpolation pipeline (programmatic use).
- ``main`` — the command-line entry point (the ``ppgrid`` console
  script; ``python -m ppgrid``).
- ``version`` — the installed version string.
- ``__version__`` — the same string, for ``import ppgrid`` consumers.

Only the names in ``__all__`` are public. Everything else in this
package is internal implementation and may change without notice:
``ppgrid.pipeline`` (its module-level functions, tables and constants
beyond ``Pipeline``, ``main`` and ``TurboInfeasibleError``),
``ppgrid.turbop``, ``ppgrid.ui``, ``ppgrid.zstdmt``, ``ppgrid.calibrate``,
``ppgrid.pullpush``, ``ppgrid._cinterp``, ``ppgrid._tempfiles``,
``ppgrid._platform``, ``ppgrid._prof``, and ``ppgrid.idwgrid`` (a
back-compat alias for ``ppgrid.pipeline``). Import them only if you
accept breakage between versions.

``import ppgrid`` is lazy (PEP 562): it pulls in nothing beyond the
standard library — no ui, no rich, no rasterio, no pyproj, no numpy,
no pandas. ``Pipeline`` and ``main`` load on first attribute access,
and the heavy geospatial/terminal stacks load on first pipeline use.

``baselines`` joins ``__all__`` when B-10 lands (beta plan).
"""

from importlib.metadata import version as _version

__version__ = _version("ppgrid")

__all__ = ["Pipeline", "main", "version"]
# ``baselines`` joins __all__ when B-10 lands (beta plan).

_LAZY_ATTRS = ("Pipeline", "main", "TurboInfeasibleError")


def version() -> str:
    """Return the installed ppgrid version string.

    Resolved once at import time from package metadata
    (importlib.metadata) into ``ppgrid.__version__``; this call just
    returns it, so it cannot fail at call time.

    Returns:
        The version string, e.g. ``"0.4.2"``.

    """
    return __version__


def __getattr__(name: str) -> object:
    """Lazily expose Pipeline / main / TurboInfeasibleError (PEP 562).

    They live in ppgrid.pipeline, whose import pulls in pandas (and, on
    first pipeline use, rasterio + pyproj); deferring it keeps
    ``import ppgrid`` cheap (issue #109).

    Raises:
        AttributeError: If name is not a lazily exposed attribute.

    """
    if name in _LAZY_ATTRS:
        from . import pipeline

        return getattr(pipeline, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg) from None


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))
