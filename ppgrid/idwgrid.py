"""Back-compat shim: the pipeline moved to :mod:`ppgrid.pipeline`.

This module re-exports the public names so existing imports
(``from ppgrid.idwgrid import Pipeline``) and the ``ppgrid.idwgrid:main``
console entry point keep working. New code should import from
:mod:`ppgrid.pipeline` (or the package root) directly.
"""

from __future__ import annotations

from .pipeline import (
    DEFAULT_BLOCK,
    DEFAULT_CALIB_MAX_POINTS,
    DEFAULT_CAP_KM,
    DEFAULT_RES,
    DEFAULT_SATURATION,
    DEFAULT_SCALE,
    DEFAULT_WORKERS,
    INT16_MAX,
    NODATA,
    OUT_CRS,
    SRC_CRS,
    TILE_PX,
    WORK_CRS,
    Pipeline,
    main,
    run,
)

__all__ = [
    "DEFAULT_BLOCK",
    "DEFAULT_CALIB_MAX_POINTS",
    "DEFAULT_CAP_KM",
    "DEFAULT_RES",
    "DEFAULT_SATURATION",
    "DEFAULT_SCALE",
    "DEFAULT_WORKERS",
    "INT16_MAX",
    "NODATA",
    "OUT_CRS",
    "SRC_CRS",
    "TILE_PX",
    "WORK_CRS",
    "Pipeline",
    "main",
    "run",
]
