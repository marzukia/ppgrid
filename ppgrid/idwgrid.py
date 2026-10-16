"""Backwards-compat shim: the pipeline moved to ppgrid.pipeline.

`import ppgrid.idwgrid as ig`, `from ppgrid.idwgrid import X`, and the old
`ppgrid.idwgrid:main` console-script target all keep working — this module
*is* ppgrid.pipeline (alias in sys.modules). New code should import from
ppgrid.pipeline.
"""

import sys as _sys
from importlib import import_module as _import_module

_sys.modules[__name__] = _import_module("ppgrid.pipeline")

if __name__ == "__main__":
    # `python -m ppgrid.idwgrid` (audit #84 a1): the sys.modules aliasing
    # above re-exposes the module but does not re-run the pipeline's own
    # __main__ guard, so without this the shim was a silent no-op.
    from .pipeline import main as _main

    _main()
