"""ppgrid — Pull-push scattered-data interpolation."""

from importlib.metadata import version

__version__ = version("ppgrid")

# Pipeline comes last: it does `from . import __version__` at import time.
from .pipeline import Pipeline

__all__ = ["Pipeline", "__version__"]
