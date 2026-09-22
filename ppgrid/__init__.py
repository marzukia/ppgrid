"""ppgrid — Pull-push scattered-data interpolation."""

from importlib.metadata import version

from .pipeline import Pipeline

__version__ = version("ppgrid")

__all__ = ["Pipeline", "__version__"]
