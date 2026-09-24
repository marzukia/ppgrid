"""Build config: the sdist stays slim (generated rasters and images do not ship)."""

from __future__ import annotations

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _exclude() -> list[str]:
    cfg = tomllib.loads(PYPROJECT.read_text())["tool"]["hatch"]["build"]
    return cfg["exclude"]


def test_sdist_excludes_generated_examples() -> None:
    """Generated GeoTIFFs/PNGs/JPGs and the 8.3 MB equake CSV must not ship."""
    excluded = _exclude()
    assert "examples/melb/**" in excluded
    assert "examples/equakes/**" in excluded
    assert "examples/*.png" in excluded
    assert "examples/*.jpg" in excluded
    assert "data/all_equakes.csv" in excluded


def test_sdist_keeps_source_data() -> None:
    """The small sample CSV ships in the sdist."""
    assert "data/melb_houses.csv" not in _exclude()
