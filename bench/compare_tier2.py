"""Bit-identity comparison for tier 2 bench outputs (issue #3).

Compares value.tif / support_km.tif arrays (tiled, memory-safe for multi-GB
rasters) between two tagged run directories, and against committed examples.

Usage:
    uv run python bench/compare_tier2.py /tmp/tier2/baseline /tmp/tier2/after
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import rasterio
from rasterio.windows import Window

REPO = Path(__file__).resolve().parent.parent
TILE = 1024

# Committed-example anchors: (config, committed value tif, committed support tif)
ANCHORS = [
    ("melb10", "examples/melb/10m/value.tif", "examples/melb/10m/support_km.tif"),
    ("melb500", "examples/melb/500m/value.tif", "examples/melb/500m/support_km.tif"),
    ("equakes500", "examples/equakes/value.tif", "examples/equakes/support_km.tif"),
]


def _sha(path: Path) -> str:
    """First 16 hex chars of the file sha256 (streaming)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _compare_tiled(pa: Path, pb: Path) -> tuple[bool, int]:
    """Tiled bit-compare of two GeoTIFFs. Returns (identical, diff_pixels)."""
    with rasterio.open(pa) as a, rasterio.open(pb) as b:
        if a.width != b.width or a.height != b.height:
            return False, -1
        w, h = a.width, a.height
        diff = 0
        for j in range(0, h, TILE):
            for i in range(0, w, TILE):
                win = Window(i, j, min(TILE, w - i), min(TILE, h - j))
                xa = a.read(1, window=win)
                xb = b.read(1, window=win)
                diff += int((xa != xb).sum())
        return diff == 0, diff


def compare_dirs(a: Path, b: Path) -> bool:
    """Compare every config's rasters between two run dirs."""
    ok = True
    for p in sorted(a.iterdir()):
        if not (p.is_dir() and p.name != "data"):
            continue
        for band in ("value.tif", "support_km.tif"):
            pa, pb = p / band, b / p.name / band
            if not (pa.exists() and pb.exists()):
                continue
            same, diff = _compare_tiled(pa, pb)
            if not same:
                ok = False
            tag = "IDENTICAL" if same else f"DIFFER ({diff} px)"
            print(f"  [{tag}] {p.name}/{band}  sha {_sha(pa)} vs {_sha(pb)}")
    return ok


def compare_anchor(run_dir: Path, cfg: str, band: str, committed: str) -> bool:
    """Compare one run output against a committed example."""
    p = run_dir / cfg / band
    if not p.exists():
        print(f"  [MISSING ] {cfg}/{band} (no run output)")
        return False
    same, diff = _compare_tiled(p, REPO / committed)
    tag = "IDENTICAL" if same else f"DIFFER ({diff} px)"
    print(f"  [{tag}] {cfg}/{band} vs committed {committed}")
    return same


def main() -> None:
    """Compare entry point.

    Raises:
        SystemExit: If argv count is wrong.

    """
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    print(f"== dir compare: {a} vs {b}")
    ok = compare_dirs(a, b)
    print("== anchor compare (after dir vs committed examples)")
    for cfg, cv, cs in ANCHORS:
        ok &= compare_anchor(b, cfg, "value.tif", cv)
        ok &= compare_anchor(b, cfg, "support_km.tif", cs)
    print("RESULT:", "PASS (bit-identical)" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
