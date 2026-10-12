"""Cross-OS anchor gate: byte-identity of pipeline outputs vs committed examples.

Re-runs the CI-safe tier2 configs (melb10, melb500 by default) into a temp
outdir and checks the resulting value.tif / support_km.tif against the
committed examples/ files.

On the canonical platform (x86_64 Linux, where the anchors were cut) the
gate is byte-exact: any sha256 difference fails the leg (exit 1) — that is
the point, it catches platform-specific writer drift (GDAL/zstd/libtiff build
deltas, compiler FP differences in the C interp) that array-compare would miss
(tile offsets and compressed bytes move while every pixel stays equal).

On non-canonical platforms (macOS arm64, Windows, aarch64 Linux) the vendored
PROJ/GDAL/zstd wheels emit different (functionally equivalent) tile bytes, so a
byte gate would always fail. There the gate falls back to functional
equivalence: identical grid/metadata (crs, size, dtype, nodata; transform
within last-ulp tolerance — cross-OS PROJ builds drift ~1 ULP) and
near-identical decoded pixels (<=0.5% of cells, each by <=4 DN). A sha match
still passes on any platform (the fast path).

Config selection is by name against bench/bench_tier2.py's CONFIGS table,
so bigger configs (equakes500, equakes_auto) can be added where RAM allows:
they peak at 8.1-8.4 GB on hydrogen (measured 2026-10-12) and stay out of
the CI anchor scope; they remain Linux/hydrogen-anchored via
compare_tier2.py.

Usage (from the repo root, or anywhere — it chdirs itself):
    python bench/check_anchors.py                # melb10 + melb500
    python bench/check_anchors.py melb10         # subset
    python bench/check_anchors.py equakes500     # where the box can hold it

Exit codes: 0 = all outputs match the committed anchors (byte-exact on the
canonical platform, functionally equivalent elsewhere),
1 = mismatch, missing output, unknown config, or a run failure.

No new deps: stdlib + the bench harness + the package itself.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
import tempfile
from pathlib import Path

import bench_tier2 as bt  # sibling module (the script dir is on sys.path)
import numpy as np
import rasterio

import ppgrid._cinterp as ci
from ppgrid._platform import is_canonical_platform

REPO = Path(__file__).resolve().parent.parent

# config -> (committed value tif, committed support tif), relative to REPO.
# Keep in sync with bench/compare_tier2.py's ANCHORS.
ANCHORS: dict[str, tuple[str, str]] = {
    "melb10": ("examples/melb/10m/value.tif", "examples/melb/10m/support_km.tif"),
    "melb500": ("examples/melb/500m/value.tif", "examples/melb/500m/support_km.tif"),
    "equakes500": ("examples/equakes/value.tif", "examples/equakes/support_km.tif"),
}
DEFAULT_CONFIGS = ("melb10", "melb500")

_CHUNK = 1 << 20


def _sha256(path: Path) -> str:
    """Streaming sha256 hex of a file (multi-GB safe)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _transforms_equivalent(t_a: rasterio.transform.Affine, t_b: rasterio.transform.Affine) -> bool:
    """Per-coefficient transform compare with last-ulp tolerance.

    Cross-OS PROJ/warp builds drift the pixel-size coefficients by ~1 ULP
    (e.g. 538.1210527474219 vs 538.121052747422 on macOS arm64, CI run
    38205315707); an exact transform match is unachievable across PROJ
    builds. rel_tol=1e-12 / abs_tol=1e-9 mirror the DN tolerance below.

    """
    return all(math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-9) for x, y in zip(t_a[:6], t_b[:6], strict=True))


def _functional_equivalent(got: Path, want: Path) -> bool:
    """Non-canonical anchor check: same grid/metadata, near-identical pixels.

    Cross-OS tile bytes differ (vendored GDAL/zstd encoder), so compare the
    decoded arrays with a tight tolerance instead of raw bytes: grid, CRS,
    dtype and nodata must match exactly, and the transform must match within
    last-ulp tolerance (cross-OS PROJ/warp builds compute the grid transform
    ~1 ULP apart — exact transform match is unachievable across PROJ builds).
    The DN values may differ only in a vanishing fraction of cells by a
    vanishing amount (last-ulp FP drift from the platform PROJ/warp build).
    A real regression moves many cells or moves them far; encoder/FP drift
    moves almost none by almost nothing.

    """
    with rasterio.open(got) as a, rasterio.open(want) as b:
        if not (a.width == b.width and a.height == b.height and a.count == b.count):
            return False
        if (
            a.dtypes != b.dtypes
            or a.crs != b.crs
            or not _transforms_equivalent(a.transform, b.transform)
            or a.nodata != b.nodata
        ):
            return False
        da = a.read(1)
        db = b.read(1)
    if da.shape != db.shape:
        return False
    n_diff = int((da != db).sum())
    if n_diff == 0:
        return True
    max_diff = int(np.abs(da.astype(np.int32) - db.astype(np.int32)).max())
    return (n_diff / da.size) <= 0.005 and max_diff <= 4


def _configs(argv: list[str]) -> list[tuple[str, str, dict]]:
    """Resolve the requested config names to (name, csv, kw) rows.

    Raises:
        SystemExit: If a name is unknown or has no committed anchor.

    """
    known = {name: (csv, kw) for name, csv, kw in bt.CONFIGS}
    wanted = argv or list(DEFAULT_CONFIGS)
    rows = []
    for name in wanted:
        if name not in known:
            msg = f"unknown config {name!r} (known: {', '.join(sorted(known))})"
            raise SystemExit(msg)
        if name not in ANCHORS:
            msg = f"config {name!r} has no committed anchor in examples/ (add one to ANCHORS)"
            raise SystemExit(msg)
        csv, kw = known[name]
        rows.append((name, csv, dict(kw)))
    return rows


def main() -> int:
    """Run the anchor gate. Returns the exit code."""
    os.chdir(REPO)  # the bench harness resolves "data/..." CSVs against CWD

    rows = _configs(sys.argv[1:])
    canonical = is_canonical_platform()
    gate = "byte-exact (canonical x86_64 Linux)" if canonical else "functional-equivalent (non-canonical platform)"
    print(f"anchor gate: {', '.join(r[0] for r in rows)}  backend: {ci.backend()}  [{gate}]")

    ok = True
    with tempfile.TemporaryDirectory(prefix="ppgrid-anchors-") as tmp:
        outdir = Path(tmp)
        for name, csv, kw in rows:
            try:
                bt.run_config(name, csv, kw, outdir)
            except Exception as exc:  # ruff: ignore[blind-except] - a run failure is a gate failure
                print(f"[FAIL] {name}: run crashed: {exc!r}")
                ok = False
                continue
            for band, anchor_rel in zip(("value.tif", "support_km.tif"), ANCHORS[name], strict=True):
                got = outdir / name / band
                want = REPO / anchor_rel
                if not want.is_file():
                    print(f"[FAIL] {name}/{band}: missing committed anchor {anchor_rel}")
                    ok = False
                    continue
                if not got.is_file():
                    print(f"[FAIL] {name}/{band}: no output file")
                    ok = False
                    continue
                sg, sw = _sha256(got), _sha256(want)
                if sg == sw:
                    print(f"[ok]   {name}/{band:<14} sha256 {sg[:16]}...")
                elif not canonical and _functional_equivalent(got, want):
                    print(f"[ok]   {name}/{band:<14} functional (bytes differ on non-canonical platform)")
                else:
                    print(f"[FAIL] {name}/{band:<14} sha256 {sg[:16]}... != anchor {sw[:16]}...")
                    ok = False

    verdict = "PASS (byte-identical)" if (ok and canonical) else ("PASS" if ok else "FAIL")
    print("anchor gate:", verdict)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
