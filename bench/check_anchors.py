"""Cross-OS anchor gate: byte-identity of pipeline outputs vs committed examples.

Re-runs the CI-safe tier2 configs (melb10, melb500 by default) into a temp
outdir and requires the resulting value.tif / support_km.tif to be
sha256-identical to the committed examples/ files. Any byte difference on
any OS or Python version fails the leg (exit 1) — that is the point: it
catches platform-specific writer drift (GDAL/zstd/libtiff build deltas,
compiler FP differences in the C interp) that array-compare would miss
(tile offsets and compressed bytes move while every pixel stays equal).

Config selection is by name against bench/bench_tier2.py's CONFIGS table,
so bigger configs (equakes500, equakes_auto) can be added where RAM allows:
they peak at 8.1-8.4 GB on hydrogen (measured 2026-10-12) and stay out of
the CI anchor scope; they remain Linux/hydrogen-anchored via
compare_tier2.py.

Usage (from the repo root, or anywhere — it chdirs itself):
    python bench/check_anchors.py                # melb10 + melb500
    python bench/check_anchors.py melb10         # subset
    python bench/check_anchors.py equakes500     # where the box can hold it

Exit codes: 0 = all outputs byte-identical to the committed anchors,
1 = mismatch, missing output, unknown config, or a run failure.

No new deps: stdlib + the bench harness + the package itself.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path

import bench_tier2 as bt  # sibling module (the script dir is on sys.path)

import ppgrid._cinterp as ci

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
    print(f"anchor gate: {', '.join(r[0] for r in rows)}  backend: {ci.backend()}")

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
                else:
                    print(f"[FAIL] {name}/{band:<14} sha256 {sg[:16]}... != anchor {sw[:16]}...")
                    ok = False

    print("anchor gate:", "PASS (byte-identical)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
