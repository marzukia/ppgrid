"""Anchor gate for installed-artifact e2e outputs (issue #98).

bench/check_anchors.py re-runs the pipeline itself; this gate runs nothing.
It compares an existing output directory -- produced by the installed
console script in a clean venv (scripts/check_artifact.sh) -- against the
committed melb500 anchors with the same cross-OS rules as check_anchors:

- canonical x86_64 Linux (where the anchors were cut): sha256 byte-exact;
- any other OS/arch: functional equivalence (grid/metadata + decoded
  pixels), per check_anchors._functional_equivalent.

Usage (from the repo checkout):
    python bench/check_artifact.py --out <e2e-output-dir>

Exit codes: 0 = outputs match the committed anchors,
1 = mismatch, missing output file, or a non-canonical functional failure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from check_anchors import ANCHORS, _functional_equivalent, _sha256

from ppgrid._platform import is_canonical_platform

REPO = Path(__file__).resolve().parent.parent
CONFIG = "melb500"  # must match the e2e command line in scripts/check_artifact.sh


def main() -> int:
    """Compare --out against the committed melb500 anchors. Returns the exit code."""
    parser = argparse.ArgumentParser(description="installed-artifact anchor gate")
    parser.add_argument("--out", required=True, help="directory holding value.tif + support_km.tif")
    args = parser.parse_args()

    out = Path(args.out)
    canonical = is_canonical_platform()
    gate = "byte-exact (canonical x86_64 Linux)" if canonical else "functional-equivalent (non-canonical platform)"
    print(f"artifact anchor gate: melb500  [{gate}]")

    ok = True
    for band, anchor_rel in zip(("value.tif", "support_km.tif"), ANCHORS[CONFIG], strict=True):
        got = out / band
        want = REPO / anchor_rel
        if not got.is_file():
            print(f"[FAIL] {band:<14} missing in {out}")
            ok = False
            continue
        sg, sw = _sha256(got), _sha256(want)
        if sg == sw:
            print(f"[ok]   {band:<14} sha256 {sg[:16]}...")
        elif not canonical and _functional_equivalent(got, want):
            print(f"[ok]   {band:<14} functional (bytes differ on non-canonical platform)")
        else:
            print(f"[FAIL] {band:<14} sha256 {sg[:16]}... != anchor {sw[:16]}...")
            ok = False

    verdict = "PASS (byte-identical)" if (ok and canonical) else ("PASS" if ok else "FAIL")
    print("artifact anchor gate:", verdict)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
