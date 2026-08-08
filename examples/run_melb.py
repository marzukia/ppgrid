#!/usr/bin/env python3
"""Reproduce the Melbourne housing example outputs.

Usage:
    python examples/run_melb.py
    python examples/run_melb.py --res 500  # single resolution
"""

import argparse
from pathlib import Path

from ppgrid.idwgrid import Pipeline


def run(resolution: float, out_dir: Path) -> None:
    """Run the pipeline at a given resolution."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data_csv = Path(__file__).resolve().parent.parent / "data" / "melb_houses.csv"

    p = Pipeline(
        str(data_csv),
        "price",
        "longitude",
        "latitude",
        str(out_dir),
        res=resolution,
        cap_km=10.0,
        skip_calibration=True,
    )
    vpath, spath = p.run()
    print(f"  value.tif:      {vpath}")
    print(f"  support_km.tif: {spath}")


def main() -> None:
    """Parse arguments and run the pipeline at requested resolutions."""
    parser = argparse.ArgumentParser(description="Reproduce Melbourne housing example.")
    parser.add_argument(
        "--res",
        type=float,
        nargs="*",
        default=[10.0, 25.0, 50.0, 100.0, 250.0, 500.0],
        help="Resolutions in metres (default: all 6)",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent / "melb"
    for res in args.res:
        label = f"{int(res)}m"
        print(f"Running {label}...")
        run(res, base / label)

    print("Done. Outputs in examples/melb/")


if __name__ == "__main__":
    main()
