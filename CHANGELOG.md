# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.1.4] - 2026-08-08
- Fixed PyPI project URLs to point to the renamed GitHub repository (`ppgrid`).
- Added homepage link between PyPI and GitHub.

## [0.1.3] - 2026-08-08
- Fixed error message tuple (trailing comma in `pullpush.py`).
- Removed dead `_init_worker` function.
- Added index validation to `bin_points`.
- Added 4 acceptance tests (blocked=whole, coverage, georef, validity).
- Fixed README grammar, install section, structure.
- Removed `.DS_Store` files from git.
- Fixed all ruff warnings and errors.

## [0.1.2] - 2026-08-08
- Re-published after making repository public for README images.

## [0.1.1] - 2026-08-08
- Fixed README image paths for PyPI rendering.

## [0.1.0] - 2026-08-08
- Initial release of ppgrid.
- Fast continent-scale raster interpolation for scattered point data.
- IDW interpolation with pull-push mipmap pipeline.
- Windowed reprojection (no OOM on large datasets).
- CLI with validation and --version flag.
- Calibrate module for transform selection + cross-validation.
- 17 tests across 3 test files.
- Melbourne housing example (13.5k points at 6 resolutions).
