# Changelog

All notable changes to this project will be documented in this file.

## [0.2.2] - 2026-09-25
- Code hygiene (no behavior change; output rasters remain byte-identical) — issues #4/#5:
  - Renamed `ppgrid/idwgrid.py` -> `ppgrid/pipeline.py`; `ppgrid/idwgrid` is now a thin backwards-compat shim, `ppgrid.Pipeline` is exported from the package root, and the `ppgrid` console script points at `ppgrid.pipeline:main`.
  - Extracted the nested `_reproject_band` closure to a module-level function.
  - Magic numbers in `pipeline.py`, `calibrate.py`, `pullpush.py` promoted to named constants (CRS defaults, int16 support-band encode/decode pair, tile size, memmap band indices, CLI defaults, float32 exact-integer limit, count epsilon, unresolved-support sentinel).
  - Worker context: `_CTX` now holds the `_WorkerConfig` dataclass object (no per-field dict copy); adding a field is one edit.
  - Deduplicated the 3x3-block neighbourhood / window / snapped-halo-box logic into shared helpers used by the per-block, shared-field, and test paths.
  - Dropped the unused 4th point-memmap band.
  - CLI: `--out` now defaults to `out/` (was `examples/`, which wrote build artifacts into the docs directory); argparse validation errors tested per flag.
  - `ruff`: removed the `global-statement` ignore (no global statements left); COM812 remains ignored (pre-existing, conflicts with the formatter).
- Performance (issue #10): vectorized the output-band reproject warp; output byte-identical.
- Performance/correctness: shared full-field pyramid descent + fast box-count, calibrate hoisted out of the block loop; fixed `_CTX` not being cleared between runs; capped the shared path at 2.5e8 cells; stopped memmap descent at level `min(2, levels)` for levels=1 grids.
- Release/docs pass (issues #17/#19/#20/#22): slimmed the sdist (generated example rasters/images and the 8.3 MB `data/all_equakes.csv` no longer ship; ~87 MB -> <1 MB, `data/melb_houses.csv` still ships), added the ruff check + format gate to CI, documented the Python 3.11+ requirement in the README install section, fixed stale docs (test count, module names, closed PR ref), removed stray `examples/melb/value.tif.aux.xml`, untracked `.vscode/settings.json`, added `out/` to `.gitignore`.

## [0.2.1] - 2026-09-22
- Added `--percentile-step` CLI flag: rounds output percentiles to the nearest step (e.g. `5` -> 90/95/100), clamped to 0-100. Recorded as a `percentile_step` tag on the value GeoTIFF.
- Fixed `percentile_step` guard: `if pct_step:` -> `if pct_step is not None:` so a falsy step is handled correctly and the validation is consistent with the docstring.
- README: added hero image (multi-resolution output), replaced the single wall-time-vs-resolution benchmark with a head-to-head vs `gdal_grid`/`gdal_rasterize` (100K points) and a wall-time-vs-point-count scaling chart (generated with `charted`), plus side-by-side comparison images. Fixed the source-clone URL (`pullpush` -> `ppgrid`).

## [0.2.0] - 2026-08-08
- Added `__version__` to package.
- Moved `pyarrow` to optional `[parquet]` dependency.
- Added `CHANGELOG.md`.
- Added `SECURITY.md` and `SUPPORT.md`.
- Added `CODEOWNERS` and pre-commit config.
- Added `examples/run_melb.py` reproducible example script.
- Added CI and PyPI badges to README.
- Stated benchmark hardware in README.
- Fixed README project structure and CLI table.
- Added `python` keyword and Python 3.14 classifier.
- Fixed emdash in pyproject.toml description.
- Regenerated `uv.lock`.

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
