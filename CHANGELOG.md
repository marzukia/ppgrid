# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- Turbo: write path + budget cap + CLI (issue #33):
  - `--turbo` (optional preset `16|32|64|128`; bare = auto `min(physical, cgroup) - 8 GB`), `--max-ram <GB>` (implies turbo), `--ram-gb` alias, `--turbo-strict`.
  - Pre-check after `grid()`: prints the cap/budget/regime actually used; shared infeasible (per-box regime) -> strict: error + exit 3, else `[warn]` + continue on the per-box path with turbo warp/write (exit 0, correct output).
  - `_write_rasters` turbo: skips the work-CRS intermediate files, keeps the quantised int16 DN fields in RAM, and reprojects tile-by-tile from arrays through the same GDAL warp kernel (byte-identical to the file-based path, sha256-tested). Optional parallel per-tile ZSTD via `zstdmt.compress_tiles`, gated by the compression oracle (mismatch -> serial GDAL write, byte-exact). In-RAM DN is budget-gated; oversized grids fall back to the file-based reproject path with MT warp.
  - Budget-derived shared cap (S3.3) replaces the hard 2.5e8 cell cap in turbo mode; non-turbo keeps 2.5e8; the 1e8 in-RAM/memmap tier is unchanged.
  - `_prepare_shared` / `box_count_banded` run threaded under the budgeted worker count when turbo (serial at `n_threads == 1`, A8 preserved).
  - Fixed `turbop._read_cgroup_max_gb`: now walks the process cgroup hierarchy to the root and takes the smallest finite `memory.max` (a slice limit above the service cgroup binds; previously only the root cgroup was read, so capped user slices saw the full physical RAM and the auto cap could plan a regime that OOMs the slice).
  - Regenerated stale anchor `examples/melb/10m/value.tif` (byte-identical to current `main` output; the committed file predated a pipeline change).

## [0.3.0] - 2026-10-06
- Turbo: pipeline read-side MT (issue #32) behind a new `Pipeline(n_threads=...)` kwarg (default `1`: the serial code path, byte-identical output):
  - `_reproject_band`: the 2048px warp tiles of each row band run on a `ThreadPoolExecutor` (the GDAL warp runs under `nogil`; each tile opens its own source handle, GDAL dataset handles are not thread-safe). The 512px flush loop stays serial raster-scan, so the on-disk bytes never change (sha256-equal to the serial run is test-covered).
  - `ingest`: CSVs with enough rows (>= 100k per chunk) are split into complete-record byte ranges (newline scan with quote-parity guards; any doubt falls back to the serial parse) and parsed in a `ProcessPoolExecutor` (`read_csv` holds the GIL, so threads cannot overlap the parse). Each chunk parses the wanted columns with explicit float64 dtypes and asserts per-chunk dtype equality; chunks concatenate in original row order (`bin_points` is order-sensitive). A row-count mismatch re-runs the serial parse.
  - New `tests/test_turbo.py` (6 tests): MT reproject sha256-equality vs serial, MT ingest bit-identity vs the single whole-file parse on a 500k-row edge-row fixture (2**53 +/- 1, empty/inf rows, int-only column, quoted fields, non-adjacent columns), serial-parse fallbacks (small file, quoted embedded newline), end-to-end full-run sha256 equality `n_threads=1` vs `4`, and `n_threads` validation.
  - Ignores rasterio's benign `NotGeoreferencedWarning` (MemoryDataset initial transform read): CPython warning filters are process-global and not thread-safe, so the library's nested suppression can leak under concurrent warps.
  - Bench harnesses: `bench/turbo_bench.py` (full paths), `bench/turbo_decomp.py` (parallelisable part in isolation).

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
