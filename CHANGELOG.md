# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- Ingest (audit P0-2, 2026-10-09, decision (a)): the serial CSV read now pins `dtype=float64` on the wanted columns, mirroring the chunked parse. Both thread counts take pandas' identical C-parser float path, restoring the S3 bit-identity guarantee (pre-fix, `n_threads=1` vs `4` returned values 1-2 ULP apart on the pattern below). **Baseline change vs 0.4.1:** for columns that are entirely integer literals with |x| >= 2^58, `n_threads=1` results change by up to 2 ULP (1 ULP on most affected rows; e.g. the `9223372036854775807` literal now parses to 0x43e0000000000001 (2^63+2048) instead of the correctly-rounded 2^63). Columns with any non-integer literal, or all-integer with |x| < 2^53, are unchanged. Regression pins in `tests/test_turbo.py` (serial-vs-chunk bit-identity, per-token float-path reference, off-pattern guards).
- CLI contract fixes (#77):
  - "stdout stays clean" now holds for every run: the turbo precheck summary goes through the `ppgrid` logger (stderr, INFO; suppressed by `--quiet` like all other progress lines) instead of stdout, and the sparse-output tile-skip counter in `_reproject_band` is a DEBUG log line (visible with `--verbose`), not a print.
  - Stale-outputs warning (#40) coverage: now fires on argparse validation failures (exit 2), the `--plan` branch, and the out-is-a-file error (preflight wrapped in main), and no longer fires when the rasters were published by the failed run itself (a post-publish failure such as a `--json` write error — the write phase records published names on the Pipeline).
  - `--json` summary write is atomic (fresh 0600 tmp + `os.replace`, like the raster finals) and a write failure (e.g. `run_summary.json` pre-existing as a directory) is a named error — `error: cannot write run summary: ...`, exit 1 — instead of a raw traceback.
  - Unknown EPSG codes exit 2 ("bad flag value") for all three of `--src-crs` / `--work-crs` / `--out-crs` (previously pyproj's CRSError exited 1; rasterio's is a ValueError subclass and already landed on 2 via the generic branch). Messages still name the bad code. `_map_pipeline_errors` catches both CRSError types before the ValueError branch.
  - TIFF staging files are created without ever following a symlink (the issue-#15 comment is now accurate): a new `_staging_fd` (unlink + `O_CREAT | O_EXCL | O_NOFOLLOW`, 0600, one retry on ELOOP/EEXIST) and `_open_tiff_staging` (same, plus a post-open symlink re-check with one retry) cover the serial work-CRS intermediates, the reprojected finals (`_reproject_core`), the turbo refhead rasters, per-thread stock-compression scratch, and the parallel writers' final byte writes. The pre-unlink loops in both write phases are gone (the opens create fresh files).
  - Tests: +7 runs (`tests/test_cli_polish.py` #77 section): stdout-empty pins for a `--turbo` run (summary asserted as an INFO log record) and for the sparse-tile-skip repro (2 far-apart points, `--res 100 --cap-km 20`, skip counter asserted as a DEBUG record), `--plan` failure + argparse failure over a populated out dir (both warn), `--json` dir-conflict (exit 1, named error, no traceback, no stale warning, no `.tmp` left), and unknown-EPSG exit-2 alignment parametrized over all three CRS flags.
- calibrate: user-supplied `calibration.json` schema validation (#79): after `json.load`, the file must be a JSON object whose `transform` / `transform_state.name` are known transform names, and any percentile quantile list (`transform_state.quantiles`, `percentile_quantiles`) must be a list of `PercentileTransform.NQ` (1001) finite numbers. A violation is one friendly exit-2 message naming the file and the bad field (new `validate_calibration`), replacing three raw-traceback modes: `AttributeError` on `None.tolist` in `PercentileTransform.state()`, `StopIteration` in the unknown-name refit branch, and the `np.interp` fp/xp length mismatch that used to die in the write phase after all the grid work.
- calibrate: `choose_transform` guards (#79): empty `values` raises `ValueError("choose_transform: empty values")` and all-candidates-rejected raises a `ValueError` naming the cause (non-finite input with NaN/Inf counts) — no more bare `IndexError` / zero-size `minimum` from the importable module.
- calibrate: blocked-CV bootstrap 0/0 fix (#79): a bootstrap draw where the baseline is exactly constant (`act == mean` in float64, dominant-mode data) is skipped instead of NaN-ing the CI (`RuntimeWarning: invalid value encountered in scalar divide`), and a bin where every draw is skipped keeps its point skill as the CI. The fill-cap prefix loop additionally falls back to the point skill on a non-finite CI instead of `break`-ing on `NaN > min_skill` — a NaN in the first bin no longer discards later healthy bins and silently returns the 25 km default cap.
- ingest: compressed CSV + `n_threads > 1` (#80): `_csv_chunk_tasks` sniffs the gzip/bz2/zip/xz magic and returns `None` (serial fallback, which reads compressed CSV via the path) instead of planning byte-offset chunks that pandas cannot parse from a compressed buffer (was `UnicodeDecodeError` on `.csv.gz`). `MemoryError` joins the planner's fallback `except` tuple so the 2x-file-size planning allocation OOMing a capped slice falls back to the serial parse instead of killing the run (streaming O(1) scan left as follow-up).
- pullpush: `bin_points_banded` empty input (#82): the early return now fills `s_out`/`c_out` with 0, matching the in-place fill contract and the pipeline's pre-fill expectation — a caller that skips the pre-fill can no longer read an uninitialised memmap as data. `bin_points` still raises on empty input.
- Tests: +21 (238 -> 259; `tests/test_calibrate.py`: choose_transform guards, `validate_calibration` accept/reject x8, bootstrap 0/0 stub repro, fill-cap non-finite-CI; `tests/test_pipeline.py`: three malformed `--calibration` files exit 2 naming the file; `tests/test_turbo.py`: `.csv.gz` magic sniff + 250k-row serial fallback + planner MemoryError; `tests/test_tier2_shared.py`: banded empty-input fill + flat-twin raise).
- Write-phase projection-domain validation (#74, P0): both `calculate_default_transform` call sites (serial + turbo write path) now validate the result. A non-finite transform or non-positive raster size — what GDAL returns when the work-CRS grid extent crosses the output projection's domain (Wagner VII pole at y = +/-7,342,230 m -> Web Mercator: NaN transform, width = height = -2147483648) — raises a named error: "work-CRS grid extent (EPSG:6933) exceeds the output projection's domain (EPSG:3857): … Retry with --out-crs 4326, a coarser --res, or a clipped input extent." (ValueError -> exit 2, the user-data-derived convention) instead of dying at `rasterio.open` with "Attempt to create -2147483648x-2147483648 dataset is illegal" (exit 1).
- Coordinate domain validation in ingest (#76):
  - Out-of-domain raw geographic coords (lat outside [-90, 90] / lon outside [-180, 180] when `--src-crs` is geographic) are now dropped with a counted summary naming the first offender row, instead of pyproj silently mapping lat 95 to inf (the int64-min "ix indices out of range" / "cannot convert float NaN to integer" crash in `grid()`) or lon 181 silently ballooning the grid to global span for 3 points.
  - Rows whose projected x/y is non-finite after `tr.transform` (e.g. the APRS pole singularity at lat -90 with `--work-crs 3408`) are likewise dropped with a count + named source coords.
  - The ingest summary (plan text, INFO log, DEBUG volumetrics, `--json`) reports the breakdown: `dropped_nan_inf` / `dropped_out_of_domain` / `dropped_nonfinite_proj` (the two new JSON keys appear only when non-zero). In-domain runs are unchanged: a 0-drop summary keeps the old format and the value band is bit-identical (sha256-pinned against the pre-fix 3793736 anchor).
- Tests: +9 (`tests/test_pipeline.py`): out-of-domain lat/lon counted drops + example naming, lon=181 grid parity vs the in-domain subset, non-finite-projection drop, all-rows-out-of-domain named error, named transform-error e2e (serial + turbo, exit 2, no "Attempt to create"/traceback), `_check_default_transform` unit pins, lat95 full-run CLI (no int64-min error), in-domain value-band sha256 anchor).
- Turbo robustness + compression audit fixes (#78, #81):
  - #78 P1-2: cap-floor + budget-ceiling visibility. `resolve_cap`/`detect_cap_gb` report `floored to 8 GB` when the 8 GB `CAP_FLOOR_GB` lifts a smaller preset/auto cap (before: planned an 8 GB / 6.8 GB budget while printing `preset 6 GB`). The turbo pre-check now receives the process's real memory ceiling `min(physical, cgroup)` and warns `[warn] ... budget X GB exceeds the process memory ceiling Y GB (cap Z GB); OOM risk` on every path (shared, per-box, strict error) — before, a 6 GB slice under the floored 8 GB cap passed the pre-check green and OOM-killed mid-run.
  - #81 F-1: the parallel ZSTD writer and the compression oracle zero-pad partial edge tiles to the declared block before compression (GDAL pads them with zeros before the codec runs). Rasters whose width/height are not multiples of the tile size previously produced frames that decompress to less than the declared tile size. The A.7 calibration probe is now a 500x512 partial tile, not a full 512x512, so the oracle covers the padded path.
  - #81 F-2/F-3: `zstdmt` hardening. `_tif_info` raises `ZstdmtError` on a next-IFD pointer that revisits a parsed IFD (cyclic chain: infinite loop before). `zstdmt.available()` is a true boolean predicate: a vanished libgdal mapping (OSError from dlopen) and an unreadable `/proc/self/maps` report False instead of escaping, and the pipeline oracle gate catches that failure (fail closed to the stock parallel writer).
  - Bug fixes: `_turbo_write_stock_parallel` no longer masks a failed reference-head write as `UnboundLocalError` (the `finally: del zero` ran before `zero` bound) (#78 P1-1). The pre-check's shared-regime selection is regime-only: the tautological `est <= budget*(1+1e-12) + 4096` re-check is deleted with a comment pinning the semantics (#78 P3-1). The pass-3 no-copy `MemoryDataset` warp source (one shared `copy=False` handle) is ported to the two sibling writers `_turbo_write_parallel` and `_reproject_band_array`, which re-copied the whole band per warp call (~3.1 GB each at full-AU scale) (#78 P1-3).
  - Tests: +10 (one per finding, mutation-checked): floor/ceiling source + pre-check warning, regime-only shared gate, edge-tile zero padding (writer + oracle), one MemoryDataset wrap per writer, head-failure not masked, cyclic-IFD raise, available() predicate. Turbo S3 bit-identity (sha256 pins) re-verified.
- CLI polish smalls (#52-#60), stacked on the help/logging PR (#51):
  - #52 `--quiet` / `-q`: suppress stderr below WARNING. Precedence: explicit `--log-level` > `--quiet` > `--verbose` > `LOGGING` > info.
  - #53 `--plan`: resolve the full run plan (input, grid extent/cells/levels, transform, saturation, turbo decision + RAM budget, workers, out dir) and print it, then exit 0 before running — creates no files or directories (new `Pipeline(dry_run=True)` skips the `calibration.json` and `_points.npy` writes; the turbo precheck stays silent, a `--turbo-strict` infeasibility still exits 3).
  - #54 overwrite guard + `--force`: an out dir already containing `value.tif` / `support_km.tif` / `calibration.json` / `run_summary.json` exits 2 listing the offending files (with a `--force` hint), unless `--force` is given. `--plan` skips the guard (it writes nothing).
  - #55 `--json`: machine-readable run summary at `<out>/run_summary.json` — phase timings (ms, incl. sub-phases), volumetrics (rows/points/dropped, cells per level, bytes raw vs compressed + ratio per band), grid metadata, calibration params, resolved CLI params, seed, git commit hash. stdout stays clean.
  - #56 `--help`/`ppgrid help` epilog now documents the exit-code contract: 0 ok, 1 pipeline/IO error, 2 validation error, 3 `--turbo-strict` infeasible.
  - #57 friendly column errors: a `--value-col` / `--lng-col` / `--lat-col` name absent from the input exits 2 listing the actual columns plus `difflib.get_close_matches` suggestions (preflight for both the run and `--plan` paths).
  - #58 `--src-crs` / `--work-crs` / `--out-crs` accept `EPSG:4326`-style strings (prefix stripped, case-insensitive) as well as bare ints; invalid strings are a parser error (exit 2).
  - #59 `--workers auto`: resolves to `os.cpu_count()` (default stays 4; ints unchanged).
  - #60 `--seed N`: the pipeline has stochastic steps (calibration subsampling above `--calib-max-points`, and the blocked-CV fold permutation + bootstrap in `calibrate_fill_cap`), all previously hard-seeded to 0. `--seed` now threads the user's seed through both (default 0 = historical behaviour); the seed is recorded in the run summary and the plan printout.
  - Outputs stay byte-identical for unchanged flag combos (sha256-verified on a 2-point CSV with/without each new flag, incl. `--workers auto` and the EPSG strings).
  - Tests: +27 (`tests/test_cli_polish.py`): quiet suppression/precedence, plan exit-0/no-files (+turbo decision), guard exit-2/force, JSON keys + clean stdout, exit-code epilog, column errors (+closest match), EPSG parsing, workers auto, seed, and per-flag byte-identity pins.

## [0.4.1] - 2026-10-09
- CLI: `ppgrid help` command (exit 0) and an examples epilog on `--help` (RawDescriptionHelpFormatter). Bare `ppgrid` with no input now exits 2 with a concise error + hint to `ppgrid help` (the positional `input` is `nargs="?"`; `--help`/`--version` unchanged).
- Leveled logging: progress goes to stderr via the `ppgrid` logger (root untouched, stdout stays clean). `--verbose` (shorthand for debug) + `--log-level {error,warning,info,debug}` (default info); env `LOGGING` (value `verbose` = debug). Precedence: flag > LOGGING > info. INFO = one line per phase + final summary; DEBUG = per-phase wall-time (ms) breakdown (time.perf_counter in `Pipeline.run`), volumetric detail (rows read, points ingested, NaN/inf dropped, cells/level, output bytes raw vs compressed + ratio, resolved plan).
- Tests: +18 (`tests/test_cli_logging.py`): parser flag tests, level-precedence table, `help`/no-input dispatch via capsys, caplog INFO/DEBUG line assertions, and byte-identical GeoTIFFs with and without `--verbose` (logging is side-channel).

## [0.4.0] - 2026-10-08
- Performance (sparse output, low-RAM scope): `_reproject_band` skips the reproject warp for coarse output tiles whose work-CRS source footprint (conservatively rounded out) intersects no task block, writing a plain NoData buffer instead — RAM-free, and a no-op when no tile is skippable (the auto-cap case). The skip is guarded end-to-end against a no-skip reference with vertically asymmetric tasks (bit-identical; `tests/test_reproject_skip.py`). `_descent_banded` row-bands within a level now run on a thread pool (levels stay sequential): bit-identical output, `--max-band-parallel` flag (default `min(workers, nbands, 8)`), and a per-level `/proc/meminfo` gate that falls back to single-threaded when available RAM is below peak RSS plus two bands of float32 buffers.
- Write-path hardening (production polish round):
  - Atomic finals: `value.tif`/`support_km.tif` publish only via `.tif.tmp` rename (serial + turbo) — a mid-write crash leaves the previous run's finals byte-identical and warns (#40).
  - Stale-output warning on any non-zero exit over pre-existing rasters in the out dir; reviewer repro pinned as a test (#40).
  - All 4 unguarded `w+` memmap sites routed through `_open_fresh_memmap` (new shared `ppgrid/_tempfiles.py`) (#45).
  - `_pull_push_descent` `_band_bufs` off-by-one for odd `band_rows` (+4 -> +5) with bit-exact MT regression test (#41).
  - Dead `downsample_sum(out=)` parameter removed (#43).
  - ZSTD oracle mismatch warning unambiguous ("bytes differ at tile N"), names the stock PARALLEL writer as the fallback.
- CI: PRs on a pinned 3.11-3.14 matrix (ubuntu-latest); main single leg; publish.yml guard — release tag must match pyproject version before upload; `workflow_dispatch` dry_run input -> `uv publish --dry-run`.
- Tests: 176 -> 190 (stale/mid-write/odd-band regressions + low-RAM bit-identity guards incl. the end-to-end asymmetric-tasks repro).

## [0.3.0] - 2026-10-08
- Full-AU (16M points, `--res 100`): 30:45 (quiet) to 50:46 (loaded) with the same pre-PR turbo code, load-dependent -> 2:33 wall time (~12-20x) with sha256-identical outputs (marzuki-hydrogen, `--turbo 64`, regime A).
- Turbo: write path + budget cap + CLI (issue #33):
  - `--turbo` (optional preset `16|32|64|128`; bare = auto `min(physical, cgroup) - 8 GB`), `--max-ram <GB>` (implies turbo), `--ram-gb` alias, `--turbo-strict`.
  - Pre-check after `grid()`: prints the cap/budget/regime actually used; shared infeasible (per-box regime) -> strict: error + exit 3, else `[warn]` + continue on the per-box path with turbo warp/write (exit 0, correct output).
  - `_write_rasters` turbo: skips the work-CRS intermediate files, keeps the quantised int16 DN fields in RAM, and reprojects tile-by-tile from arrays through the same GDAL warp kernel (byte-identical to the file-based path, sha256-tested). Optional parallel per-tile ZSTD via `zstdmt.compress_tiles`, gated by the compression oracle (mismatch -> byte-exact stock-codec parallel write; the serial per-band write is the last resort only if that writer's layout guards fail). In-RAM DN is budget-gated; oversized grids fall back to the file-based reproject path with MT warp.
  - Budget-derived shared cap (S3.3) replaces the hard 2.5e8 cell cap in turbo mode; non-turbo keeps 2.5e8; the 1e8 in-RAM/memmap tier is unchanged.
  - `_prepare_shared` / `box_count_banded` run threaded under the budgeted worker count when turbo (serial at `n_threads == 1`, A8 preserved).
  - Fixed `turbop._read_cgroup_max_gb`: now walks the process cgroup hierarchy to the root and takes the smallest finite `memory.max` (a slice limit above the service cgroup binds; previously only the root cgroup was read, so capped user slices saw the full physical RAM and the auto cap could plan a regime that OOMs the slice).
  - Regenerated stale anchor `examples/melb/10m/value.tif` (byte-identical to current `main` output; the committed file predated a pipeline change).
  - RAM budget `Bt = 0.85 * C - 4 GB` (`C` = effective RAM cap) drives regime selection: A = full in-RAM shared, B = input in RAM + banded val/sup memmap, C = shared all-memmap banded descent (input + val/sup file-backed memmap, 8 GB page-cache floor in its peak accounting), per_box = per-box warp/write fallback taken when regime C has < 2 in-flight units (the only genuine low-RAM mode). `--ram-gb`/`--max-ram` cap this shared-field budget, not process RSS.
- Turbo: speed pass 3 — regime-A in-RAM shared path + byte-exact parallel writer (issue #39):
  - Regime A keeps the shared full fields (val_full/sup_full) and the quantised DN fields (val_dn/sup_dn) on tmpfs memmaps (reclaimable as clean file cache) instead of anon RAM; `_pull_push_descent` gained optional `out_val/out_sup` and writes the final descent level straight into the shared fields (bit-exact in both the threaded and serial branches).
  - Byte-exact parallel tile writer: full-thread-id TID scratch paths (low-16-bit id collisions truncated a live scratch file mid-run), tile frames reassembled into global 512 raster-scan (block-major assembly was wrong), reference head written as one 512^2 tile. sha256-identical to the committed serial anchors.
  - Reproject: the source band is wrapped once in a no-copy `MemoryDataset` — rasterio's ndarray source form copied the whole band into a fresh MEM dataset per warp call (3 GB per 2048^2 block at full-AU; 8 in-flight copies OOM-killed the run).
  - `_quantize` rewritten in-place (same IEEE ops/order, bit-identical) with per-thread f32+f64 scratch; the int16 outputs are owned per call (thread-local scratch raced the `ex.map` prefetch and corrupted ~10% of blocks).
  - Fixed the support blend reusing the value `om` buffer across blends (computed `(1-a)*pv*ps` instead of `(1-a)*ps`). User-visible: `support_km.tif` for multi-level grids may differ vs 0.2.2 (now correct); `value.tif` unchanged. `MADV_NOHUGEPAGE` on large fresh arrays (THP=always with sync defrag made first-touch faults 20-40 us/page).
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
