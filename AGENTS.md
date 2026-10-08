# AGENTS.md

Working notes + rules for AI agents (and humans) touching this repo. Read this before editing.

## What this is

`ppgrid` — pull-push mipmap scattered-data interpolation. Turns `N` geolocated points
into a pair of int16 GeoTIFFs (value percentile + support-km) at continent scale in
minutes, no GPU. It is an **IDW approximation**: fast, visually clean, and
deliberately *not* a spatially exact interpolator (the output is for visualisation,
not downstream calculation).

- Python 3.11+, `uv` managed. Entry point `ppgrid` (`ppgrid.pipeline:main`; `ppgrid.idwgrid` is a back-compat shim).
- Deps: numpy, pandas, pyproj, rasterio. Optional `[parquet]` (pyarrow).
- Tests: `uv run pytest` (189 tests). Lint/format: `ruff` (line-length 120).

## Repo layout

- `ppgrid/pipeline.py` — the Pipeline + CLI. All the logic lives here. `ppgrid/idwgrid.py` is a thin back-compat shim (module alias to `ppgrid.pipeline`).
- `ppgrid/calibrate.py` — transform selection + spatially-blocked CV fill-cap.
- `data/` — example inputs: `melb_houses.csv` (13,580 pts), `all_equakes.csv` (44,376 pts).
- `examples/` — committed outputs + repro scripts + README images (see "Images" below).
- `tests/` — pytest. `examples/run_melb.py` regenerates the Melbourne example rasters.

## Versioning / release

- `pyproject.toml` `version` is the source of truth; `__version__` reads it via
  `importlib.metadata`. Keep `CHANGELOG.md` in step with every version bump.
- Releasing: bump `version` + `uv.lock`, update CHANGELOG, push a tag, the
  `publish.yml` workflow uploads to PyPI. The README PyPI badge shows the *published*
  version, so it lags the repo until a tag is cut.

## Rules — do not fuck these up

### Images in the README (this bit bites)
- The README is the PyPI `long_description`. **Every image MUST be an absolute
  `https://raw.githubusercontent.com/marzukia/ppgrid/main/examples/...` URL.**
  Relative `examples/...` paths render on GitHub but 404 on PyPI's static host.
- README images are committed under `examples/`. They resolve from `main`, so a
  new image is invisible on the repo's README **until the branch merges to main**.
- The benchmark charts (`bench_headtohead.png`, `bench_scaling.png`) are generated
  with **`charted`** (github.com/marzukia/charted — zero-dep Python SVG/PNG). Regenerate,
  don't hand-edit:
  ```bash
  uvx --from "charted[png]" python  # build BarChart / LineChart, .to_png_bytes()
  ```
  Data for those charts lives in the blog post
  (`mrzk.io` ppgrid post, `fig-bar-benchmark.json` / `fig-line-scaling.json`).

### The `--percentile-step` guard
- `pipeline.py` uses `if pct_step is not None:` (NOT `if pct_step:`). A step value of
  `0` is invalid (validated to `(0, 100]`) but the `is not None` form is the correct
  guard and matches the docstring. Do not "simplify" it back to a truthiness check.

### CRS defaults
- Working CRS `EPSG:6933` (Wagner VII, equal-area, metres are true) for interpolation.
- Output CRS `EPSG:3857` (Web Mercator) for the final GeoTIFF.
- Input `EPSG:4326`. All three are CLI-overridable (`--work-crs`, `--out-crs`,
  `--src-crs`). Don't hardcode a different default.

### Output encoding is opinionated — leave it
- Value band = **percentile, not raw value**: `percentile = DN / scale`, default
  scale 100 so DN 5000 = 50th percentile. int16 fits 0-10000. Raw values are restored
  via the `calibration.json` LUT. This is deliberate (clean vectorisation/MVT, rounds
  to whole numbers). If you "fix" it to store raw values you break the int16 design
  and the decoding example in the README.

### Data provenance
- `melb_houses.csv`: Kaggle Melbourne Housing Snapshot, **CC BY-NC-SA 4.0**.
- `all_equakes.csv`: USGS FDSN, **Public Domain**.
- Keep the Data Lineage table in the README accurate if the equake download window
  or filters change.

## Git / pushing

- The repo lives under the `marzukia` GitHub org. The `monkytheluffy` bot identity is
  NOT a collaborator (push 403s). Push with the `marzukia` PAT:
  ```bash
  PAT=$(cat ~/.config/marzukia-pat)
  git push "https://x-access-token:${PAT}@github.com/marzukia/ppgrid" <branch>
  ```
- Branches go to a PR, not main, per the fleet merge gate.

## What changed recently (2026-10-08, sparse-output branch reconciled onto main post-#50)

- PR #27 (`sparse-output-lowram`) merged `origin/main` (0.3.0, turbo) and reconciled: **main's turbo behaviour is the baseline; #27's low-RAM optimisations layer on top.**
  - `_reproject_band` gained `task_info` (work-grid block geometry + populated-block set from `grid()`): a coarse 2048px dst tile whose source footprint intersects no task block is written as a plain NoData buffer without a GDAL warp (conservative round-out, so never a false empty). File-source serial reproject only; the turbo array reproject (`_reproject_band_array`) is untouched.
  - `_descent_banded` (main's structure: `_thread_bands` + `_descent_band_body` per-thread buffers) gained the #27 per-level RAM gate: pool clamped to `min(n_threads, nbands, 8)`, and single-thread when `/proc/meminfo` MemAvailable < peak RSS + two bands of f32 buffers. `nworkers` is an alias of `n_threads` (both test suites call it).
  - Non-turbo (low-RAM) shared descent is now parallel by design: `--max-band-parallel` caps the pool, default `workers` (main's baseline for this path was serial `n_threads = 1`). Turbo regime A still never calls `_descent_banded` (full in-RAM descent) — unchanged.
  - Bit-identity holds at every thread count (S3 guarantee; `tests/test_pullpush_mt.py` + the #27 bit-identical/pool/RAM-gate tests all green). 180 (main) + 9 (#27) = 189 tests.

## What changed recently (2026-10-06→08, turbo speed passes 2/3 + 0.3.0 release)

- **Speed passes 2/3 (0a6fe90 + 941f4e5, both in the #39 merge history): full-AU 50:46 → 2:33, byte-identical** (regime A, `--turbo 64`, 43 GB; per-box 8.5 GB: 3:20; all outputs sha256-identical to the serial anchors). Pass 2 (0a6fe90) is the dnfill/write-path base; pass 3 (941f4e5) hit the wall target.
  - Regime A keeps the shared full fields + quantised DN fields on tmpfs memmaps (reclaimable clean file cache) instead of anon RAM; `_pull_push_descent(out_val=/out_sup=)` writes the final descent level straight into the shared fields (bit-exact, threaded and serial branches).
  - Parallel stock writer (`_turbo_write_stock_parallel`): fixed three silent serial-fallback bugs — TID scratch collision (`get_ident() & 0xFFFF`), block-major vs global 512 raster-scan tile order (608 vs 616 tiles), 3 GB zero-placeholder reference head (now one 512² tile).
  - Reproject: one no-copy `MemoryDataset(copy=False)` + MultiBand tuple form, shared read-only across warp threads (rasterio's ndarray form copied 3.1 GB per warp call per thread; 8 in-flight copies OOM-killed the run).
  - `_quantize` in-place (same IEEE ops/order, bit-identical), per-thread f32/f64 scratch, owned int16 outputs (the shared thread-local scratch raced the `ex.map` prefetch and corrupted ~10% of blocks).
  - Support-blend fix: `om` buffer reused across blends — computed `(1-a)*pv*ps` instead of `(1-a)*ps`. Correctness fix on the default multi-level path: `support_km.tif` for multi-level grids may differ vs 0.2.2 (now correct), `value.tif` unaffected.
  - `MADV_NOHUGEPAGE` on large fresh arrays (THP=always + sync defrag made first-touch faults 20-40 us/page).
  - Known state: on this GDAL stack the zstd oracle mismatches libtiff streaming, so turbo runs take the byte-exact stock-parallel write fallback; the parallel ZSTD path is unit-tested and activates where the stack matches.
- **Budget semantics (measured):** `--ram-gb` caps the shared-field budget, NOT process RSS — regime-C runs peaked at 39.4 GB under a 20 GB budget (write/descent buffers sit outside Bt sizing). per_box is the only genuine low-RAM mode.
- **0.3.0 released 2026-10-08** (tag v0.3.0 @ f9e0380): CHANGELOG consolidated into one [0.3.0] entry; release notes live on the GitHub release; `publish.yml` publishes to PyPI on `release: published`. Issues #29-#33 closed as landed in #39.
- Review findings from the #39 adversarial review are filed as #40-#45 (all minor, post-release): stale-output warning gap (#40), odd `band_rows` off-by-one (#41), `_turbo_zstd_ok` docstring (#42), dead `downsample_sum(out=)` param (#43), WIP-labelled 0a6fe90 in history (#44), DN temp-file hygiene vs `_open_fresh_memmap` guard (#45).

## What changed recently (2026-10-07, PR #39 review fixes: M-1..M-3, Y-1)

- M-1: equakes anchors regenerated under the rasterio 1.5.1 lock (the committed
  files were cut on 1.5.0; the vendored GDAL emits different BigTIFF ZSTD tile
  bytes — arrays equal, tile offsets moved). New shas in the A.6 table:
  `examples/equakes/value.tif` `99c8a01c…`, `support_km.tif` `5e545742…`.
  `bench/compare_tier2.py` re-run: bit-identical (melb anchors unchanged).
- M-2: `turbop.resolve_cap` clamps a preset against `min(physical, cgroup)`
  (a capped slice OOMs at its own limit); the pipeline wires both reads via
  `Pipeline._resolve_turbo_cap` (before: preset 32 on a 25.8 GB slice planned
  32 GB; now clamps to 17.8 GB).
- M-3: `turbop.per_box_peak_bytes` (64 B/cell bottom-up + 2 GB fixed, x
  in-flight blocks + points/DN extra) + `precheck(per_box_peak_bytes=…)`;
  the pipeline computes it from the worst snapped box after `grid()`.
  Non-strict per-box over budget now warns "best-effort, OOM risk, output may
  be partial" instead of promising exit-0 correct output; strict names the
  per-box infeasibility. Verified: melb @1m --max-ram 8 warns 189.7 GB est >
  6.8 GB budget, then dies 137.
- Y-1: BigTIFF slot widths in the parallel ZSTD write. `_tif_parse_ifd` sizes
  tags per TIFF type (type 16 LONG8 added; unknown type = hard error) and
  returns `sz324/sz325/fmt324/fmt325`; `_turbo_write_parallel` patches 324
  (8-byte slots) and 325 (4-byte slots) at their real strides, shared
  `_TIFF_TYPE_SIZES` with the overlap guard. Pre-fix, one 8-byte stride
  corrupted both arrays on every real BigTIFF output (7888 tiles: 325 sizes
  read `2000, 0, 2001, 0, …`). Pinned: BigTIFF parallel byte-identity test +
  IFD parse value asserts.
- Tests: 170 -> 176 (resolve_cap cgroup clamp, per-box peak model + gate,
  pipeline wiring, BigTIFF parallel, IFD value pins).

## What changed recently (2026-10-07, turbo write path + CLI, issue #33)

- CLI: `--turbo` (optional preset `16|32|64|128`; bare = auto cap), `--max-ram <GB>` (implies turbo), `--ram-gb` (alias of `--max-ram`), `--turbo-strict`. Pre-check after `grid()` prints cap/budget/regime used; shared infeasible (per-box regime) -> strict: stderr error + exit 3, else `[warn]` + continue on per-box with turbo warp/write (exit 0, correct output).
- `_write_rasters` turbo path: no work-CRS intermediate files. Quantised int16 DN fields stay in RAM (budget-gated: `nx*ny*4 > budget//2` -> file-based reproject path with MT warp) and are reprojected tile-by-tile through the same GDAL warp kernel as the serial path (`_reproject_band_array`; same 2048px tiles, same 512px serial flush, same metadata) -> byte-identical output, sha256-tested.
- Parallel per-tile ZSTD (`_turbo_write_parallel`): GDAL zero-filled reference head serialises the exact IFD/tag layout; the pipeline patches TileOffsets/TileByteCounts + oracle-verified zstd frames. Oracle mismatch (e.g. CPL zstd pfn vs libtiff streaming params differ on a stack) -> byte-exact stock-codec parallel write (the serial per-band write is the last resort only if that writer's layout guards fail).
- Budget-derived shared cap (S3.3): turbo mode derives the cells cap from `Bt = 0.85*C - 4 GB`; non-turbo keeps `_SHARED_MAX_CELLS = 2.5e8`; the `1e8` in-RAM/memmap tier is unchanged.
- `_prepare_shared` / `box_count_banded` run threaded under the budgeted worker count when turbo (`plan.workers`); `n_threads == 1` keeps the exact serial path (A8).
- `turbop._read_cgroup_max_gb` fixed: walks the process cgroup hierarchy to the root and takes the smallest finite `memory.max` (a slice limit above the service cgroup binds). Previously only the root cgroup was read, so a memory-capped user slice saw full physical RAM and auto `--turbo` could plan a regime that OOMs the slice.
- `examples/melb/10m/value.tif` regenerated: the committed anchor predated a pipeline change; current `main` and this branch produce the new bytes (verified against pre-turbo HEAD).
- Tests: `tests/test_turbo.py` now 22 (A9 regime pins, strict exit 3, non-strict continue, CLI flags/wiring, array-vs-file reproject sha, parallel-zstd byte-identity, 4GiB guard, IFD classic+BigTIFF parse, e2e in-RAM identity, budget-gate fallback, out==work CRS, `box_count_banded` MT, `test_cgroup_reader` walk-up).

## What changed recently (2026-10-06, turbo read-side MT, 0.3.0)

- New `Pipeline(n_threads=...)` kwarg (default `1`, serial path unchanged, byte-identical output). NOT a CLI flag.
  - `_reproject_band`: 2048px warp tiles per row band run on a `ThreadPoolExecutor` via the module-level `_warp_dst_tile` helper (per-tile source open: GDAL handles are not thread-safe). The 512px flush loop STAYS serial raster-scan (file bytes depend on tile order). Do not "parallelise" the flush.
  - `ingest`: CSVs >= 100k rows per chunk split into complete-record byte ranges (`_csv_chunk_tasks`, newline scan + quote-parity guards, `None` on any doubt -> serial parse) and parsed in a `ProcessPoolExecutor` (`_read_csv_chunk`, explicit float64 dtypes + per-chunk dtype assert). Chunks concat in original row order (`bin_points` is order-sensitive); row-count mismatch re-runs the serial parse.
  - rasterio's benign `NotGeoreferencedWarning` is ignored at import (CPython warning filters are process-global, not thread-safe; the library's nested suppression leaks under concurrent warps). A matching `filterwarnings` entry in `[tool.pytest.ini_options]` restates it for pytest.
  - Tests: `tests/test_turbo.py` (6, sha256/bit-identity + fallbacks). Bench: `bench/turbo_bench.py`, `bench/turbo_decomp.py`.
  - Design doc in-repo: `docs/turbo-design.md` (accepted criteria A4-A8).

## What changed recently (2026-09-22, the "polish" pass)

- Bumped to **0.2.1**.
- `--percentile-step` flag (round output percentiles to a step, e.g. 5 → 90/95/100)
  + the `is not None` guard fix.
- README: added hero image (`examples/melb_4panel.jpg`), replaced the old single
  wall-time-vs-resolution chart with two `charted` benchmark charts (head-to-head vs
  gdal at 100K pts + wall-time-vs-point-count scaling), added side-by-side/zoom
  comparison images, fixed the clone URL (`pullpush` → `ppgrid`). Removed stale
  `bench_combined/time/size.png`.
- All README image refs converted to absolute raw.githubusercontent URLs for PyPI.
