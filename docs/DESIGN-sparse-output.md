# Design: sparse tiled output for large-area rasters

Status: LANDED — the recommended Option C items shipped in 0.3.0–0.4.x (reproject
NoData-tile skip, `tests/test_reproject_skip.py`; non-turbo shared descent parallel by
design with `--max-band-parallel`; per-level RAM gate). Date: 2026-09-29. Author: monky worker.

Dataset: `data/au_gcc_sparse.csv` (949,809 WGS84 points, Australia/GCC).
Resolution target: ~100 m (>= 1 B cells) with `out_crs != work_crs`.
Benchmark: `bench/sparse_bench.py` (see §11). Raw results: `/home/monky/tmp/sparse_bench/` (auto cap) and `/home/monky/tmp/sparse_bench_cap2/` (cap 2 km).

## 1. TL;DR

- The issue's two cost premises are only half right. **"Compute is already sparse":
  refuted** for this dataset — 0 of 399 blocks at 100 m are empty; every block runs a
  full pull-push descent. **"99.95 % NoData": refuted at the calibrated cap** — the
  auto cap is 25 km and the measured NoData fraction is **0.70 %** at every
  resolution. 99.94 % NoData is only true in the cap→0 limit (cells that hold a point
  are 949,809 / 1,565,900,000 = 0.06 %).
- Measured 100 m breakdown (479.6 s total, `out_crs=3857`):
  **per-block compute 239.7 s (50 %) > reproject 127.2 s (26.5 %) > work-CRS GeoTIFF
  write 89.3 s (18.6 %) > calibrate 15.6 s (3.3 %)**.
- Option A (crop to populated envelope) is **structurally a no-op**: the output grid
  is already the tight point bbox (`pipeline.py:639-641`), and the populated-block
  envelope equals the full grid at all three measured resolutions (`env_task_frac = 1.0`).
- Recommendation: **Option C**, implemented as (ii) reproject NoData-tile skip +
  (iii) 2048 px output tiles + (iv) direct block→dst reproject (drop the work-CRS
  intermediate file). Expected 100 m: **479.6 s → ~350 s (−27 %)**, peak RSS 8.8 →
  ~15.5 GB, no contract change, final GeoTIFFs cell-identical on all data cells.
  Option B (per-tile files) is deferred: it changes the output contract and buys
  little wall time on this dataset.

## 2. Problem statement

`Pipeline.grid()` sizes the output grid from the full input point bbox
(`pipeline.py:632-641`):

```
self.x0 = self.x.min()
self.nx = int((self.x.max() - self.x0) // self.res) + 1   # 41005 at 100 m
self.ny = int((self.y.max() - self.y0) // self.res) + 1   # 38189 at 100 m
```

At 100 m this is 1.566 B cells. `_write_rasters()` then:
1. allocates the full work-CRS (EPSG:6933) raster (both bands),
2. writes every block (or the whole shared field) into it — 6.26 GB raw int16,
3. renames it to `_value_tmp.tif` and calls `_reproject_band()` **per band over the
   full destination extent** (EPSG:3857, 41264×37909 = 1.564 B cells) — another
   6.26 GB written to the final GeoTIFFs.

The issue asks whether the output side can stop paying full-bbox costs when the
raster is sparse, via: (A) crop to the populated envelope, (B) per-tile sparse
output, (C) hybrid single-TIFF (envelope + NoData-tile skip in reproject + 2048 px
blocks).

## 3. Method

- Harness: `bench/sparse_bench.py`. Runs the real `Pipeline` per resolution with
  `out_crs=3857`, `work_crs=6933`, `workers=4`, `block=2048`. Instrumentation is
  done by monkey-patching `Pipeline._process_block`, `Pipeline._reproject_band`,
  `Pipeline._prepare_shared`, `ThreadPoolExecutor.map`, `rasterio.io.DatasetWriter.write`
  and `rasterio.open` (write mode) — **no pipeline code modified**.
- Machine: marzuki-hydrogen, 12 cores, 62 GB RAM. Contended: loadavg 12–23 across
  the runs (vLLM + Palworld + monitoring). `loadavg_start/end` recorded per run in
  the JSON. Absolute wall times scale with load; the *breakdown* is stable
  (repeated 500 m runs agreed within 20 %).
- Calibration variance note: one early 500 m run took 117.5 s for `calibrate()`
  under load 14→17; later runs took 15.6–19.0 s. Calibrate cost is resolution-
  independent (blocked CV on a 1 km grid) and load-sensitive.
- Runs completed within the 15-min budget at all resolutions (100 m auto: 480 s),
  so **no extrapolation was needed** for the baseline.

## 4. Data profile (`au_gcc_sparse.csv`)

| fact | value |
|---|---|
| rows | 949,809 (`value,lng,lat`) |
| values | uniform 1000–10000 (integer) |
| extent (EPSG:6933) | 4100.5 × 3818.9 km |
| point mass | top 50 of 1,603 100-km cells hold ~87 % of points |
| background density | median 32 pts / 100-km cell (a continental background, not islands) |
| auto calibration | transform `log10`, cap **25.0 km** (fallback `FILL_CAP_DEFAULT_KM` — uniform values give CV skill ≈ 0) |

Block occupancy (2048 px blocks):

| res | grid (6933) | blocks | blocks with ≥1 point | tasks (3×3 rule) | empty blocks |
|---|---|---|---|---|---|
| 500 | 8201×7638 (62.6 M) | 5×4=20 | 20 | 20 | **0** |
| 200 | 20503×19095 (391.5 M) | 11×10=110 | ~110 | 110 | **0** |
| 100 | 41005×38189 (1565.9 M) | 21×19=399 | 382 | 399 | **0** |

The dataset is *point-sparse* (0.06 % of cells hold a point) but **not
output-sparse** at the calibrated cap: points are close enough that a 25 km fill
reaches ~99.3 % of cells. There are no empty regions at block scale anywhere in
the grid.

## 5. Baseline measurements (BEFORE)

Auto cap (25 km), `out_crs=3857`, 4 workers, block 2048:

| res | cells | calib | shared descent | block wall | block CPU | work write | reproj (v+s) | close | **total** | peak RSS | final size (v+s) | NoData |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 500 | 62.6 M | 0.008* | **27.1 s** | 0.9 s | 3.4 s | 0.7 s | 3.7+2.8 = 6.6 s | 4.6 s | **40.5 s** | 3.4 GB | 53.4+13.2 MB | 0.70 % |
| 200 | 391.5 M | 19.0 s | — | 33.1 s | 130.6 s | 19.1 s | 17.9+14.4 = 32.3 s | 8.3 s | **112.3 s** | 3.4 GB | 214.5+59.6 MB | 0.70 % |
| 100 | 1565.9 M | 15.6 s | — | **239.7 s** | 957.2 s | **89.3 s** | 73.8+53.4 = **127.2 s** | 6.6 s | **479.6 s** | 8.8 GB | 599.4+162.3 MB | 0.70 % |

\* 500 m reused `calibration.json`; a fresh run is 15–117 s (load-dependent).
Shared path = full-field descent (`_SHARED_MAX_CELLS = 2.5e8`, `pipeline.py:100`):
active at 500 m (62.6 M cells), disabled at 200/100 m (391 M / 1566 M cells).

Scaling: cells ×4 → total ×2.8–2.9 (sub-linear; compute is ~O(cells), output
I/O ~O(cells) with shared contention).

100 m internal detail (the decisive case):

| phase | time | share | notes |
|---|---|---|---|
| per-block pull-push (`_process_block` × 399) | 239.7 s wall / 957 s CPU | 50.0 % | 4 workers, 2.4 s CPU/block, p50 2.5 s, p95 4.1 s, max 6.0 s |
| reproject both bands (`_reproject_band`) | 127.2 s | 26.5 % | = ~50 s GDAL warp (798 × 2048² coarse tiles) + **76.8 s flush writes** (12,150 × 512² `dst.write` calls @ 6.3 ms each) |
| work-CRS GeoTIFF writes (both bands) | 89.3 s | 18.6 % | 798 full-array `dst.write`, 6.26 GB raw int16 → 798 MB ZSTD on disk |
| calibrate | 15.6 s | 3.3 % | blocked CV, resolution-independent |
| open/close writers, grid, ingest, misc | ~8 s | 1.7 % | |

Micro-benchmarks (harness-adjacent, real data where noted):

- **Warp cost is content-independent.** Nearest-neighbour `reproject` of a 2048²
  tile: 0.038 s (96 % data tile) vs 0.041 s (99.9 % NoData tile) vs 0.044 s
  (100 % NoData). GDAL warps NoData no faster than data. Skipping the warp for
  NoData tiles saves ~60 ms/tile only.
- **512² vs 2048² write tiles, real data (8192² region, ZSTD lvl 3, predictor 2):**
  512² = 256 calls / 0.10 s (0.41 ms/call), 28.1 MB; 2048² = 16 calls / 0.09 s
  (5.53 ms/call), 28.0 MB. **Total time ≈ equal; per-call overhead 13×.** On
  90 %-NoData data the 2048² tiles compress ~20 % *worse* (less local redundancy).
  Production 512² calls ran at 6.3 ms each (15× the micro number — machine load).
- **NoData tile inventory (2048 px dst tiles, final 3857 rasters):**
  auto cap: **0/20, 0/110, 0/399** all-NoData at 500/200/100 m.
  cap 2 km: 17/399 (4.3 %) at 100 m, 3/20 (15 %) at 500 m.

Sparse scenario (`--cap-km 2`, the case where output sparsity actually exists):

| res | calib | block wall | work write | reproj | close | **total** | peak RSS | final size | NoData |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 0.4 s | 196.1 s | 37.7 s | 71.2 s | 2.8 s | **309.4 s** | 7.5 GB | 87.6+49.7 MB | **90.07 %** |
| 500 | 0.4 s | 0.7 s | 0.7 s | 3.2 s | 1.1 s | **75.0 s** | 7.5 GB | 4.5+3.4 MB | 88.9 % |

At cap 2 km the *current code already* reaps most of the sparsity benefit: work
write −58 %, final size −85 %, close −58 %. What it does **not** save: compute
(only −18 %: 5 pyramid levels instead of 8) and reproject warp/flush (−44 % from
cheaper flush writes, warp unchanged).

## 6. Verdict on the three suspected cost sites

1. **Full-bbox raster allocation/write** — **confirmed, major, but mis-attributed.**
   Allocation is lazy; the cost is ZSTD-encoding 6.26 GB raw int16 into the
   work-CRS intermediate: 89.3 s at 100 m (18.6 %). The intermediate is pure waste
   when `out_crs != work_crs`: it is written (89 s), read back during reproject
   (part of the ~50 s warp), then deleted.
2. **Full-field reproject** — **confirmed, #2 cost.** 127.2 s at 100 m (26.5 %):
   ~50 s GDAL warp over the whole 1.564 B-cell destination + 76.8 s of 512² flush
   writes. The warp is content-independent (NoData warps at data speed), so
   "sparse reproject" only works by *not warping and not writing* fully-NoData
   dst tiles — which requires dst tiles that are fully NoData, and this dataset
   has **zero** of them at the auto cap (and only 4.3 % at cap 2 km).
3. **NoData disk I/O** — **refuted as a separate cost at the calibrated cap**
   (0.70 % NoData: there is nothing to skip or compress away), **confirmed as the
   dominant I/O property only for small caps** (cap 2 km → 90 % NoData, where the
   existing ZSTD setup already captures most of the win: size −85 %, write time
   −58 %).

The actual dominant cost — per-block compute, 50 % — was not in the issue's list.
It exists because all 399 blocks are tasks: the 3×3 neighbourhood rule
(`_block_neighbourhood_nonempty`, `pipeline.py:316-324`) plus a continental
background means "no block is empty" below ~200 m. Note the per-box geometry is
already near-minimal: box = block + 2×halo (2560² at 100 m, `_snapped_box`
`pipeline.py:176-196`), only 1.67× the grid area in redundant work; the 3×3
neighbourhood is used for *point gathering* (`_block_points`, `pipeline.py:198-215`)
and is then masked to the box (`pipeline.py:296-301`).

## 7. Option A — crop output to the populated envelope (+ `--full-bbox`)

**Design.** In `grid()` (`pipeline.py:632-690`) compute the min/max of task blocks
and size `nx/ny/x0/y0` to that envelope instead of the point bbox; plumb through
`_write_rasters` profile (`pipeline.py:836-846`); reproject dst dims follow
automatically via `calculate_default_transform` (`pipeline.py:931-942`).

**Measured effect on this dataset: ≈ 0.**
- The grid is *already* the tight point bbox (`x0 = x.min()`,
  `pipeline.py:639`), so no extent outside the point hull exists to trim.
- The first/last block row/col always contains the extreme points → the
  populated envelope is the full grid at every resolution:
  `env_task_frac = 1.0`, `env_pop_frac = 1.0` (harness `env` stats, all runs).
- The only trimmable NoData is the small set of grid *corners* beyond the
  diagonal point extremes — O((cap/res)²) cells, < 0.5 % of the grid.

**Effect on genuinely gappy data** (e.g. a single city input at continent-scale
res): the point bbox is *already* the city — the grid is small by construction.
A only helps when the output extent is defined *larger* than the point hull,
which the current API does not do.

**Risks.** Origin/extent change shifts the GeoTIFF transform (tile-server
alignment), breaks byte-identity of all committed examples, and interacts subtly
with the task-vs-None distinction (16/399 task blocks return `None` at 100 m —
"populated" must mean task, not data, or the envelope re-includes them).

**Effort.** S (1–2 days). **Verdict: reject.** Build effort for a measured
no-op under the current sizing model. Keep the idea in the follow-ups list for a
future `--extent` flag (user-defined output extent), where envelope-cropping
becomes real.

## 8. Option B — sparse tiled / mosaic output (one GeoTIFF per populated block)

**Design.** `_write_rasters` (`pipeline.py:828-958`) writes each 2048² block
output as its own GeoTIFF (`value_b{bx}_{by}.tif` / `support_…`) instead of one
full-bbox file, plus an `index.json` (block → bounds/transform/size) or a
rasterio VRT. Reproject per tile (each 2048² work tile → its dst window; the
global dst grid is computed once from the full extent, so tiles share a coherent
dst lattice), or skip reproject and emit work-CRS tiles.

**Measured effect.** Wall time ≈ unchanged: the same cells are computed and
encoded (399 files × 2048² vs 1 file × 1566 M cells is the same byte count;
per-file open/close overhead adds a few seconds). Final *size* ≈ unchanged at
auto cap (0.7 % NoData); −85 % at cap 2 km, which the single-TIFF path already
gets. The real benefit is **downstream**: an MVT tile server can open one 8.4 MB
raw tile instead of streaming a 600 MB file, and can skip 96 of 99 tiles per
zoom level on gappy data.

**Risks (this is the expensive option).**
- Contract break: README decode example, MVT consumers, `tests/`, and every
  committed example output. The single `value.tif`/`support_km.tif` pair is the
  product's API; a directory + index is a new API.
- Seams: per-tile reproject with nearest neighbour can disagree with a
  whole-field warp at tile edges (a dst pixel whose footprint crosses two
  src tiles). Must be handled by warping each dst tile from the *merged*
  footprint region, not per src tile, or accept 1-px edge artifacts.
- 399×2 files at 100 m is fine; a 50 m continental run is ~1,600×2 — file-count
  pressure on the out dir.
- `calibration.json` and the tmp-file lifecycle (`pipeline.py:909-925`) need
  rework.

**Effort.** L (1–2 weeks with tests + example regeneration). **Verdict: defer.**
Only worth it if the downstream MVT pipeline actually needs lazy per-tile reads;
that is a consumer-side requirement not evidenced by the issue. Revisit with
`--tiles` flag if requested.

## 9. Option C — hybrid single-TIFF (recommended)

Single GeoTIFF contract preserved. Four sub-parts, each independently
measurable by the harness:

### C(i) — envelope sizing
Same as Option A. **Measured ≈ 0 on this dataset** (§7). Ship only if a future
`--extent` flag makes it non-vacuous. *Effort S. Not required for the fix.*

### C(ii) — skip fully-NoData dst tiles in `_reproject_band`
**Design.** `_reproject_band` (`pipeline.py:327-391`) loops 2048² coarse dst
tiles (`warp_tile`, `pipeline.py:360-366`) and flushes 512² sub-tiles
(`pipeline.py:381-391`). For each coarse dst tile, compute its source footprint
in work-grid cell coords (inverse of `dst_transform` — cheap analytic bound for
equal-area→mercator at this scale, or `transform_bounds`); if the footprint
intersects **no task block** (the populated set from `self.tasks`,
`pipeline.py:678-687` — an over-approximation that is always safe), skip the
warp **and** write a plain NoData buffer to the tile (writing the NoData tile is
required: a fresh GTiff file is not guaranteed to be NoData-initialized, and
ZSTD-encoding a constant tile is ~free — micro-bench: the flush cost of NoData
tiles is what makes cap-2 reproject already 44 % cheaper).

**Measured effect.**
- Auto cap (0.7 % NoData): **0/399 tiles skippable at 100 m → 0 s saved.**
- Cap 2 km: 17/399 skippable → ~1–3 s saved (warp ~60 ms/tile + NoData flush is
  already cheap).
- For data with true voids (single-country input, equake-style arcs): this is
  where the option pays — every 2048² void tile skips ~60–85 ms of warp + its
  flush. A continent 50 % void saves ~50 % of reproject.

**Risks.** Low. Skip decision uses the task set (superset of data cells) →
never skips a data tile. The footprint bound must be conservative (round out);
a false "has data" costs one warp, a false "empty" costs a wrong pixel — use
`transform_bounds` on the dst window to be safe. No contract change: dims,
transform, values all identical.

**Effort.** S–M (1 day incl. test with a gappy fixture).

### C(iii) — 2048 px output tiles
**Design.** `TILE_PX = 512` (`pipeline.py:75`) → 2048 for the *output* rasters
(work tmp + final). Affects the profile (`pipeline.py:846`) and the flush loop
(`pipeline.py:381-391`, already coarsened by `warp_tile = 2048`).

**Measured effect.** Real-data micro-bench (§5): total write time ≈ equal
(0.10 s vs 0.09 s per 8192²), per-call overhead 13× higher, file size identical
on dense data, ~20 % *worse* compression on 90 %-NoData data. Extrapolated to
100 m: flush 76.8 s → ~60–65 s (−12 to −17 s), work write 89.3 s → ~85–95 s
(±5 s). Net: **−10 to −20 s (−2 to −4 %)** at 100 m.

**Risks.** The `TILE_PX` docstring (`pipeline.py:75-79`) states 512 is load-
bearing for **byte-identity with the committed baseline** — changing it changes
on-disk layout of *every* output (values identical, bytes differ). All
byte-identity tests and committed examples must be regenerated. On very NoData-
heavy data the larger tile compresses ~20 % worse (small size regression).

**Effort.** S (half day + example regen). **Only worth bundling if C(iv) is
shipped** — then the flush loop is rewritten anyway.

### C(iv) — direct block→dst reproject (drop the work-CRS intermediate)
**Design.** This targets the *measured* double-write: 89.3 s writing the
work-CRS tmp + its read-back inside the warp (the tmp is 628 MB at 100 m and is
deleted immediately after reproject, `pipeline.py:949-953`).
1. Keep the pool's block results in a `{(bx,by): (vq, rq)}` dict instead of
   discarding after `vd.write` (`pipeline.py:895-907`). RAM: 399 × 2 × 2048² ×
   2 B = **6.7 GB** at 100 m (measured peak RSS 8.8 GB → ~15.5 GB; acceptable on
   the 62 GB box, gate behind available-RAM check with fallback to the current
   file path — keep the flag `--direct-reproj`, default on when RAM allows).
2. Compute the dst grid once (`calculate_default_transform`, unchanged,
   `pipeline.py:931-942`), open both dst GeoTIFFs once.
3. For each 2048² coarse dst tile (same loop as C(ii)): compute the src
   footprint, gather the ≤ 9 overlapping block arrays into one contiguous
   src buffer (copy ≤ 6.6 M cells ≈ 13 MB, ~5 ms), build the src transform for
   that region, `reproject` from the **array** (rasterio accepts ndarray +
   `src_transform`/`src_crs`) into the dst tile, flush at C(iii) tile size,
   apply the C(ii) skip test first.
4. Delete the work-tmp write entirely when `out_crs != work_crs`.

**Measured/derived effect at 100 m:** removes 89.3 s (work write) + ~20–30 s
(file read-back inside the warp; array source skips GDAL read) and keeps the
flush (now at 2048²: ~62 s). Warp itself stays ~30–40 s (content-independent).
**Estimated total: 479.6 s → ~340–360 s (−25 to −29 %)**, and cap 2 km
309.4 s → ~250 s (−19 %). Disk: 798 MB of tmp traffic gone per run.

**Risks.**
- **Seams.** A dst tile whose src footprint crosses block edges gathers the
  neighbouring blocks — with C(iv) the gather is from the *same* block outputs
  the file path used, so values are bit-identical to today's file-based warp
  (same GDAL kernel, same src pixels, same resampling). The 3×3 point-gather
  halo already guarantees block outputs agree on shared pixels outside one
  pyramid step of each box edge, and the box halo ≥ one step
  (`_snapped_box`, `pipeline.py:176-196`) — so the gathered src buffer is
  internally consistent. Verify with the harness `--compare` (cell-identity).
- **RAM.** +6.7 GB at 100 m. Gate on available memory (`/proc/meminfo` or
  `resource`); fall back to the file path. At 50 m continental (~6 B cells,
  ~1,600 blocks) the dict is ~27 GB — the fallback keeps today's behavior.
- **Shared path (500 m).** `_prepare_shared` writes `_val_full.npy` etc.; C(iv)
  should not activate on the shared path (its per-block "outputs" are slices of
  the shared field) — keep the file path there. The 500 m case is fast anyway
  (40.5 s total).
- **Exception paths.** Current reproject failure renames the tmp back to
  `value.tif` (`pipeline.py:944-948`) so the work-CRS file is a valid fallback
  output. With C(iv) there is no tmp: on reproject failure, fall back to
  writing the work-CRS files (re-run the block writes from the kept dict).

**Effort.** M (2–3 days incl. fallback, gating, tests).

### Option C — combined expected BEFORE → AFTER

| scenario | total (measured BEFORE) | est. AFTER (ii)+(iii)+(iv) | delta | RSS | final size |
|---|---|---|---|---|---|
| 100 m auto | 479.6 s | **~340–360 s** | **−25 to −29 %** | 8.8 → ~15.5 GB | unchanged (762 MB) |
| 100 m cap 2 | 309.4 s | ~250 s | −19 % | 7.5 → ~14 GB | unchanged (137 MB) |
| 200 m auto | 112.3 s | ~85–95 s | ~−20 % | 3.4 → ~5 GB | unchanged (274 MB) |
| 500 m auto | 40.5 s | ~40 s (shared path, no change) | ~0 | 3.4 GB | unchanged (67 MB) |
| gappy data (voids ≥ 50 %) | — | reproject −50 %, plus A-like extent trim if `--extent` added | data-dependent | | |

The output side drops from ~240 s (50 %) to ~160 s at 100 m. **Compute stays the
dominant cost (240 s, 50 % → ~70 % of AFTER).** That is an algorithm property
(§6), not an output property — none of A/B/C fix it.

## 10. Recommendation

**Ship Option C as (ii) + (iv) (+ iii bundled with iv), default-on with flags.**

Rationale, in priority order:
1. **(iv) is the only sub-part that attacks a measured 89 s cost** with no
   contract change and bit-identical values (same GDAL warp, array source).
2. **(ii) is cheap insurance** for the sparsity the issue is actually about:
   zero cost on this dataset, and it is the mechanism that makes continent-scale
   runs with real voids (the issue's premise, correct for gappy data) scale.
3. **(iii) is a tie-breaker** for the flush loop (−12 to −17 s) accepted
   together with (iv)'s loop rewrite; it costs byte-identity with the committed
   baseline (regenerate examples — the `TILE_PX` docstring, `pipeline.py:75-79`,
   must be updated).
4. **A is rejected on evidence** (§7: measured no-op under bbox sizing).
5. **B is deferred** (§8: contract break, wall time ≈ unchanged; revisit if the
   MVT consumer needs lazy tiles).

Follow-ups (separate work items, not part of this fix):
- **F1 (biggest remaining cost):** per-block compute is 50 % of 100 m wall and
  is near the algorithm floor for the current design (1.67× grid-area
  redundancy, 4-way parallel, ~370 ns/cell — the shared path is slower at this
  size: 433 ns/cell single-threaded at 500 m, §5). Real options: parallelize the
  banded shared descent (it is single-threaded), or reduce per-block box cost
  with an early exit when the box's point count is tiny (many blocks at 100 m
  hold < 100 points). Estimate: −50 to −150 s at 100 m.
- **F2:** `--extent` (user-defined output extent) — makes envelope-cropping (A)
  non-vacuous and serves "interpolate this region" use cases.
- **F3:** revisit B if MVT lazy-reading is confirmed as a requirement.

## 11. Benchmark harness spec

`bench/sparse_bench.py` (created for this design; not committed to the repo —
the harness lives on the author's box; no pipeline code modified —
instrumentation is monkey-patching).

```
cd <worktree>
PYTHONPATH=$PWD /home/monky/projects/ppgrid/.venv/bin/python bench/sparse_bench.py \
    --res 500,200,100 --outdir /home/monky/tmp/sparse_bench [--reuse-calib] \
    [--cap-km auto|<km>] [--out-crs 3857] [--work-crs 6933] [--workers 4] \
    [--block 2048] [--data data/au_gcc_sparse.csv] [--no-nodata] \
    [--compare BEFORE_VALUE_TIF BEFORE_SUPPORT_TIF]
```

- Runs the real `Pipeline` per resolution, `out_crs != work_crs`, `cap_km=auto`
  by default (the sparse scenario is `--cap-km 2`).
- Patches (no source edits): `Pipeline._process_block` (per-block wall/CPU,
  None-result count), `Pipeline._reproject_band` (per-band wall),
  `Pipeline._prepare_shared` (shared-path wall), `ThreadPoolExecutor.map`
  (block-loop wall), `rasterio.io.DatasetWriter.write` (bucketed by phase:
  empty-block / task / reproject-flush; call counts, byte counts),
  `rasterio.open` write mode (open/close timing),
  `rasterio.warp.calculate_default_transform` (timing).
- Records: phase timings, grid/dst dims, block counts, cap/transform chosen,
  block time p50/p95/max, writer open/close, reproject per band + flush
  sub-phase, work-tmp sizes, final GeoTIFF sizes, **final NoData fraction**,
  **all-NoData 2048 px dst-tile count** (the C(ii) skippable set), envelope
  stats (min/max of task blocks and of populated blocks — the A metric),
  peak RSS, loadavg start/end. SIGTERM/SIGINT dump a partial record.
- `--compare`: after the run, cell-wise compares both final tifs against a
  BEFORE baseline: `both_data_mismatch` (values differ where both are data —
  must be 0) and `nodata_flip` (data↔NoData — must be 0 unless the option
  intentionally changes extent). Writes `compare.json`.
- Output: `<outdir>/sparse_bench.json` (per-res records) + console table.
  Self-test of `--compare`: identity compare of a file with itself passes
  (`both_data_mismatch=0`, `nodata_flip=0`); mismatched dims are reported.

**Validation protocol for the fix:**
1. BEFORE: the two runs already captured (`/home/monky/tmp/sparse_bench/`,
   `/home/monky/tmp/sparse_bench_cap2/`), kept as baseline.
2. AFTER: same commands on the fixed tree into fresh outdirs, plus
   `--compare <before>/f100m/value.tif <before>/f100m/support_km.tif` (and 200 m).
3. Acceptance gates:
   - 100 m auto: `t_total ≤ 400 s` (≥ 17 % faster than 479.6),
     `peak_rss_mb ≤ 16000`, compare `pass=true` for both bands.
   - 100 m cap 2: `t_total ≤ 280 s`, compare `pass=true`.
   - 500 m: `t_total ≤ 50 s` (no regression on the shared path).
   - `uv run pytest` green (53 tests) — byte-identity tests regenerated only if
     C(iii) ships (expected: the value/support *values* are identical; on-disk
     bytes differ).
4. Re-measure under stable load when possible (record loadavg; the 100 m
   auto baseline ran at loadavg 12–13, the cap 2 run at 16–23 — note the skew
   when comparing absolute numbers, prefer the breakdown percentages).

## Appendix — evidence index

| file | content |
|---|---|
| `/home/monky/tmp/sparse_bench/sparse_bench.json` | baseline: 500/200/100 m, auto cap, full phase detail |
| `/home/monky/tmp/sparse_bench_cap2/sparse_bench.json` | sparse scenario: 500/100 m, cap 2 km |
| `bench/sparse_bench.py` | harness (this design's benchmark) |
| `ppgrid/pipeline.py` | `TILE_PX:75`, `_SHARED_MAX_CELLS:100`, `_snapped_box:176`, `_block_points:198`, `_quantize:217`, `_process_block:271`, `_block_neighbourhood_nonempty:316`, `_reproject_band:327`, `grid():632-690`, `_prepare_shared:730`, `_write_rasters:828-958` |

Machine: marzuki-hydrogen (12C/62 GB), contended (loadavg 12–23, vLLM + Palworld
+ monitoring). Run times are wall-clock under that contention; per-phase shares
are stable across repeated runs.

---

## 12. LOCKED SCOPE (Andryo, 2026-09-29) — low-RAM-first

**Constraint (hard):** ppgrid targets low-RAM environments. Do NOT hold more RAM
than the current path needs. C(iv) (direct block->dst reproject, keeps all block
results in a dict) is **DROPPED** — wrong shape for low-RAM regardless of the
exact number (reviewer corrected the peak to ~8.8 GB, but it still holds all 399
results alive).

**Ship:**
1. **C(ii) — NoData-tile skip in `_reproject_band`.** RAM-free. For each coarse
   dst tile, compute its work-grid source footprint (conservative, round OUT);
   if it intersects no task block, skip the `reproject()` warp and write a plain
   NoData buffer to the tile. Uses `self.tasks` (superset of data cells) -> never
   skips a data tile. Zero cost when nothing is skippable (the auto-cap case).
   This is the mechanism that makes gappy data scale.
2. **Parallelize `_descent_banded` across row-bands within a level.** The bands
   within a level are independent (only the LEVELS are sequential: k-1 reads k).
   Replace the sequential `for r0 in range(0, n0, band_rows)` with a thread pool
   over bands. This attacks the low-RAM compute penalty (the single-threaded
   memmap descent is the slow path low-RAM boxes are forced onto). MUST stay
   bit-identical (bands are independent; same per-band math).
3. **RAM BUDGET GATE (the key requirement):** both changes must be proven to use
   NO MORE RAM than the current path. For (2): the thread pool holds `nworkers`
   bands' worth of intermediate buffers concurrently (not the whole grid) — bound
   it so peak RAM <= current single-threaded peak + one band's worth. Add a
   `--max-band-parallel` flag (default = min(workers, n0//band_rows)) and a
   `/proc/meminfo`-based fallback that drops to single-threaded when available
   RAM is below a threshold. For (1): the NoData buffer is one tile (4 MB),
   negligible.

**Defer / separate tickets:**
- C(iii) 2048px output tiles: only if RAM-neutral AND byte-identity regen is
  accepted. Default: OFF.
- C(iv) direct reproject: dropped (RAM shape).
- A envelope crop: rejected (measured no-op).
- F1 compute (per-block, the 50% cost): the banded-descent parallelization is
  part of this; the per-block early-exit-for-tiny-point-counts is a SEPARATE
  follow-up ticket.

**Acceptance (low-RAM):** on `data/au_gcc_sparse.csv`, 100 m, `out_crs=3857`:
- `peak_rss_mb` AFTER <= BEFORE (the current 8.8 GB) — MUST NOT increase.
- C(ii): reproject time drops when tiles are skippable (cap-2 km case: 17/399
  skippable); unchanged at auto cap (0 skippable).
- banded descent: wall time drops (parallel), bit-identical output (harness
  `--compare` vs baseline: `both_data_mismatch=0`, `nodata_flip=0`).
- 53 tests green (`uv run --extra dev pytest`).
