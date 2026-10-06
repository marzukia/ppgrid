# ppgrid `--turbo` mode, design

Repo: `~/projects/ppgrid` @ `74d9db8` (v0.2.2, worktree `~/.pi-bg-wt/ppgrid/20260929-063203-1812974`). Provenance note (review F5): that worktree is at d29210a = 74d9db8 + an unpushed partial 'parallel banded descent' implementation, and it holds the UNTRACKED `docs/DESIGN-sparse-output.md` (the 479.6 s anchor, measured on a different 64 GB machine). All appendix runs used `~/projects/ppgrid` @ 74d9db8 (clean).
Design only, no code, no PR. All claims cite code (`file:line`) or a run output (appendix A).

**Operator intent:** sacrifice the low-RAM constraint and parallelise the bottleneck
phase. Target deployment: 64 GB box.

**Byte-identity definition (repo convention, commit `2531d36`
"perf(idwgrid): vectorize band reproject warp, keep output byte-identical (#10)"):**

- *Array identity*, DN grids `np.array_equal` to the non-turbo path
  (`tests/test_tier2_shared.py:209` `test_melb10_anchor_bit_equal`,
  `tests/test_tier2_shared.py:156` `test_shared_field_matches_perbox`).
- *File identity*, GeoTIFF sha256 equal to the non-turbo file
  (`bench/compare_tier2.py:30` `_sha`, `:73` `compare_anchor`,
  `bench/compare_tier2.py:55` `compare_dirs` tiled compare).

Both must hold for `--turbo`. File identity depends on: same driver profile
(incl. `predictor=2`, `pipeline.py:850-851`), same 512² tile layout (`TILE_PX`),
same raster-scan flush order (`pipeline.py:383-396`), same ZSTD codec invocation
(libtiff 6.2 legacy zstd path, appendix A.6). File bytes therefore depend on the exact codec invocation as well.

---

## Plain English, no math

ppgrid takes scattered data points, each a place plus a number like a house
price, and turns them into a map. Every square of the map carries an estimate
of the number at that place. This document designs a new flag, `--turbo`, that
makes the map much faster without changing a single byte of the output.

The reason turbo exists is that at large scale, ppgrid has two problems. The
slowest steps run on one thread, and a hard cap on how many squares may live in
memory forces very large maps onto a slow per-region path. On the anchor case,
a full Australia map, the per-region path pays the output work once per region,
roughly four hundred times, instead of once. Turbo attacks all three: it
parallelises the single-threaded steps, and it raises the memory cap in a
measured, budgeted way for a 64 GB box.

Every speed-up has to preserve byte identity. The output files of a turbo run
must hash equal to the output files of today's run, on every benchmark
configuration. That is the acceptance rule, and it is why the design keeps the
per-pixel math unchanged and only reorders or parallelises work that is
independent.

The newest part of this document is section 3.6, the RAM scaling modes. One
knob sizes the memory-hungry settings to the box: presets for 16, 32, 64, and
128 GB, or auto, which reads the box. The plan then picks a regime. Regime A
runs the whole map in RAM. Regime B keeps the input in RAM and lets the output
spill to disk. Regime C keeps everything on disk with a small page-cache floor.
If even that does not fit, the run falls back to the per-region path, but with
turbo's fast parallel output steps, which is still a big win over today.

The acceptance tests (section 4, and the A-series matrix in 3.6.5) prove four
things: the time targets, a RAM ceiling on the big case, byte identity against
today's output, and the fallback behaviour when the box is too small. Wall
times are medians of at least three runs, because single runs drift a lot.

The numbers in section 3.6 are derived, not yet measured. The test matrix T0
through T4 on one machine, at caps from 128 GB down to 16 GB, is what turns
them into measured values. The open questions (3.6.6) are five items waiting on
review or on those runs: flag naming, keeping the bare flag, the fixed
headroom, whether regime C earns its keep, and one size to confirm.

## 1. Bottleneck phase

Plain English: this section finds where the time goes today. The answer is that
the slow part depends on map size. Small maps are slowed by two single-threaded
steps, the point-counting step and the final write. Large maps are slowed by
the per-region output work, which repeats once per region. Turbo has to fix all
three.

The bottleneck is **scale-dependent**. `Pipeline._prepare_shared`
(`pipeline.py:730-816`) takes the shared in-RAM path only while
`nx_padded * ny_padded <= _SHARED_MAX_CELLS` (`= int(2.5e8)`, `pipeline.py:100`;
check at `pipeline.py:747`). Below that, `box_count` + the final write/reproject
dominate. Above it, the per-box pull-push block loop dominates (already
thread-pooled over boxes via `--workers`, `pipeline.py:900-917`).

### 1.1 Shared path (≤ 2.5e8 padded cells), measured on this box (48 threads)

Plain English: the measured timing breakdown for the small-map case, 16 million
points on a 32.8 by 47 km box. The counting step takes the largest slice, and
the final write is stuck on one thread because of Python's GIL. The table holds
a clean run (no other load) and a contended run (two CI jobs stealing CPU), and
only the clean one counts.

16M pts, 32.8 × 47 km @ 10 m, padded 8192² (67.1M cells), cap 64 → single block
(run `fine16m_clean`, appendix A.2; first contended run `fine16m`, A.2).
**Clean total: 39.88 s, peak RSS 4460 MB.** The 75.8 s contended figure
overstates `grid()` (34.4 s there vs 7.4 s clean, CPU contention with two
qemu CI runners); the contended run is kept as evidence only.

| phase | clean (s) | contended (s) | code |
|---|---|---|---|
| ingest (read_csv + pyproj + tv) | 17.4 (fine16m_clean, warm csv) | 25.487 (cold csv) | `pipeline.py:493` |
| calibrate | 0.84 | 1.269 | `pipeline.py:529` |
| grid (geometry + argsort + memmap) | 7.4 (fine16m_clean; A.3 probe sums 7.96) | 34.431 (CPU contention) | `pipeline.py:632-697` |
| `_prepare_shared` (bin + box_count + descent) | 10.34 (67.1e6) / 12.85 (1.05e8 banded) | 11.174 (67.1e6 in-RAM) | `pipeline.py:730` |
| block loop (slice + quantize) | 1.04 | 1.127 | `pipeline.py:271,252` |
| reproject (warp + flush) | 9.35 (1.05e8) | 1.304 (67.1e6) | `pipeline.py:327` |
| **total** | **39.9** (fine16m_clean, 67.1e6) / **51.2** (wide10_cap2, 1.05e8) | **75.8** (fine16m, contended) | |

In words: each row is one phase of the run; read the clean column. The
contended column is the same work with two CI jobs stealing CPU, kept as
evidence only. `box_count` inside `_prepare_shared` is the largest
single-threaded slice, and the final write is the GIL-bound one.

Where the time actually goes:

- **`box_count` is the dominant `_prepare_shared` cost.** 4-pass separable box sum
  (`pullpush.py:107-146`): row cumsum `p` + row-window subtract + column cumsum
  `q` + window subtract. Single-threaded numpy. Measured probe (appendix A.3):
  8192² @ r=6400, int64 fallback path = **11.68 s**; float32 path (active below
  2²⁴ total points, `pullpush.py:26`) ≈ 8-9 s of the 11.174 s measured in the
  real run. The axis-0 cumsum (`q[1:] = np.cumsum(w, axis=0)`, `pullpush.py:141`)
  is strided (C-order array, column-wise), the slow half.
- **Grid index** is second (`pipeline.py:662-666`): `argsort` 3.30 s +
  cell-divide 3.46 s + searchsorted 0.24 s + memmap write 0.96 s at 16M
  (appendix A.3).
- **Ingest** is third (read_csv 7.15 s + pyproj 1.22 s at 16M, pyproj already
  releases the GIL, A.3; the pandas→numpy TV build is the rest).
- **Final write/reproject** is GIL-bound: `DatasetWriter.write` held the GIL in
  the 4-thread micro-bench (0.91× scaling, A.4); the warp (`reproject`) releases
  it (2.88× at 4 threads, A.4).

### 1.2 Per-box path (> 2.5e8 padded cells), repo anchor + this-box measurement

Plain English: the large-map case, where the whole field is processed region by
region. The measured anchor, a full Australia map, takes 479.6 seconds, and the
cost sits mostly in writing and reprojecting each region's output, work that
runs single-threaded no matter how many workers you add. That serial output
work is what turbo removes.

Repo anchor (`docs/DESIGN-sparse-output.md`, untracked in the worktree, measured on a different 64 GB machine, review F5; 4 workers):
**AU 949K pts @ 100 m, 41005 × 38189 grid, 399 boxes: 479.6 s total, 39.8 GB RSS**
= per-block 239.7 s (50%) + reproj 127.2 s (26.5%) + work-write 89.3 s (18.6%).

This-box measurement (appendix A.2, run `au50m`, clean, workers=1):
16M pts, 2500 × 2000 km @ 100 m, padded 5.07e8, 130 boxes, cap 25:
**176.2 s total, peak RSS 3.69 GB** = block loop 68.2 s (wall = cpu, 1 worker)
+ reproject 54.5 s + work-write 68.8 s (3961.8 MB written) + ingest 6.8 s (warm
csv) + grid 3.4 s. Per-box compute ≈ 525 ms at 208K pts/box (dense regime).

Extrapolation to full AU 16M @ 100 m (41005 × 38189, 399 boxes, 1714 pts/box,
sparse regime): per-box compute is cells-dominated (4.2M cells/box × ~15 level
steps ≈ 0.3-0.5 s/box CPU), pooled over 4 workers ≈ 30-50 s. The per-box
reproject + work-write is **main-thread serial over all boxes**
(`pipeline.py:943-946`, after the pooled block loop; the write is GIL-serial,
A.4) and output-sized ≈ 3× au50m's 123 s ≈ 370 s at ANY worker count:
cross-check: 0.95 s/box on this box (au50m, 1 worker) vs 0.54 s/box on the
DESIGN box (4 workers): the per-box rate is worker-invariant.
**≈ 420-460 s total on this box class; ≈ 480 s on the DESIGN box**
(DESIGN doc anchor vs au50m extrapolation). (Review F2: the original
"≈ 150-200 s" figure wrongly applied `--workers` to the serial phase.)

In words: 399 regions, each pays its own write and reproject, and that part
runs on the main thread alone. More workers do not touch it. That is why
479.6 seconds is mostly output, and why moving the Australia-scale map off the
per-box path is the big win.

### 1.3 Verdict

Plain English: the conclusion in three lines. Below the cap, fix the counting
step and the serial write. Above the cap, fix the per-region output. The
biggest win is raising the cap so Australia-scale grids stop paying per-region
at all.

- ≤ ~2.5e8 padded cells: **`box_count` + serial GIL-bound write/reproject** are
  the bottleneck (both single-threaded).
- > 2.5e8: **per-box output-bound work** (per-box tmp write + serial reproject,
  `pipeline.py:944-947`), the compute inside boxes is already parallel.
- `--turbo` must attack all three: parallel `box_count`, parallel final
  write/reproject, and (the big win at AU scale) **raise the shared cap so
  2.5-1.5e8-cell grids stop paying 399× per-box write+reproject at all**.

---

## 2. Current RAM model

Plain English: this section maps where the low-RAM promise lives in code, and
how much RAM the shared path really uses. The answer: the 250 million cell cap
is a hard switch, not a memory necessity. A 64 GB box holds the
Australia-scale field in RAM with room to spare, which is exactly the
constraint the operator wants sacrificed.

### 2.1 Where low-RAM is enforced

Plain English: the list of switches that keep memory low, plus the one that
actually bites. The disk-backed path throttles about 20x at very high cell
counts, but only in that tier. The in-RAM tier has no such throttle, which is
why a RAM-budgeted in-RAM cap is the safe relaxation.

| guard | value | code |
|---|---|---|
| shared-path cap | 2.5e8 padded cells | `pipeline.py:100`, check `:747` |
| in-RAM vs memmap tier (inside shared) | 1e8 padded cells | `pipeline.py:104`, branch `:767` (memmap) / `:787` (in-RAM) |
| row-banded bin/box_count | `band_rows=4096` | `pullpush.py:403,149` |
| row-banded descent, 1-row overlap | `band_rows=1024` | `pullpush.py:264-315` |
| free pyramid levels as consumed | `free_levels=True` | `pipeline.py:796` |
| points on disk, workers mmap | `_points.npy` | `pipeline.py:664-670,897` |

In words: these are the switches that keep the shared path small. The 2.5e8
cap is the big one; the 1e8 line splits in-RAM from disk-backed inside the
shared path.

Below 2.5e8 cells the finest grids go memmap + banded above 1e8, so the shared
path already runs 1e8-2.5e8-cell grids in ~8 GB (appendix A.2: 1.05e8-cell run
peaked at 3.59 GB RSS). The cap's own rationale (`pipeline.py:91-100`
comment): at ~2e9 cells the **memmap** banded descent hit page-cache
writeback throttling (~20× slower) on a 12.88 GB cgroup. That constraint
binds only the memmap tier; the in-RAM tier (≤ 1e8 today) has no such
writeback, which is why a RAM-budgeted in-RAM cap is the right relaxation
(§3.3). **The 2.5e8 cap is a hard gate, not a memory necessity for the
in-RAM field**
above it, ppgrid switches to per-box even though a 64 GB box would hold the
in-RAM field easily (that is the constraint the operator wants sacrificed).

### 2.2 Working sets (derived from code + anchored to measured RSS)

Plain English: the cost of a map square in RAM, broken into pieces, then the
model's peaks checked against measured runs. The peak moment is the counting
step, when four full-size temporary arrays are alive at once. Full Australia
in RAM peaks at about 40 GB, which fits a 64 GB box.

Per padded cell, in-RAM shared field: `s0` f32 (4 B) + `c0` f32 (4 B) +
`val_full` f32 (4 B) + `sup_full` f32 (4 B) + `near_full` bool (1 B) = 17 B at
level 0; pyramid `s+c` over `levels` ≈ 8 B × 1.333 (geometric sum,
`pipeline.py:783-789`) = 10.67 B/cell. Transient: `box_count` f32 prefix temps
are **four** 4 B/cell arrays, `p` (:128), `w = p[:,hi]−p[:,lo]` (:135),
`q` (:139), return `q[hi]−q[lo]` (:144), ≈ +16 B/cell (measured HWM delta
14.33 B/cell, review probe), freed before the descent; `near_full` bool is
allocated on top of the live return (`pipeline.py:792`). Points: 16M × 3 f64 bands
(`PTS_NBANDS`, `pipeline.py:88`) = 384 MB memmap (file-backed, reclaimable)
+ `ix`/`iy` f64 256 MB transient (`pipeline.py:760-761`).

In words: a square costs 17 bytes at level 0 of the pyramid and about 28 bytes
across all levels; the counting step adds roughly 16 more bytes per square
while it runs, then frees them. The peak moment is input data plus four
temporary counting arrays alive at once, which is why the cap formula in §3.3
prices the peak at 24.5 bytes per cell.

| case | padded cells | in-RAM shared peak (model) | measured |
|---|---|---|---|
| 1M pts @ 10 m, 32.8 × 47 km, cap 64 | 67.1M | ~3.2 GB | 3.6 GB (`fine1m` RSS 3570 MB) |
| 16M pts @ 10 m, same box, cap 64 | 67.1M | ~3.6 GB | **4.40 GB** (`fine16m`, A.2) |
| 16M pts @ 10 m, 100 × 100 km, cap 2 | 105M | ~4.0 GB (banded tier) | **3.59 GB** (`wide10_cap2`, A.2) |
| 16M pts @ 100 m, 2500 × 2000 km, cap 25 | 507M | per-box ~3.7 GB | **3.69 GB** (`au50m`, A.2) |
| 16M pts @ 100 m, full AU 4100 × 3819 km, cap 25 | **1.58e9** | **~32-45 GB** (below) | OOM-killed locally (12 GB cgroup, A.5) |
| 949K pts @ 100 m, full AU, cap 25 | 1.58e9 | per-box | 39.8 GB (DESIGN doc, 64 GB box) |

In words: same geometry, more data. The shared path holds 67 to 105 million
cell grids in 3.6 to 4.4 GB. The full-Australia case OOM-killed on the 12 GB
cgroup, and the model below says it would fit a 64 GB box.

Full-AU in-RAM model (1.58e9 cells, levels = 9 → step 256, `pipeline.py:644`):
- phase A (bin + box_count): `s0+c0` 12.6 GB + 4 box_count temps 25.3 GB +
  `near` 1.6 GB + points 0.8 GB ≈ **40 GB** ← **peak** (review F1: the original
  model counted only `p`+`q` = 12.6 GB; measured-transient scaling 14.33 B/cell
  → 37.7 GB)
- phase B (pyramid build): s+c pyramid 16.8 GB + near 1.6 + points 0.8 ≈ **19 GB**
- phase C (in-RAM descent → `val/sup`): pyramid 16.8 + val/sup 12.6 + near 1.6
  + points 0.8 ≈ **32 GB**
- phase D (block slice + write): val/sup 12.6 + near 1.6 + int16 DN buffer
  3.1 GB + GDAL ≈ **18 GB**

In words: A is the counting step (the peak, about 40 GB), B builds the pyramid
(about 19 GB), C produces the values (about 32 GB), D writes the file (about
18 GB). The phases do not run at once, so the peak is the largest one, A.

→ **peak ≈ 38-40 GB on a 64 GB budget (fits, ~15 GB margin under the 54.4 GB
budget line).** 16M pts stays
under the 2²⁴ float32-exact-integer limit (`pullpush.py:26`, pinned by commit
`9d81801`), so `box_count` uses the exact float32 path, integer sums are
order-independent, which is what makes parallel box_count bit-exact (S3.1).

Box note (review F6): when A.5 was measured (2026-10-06) monky's user cgroup
was capped at 12 GB (`memory.max = 12884901888` on `user-1003.slice`, the
full-AU run OOM-killed); it was raised to 64 GB the same night (Andryo), so
**A4/A5 are now directly runnable on this box**. The 64 GB box is
the deployment target; all sizing above is for it.

---

## 3. Turbo design

Plain English: the plan. Five parallelisation moves, each built so the
per-pixel math never changes. A RAM-budgeted cap replaces the hard cap. A
pre-check sizes the plan against the box before any big allocation, and degrades
to today's path if it does not fit. Byte identity is the acceptance rule
throughout.

### 3.1 Flag semantics

Plain English: what each new flag means at the command line. Off means zero
change to today's behaviour. The RAM number is a planning budget, not a
reading of the box, because this box proved its own cap can be smaller than the
machine.

```
--turbo           enable turbo mode (parallel phases + RAM-budgeted shared cap)
--ram-gb FLOAT    RAM budget for the pre-check (default: auto-detect cgroup/total)
--turbo-strict    hard-fail (exit 3) instead of auto-fallback when the estimate
                  exceeds the budget
```

- `--turbo` off ⇒ **zero diff**: every code path is identical to today (same
  functions, same order). The only shared code is the parallel primitives, which
  run with `n_threads=1` and must be bit-identical at 1 thread.
- `--ram-gb` is a **user-specified budget for planning only**, never the local
  cgroup (this box proves that: it sat at 12 GB on a 125 GB machine until 2026-10-06, then was raised to 64 GB). Default
  auto-detect = min(cgroup `memory.max`, physical) × 0.85, and the pre-check
  prints the value it used so a 64 GB deploy can see `--ram-gb 64` honoured.
- `--workers` unchanged (per-box parallelism, default 4). Turbo adds threads
  *inside* phases, bounded by `min(8, os.process_cpu_count())`.

### 3.2 Parallelisation strategy (each item preserves per-element op order)

Plain English: the five moves, in code order. The pattern in all of them is the
same: split work into chunks that do not touch each other's output, run the
chunks on threads, and keep the per-pixel math exactly as it is. The write
order that the file bytes depend on stays serial.

**S3.1, Parallel `box_count` (the #1 shared-path cost, §1.1).**
Restructure `pullpush.py:107-146` into a chunked 2D scan:

1. Row-window pass: per row, exclusive column prefix `p[:, 1:] = cumsum(c, axis=1)`
  , rows are independent → thread pool over row-chunks (bit-exact: each row's
   cumsum is the identical sequential op; in the float32 regime partial sums ≤
   total < 2²⁴ are exactly representable, so no rounding-order issue).
2. Column-window pass: make it contiguous, cumsum over `w.T` rows instead of
   strided axis-0 (values identical: same addition sequence per element). The
   per-row offsets for the band scan are plain integer sums (order-independent
   in both the f32-exact and int64 regimes).
3. Final window subtract: elementwise, parallel over row-chunks.

Target: ≥ 2× (bandwidth + allocation overlap; the axis-0 stride fix alone
should be a big part of it).

In words for the target: at least twice as fast on the counting step, with the
big part coming from fixing a slow memory access pattern, not from the threads.

**S3.2, Parallel row-banded descent.** `_descent_banded` (`pullpush.py:264-315`)
already computes level k in 1024-row bands; output row r reads input rows
r//2 and r//2+1 (even rows also tap r//2−1, covered by the band overlap
`e0 = r0//2−1`, `pullpush.py:264-315`; read-only) → bands are independent →
run the band loop
in a thread pool. In-RAM path (`_pull_push_descent`, `pullpush.py:241-250`):
same per-level row independence, add a threaded row-chunk variant. Bit-exact:
per-element ops unchanged; no cross-row writes.

**S3.3, Parallel warp in `_reproject_band`** (`pipeline.py:327-396`).
The warp loop over 2048-px dst tiles (the `coarse` dict build, `:359-379`) is
pure per-tile nearest-neighbour (`reproject` releases the GIL, 2.88× measured,
A.4) → thread pool over `i0` tiles within the row band. The 512² flush loop
(`:383-396`) stays **serial, raster-scan** (file bytes depend on tile order,
`pipeline.py:339-342` docstring). Bit-exact: per-pixel warp function of the dst
grid, unchanged.

**S3.4, Parallel final write (advanced, gated).** Two sub-parts:

- **Skip the work-CRS intermediate file** when `out_crs != work_crs`: today
  `_write_rasters` writes `value.tif`/`support_km.tif` (work CRS), renames to
  `_*_tmp.tif`, reprojects from disk, unlinks (`pipeline.py:922-951`). Turbo
  keeps the quantised int16 DN field in RAM (3.1 GB for full-AU, fits, §2.2)
  and reprojects tile-by-tile from the array through the same GDAL warp kernel
  → identical pixels, minus one 3.1 GB zstd write + read-back.
- **Parallel per-tile ZSTD compression** for the final 512² flush. Measured
  (A.6): GDAL 3.12.4 exposes the CPL compressor registry
  (`CPLGetCompressor("zstd")->pfnFunc`, ~2.4 ms/tile on a 512² int16 tile,
  thread-safe, deterministic; 1.93× under the 2026-10-06 contended window,
  6.11× idle on review re-measure, F4). Caveat found: the CPL "zstd" pfn
  emits a single-segment frame + 4-byte FCS, while the bytes ppgrid ships
  today come from **libtiff 6.2's ZSTD codec** (Ctx streaming API, default
  level 3, A.6); the frame headers differ, 16 B/tile on the report's DN tile
  (0 B on incompressible random tiles, F4). Resolution: a **calibration oracle** at startup
  (appendix A.7): compress one 512² tile through the turbo path, byte-compare
  with a tile written by the stock GDAL path on the same raster; on mismatch
  fall back to serial GDAL write (byte-exact, just slower). The oracle is the
  version-drift guard for libtiff/libzstd upgrades.

**S3.5, Parallel ingest** (`pipeline.py:493-527`): process pool over CSV
row-chunks (pandas `read_csv` holds the GIL; pyproj is already GIL-free).
**Pin the schema: parse every chunk with explicit float64 dtypes and assert
per-chunk dtype equality**, today's single parse infers dtypes per whole
file; per-chunk inference can disagree on edge rows (int64 vs float64,
divergent doubles for |x| ≥ 2⁵³), so the bit-exactness claim otherwise rests
on an unstated premise (review F3). Concatenate in original row order before
the TV transform → `bin_points`
order preserved (`pullpush.py:368-399` scatter is order-sensitive in its
float64 bincount accumulation) → bit-exact. Expected ~2-3× on the read_csv half
(7.15 s of 25.5 s cold at 16M, A.3).

**S3.6, Grid index**: leave `argsort` stable-sequential (3.3 s, 16M); the
cell-divide + memmap-write half can go to S3.1-style row-chunk threads (minor).

### 3.3 RAM relaxations

Plain English: replace the fixed 250 million cell cap with a cap derived from
the RAM budget. At 64 GB the cap becomes about 2 billion cells, so the
full-Australia map (1.58 billion) takes the shared path. The disk-backed
tier's guard is unchanged.

- Replace the hard `_SHARED_MAX_CELLS = 2.5e8` (`pipeline.py:100`) with a
  **budget-derived cap** from the phase-max model (§2.2, corrected per review
  F1: per-cell peak ≈ 24.5 B phase A with the 4-array box_count transient,
  ≈ 19.7 B phase C pyramid+val/sup, **phase A dominates**):
  `cells_max = (ram_budget × 0.85 − 4 GB slack) / 24.5` ⇒ **≈ 0.95e9 cells at
  32 GB, 1.5e9 at 48 GB, 2.06e9 at 64 GB**. Full AU @ 100 m (1.58e9, est
  peak ~38-40 GB, phase A) is shared for `--ram-gb ≥ ~48`; it stays per-box at
  32-44 GB (the 42-44 GB band admits 1.58e9 yet OOMs, the pre-check exists
  for exactly this, so the coefficient must be right).
  The 1e8 in-RAM/memmap tier (`pipeline.py:104`) stays as-is (the memmap
  tier's writeback throttle, §2.1, is unchanged).
- Turbo additionally allows keeping `val_full/sup_full` and the int16 DN field
  in RAM through the write phase (no memmap round-trip, §3.4).

In words for the formula: take the RAM budget, set aside 15 percent plus 4 GB
for the box, divide what is left by the 24.5 bytes a cell costs at its peak
moment, and you get the cap. 32 GB admits about 0.95 billion cells, 48 GB
about 1.5 billion, 64 GB about 2 billion. The full-Australia map at 100 m is
1.58 billion, so it goes shared at 48 GB and up; at 32-44 GB it stays per-box,
which is exactly the band the pre-check in §3.5 exists to catch.

### 3.4 Memory model at 16M pts / 10 m / 64 GB

Plain English: the three real deployments and their peaks. The defining case is
the Australia-scale map at 100 m, about 40 GB peak on a 64 GB box. The
Australia-scale map at 10 m is out of scope at any RAM and stays per-box,
collecting the turbo write wins.

- **10 m, 100 × 100 km (2.68e8 padded, cap 64):** per-box today; shared under
  turbo. Peak (phase A): s0+c0 2.1 GB + 4 box_count temps 4.3 GB + near 0.3 GB
  + points 0.8 GB ≈ **~7.5 GB** (phase C ≈ 6.1 GB).
- **100 m, full AU (1.58e9, cap 25):** **peak ≈ 38-40 GB** (§2.2, phase A):
  the defining turbo case.
- 10 m / full AU (1.57e11 cells, 626 GB int16 field) is out of scope:
  per-box at any RAM; turbo's per-box wins (S3.3, S3.4-skip-tmp) still apply.

### 3.5 Graceful failure + auto-fallback

Plain English: an out-of-memory kill is a hard kill, the process just stops. So
before any big allocation, the plan estimates its own peak from the §2.2
formulas and compares it to the budget. If it does not fit, it logs a warning
and runs today's path with correct output. The strict flag turns that into an
error.

cgroup OOM is **SIGKILL, uncatchable** (measured: journalctl `oom-kill`,
A.5). Therefore the pre-check runs **before** the first heavy allocation
(after `grid()`, which knows n, cells, levels):

```
est_peak = max(phase A..D estimates, §2.2 formulas)
budget   = ram_gb × 0.85
if est_peak > budget:
    --turbo-strict → stderr "error: --turbo est peak 51.3 GB > budget 54.4 GB" ; exit 3
    else           → log "[warn] --turbo: est peak 51.3 GB > 54.4 GB budget; using per-box"
                     continue on the non-turbo path (exit 0, correct output)
```

In words: estimate the worst case, compare it to 85 percent of the budget, and
if it is over, either hard-fail (strict) or warn and fall back to today's path.
The estimate is the max of the four phases, not their sum, because the phases
do not run at once.

The 0.85 factor covers Python/OS slack; the per-phase max (not sum) is the
conservative bound. A runtime backstop: if the shared path is taken and
`resource.getrusage` RSS exceeds `budget × 0.95` mid-run, log `[warn]` once
(performance only, the run completes; the pre-check is the real gate).

---

### 3.6 Turbo modes / RAM scaling (addendum 2026-10-06)

Plain English: this addendum adds one knob that scales turbo to the RAM of the
box it runs on. Think of it like a max-vram toggle, but for system RAM, and
the name is deliberate, ppgrid never touches a GPU. The knob sizes the
memory-hungry settings (how many bands and chunks may be in flight, how many
workers, how many compression contexts, how deep the write queue goes) so the
same plan works on a 16 GB laptop and a 128 GB box.

The three regimes, in everyday words. Regime A is the big-box case: the whole
map lives in RAM, nothing spills to disk, and all eight workers run. A 64 GB
box reaches it on the full-Australia map, and so does 128 GB. Regime B is the
middle case: the input data (the per-cell sums, counts, and near flags) stays
in RAM, but the output values and the big counting arrays do not fit, so they
go to disk and the work runs in staged bands with a handful of chunks in
flight, sized from the leftover budget. A 32 GB box runs regime B on the
Australia map. Regime C is the small-box case: even the input goes to disk,
and the machine survives on an 8 GB page-cache floor that keeps the disk path
from throttling. It only works if at least two chunks can be in flight. A
16 GB box cannot reach even regime C on the Australia map, so it takes the
per-box fallback, but with turbo's parallel warp and write, which is still
much faster than today.

Bt, the budget the plan works against. C is the cap you set. Bt is 0.85 times
C minus 4 GB. The 0.85 keeps 15 percent in reserve for the box, and the 4 GB
is a base floor for the operating system and the runtime. Everything else in
this section is the act of dividing Bt into slices: what the input costs per
cell, what the counting arrays cost, what a descent band costs, what the tile
cache and the compression contexts cost.

How to read the worked table below. Each row is one box size running the same
full-Australia map. The regime column says which of the three ways the machine
runs. In-flight says how many counting or descent chunks may be alive at once.
Workers says the thread count. Estimated peak RAM is what the plan predicts the
process will use. Estimated wall is the predicted total runtime. The last
column is that runtime against today's 479.6 second anchor: less than 1 means
faster. 16 GB lands at 0.6 to 0.79 times, 32 GB at 0.5 to 0.69 times, 64 GB
and 128 GB at 0.24 to 0.33 times, so 64 GB runs about three times faster than
today, and 128 GB adds headroom for bigger maps but no speed for this one.

One knob. It sizes the memory-hungry turbo parameters (descent bands in
flight, box_count chunks in flight, workers, ZSTD context pool, write tile
cache) to a RAM cap, so the plan works on 16 GB laptops and 128 GB+ boxes.
Andryo asked for a "max-vram style toggle" (his words). This is system RAM,
so the flag is `--max-ram <GB>`.

**Not in scope:** GPU memory (VRAM) is not what this knob sizes. ppgrid is
CPU-only; no phase touches VRAM. The name `--max-ram` is deliberate.

Number convention: every new number in this section is **derived, pending
A/B validation** (§3.6.5). Measured values are cited by section, not
restated.

#### 3.6.1 CLI surface

Plain English: the flags. `--turbo` takes an optional preset (16, 32, 64, 128
GB); bare `--turbo` means auto, which sizes to the box minus an 8 GB
headroom. `--max-ram` sets the cap directly and implies turbo. The knob sizes
five parameters, and none of them change the output bytes.

```
--turbo {auto|16|32|64|128}   RAM-scaling turbo mode. Bare `--turbo` stays
                              valid (= auto) via argparse nargs="?"
                              const="auto".
--max-ram <GB>                explicit cap (float GB). Overrides the preset.
                              Implies `--turbo` (a cap is only meaningful
                              in turbo).
--turbo-strict                unchanged (§3.1): est peak > budget, exit 3.
--ram-gb <GB>                 kept as alias of `--max-ram` (fold in v2,
                              §3.6.6).
```

Precedence: `--max-ram` > `--turbo <preset>` > auto.

**auto** = min(physical RAM, cgroup memory.max) − 8 GB headroom. One
formula for both spellings: the `--ram-gb` alias and the `--max-ram`
auto-default (bare `--turbo`) both use it, superseding the §3.1
auto-detect formula (× 0.85), which the alias would otherwise carry
silently. The headroom covers what is outside the ppgrid process
(derived): co-tenant anon 3 GB (dedicated box; ~10 GB on hydrogen with
vLLM + CI runners, Appendix A box caveat), OS/kernel 1 GB,
Python/numpy/GDAL runtime 1.5 GB, CSV + points ingest transient 1.5 GB,
allocator + page-cache reclaim floor 1 GB = **8 GB fixed**. It is
fixed, not a percentage: the 0.85 planning factor (§3.1) already covers
in-process slack, so the headroom must not double-count it. Floor:
`C = max(C, 8)`; below 8 GB the sizing budget `Bt = 0.85*C − 4` is near
zero (negative below 4.71 GB), so a sub-floor cap takes the per-box
path with a logged note.

In words for the headroom: 8 GB is what the box keeps for itself, other
processes, the operating system, the Python runtime, the data ingest, and a
floor for the allocator. It is a fixed number, not a percentage, because the
0.85 factor already covers slack inside the process and the headroom must
not count it twice.

Clamp rule: when the preset is ≥ physical RAM, `C = physical − 8 GB`
(preset = whole box, keep OS headroom). Below physical, the preset stands
as given (explicit user cap on a bigger box).

The knob sizes five parameters. All are byte-invariant (§3.6.3):

| parameter | sized from |
|---|---|
| descent bands in flight (height stays 1024, `pullpush.py:275`) | C |
| box_count chunks in flight (height 4096, `pullpush.py:149`) | C |
| workers (threads, cap min(8, cpu), §3.1) | C |
| ZSTD context pool (one context per worker) | C |
| write tile cache (queue depth; flush stays serial raster-scan, §3.3) | C |

In words: the knob sizes five things, how many descent bands and counting
chunks may be alive at once, how many threads, how many compression contexts,
and how deep the write queue goes. They change the parallelism, never the
math.

ZSTD **window**: held at the libtiff 6.2 default (level 3, A.6) for byte
identity. The knob scales the context count, not the window. Any window
change must pass the A.7 oracle (1 tile, ~5 ms, serial fallback).

#### 3.6.2 Derivation

Plain English: the arithmetic that turns a RAM cap into a runnable plan. A few
named constants (B0, Bt, M_in, and friends) are a price list: how many bytes
each piece of the map costs per cell. The three regimes are the three answers
to "what fits?", and the est_peak lines are the plan's predicted worst case.
Every number here is derived, not yet measured; the T-matrix in §3.6.5 is what
validates them.

Inputs: `C` (GB cap), `N` (total padded cells), `Wc` (padded columns),
`r` (cap radius in cells), `cpu`.

```
B0     = 4 GB            non-turbo base: OS + runtime + ingest transient
                         (derived; same slack as §3.3)
Bt     = 0.85*C − 4      turbo budget (coefficient from §3.1/§3.3)
M_in   = 9 B/cell        s0+c0 f32 + near bool (§2.2: 8 + 1)
M_vs   = 8 B/cell        val+sup f32 (§2.2)
M_box  = 12 B/cell over (4096 + 2r) rows   p, w, q f32 temps
                         (`pullpush.py:179-191`, derived)
M_band = 40 B/cell over 1024 rows + 8 B/cell over 512 parent rows
                         descent slices a/local/pv/ps
                         (`pullpush.py:304-313`, §2.2 terms, derived)
M_tile = 1 MB           0.5 MB raw + 0.44 MB zstd per 512^2 int16 tile
                         (A.6: 438,584 B shipped)
M_z    = 2 MB          per libzstd 1.5.7 level 3 streaming context
                         (derived; A.6 codec identity)
```

In words, the price list: B0 is the 4 GB the box always keeps for itself. Bt
is what the map may use after that. M_in is 9 bytes per cell for the input
data. M_vs is 8 bytes per cell for the output values. M_box is the temporary
counting arrays, 12 bytes per cell but alive only for a 4096-row window at a
time. M_band is one descent band's slice. M_tile is one compressed output
tile. M_z is one compression context. Read the rest of this section as the
act of buying as much of each as Bt allows.

Regimes (full-AU geometry: `N = 1.58e9`, `Wc = 41472` (512-padded, §2.2),
`r = 250` (§1.2). All values below derived, pending A/B):

- **A. Full in-RAM.** `N ≤ (0.85*C − 4)/24.5` (§3.3 formula; 24.5 B/cell
  phase A peak, §2.2). Whole field in RAM. Peak `= 24.5 B × N + B0`
  → **42.7 GB** at full AU by formula (the §2.2 RSS sum is 38-40 GB
  without B0, anchored to measured HWM; both bases printed in the
  table). workers `= min(8, cpu)`.
  In words: the whole map fits in RAM, so run everything at full threads;
  64 GB and up reach this on the Australia map.
- **B. Banded, input in RAM.** `Bt ≥ N×M_in + max(M_box, M_band)`.
  s0/c0/near in RAM (14.2 GB at full AU). In-flight sizing reserves the
  tile cache first (`T = clamp(floor(0.1*Bt / M_tile), 8, 256)` tiles),
  then `c = floor((Bt − N×M_in − T) / M_box)`,
  `b = floor((Bt − N×M_in − T) / M_band)`,
  workers `= min(8, cpu, c, b)`. val/sup in RAM when
  `Bt ≥ N×(M_in + M_vs) + T + in-flight`, else memmap (file-backed, §2.1;
  `pullpush.py:264-315` level 0 goes to memmaps or arrays).
  In words: the input fits, the output does not, so the output goes to disk
  and the work runs in bands with a few chunks in flight, sized from what is
  left in Bt after the input and the tile cache are paid for; 32 GB runs this
  on the Australia map.
- **C. Banded, all memmap.** `Bt < N×M_in`. s0/c0/near/val/sup all memmap
  (26.8 GB file-backed working set at full AU, §2.2 terms). Needs an 8 GB
  page-cache floor (derived) to avoid the §2.1 writeback throttle (~20×
  at a 12.88 GB cgroup). Viable only with ≥ 2 in-flight units; else
  per-box fallback with turbo warp/write (S3.3, S3.4).
  In words: even the input goes to disk, and the machine survives on an 8 GB
  page-cache floor that keeps the disk path from throttling; it only works
  if at least two chunks can be in flight, otherwise the run falls back to
  per-box with turbo's fast write.
- ZSTD contexts `= workers × M_z`.

Per-regime est_peak (all derived, pending A/B; the regime-aware
pre-check below uses these):

- **A:** `24.5 B * N + B0` (full AU: 42.7 GB). The §2.2 RSS sum is
  38-40 GB without B0; the table prints both bases.
- **B:** `N*M_in + T*M_tile + c*M_box + b*M_band + B0` (component
  bound). In-flight sizing fits Bt, so the planned peak is bounded by
  `Bt + B0 = 0.85*C` (C = 32: 27.2 GB).
- **C:** file-backed `N*(M_in + M_vs)` (26.86 GB at full AU, memmap) +
  in-flight units + 8 GB page-cache floor + B0. Only the in-RAM terms
  count against the budget: in-flight + 8 + B0 = 16.6 GB at C = 20
  (in-flight is the max of c*M_box and b*M_band, separate phases, not
  the sum). The file-backed set is reclaimable (§2.1).

In words: these are the predicted worst cases the pre-check compares against
the budget. A is the whole field at its peak price plus the base floor. B is
bounded by the budget by construction, because the in-flight counts were
chosen to fit. C only counts the in-RAM terms, because the disk-backed set
can be reclaimed if the box needs it.

Pre-check (regime-aware; this addendum replaces the plain §3.5 gate
for turbo): select the regime from C first (A if
`N <= (0.85*C - 4)/24.5`, else B if `Bt >= N*M_in + max(M_box,
M_band)`, else C, else per-box when C has < 2 in-flight units), then
compare that regime's est_peak to `budget = 0.85*C`. A feasible regime
fits the budget by construction (the sizing above); the gate's job is
the regime selection and the per-box fallback. Consequences: at C = 32
the gate passes regime B (27.2 GB = budget), so the §3.6.2 row is
reachable; at C = 20 it passes regime C (16.6 GB <= 17.0 GB), shared
memmap, not per-box. Budget note: the §3.5 pre-check budget `0.85*C`
and the sizing budget `Bt = 0.85*C - 4` differ by exactly `B0 = 4 GB`
(the non-turbo baseline floor, intentional); the pre-check counts B0
inside the budget, so a regime sized to Bt fits 0.85*C.

In words: pick the regime from the cap first, then check that regime's
predicted peak against the budget. A regime that fits by construction always
passes, so the gate's real jobs are choosing the regime and deciding when to
drop to per-box.

Worked table at full AU (anchor geometry, all derived, pending A/B):

| C (GB) | regime | in-flight c/b | workers | est peak RAM | est wall | vs 479.6 s anchor |
|---|---|---|---|---|---|---|
| 16 | C fails → per-box | 0/0 | 4 (per-box pool) | ~4-6 GB | 290-380 s | 0.60-0.79× |
| 32 | B, val/sup memmap | 3/4 | 3 | ~27 GB (+12.6 GB file-backed) | 240-330 s | 0.50-0.69× |
| 64 | A, full in-RAM | full field | 8 | 42.7 GB (formula 24.5N+B0) / 38-40 GB (§2.2 RSS sum, no B0) | 115-160 s | 0.24-0.33× |
| 128 | A, full in-RAM | full field | 8 | 42.7 GB (formula 24.5N+B0) / 38-40 GB (§2.2 RSS sum, no B0) | 115-160 s | 0.24-0.33× |

Plain reading, one line per row:

- **16 GB:** the whole map does not fit at all, so the machine runs today's
  per-box path, but with turbo's parallel output steps. Predicted 290-380 s
  against today's 479.6 s, and low RAM.
- **32 GB:** the input fits in RAM, the output spills to disk, three counting
  chunks and four descent bands in flight, three workers. Predicted 240-330 s,
  about 27 GB RAM.
- **64 GB:** the whole map lives in RAM, nothing spills, eight workers.
  Predicted 115-160 s, peak about 42.7 GB by the formula (38-40 GB by the
  measured-anchored sum).
- **128 GB:** the same run as 64 GB. The extra RAM buys room for bigger maps,
  not a faster run of this one.

Table notes:
- 16 GB peak: A.2 `au50m` measured 3.69 GB RSS (per-box) + turbo write
  buffers (derived).
- Wall model (derived, calibrated to the A4 target): `wall(W) = 30 +
  740/W` s, W = workers. 30 s = serial floor (ingest, calibrate, grid,
  serial flush; derived from the §1.1 phase table). 740 s = 1-thread
  parallel equivalent (A4 target ≤ 215 s at 4 workers = 30 + 740/4, §4).
  Ranges are the model ± 15%, widened where the A.4 scaling factors are
  sub-linear (S3.1 box_count target ≥ 2×, warp 2.88×, zstd 1.93×-6.11×).
- 16 GB row: per-box gains from the 479.6 s DESIGN anchor (§1.2 split):
  reproj 127.2 s → 62 s (S3.3 warp 2.88× on ~100 s, flush ~27 s serial,
  A.4), zstd 29.2 s @1 thread → 7.3 s @4 (A.6: 12150 tiles × 2.4 ms),
  skip work file 89.3 s (S3.4). 479.6 − 65 − 22 − 89 ≈ 304 s.

In words for the wall model: total time is a 30 second serial floor (the steps
that cannot parallelise) plus 740 seconds of parallel work divided by the
worker count. Four workers: 215 s. Eight workers: 122.5 s. The ranges are that
plus or minus 15 percent, widened where the measured speedup is less than the
worker count.

Reading the table:
- **16 GB** runs per-box + turbo warp/write. Regime B (in-RAM input)
  ends at C* = 24.13 GB (Bt = 16.51 = N*M_in + max(M_box, M_band));
  regime C (all memmap) stays viable from ~18.3 GB (>= 2 in-flight
  units after the 8 GB floor). T2b (24 GB, Bt = 16.40) sits 0.11 GB
  (0.7%) below the B/C boundary; the T2b probe settles it.
- **32 GB is the RAM viability knee** of the shared path at full AU
  (first tier with ≥ 3 in-flight units; val/sup still memmap).
- **48 GB (not a preset):** regime B, val/sup in RAM (36.8 >= 26.86 +
  0.25), c = 9, b = 12, workers 8, wall 115-160 s (model 30 + 740/8 =
  122.5 s, widened per the sub-linear clause), derived. Presets skip 48
  for UI simplicity, not speed.
- **64 GB is the speed knee:** first tier at full in-RAM (cells_max
  2.06e9 ≥ 1.58e9, §3.3) and 8 workers.
- **128 GB adds margin, not speed:** same 8-worker wall as 64 GB. T
  clamps at 256 tiles for every C >= 32 (0.1*Bt >= 2.32 GB > 0.25 GB),
  so 64 → 128 buys no bigger tile cache at this geometry; 128 GB buys
  cells_max headroom (4.28e9 vs 2.06e9 cells, larger grids) + margin.
- **Laptop-sized grid at 16 GB** (fine16m, 67.1M cells, §1.1): regime A
  even at the C = 8 clamp, peak 4.4 GB (A.2, measured), wall 12-25 s
  derived (1.6-3× of the 39.9 s A.2 clean). The knob pays off most where
  the grid fits.

#### 3.6.3 Why the bytes do not change

Plain English: why the speed-ups do not change the answer. Every knob this
section sizes changes how much work happens at once, never what math happens
per pixel, and the write order that the file bytes depend on stays serial. A
startup oracle test still gates every run.

Band height, chunk height, in-flight counts, worker count, tile queue
depth, ZSTD context count: none change the per-element ops. Descent bands
are read-only overlap (`pullpush.py:304-313`, bit-identical by
construction, `pullpush.py:264-315` docstring). box_count chunks
re-derive identical row prefixes (f32 exact below 2^24, `pullpush.py:26`).
Warp is a per-pixel function of the dst grid (§3.3 S3.3). Flush stays
serial raster-scan (§3.3 S3.3). ZSTD window stays at the libtiff default
(A.6). The A.7 oracle still gates every run. Cross-tier byte identity is
an acceptance gate (§3.6.5, gate 4).

#### 3.6.4 Degradation (runtime allocation failure)

Plain English: what happens if the plan is wrong and an allocation fails
mid-run. The phase catches the failure, logs it, re-plans at the next smaller
tier, and restarts the phase from a clean boundary. It never retries the same
tier. Worst case lands on today's path with correct output.

cgroup OOM is SIGKILL, uncatchable (A.5). The pre-check (§3.5) is the
primary gate. Step-down is the backstop:

1. Phase owner catches `MemoryError` / mmap fail.
2. Log `[warn] turbo: alloc fail at <phase> under tier <C>; stepping down
   to <C'>`.
3. Re-derive §3.6.2 parameters at the next tier (128 → 64 → 32 → 16 →
   per-box), re-run the phase from its last clean chunk/band boundary.
4. Never re-try the same tier. At per-box, a further failure takes the
   §3.5 auto-fallback (non-turbo, exit 0, correct output).
5. RSS backstop (extends §3.5): at `budget × 0.90` step down
   pre-emptively; at `0.90-0.95` log once (performance only).

A step-down never changes output bytes (§3.6.3), so a stepped-down run
still passes gate 4. The host is never OOM-killed: the pre-check keeps est
peak under budget, and the backstop keeps RSS under budget × 0.90.

#### 3.6.5 Validation plan (A/B)

Plain English: the test plan that turns "derived" into "measured". One
machine, capped at 128, 64, 32, 24, and 16 GB in turn (plus a 16 GB
container for the laptop case), the same Australia map each time, three runs
per point. Passing means no OOM kill, RAM under the cap, time within 20
percent of the table row, output byte-identical to the 64 GB run, and the
fallback behaving as designed.

Machine: hydrogen (2× RTX PRO 5000, 128 GB, 48 threads; same class as the
A.2 runs). Tier caps via cgroup `MemoryMax` on a test slice (technique
from A.5). One 16 GB sanity target via podman (`--memory=16g`, plain
`podman exec`, no `-it`).

Inputs: full-AU 949K pts @ 100 m (anchor geometry, §1.2) + fine16m
16M pts @ 10 m (A.2; realistic laptop case for the 16 GB target).

Matrix (full AU unless noted), median of ≥ 3 runs per the §4 protocol,
low-contention window, fixed workers:

| run | cap | flags | expected path |
|---|---|---|---|
| T0 | 128 GB native | `--turbo auto` | A |
| T1 | 64 GB slice | `--max-ram 64` | A (reference anchor run) |
| T2 | 32 GB slice | `--max-ram 32` | B |
| T2b | 24 GB slice | `--max-ram 24` | C or per-box (knee probe) |
| T3 | 16 GB slice | `--max-ram 16` | per-box |
| T4 | 16 GB podman, fine16m | `--turbo auto` | A |

Acceptance gates (per tier):
1. **No OOM kill:** `memory.events` oom_kill counter unchanged (A.5).
2. **Peak RSS** (`memory.peak`) ≤ cap − 0.5 GB.
3. **Wall** (median of 3) within **±20%** of the §3.6.2 table row. 20%
   because §4 measured ±40% single-run drift on the old box; a median of
   3 on a quiet box should halve it.
4. **Byte identity:** sha256 of value.tif, support_km.tif,
   calibration.json equal to the T1 (64 GB) run, all tiers.
5. **Fallback behaviour:** T3 shows the per-box log line, exit 0; T4
   confirms regime A; `--turbo-strict` with a shared-forcing override at
   T3 → exit 3 (pre-check, §3.5).

T1 doubles as the A4/A5 acceptance run on this box (§4).

#### 3.6.6 Open questions (for review)

Plain English: five open questions with reviewer recommendations, unchanged.
Two wait on the T-matrix runs (the 24 GB knee probe and the 16 GB tier),
three are naming and headroom decisions. This block is quoted verbatim for
review.

1. `--ram-gb` (§3.1) vs `--max-ram`: alias in v1, fold `--ram-gb` into
   `--max-ram` in v2? Proposal: alias now, deprecate at the v0.3.0 cut.
   reviewer recommendation (2026-10-06): AGREE with the proposal, but
   fix m4 first: one auto formula (- 8 GB) for both spellings, or the
   alias silently changes --ram-gb's documented default.
2. Bare `--turbo` → `--turbo auto` via `nargs="?" const="auto"`: keep the
   bool form working? Proposal: yes.
   reviewer recommendation (2026-10-06): AGREE, keep; correct argparse,
   backward-compatible, zero break.
3. Auto headroom: 8 GB fixed vs `min(8 GB, 25% of RAM)`. On a 16 GB
   laptop, 8 GB is 50%. Proposal: fixed for v1 (full AU is per-box below
   24 GB anyway); revisit after T3/T4.
   reviewer recommendation (2026-10-06): AGREE fixed for v1; the 0.85
   factor already adds 0.15C (>= 9.6 GB at C = 64) on top, so even
   hydrogen's ~10 GB co-tenant anon is covered, and full AU is per-box
   below ~24 GB anyway; revisit after T3/T4.
4. Regime C viability: is the all-memmap banded path (20-24 GB) worth
   keeping, or skip straight to per-box? T2b answers it.
   reviewer recommendation (2026-10-06): KEEP for v1, let T2b decide;
   the arithmetic says C is viable from ~18.3 GB (2+ in-flight units
   after the 8 GB floor), so it is not dead code, and the T2b
   knife-edge (16.40 vs 16.51 GB, 0.7% margin) is exactly what a knee
   probe should settle; if T2b shows the writeback throttle despite
   the 8 GB floor, drop C.
5. ZSTD context budget 2 MB: confirm by measuring per-context RSS in T0
   (A.6 pins the codec, not the context size).
   reviewer recommendation (2026-10-06): AGREE, measure in T0; the pool
   is at most 8 contexts × 2 MB = 16 MB, so even a 2× miss costs ~8 MB,
   low risk, cheap to close.

---

## 4. Acceptance criteria

Plain English: the pass/fail list for shipping turbo. Time targets on five
benchmark cases, a RAM ceiling on the big one, byte identity on every
configuration, and the fallback behaviour. Wall times are medians of at least
three runs, because single runs on this box drift by 40 percent, which would
swamp the margins.

Machine: 64 GB target box (numbers below re-anchored to this 48-thread box
where the case fits in the 64 GB cgroup (raised 2026-10-06, F6); DESIGN-box wall times in parentheses).

| # | criterion | target | baseline (measured) |
|---|---|---|---|
| A1 | `fine1m` 1M pts @ 10 m (67.1M cells, shared) total | **≤ 7 s** (≥ 1.8×) | 12.95 s (`bench_tier2.py:36`) |
| A2 | `fine16m` 16M pts @ 10 m (67.1M cells, shared) total | **≤ 20 s** (2.0×) | **39.9 s clean** (A.2) |
| A3 | `wide10_cap2` 16M pts @ 10 m, 100 km box (1.05e8, shared banded) | **≤ 30 s** (≥ 1.7×) | 51.2 s (this box, clean) |
| A4 | full-AU 16M pts @ 100 m (1.58e9) per-box → shared | **≤ ~215 s (2× of ~430 s)** | ~420-460 s est. this box (serial reproj+write, F2) / 479.6 s DESIGN box (949K) |
| A5 | RAM, full-AU 16M @ 100 m shared | **≤ 50 GB** on 64 GB budget | model ~38-40 GB peak, phase A (§2.2, F1) |
| A6 | byte-identity: `--turbo` vs non-turbo sha256 equal on **all** `bench_tier2.py` configs + committed anchors (`compare_anchor`) | 100 % |, |
| A7 | array-identity: turbo shared field vs per-box field `np.array_equal` (extend `test_tier2_shared.py:156` pattern) at a ≥ 2.6e8-cell case | 100 % |, |
| A8 | `--turbo` off: bit-identical output and same code path (1-thread primitives) | 0 diff |, |
| A9 | pre-check fallback (regime-aware, §3.6.2): `--ram-gb 20` on A4 case → `[warn]` (regime A est peak 38.7 GB > budget 17.0 GB) + regime C shared memmap (c = 2, b = 2 after the 8 GB floor) + exit 0; `--turbo-strict` at 20 GB → exit 0 (selected regime C fits the budget; the strict exit-3 path is pinned at T3, 16 GB, §3.6.5 gate 5) | both |, |

In words: A1 to A3 are the small-map speed targets (1.7 to 2 times faster).
A4 is the big win: the Australia map goes from per-box to shared, about 2
times. A5 is the RAM ceiling. A6 to A8 are the byte-identity rules. A9 is the
fallback. The baseline column is what was measured today.

Acceptance protocol (review F9): every wall-time target is the **median of ≥ 3
runs** in a low-contention window with fixed workers; single-run baselines
drift ±40% on this box, which swamps the A1/A3 margins as written.

Speedup accounting (why the targets are honest): shared-path A1/A2/A3 gains
come from S3.1 (box_count ~2-3×), S3.3 (warp 2.88×), S3.4 (zstd 1.93× contended / 6.11× idle + skip
tmp file), S3.5 (ingest 2-3× on read_csv). A4's gain is structural: one
shared write+reproject instead of 399 per-box ones (§1.2: 216 s of the 479.6 s
DESIGN anchor is per-box reproj+work-write).

---

## 5. Implementation sketch

Plain English: where the code lands. One table of sites, changes, and sizes,
roughly 660 lines, plus a test plan that pins byte identity and a 1-thread
regression, and a short risk register.

| site | change | ~LOC |
|---|---|---|
| `pullpush.py:107` (+ new `box_count_mt`) | chunked 2D scan, thread pool, f32 + int64 paths | 90 |
| `pullpush.py:264` `_descent_banded` | thread the band loop (bands independent, §3.2) | 40 |
| `pullpush.py:205` `_pull_push_descent` | threaded row-chunk variant (in-RAM path) | 40 |
| `pipeline.py:327` `_reproject_band` | parallel warp over `i0` tiles, serial flush | 25 |
| `pipeline.py:493` `ingest` | process-pool row-chunk read + ordered concat | 60 |
| `pipeline.py:828` `_write_rasters` | turbo write: RAM int16 DN field, skip tmp, parallel tile compress, serial flush | 80 |
| `pipeline.py:100-104` | budget-derived cap from `--ram-gb` (keep 2.5e8 as the non-turbo cap) | 40 |
| `pipeline.py:730` `_prepare_shared` | use `box_count_mt` / threaded descent when turbo | 15 |
| new `ppgrid/turbop.py` | pre-check estimator (§3.5), thread bound, calibration oracle (A.7) | 120 |
| new `ppgrid/zstdmt.py` | ctypes `CPLGetCompressor` pfn, per-tile pool, oracle compare | 120 |
| `pipeline.py:~1010` CLI (`main`) | `--turbo`, `--ram-gb`, `--turbo-strict` + validation | 25 |
| `tests/test_turbo.py` (new) | see test plan | ~180 |

**Test plan** (`tests/test_turbo.py`):

1. `box_count_mt` vs `box_count`: f32 regime (< 2²⁴ pts) and int64 regime
   (≥ 2²⁴, seed a 17M-count grid), `np.array_equal`, 4 threads.
2. `_descent_banded` threaded vs serial: `np.array_equal` on val/sup (1e8
   cells, 8 levels).
3. Parallel `_reproject_band` vs serial: **file sha256 equal** (the A6 unit).
4. ZSTD oracle: 1 tile turbo-compressed vs stock-GDAL-written, bytes equal;
   force-mismatch path → serial fallback selected.
5. Pre-check: mock budgets → per-box fallback + `[warn]` (caplog), strict
   `SystemExit(3)`.
6. End-to-end `--turbo` vs non-turbo sha256 on `fine1m` (12.95 s × 2, CI-safe)
   and on the `au50m` 16M per-box case (176 s, mark `slow`).
7. A8 regression: run the whole existing suite (53 tests, 80 s baseline green,
   appendix A.1) with turbo primitives forced to 1 thread.

**Out of scope (explicitly):** re-architecting pull-push; changing tile size /
predictor / codec; multi-process block workers (threads suffice, the GIL is
released by every heavy op, A.4); 10 m / continent-scale extents (per-box only).

**Risk register:**
- libtiff/libzstd upgrade changes the shipped zstd bytes → oracle gate
  (S3.4) falls back to serial write; output stays byte-identical, speed
  regresses only on the final write.
- Float32 2²⁴ boundary crossed by a future point count → int64 box_count path
  (2.5× slower), parallelised by S3.1 too; boundary already pinned
  (`9d81801`).
- Under-budget box without `--ram-gb` → auto-detect picks the cgroup cap →
  conservative fallback (safe direction).

---

## Appendix A, evidence

Plain English: the raw evidence every number above rests on. Test-suite
baseline, phase-timed runs, sub-phase probes, the GIL micro-bench, the OOM
journal entries, and the ZSTD byte-format findings that forced the
calibration oracle. All runs are on this box at the pinned commit.

All runs: `~/projects/ppgrid` @ `74d9db8`, `uv run python /tmp/turbo-bench/run.py`
(phase-timed clone of `bench/bench_tier2.py` instrumentation, identical patches).
Box: Ryzen 9 3960X 24C/48T, 125 GB RAM, Fedora 42. **Caveat: monky's user
cgroup is capped at 12 GB** (`user-1003.slice` `memory.max=12884901888`) while
vLLM + 2 qemu CI runners + PalServer hold ~10 GB anon, large per-box cases
were run at reduced workers; wall times on the 64 GB target will differ.

### A.1 Baseline test suite

In words: the test suite before any change, all green. 53 tests in 80 seconds
is the regression baseline.

```
$ uv run pytest tests/          # 2026-10-06, before any bench
53 passed in 80.44s
```

`bench_tier2.py` configs: `fine1m` 1M pts @ 10 m cap 64 (`:36`), `wide1m` 1M
@ 10 m 100 km cap 2 (`:38`), `melb10` 1M price @ 10 m cap 10 (`:41-43`),
`equakes500` 127K @ 500 m cap 25 (`:51-56`). This-box runs (workers=4):
`melb10` 17.69 s, `equakes500` 69.89 s (per-box block loop 41.46 s, RSS
6379 MB), `fine1m` 12.95 s.

### A.2 Phase-timed runs (this box, 2026-10-06)

In words: four real runs with per-phase timers. The clean 39.88 s run and the
176 s per-box run are the anchors for the §4 targets. The full-AU run
OOM-killed twice, which is why that number is estimated, not measured.

`fine16m_clean`, 16M pts, 32.8×47 km @ 10 m, cap 64, workers=4, shared in-RAM
(67.1M padded cells, 1 block):

```
t_ingest 17.403  t_calibrate 1.364  t_grid 7.402  t_prepare_shared 10.342
t_blockloop_wall 1.035  t_warp 0.55 (12 calls)  t_reproj_band 1.368
t_write_all 1.641 (122 calls, 118.7 MB)  t_write_rasters 13.713
t_total 39.882  peak_rss_mb 4460.2
```

(first run `fine16m`, contended: t_total 75.8 s, t_grid 34.431, t_ingest
25.487, same geometry, CPU contention only.)

`wide10_cap2`, 16M pts, 100×100 km @ 10 m, cap 2, workers=2, shared memmap
banded (1.05e8 padded cells, 25 blocks):

```
t_ingest 14.865  t_calibrate 0.843  t_grid 3.055  t_prepare_shared 12.849
t_blockloop_wall 3.276 (cpu 6.367)  t_warp 3.479 (60)  t_reproj_band 9.348
t_write_all 7.243 (842 calls, 792.9 MB)  t_write_rasters 32.396
t_total 51.159  peak_rss_mb 3592.6
```

`au50m`, 16M pts, 2500×2000 km @ 100 m, cap 25, workers=1, per-box
(5.07e8 padded cells, 130 blocks):

```
t_ingest 6.788 (warm csv)  t_calibrate 0.831  t_grid 3.411  t_prepare_shared 0.0
t_blockloop_wall 68.216 (cpu 68.211, 525 ms/box)  t_warp 16.632 (260)
t_reproj_band 54.473  t_write_all 68.843 (4136 calls, 3961.8 MB)
t_write_rasters 165.13  t_total 176.16  peak_rss_mb 3693.1
```

`au100`, 16M pts, full AU 4100×3819 km @ 100 m, cap 25, per-box
(1.58e9 padded cells, 399 blocks): **OOM-killed twice** (A.5). Second run
(workers=2) peaked at RSS 8039.8 MB then SIGKILL mid-reproject; output tiles
contain real data (decompressed tile: min −7545, max 9539, 5011 unique DN),
4708/12150 frames written → pipeline output is correct, the run is
resource-bounded. Full-AU wall time is therefore estimated, not measured, on
this box (§1.2).

### A.3 Sub-phase probes (16M points, this box)

In words: zoom in on the slowest steps to see which internal line takes the
time. The strided column-sum and the argsort are the named culprits.

```
grid() split:        cell-div 1.88+1.58 s, bid 0.10 s, argsort 3.30 s,
                     searchsorted 0.24 s, memmap write 0.96 s
ingest split:        read_csv 7.15 s, pyproj 1.22 s (GIL-free)
box_count probe:     8192^2 r=6400, int64 fallback path = 11.68 s (single
                     thread); float32 path ~8-9 s inside the real run
```

### A.4 GIL micro-bench (4 threads vs 1, numpy/GDAL, 512²-2048² int16 tiles)

In words: does the heavy math let go of Python's GIL? The warp does (2.88× on
4 threads), the tile write does not (0.91×, effectively single-threaded).
That split is what §3.2 parallelises.

```
rasterio.warp.reproject (nearest warp):  2.88×  → releases GIL
DatasetWriter.write (ZSTD 512² tiles):   0.91×  → holds GIL (serial floor)
```

### A.5 cgroup / OOM evidence

In words: proof that the memory cap is real. A 12 GB cap, OOM kills in the
journal, and a run that died mid-write. The cap was raised to 64 GB the same
night.

```
/sys/fs/cgroup/user.slice/user-1003.slice/memory.max      = 12884901888 (12 GB)
memory.peak                                              = 12890607616
memory.events: oom_kill 30
journalctl (2026-10-06 07:58:31): Memory cgroup out of memory: Killed process
system swap: 31/31 GB used (no reclaim headroom)
```

(At review time 2026-10-09: `memory.max` = 68719476736 (64 GB), `oom_kill`
30 → 32, cap raised 2026-10-06, F6.)

### A.6 ZSTD byte-format findings (rasterio 1.5.1, vendored GDAL 3.12.4)

In words: the shipped file bytes come from a specific ZSTD call inside
libtiff, and GDAL's alternate compressor emits a different frame header (16
bytes per tile). That is why a startup oracle test must pass before the
parallel write is trusted.

- The shipped tile bytes come from **libtiff 6.2.0's ZSTD codec**, the
  vendored `libgdal` imports the Ctx streaming API
  (`ZSTD_createCCtx/ZSTD_compressStream2/ZSTD_CCtx_setParameter`, `nm -D`),
  default level 3, libzstd 1.5.7. Frame header `28 b5 2f fd 00 60 00 00`
  (no FCS, window descriptor 0x60, streaming output).
- The GDAL CPL registry compressor `CPLGetCompressor("zstd")->pfnFunc`
  (struct +32, user_data +40; registered via `GDALAllRegister()`) is callable
  from ctypes: **~2.4 ms/tile (512² int16), deterministic (3200 frames, 1
  distinct), 1.93× at 8 threads under the 2026-10-06 contended window
  (6.11× idle, review re-measure)**, but emits a single-segment frame + FCS
  (`a0` header; 438600 B vs the shipped 438584 B per 512² DN tile, 0 B
  difference on incompressible random tiles, F4). `ZSTD_compress2` level
  sweep 1-9 does not reproduce the shipped bytes either. Implementation note
  (F4): the callback is the 6-arg bool-returning `CPLCompressionFunc`
  (input, size, void** out, size_t* outsz, options, user_data); a naive
  5-arg v1-style call segfaults.
- → S3.4's calibration oracle (compare one turbo-compressed tile against a
  stock-GDAL-written tile) is required before the parallel write may be used;
  the ctypes mechanics above are proven, only the exact codec parameters are
  pinned by the oracle at runtime.

### A.7 Calibration oracle (spec)

In words: the oracle in three steps. Write one tile the stock way, compress
the same tile the turbo way, compare the bytes. If they differ, use the stock
path. Cost: about 5 ms per run.

1. Write a 1-tile raster with the stock profile (`vprof`, `pipeline.py:850`).
2. Compress the same 512² int16 tile through the turbo compressor.
3. `bytes_equal` → turbo write enabled; else log
   `[warn] turbo zstd oracle mismatch; serial write` and use the stock path.
   Re-check cost: 1 tile ≈ 5 ms.
