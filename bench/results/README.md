# Bench results (issue #108)

`bench/regimes.py` writes one **result JSON** per run: the full config spec, the
commit hash, an environment fingerprint, and per-run cost records. This directory
holds the *committed* parts: the rerun README (this file) and the wall-gate
baseline.

## Layout

```
bench/results/
  README.md                 this file
  baseline/ci.json          committed wall-gate baseline (ci set, cut on the
                            machine the gate is meant to protect — see below)
```

Raw result JSONs are NOT committed. They land in `--outdir` (default
`/home/andryo/ppgrid-runs/<worktree-basename>/regimes` on hydrogen; `/tmp` on CI)
or in `--results-out PATH` when a stable path is wanted.

## Rerunning

### Hydrogen (via `ppgrid-run`, as andryo)

```bash
~/scripts/ppgrid-run <worktree> -- uv run --frozen python bench/regimes.py \
    --set ci --outdir /home/andryo/ppgrid-runs/<worktree-basename>/regimes
```

- `--set ci` = the 6 CI-safe configs (melb10, melb500 + four 10 km synthetic
  regimes). `--set all` adds the hydrogen-only configs (denser/sparser sparsity
  corners, equakes500, equakes100). `--ids a,b,c` selects a subset.
- `--config-file SPEC.json` (schema_version 1) replaces/extends the built-in
  table — that is how Andryo cuts new regime definitions (the uniform /
  clustered / edge families in the table are seeds, not a fixed set).
- The wall gate runs against `--baseline` (default
  `bench/results/baseline/ci.json`). It is `off` when the file is missing,
  `skipped` per config on n_cpus or spec drift, `fail` when a config's total
  wall exceeds `--wall-threshold` (default 2.0x) the baseline.

### CI (on-demand)

`github -> ppgrid -> Regimes bench -> Run workflow` runs the ci set on
ubuntu-latest (4 vCPU / 16 GB). The job fails on a wall-gate fail or a
correctness mismatch. Dispatch with **cut baseline = true** to run the set with
the wall gate off and download the result JSON as the new baseline (below).

## Cutting a baseline

1. Dispatch `regimes.yml` with `cut baseline = true` on the branch.
2. Download the `regime-results` artifact (the run's result JSON).
3. Commit it as `bench/results/baseline/ci.json` and push.

The baseline's `env.n_cpus` pins the machine class: on any runner with a
different core count the gate reports `skipped` (with the reason) instead of
passing or failing. A baseline cut on ubuntu-latest gates ubuntu-latest
dispatches; a hydrogen baseline gates hydrogen runs.

## Correctness gate

Every config with an `anchor` entry is checked before its wall counts for the
gate:

- canonical platform (x86_64 Linux): **byte-identity** (sha256) of the value +
  support_km bands vs the committed anchor;
- other platforms (macOS CI leg, arm64): **functional equivalence** — same grid
  shape/transform/CRS and bit-identical decoded arrays;
- configs without an anchor (e.g. `equakes100`) warn `no committed anchor —
  wall result unverified` and do not fail the run; a config that *claims* an
  anchor path that does not exist fails.

Anchors: `examples/melb/{10m,500m}/`, `examples/equakes/` (pre-existing) and
`examples/regimes/<id>/` (synthetic, cut on hydrogen 2026-10-15 at
f42387c, `skip_calibration` + identity transform + pinned workers).

## Result JSON schema (v1)

Top level:

```
schema_version: 1            this file's format (bump on breaking change)
tag: "ci-<commit12>"         default tag; --tag overrides
commit: "<sha>"              the commit the run executed
wall_gate: {status, baseline, threshold, per_config, note?}
  status: off | skipped | pass | fail
  per_config[<id>]: {status, wall_s?, baseline_wall_s?, wall_ratio?, reasons?}
  off = no baseline file (note explains); skipped = every config skipped
env: {hostname, os, n_cpus, n_bytes_memory, python, versions: {ppgrid, numpy,
      rasterio, gdal, pyproj, pandas}}
runs: [ per-config records, in spec order ]
```

Per-run record:

```
id, config (full spec), n_points, grid {nx, ny}, status: "ok" | "crashed"
timings_s: {ingest, calibrate, grid, write, total}   (wall, from run_summary)
peak_rss_bytes, tmp_disk_hwm_bytes, outputs_bytes {value.tif, support_km.tif,
calibration.json, ...}, shas {value.tif, support_km.tif},
correctness: {mode: bit-identity | functional | skipped, status:
pass | fail | no-anchor, bands: {value, support}}
vs_baseline: {}            reserved (issue #107 baseline column)
```

`tmp_disk_hwm_bytes` = high-water mark of `<outdir>/<id>/.work` (the pipeline's
temp-file footprint), sampled at ~200 ms by a background thread; `peak_rss_bytes`
= the worker subprocess's max RSS (one subprocess per config, so peaks are
honest and independent).

## Conservative rules (documented decisions)

- The wall gate is **skipped, never failed**, on machine (n_cpus) or config
  drift — a stale baseline must not turn a healthy run red.
- Crashed configs are excluded from the gate (skipped, reason "run crashed")
  and still fail the overall run via their `status: "crashed"` record.
- The ci set excludes equakes500 (8.1–8.4 GB peak RSS vs the 16 GB runner's
  headroom — same precedent as the anchor gate in `ci.yml`).
