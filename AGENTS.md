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
- Tests: `uv run pytest` (51 tests). Lint/format: `ruff` (line-length 120).

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
