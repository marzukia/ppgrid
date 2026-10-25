# AGENTS.md

Working notes + rules for AI agents (and humans) touching this repo. Read this before editing.

## What this is

`ppgrid` — pull-push mipmap scattered-data interpolation. Turns `N` geolocated points
into a pair of int16 GeoTIFFs (value percentile + support-km) at continent scale in
minutes, no GPU. It is an **IDW approximation**: fast, visually clean, and
deliberately *not* a spatially exact interpolator (the output is for visualisation,
not downstream calculation).

- Python 3.11+, `uv` managed. Entry point `ppgrid` (`ppgrid.pipeline:main`; `ppgrid.idwgrid` is a back-compat shim).
- Deps: numpy, pandas, pyproj, rasterio, rich (CLI presentation layer, added wave 1 2026-10-13). Optional `[parquet]` (pyarrow).
- Tests: `uv run pytest` (full suite). Lint/format: `ruff` (line-length 120).

## Repo layout

- `ppgrid/pipeline.py` — the Pipeline + CLI. All the logic lives here. `ppgrid/idwgrid.py` is a thin back-compat shim (module alias to `ppgrid.pipeline`).
- `ppgrid/calibrate.py` — transform selection + spatially-blocked CV fill-cap.
- `data/` — example inputs: `melb_houses.csv` (13,580 pts), `all_equakes.csv` (44,376 pts).
- `examples/` — committed outputs + repro scripts + README images (see "Images" below).
- `tests/` — pytest. `examples/run_melb.py` regenerates the Melbourne example rasters.
- `bench/` — perf harnesses + anchor gates: `check_anchors.py` (re-runs pipeline vs committed anchors), `check_artifact.py` (anchor gate for installed-artifact e2e outputs, issue #98).
- `scripts/` — CI helpers: `check_artifact.sh` (clean-venv install + console-script e2e, shared by ci.yml + publish.yml, issue #98).

## Versioning / release

- `pyproject.toml` `version` is the source of truth; `__version__` reads it via
  `importlib.metadata`. Keep `CHANGELOG.md` in step with every version bump.
- Releasing: bump `version` + `uv.lock`, update CHANGELOG, push a tag, the
  `publish.yml` workflow uploads to PyPI. The README PyPI badge shows the *published*
  version, so it lags the repo until a tag is cut.
- The GitHub release body is written from `docs/RELEASE_NOTES.md` (issue #100):
  mandatory sections = what's new (issue-cited), performance claims with full
  scope (hardware, versions, dataset, CRS, resolution, options), and the
  support/feedback path linking SUPPORT.md at the tag ref.

## Rules — do not break these

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

### Turbo budget semantics (measured)
- `--ram-gb` / `--max-ram` cap the shared-field budget, NOT process RSS — a
  regime-C run peaked at 39.4 GB under a 20 GB budget (write/descent buffers
  sit outside the Bt sizing). per_box is the only genuine low-RAM mode. Don't
  "fix" the budget to clamp RSS; clamp the fields instead.

### The zstd oracle fallback is expected on some stacks
- On the current hydrogen GDAL stack the zstd oracle mismatches libtiff
  streaming, so turbo runs take the byte-exact stock-parallel write fallback
  and print a `[warn] zstd oracle mismatch` line. That warning is correct
  behaviour there, not a bug: the parallel ZSTD path is unit-tested and
  activates where the stack matches. Byte-identity holds either way.

## Git / pushing

- The repo lives under the `marzukia` GitHub org. The `franktropic` identity is a
  collaborator with write (verified 2026-10-14 via API + dry-run push). Push plain:
  ```bash
  git push <branch>          # gh CLI auth (franktropic)
  gh pr create ...           # PRs are authored by franktropic
  ```
  The `monkytheluffy` bot identity is NOT a collaborator (push 403s).
- Branches go to a PR, not main, per the fleet merge gate.

## Change history

Dated change notes live in [CHANGELOG.md](CHANGELOG.md), not here — this file
is stable rules and gotchas only. If a change alters how this file's rules
read, update the rule in the same PR; do not add a "what changed" entry here.

## File size gate (harden-repo, 2026-10-20)

- File size is a load-bearing signal, checked by `python3 scripts/size_ratchet.py`
  (CI `size` job + pre-commit hook). >800 non-blank lines = must be in
  size-ratchet.json; ratcheted files may only shrink. `--update` lowers budgets
  after a split — never to raise one. .py files only; data/, examples/, dist/,
  build/, .venv, node_modules excluded.
- >2,000-line file: no NEW top-level units (top-level def / async def / class;
  CI export gate vs the PR base ref). New code goes to a sibling module.
  Test files (`tests/`, `*_test.py`): x3 budget.
- Task rule: a change adding code to a >2,000-line file must name the target
  module, not the god file. "Add X to ppgrid/pipeline.py" is an invalid task.
