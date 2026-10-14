# Release notes template

The GitHub release body for each tag is written from this template
(`publish.yml` publishes to PyPI on `release: published`). Copy the template
body below, fill every bracketed placeholder, and paste it into the release.
Sections are mandatory: a section with nothing to say gets "None this
release." — never silence.

Rules:

- **Performance claims carry full scope** — hardware, versions, dataset,
  CRS, resolution, options (s20 P0 box 3). The README bench tables already
  do this (see the 0.3.0 "Turbo benchmarks (full AU)" table); release notes
  must too. A number without scope is a claim without its hardware.
  Load-dependent numbers state the measured range and the cause.
- **The support/feedback path is visible in every note**: the note links
  SUPPORT.md at the tag ref, so it points at the doc shipped in that release
  (s30 "release").
- **Cite the issue number per change**; the CHANGELOG carries the detail.

---

## Template body (copy from here)

# ppgrid X.Y.Z

[One line, user language: what the release is for.]

## What's new

- [user-facing change] (#issue)
- [user-facing change] (#issue)

## Performance claims

[Every benchmark number in this note — or in a README bench table this note
updates — with its full scope. State ALL of:

- **Hardware**: machine name, CPU (cores/threads), RAM, OS
- **Versions**: ppgrid tag + the runtime deps that matter to the number
  (GDAL/rasterio, pyproj, numpy)
- **Dataset**: name + point count
- **CRS**: `--src-crs` / `--work-crs` / `--out-crs` as used
- **Resolution**: `--res` (and `--cap-km`)
- **Options**: every flag that changes the number (`--turbo` preset,
  `--workers`, `--skip-calibration`, ...)

Scope example (the 0.3.0 README table): "marzuki-hydrogen (Threadripper
24C/48T, 125 GB, CPU-only); full-AU dataset, 16M points; EPSG:4326 input ->
EPSG:6933 work -> EPSG:3857 output; `--res 100 --cap-km 25
--skip-calibration`; `--turbo 64`.")

## Support

Python 3.11-3.14; Linux and macOS supported and CI-covered; Windows
unsupported (match the current SUPPORT.md). Full matrix + feedback path
(issues / discussions / email):
[SUPPORT.md](https://github.com/marzukia/ppgrid/blob/vX.Y.Z/SUPPORT.md).

## Changelog

Full detail: [CHANGELOG.md](https://github.com/marzukia/ppgrid/blob/vX.Y.Z/CHANGELOG.md)
(`[X.Y.Z]` entry).

---

## House notes (keep out of the published body)

- The tag must match the pyproject version (`publish.yml` guards it); the
  README PyPI badge shows the published version and lags until the tag is cut.
- The v0.3.0 release note is the reference for scope-attached performance
  claims.
