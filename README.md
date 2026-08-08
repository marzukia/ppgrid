# pullpush — Pull-Push Scattered-Data Interpolation

Fast, continent-scale raster interpolation for scattered point data. Turns tens of millions of geolocated points into a pair of GeoTIFFs in minutes on a single machine. No GPU needed.

## What it does

You have `N` points (`N` ~ 10^7) with `(longitude, latitude, value)`. You want a raster of `M` cells where every cell within a defensible distance of real data carries an interpolated value, and everything else is nodata.

Standard IDW in QGIS/ArcGIS is `O(N×M)` — ~14 hours for 16M points. This tool uses pull-push mipmap interpolation to reduce cost to `O(M)`, independent of N.

## How it works

### Pull-Push (mipmap) Interpolation

From Gortler et al. 1996 (Lumigraph) and Kraus 2009.

Once points are snapped to a grid, IDW is exactly a normalised convolution:

```
z = (S ⊛ K) / (C ⊛ K)   K(r) = r^-p
```

Pull-push evaluates this across a mipmap pyramid so cost is `O(M)`, independent of N.

**Steps:**
1. Points are binned into `S` (sum of transformed values) and `C` (count) grids
2. A mipmap pyramid is built by repeated 2x2 block sums
3. The coarsest level seeds the interpolation
4. Descending the pyramid, each level blends local estimate vs upsampled parent
5. Local confidence is `min(C/saturation, 1)` — dense cells trust themselves, sparse cells inherit
6. A summed-area table (`box_count`) provides an exact radius fill cap

**Output bands:**
- **Value:** int16, 0-100 percentile, `percentile = DN/scale`
- **Support km:** int16, `support_km = 2^(DN/8)`, effective spatial scale of estimate

**Working CRS:** EPSG:6933 (Wagner VII) by default — global equal-area, metres are true. Configurable via `--work-crs`.

**Output CRS:** EPSG:3857 (Web Mercator) by default. Configurable via `--out-crs`.

### Calibration (optional)

Before interpolation, the tool can:
1. **Choose a transform** — tests identity/log10/sqrt/percentile and picks the one with highest intraclass correlation across coarse scales
2. **Derive a fill cap** — spatially blocked cross-validation to find the honest distance beyond which interpolation has no skill
3. **Save calibration.json** — contains the percentile-to-value lookup table for decoding the output raster back to real units

## Project Structure

```
pullpush/
  __init__.py    — package init
  __main__.py    — entry point for `python -m pullpush`
  pullpush.py    — mipmap kernels (downsample, upsample, pull_push, box_count)
  calibrate.py   — transform selection + blocked CV for fill cap
  idwgrid.py     — CLI driver with block parallel processing
examples/
  test_run/
    value.tif       — interpolated value band (percentile)
    support_km.tif  — effective support scale
    calibration.json — percentile-to-value lookup table
```

## Install

```bash
uv sync
```

Or manually:

```bash
pip install numpy pandas pyproj rasterio
```

## Quick Start

```bash
# Interpolate Melbourne house prices
pullpush data/melb_houses.csv --value-col price --res 500 --cap-km 10 --skip-calibration -o examples/melb/
```

## Usage

```bash
# Quick run (skip calibration)
python -m pullpush data.csv --value-col premium --res 500 --cap-km 64 --skip-calibration

# Full run with calibration (saves calibration.json)
python -m pullpush data.csv --value-col premium --res 100 --cap-km auto

# Custom projection and params
python -m pullpush data.csv --value-col premium --res 100 --cap-km 25 --transform log10 --workers 8 --work-crs 3857 --out-crs 3857

# Reuse existing calibration
python -m pullpush data.csv --value-col premium --calibration examples/previous/calibration.json
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `input` | (required) | CSV or Parquet input path |
| `-o, --out` | `examples/` | Output directory |
| `--value-col` | `value` | Value column name |
| `--lng-col` | `longitude` | Longitude column name |
| `--lat-col` | `latitude` | Latitude column name |
| `--res` | `500.0` | Cell size in metres |
| `--cap-km` | `auto` | Fill cap km, or 'auto' (blocked-CV derived) |
| `--transform` | `auto` | auto \| identity \| log10 \| sqrt \| percentile |
| `--saturation` | `1.0` | Counts for a cell to fully self-trust |
| `--block` | `8192` | Block size in cells |
| `--workers` | `4` | Number of parallel workers |
| `--scale` | `100.0` | DN = percentile × scale |
| `--compress` | `ZSTD` | GeoTIFF compression |
| `--calibration` | (none) | Path to existing calibration.json |
| `--calib-max-points` | `2000000` | Max points to use for calibration |
| `--src-crs` | `4326` | Input coordinate reference system |
| `--work-crs` | `6933` | Working CRS for interpolation (equal-area) |
| `--out-crs` | `3857` | Output CRS for final GeoTIFF |
| `--skip-calibration` | false | Skip calibration, use defaults |

## Decoding the output

The value band stores percentiles, not raw values. To convert back to real units, use the `calibration.json` saved in the output folder:

```python
import json, numpy as np, rasterio

with open("examples/test_run/calibration.json") as f:
    quantiles = np.array(json.load(f)["percentile_quantiles"])

with rasterio.open("examples/test_run/value.tif") as r:
    percentiles = np.array(r.read(1), copy=True) / 100.0

real_values = np.interp(percentiles, np.linspace(0, 100, len(quantiles)), quantiles)
```

## Data

`data/melb_houses.csv` — 13,580 Melbourne property sales with latitude, longitude, and price. Sourced from the [Melbourne Housing Snapshot](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot) (CC BY-NC-SA 4.0).

`data/all_equakes.csv` — 44,376 earthquake events from Jan–Aug 2026, mag ≥ 1.5.

### Data Lineage

| Step | Description |
|------|-------------|
| Source | [USGS Earthquake Hazards Program](https://www.usgs.gov/programs/earthquake-hazards) |
| API | [FDSN Event Web Service](https://earthquake.usgs.gov/fdsnws/event/1) |
| Download | Batched CSV requests by month, minmagnitude=1.5, starttime=2026-01-01, endtime=2026-08-08 |
| Processing | Concatenated monthly CSVs (deduplicated header) into single file |
| License | Public Domain (USGS federal data) |

### Columns Used

- `latitude` / `longitude` — spatial coordinates (WGS 84)
- `mag` — earthquake magnitude (continuous, for interpolation)
- `depth` — focal depth in km (optional value layer)
- `time` — event timestamp (ISO 8601)
