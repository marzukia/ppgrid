# ppgrid - Pull-Push Scattered-Data Interpolation

Fast, continent-scale raster interpolation for scattered point data. Turns tens of millions of geolocated points into a pair of GeoTIFFs in minutes on a single machine. No GPU needed.

## What it does

You have `N` points with `(longitude, latitude, value)`. You want a raster where every cell within a specified distance of real data carries an interpolated value, and everything else is nodata.

Standard IDW in QGIS or ArcGIS is `O(N*M)` (~14 hours for 16M points). This tool uses pull-push mipmap interpolation to reduce cost to `O(M)`, independent of N.

## How it works

### Pull-Push (mipmap) Interpolation

Based on Gortler et al. 1996 (Lumigraph) and Kraus 2009.

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
5. Local confidence is `min(C/saturation, 1)`. Dense cells trust themselves, sparse cells inherit
6. A summed-area table (`box_count`) provides an exact radius fill cap

**Output bands:**
- **Value:** int16, 0-100 percentile, `percentile = DN/scale`
- **Support km:** int16, `support_km = 2^(DN/8)`, effective spatial scale of estimate

**Working CRS:** EPSG:6933 (Wagner VII) by default. Global equal-area, metres are true. Configurable via `--work-crs`.

**Output CRS:** EPSG:3857 (Web Mercator) by default. Configurable via `--out-crs`.

### Calibration (optional)

Before interpolation, the tool can:
1. **Choose a transform**. Tests identity/log10/sqrt/percentile and picks the one with highest intraclass correlation across coarse scales
2. **Derive a fill cap**. Spatially blocked cross-validation to find the honest distance beyond which interpolation has no skill
3. **Save calibration.json**. Contains the percentile-to-value lookup table for decoding the output raster back to real units

## Install

```bash
pip install ppgrid
```

Or from source:

```bash
git clone https://github.com/marzukia/pullpush.git
cd pullpush
uv sync
```

## Quick Start

```bash
ppgrid data.csv --value-col price --res 500 --cap-km 10 --skip-calibration
```

This reads `data.csv`, interpolates the `price` column at 500m resolution with a 10km fill cap, and writes `value.tif` + `support_km.tif` to `examples/`.

## Usage

```bash
# Quick run (skip calibration)
ppgrid data.csv --value-col premium --res 500 --cap-km 64 --skip-calibration

# Full run with calibration (saves calibration.json)
ppgrid data.csv --value-col premium --res 100 --cap-km auto

# Custom projection and params
ppgrid data.csv --value-col premium --res 100 --cap-km 25 --transform log10 --workers 8

# Reuse existing calibration
ppgrid data.csv --value-col premium --calibration calibration.json
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
| `--transform` | `auto` | auto, identity, log10, sqrt, percentile |
| `--saturation` | `1.0` | Counts for a cell to fully self-trust |
| `--block` | `2048` | Block size in cells |
| `--workers` | `4` | Number of parallel workers |
| `--scale` | `100.0` | DN = percentile * scale |
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

with open("calibration.json") as f:
    quantiles = np.array(json.load(f)["percentile_quantiles"])

with rasterio.open("value.tif") as r:
    percentiles = np.array(r.read(1), copy=True) / 100.0

real_values = np.interp(percentiles, np.linspace(0, 100, len(quantiles)), quantiles)
```

## Benchmarks

Melbourne Housing dataset (13,580 points) at various resolutions on a single machine.

| Resolution | Wall Time | File Size |
|------------|-----------|-----------|
| 10m | 26.6s | 27.3 MB |
| 25m | 4.0s | 6.8 MB |
| 50m | 1.3s | 2.3 MB |
| 100m | 0.8s | 749 KB |
| 250m | 0.6s | 159 KB |
| 500m | 0.6s | 49 KB |

![Benchmarks](https://raw.githubusercontent.com/marzukia/pullpush/main/examples/bench_combined.png)

## Example Outputs

Melbourne Housing dataset (13,580 points) interpolated at 10m resolution:

![Melbourne Housing 10m Full](https://raw.githubusercontent.com/marzukia/pullpush/main/examples/melb/full_10m.png)

Cropped to the CBD to show resolution differences:

| 10m | 25m | 50m |
|-----|-----|-----|
| ![10m](https://raw.githubusercontent.com/marzukia/pullpush/main/examples/melb/10m/value.png) | ![25m](https://raw.githubusercontent.com/marzukia/pullpush/main/examples/melb/25m/value.png) | ![50m](https://raw.githubusercontent.com/marzukia/pullpush/main/examples/melb/50m/value.png) |

| 100m | 250m | 500m |
|------|------|------|
| ![100m](https://raw.githubusercontent.com/marzukia/pullpush/main/examples/melb/100m/value.png) | ![250m](https://raw.githubusercontent.com/marzukia/pullpush/main/examples/melb/250m/value.png) | ![500m](https://raw.githubusercontent.com/marzukia/pullpush/main/examples/melb/500m/value.png) |

## Data

`data/melb_houses.csv`: 13,580 Melbourne property sales with latitude, longitude, and price. Sourced from the [Melbourne Housing Snapshot](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot) (CC BY-NC-SA 4.0).

`data/all_equakes.csv`: 44,376 earthquake events from Jan-Aug 2026, mag >= 1.5.

### Data Lineage

| Step | Description |
|------|-------------|
| Source | [USGS Earthquake Hazards Program](https://www.usgs.gov/programs/earthquake-hazards) |
| API | [FDSN Event Web Service](https://earthquake.usgs.gov/fdsnws/event/1) |
| Download | Batched CSV requests by month, minmagnitude=1.5, starttime=2026-01-01, endtime=2026-08-08 |
| Processing | Concatenated monthly CSVs (deduplicated header) into single file |
| License | Public Domain (USGS federal data) |

### Columns Used

- `latitude` / `longitude`: spatial coordinates (WGS 84)
- `mag`: earthquake magnitude (continuous, for interpolation)
- `depth`: focal depth in km (optional value layer)
- `time`: event timestamp (ISO 8601)
