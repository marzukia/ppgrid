# idwgrid — Pull-Push Scattered-Data Interpolation

Fast, continent-scale raster interpolation for scattered point data. Turns tens of millions of geolocated points into a gapless, capped, multi-band GeoTIFF in minutes on a single machine. No GPU needed.

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

**Working CRS:** EPSG:3577 (Australian Albers). Equal-area — metres are true.

### Calibration (optional)

Before interpolation, the tool can:
1. **Choose a transform** — tests identity/log10/sqrt/percentile and picks the one with highest intraclass correlation across coarse scales
2. **Derive a fill cap** — spatially blocked cross-validation to find the honest distance beyond which interpolation has no skill

## Project Structure

```
src/
  pullpush.py     — mipmap kernels (downsample, upsample, pull_push, box_count)
  calibrate.py    — transform selection + blocked CV for fill cap
  idwgrid.py      — CLI driver with block parallel processing
outputs/
  test_run/
    value.tif       — 5km global IDW interpolation
    support_km.tif  — effective support scale
```

## Install

```bash
uv sync
```

Or manually:

```bash
pip install numpy pandas pyproj rasterio
```

## Usage

```bash
# Quick run (skip calibration)
python -m src.idwgrid data.csv --value-col premium --res 500 --cap-km 64 --skip-calibration

# Full run with calibration
python -m src.idwgrid data.csv --value-col premium --res 100 --cap-km auto

# Custom params
python -m src.idwgrid data.csv --value-col premium --res 100 --cap-km 25 --transform log10 --workers 8
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `input` | (required) | CSV or Parquet input path |
| `-o, --out` | `outputs/` | Output directory |
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
| `--skip-calibration` | false | Skip calibration, use defaults |

## Data

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