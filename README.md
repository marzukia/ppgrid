# Problem Statement

At present, I have to visualise (cheaply) a dataset containing ~16M points (for many different layers, each 16M) that contain floating (or sometimes integer) values for exmaple, insurance premiums. My current workflow (as IDW is too slow) is to create two raster heatmaps using a 100m2 grid one containing a cumsum of each grid divided by a count grid. This is fine but I feel like it doesn't look nice especially when data is sparse (e.g. Australia most propertys are on the coast, lots of empty land).

IDW was way too slow in QGIS and ArcGIS as it looks to do things serially instead of taking advantage of things like matmuls, or convolutions.

First, can you research what tools have been made or released which does things like IDW in a more technically sophisticated way than the mainstream GIS community. Also what would you call my current method of rasterisation?

## Current State

Grid binning with zonal statistics — or more colloquially, rasterized bin mean. It's the square-bin variant of a hexbin plot (like in ggplot2's geom_hex). What you're computing is essentially a weighted_mean per cell (sum of values / count).

Single-value-per-bin means empty cells are blank. No spatial smoothing, so coast/interior boundaries look hard-edged. Nothing propagates information from coast to interior at all.


# IDW — Spatial Interpolation Experimentation

Fast spatial interpolation and visualization for large point datasets (~16M points per layer). The target use case is insurance premium heatmaps over Australia, but experimentation uses USGS earthquake data.

## Problem

Grid binning with zonal statistics (rasterized bin mean: sum/count per cell) is fast but looks bad in sparse areas. Standard IDW in QGIS/ArcGIS is too slow for large datasets because it runs serially without leveraging matrix ops or convolutions.

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

## Approach

### Pull-Push (mipmap) Interpolation

From Gortler et al. 1996 (Lumigraph) and Kraus 2009.

**Core insight:** Once points are snapped to a grid, IDW is exactly a normalised convolution:
```
z = (S ⊛ K) / (C ⊛ K)   K(r) = r^-p
```
Pull-push evaluates this across a mipmap pyramid so cost is `O(M)` independent of N. 44K points → 1.6M cells in 11s with 4 workers.

**How it works:**
1. Points are binned into `S` (sum of values) and `C` (count) grids at `[ix, iy]`
2. A mipmap pyramid is built by repeated 2x2 block sums
3. The coarsest level seeds the interpolation
4. Descending the pyramid, each level blends local estimate vs upsampled parent
5. Local confidence is `min(C/saturation, 1)` — dense cells trust themselves, sparse cells inherit
6. A summed-area table (`box_count`) provides an exact radius fill cap

**Output bands:**
- **Value:** int16, 0-100 percentile, `percentile = DN/scale`
- **Support km:** int16, `support_km = 2^(DN/8)`, effective spatial scale of estimate

**Working CRS:** EPSG:3577 (Australian Albers). Equal-area — metres are true.

**Dependencies:** python ≥ 3.11, numpy, pandas, rasterio, pyproj

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