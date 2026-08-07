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
src/idw.py       — main IDW engine (scatter + Gaussian post-process)
outputs/          — generated rasters (GeoTIFF)
```

## Approach

### Scatter-IDW (current)

Instead of "for each grid cell find nearest neighbors", we reverse it: "for each data point scatter 1/r² weight to nearby cells. This scales as O(points × radius²) instead of O(grid_cells × log(points)). Much faster when the grid is much larger than the data extent.

**Implementation:**
1. Project all points to EPSG:3857 metric coordinates
2. For each batch of points, compute target cells and distances via numpy broadcasting
3. Apply 1/r² weight circular mask, accumulate into sum_w and sum_wv arrays
4. Divide sum_wv/sum_w for final weighted mean

**Benchmarks (USGS EQ data, 44K points):**
- 5km grid → 1.5s → 1.6M output cells (radius_factor=10)
- 5km grid → 7.3s → 4.0M output cells (radius_factor=30, circular)

### Alternative: kNN-IDW

Build a kD-tree, query k=16 nearest neighbors per grid cell. Slower for empty global grids (65M cells) but exact. Use pykdtree (not scipy.cKDTree).

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