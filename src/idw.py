"""
Fast IDW via scatter. Each data point scatters 1/r^2 weight to nearby cells.
Scales as O(points × radius²) not O(grid_cells × k).
"""

from __future__ import annotations

import numpy as np
from pyproj import Transformer
import sys


def idw_interpolate(
    data: np.ndarray,
    grid_resolution: float = 500.0,
    power: float = 2.0,
    radius_factor: float = 10.0,
    epsg_in: int = 4326,
    epsg_out: int = 3857,
) -> dict:
    """IDW by scatter. data: (N,3) = lon, lat, value."""
    if data.shape[1] < 3:
        raise ValueError("data needs >=3 columns: lon, lat, value")

    transformer = Transformer.from_crs(epsg_in, epsg_out, always_xy=True)
    px, py = transformer.transform(data[:, 0], data[:, 1])
    finite = np.isfinite(px) & np.isfinite(py)
    if (~finite).any():
        print(f"Drop {(~finite).sum()} pts", file=sys.stderr)

    projected = np.column_stack([px, py])[finite]
    vals = data[:, 2][finite]

    xmin, ymin = projected.min(axis=0).astype(np.float64)
    xmax, ymax = projected.max(axis=0).astype(np.float64)
    xmin = np.floor(xmin / grid_resolution) * grid_resolution
    ymin = np.floor(ymin / grid_resolution) * grid_resolution
    W = int(np.ceil((xmax - xmin) / grid_resolution)) + 1
    H = int(np.ceil((ymax - ymin) / grid_resolution)) + 1

    span = int(radius_factor) + 1
    print(f"Grid: {H}x{W}, span={span} (radius={span*grid_resolution:.0f}m)", file=sys.stderr)

    pbx = ((projected[:, 0] - xmin) / grid_resolution).astype(np.int32)
    pby = ((projected[:, 1] - ymin) / grid_resolution).astype(np.int32)
    frcx = (projected[:, 0] - xmin) / grid_resolution - pbx
    frcy = (projected[:, 1] - ymin) / grid_resolution - pby

    sum_w = np.zeros((H, W), dtype=np.float64)
    sum_wv = np.zeros((H, W), dtype=np.float64)

    BS = 2000
    for si in range(0, len(projected), BS):
        ei = min(si + BS, len(projected))
        bx = pbx[si:ei]
        by = pby[si:ei]
        fx = frcx[si:ei]
        fy = frcy[si:ei]
        v = vals[si:ei]

        sx = np.arange(-span, span + 1, dtype=np.float64)
        sy = np.arange(-span, span + 1, dtype=np.float64)

        tx = bx[:, np.newaxis, np.newaxis] + sx[np.newaxis, np.newaxis, :]  # (B,1,Sx)
        ty = by[:, np.newaxis, np.newaxis] + sy[np.newaxis, :, np.newaxis]  # (B,Sy,1)

        dxr = sx[np.newaxis, np.newaxis, :] - fx[:, np.newaxis, np.newaxis]  # (B,1,Sx)
        dyr = sy[np.newaxis, :, np.newaxis] - fy[:, np.newaxis, np.newaxis]  # (B,Sy,1)
        d = np.sqrt(dxr**2 + dyr**2)  # broadcasts to (B,Sy,Sx)

        # Broadcast tx, ty to match d shape (B,Sy,Sx) then flatten
        tx, ty = np.broadcast_arrays(tx, ty)

        # Circular influence zone (not square)
        radius_cells = radius_factor  # in grid units
        ok = (tx >= 0) & (tx < W) & (ty >= 0) & (ty < H) & (d > 0) & (d <= radius_cells)
        w = np.where(ok, 1.0 / (d ** power), 0.0)

        txf = tx.flatten().astype(np.intp)
        tyf = ty.flatten().astype(np.intp)
        wtf = w.flatten()
        vf = np.broadcast_to(v[:, None, None], w.shape).flatten()

        m = (txf >= 0) & (txf < W) & (tyf >= 0) & (tyf < H) & (wtf > 0)

        np.add.at(sum_w, (tyf[m], txf[m]), wtf[m])
        np.add.at(sum_wv, (tyf[m], txf[m]), wtf[m] * vf[m])

    with np.errstate(invalid="ignore"):
        z = np.where(sum_w > 0, sum_wv / sum_w, np.nan).astype(np.float32)

    gx = (np.arange(W) + 0.5) * grid_resolution + xmin
    gy = (np.arange(H) + 0.5) * grid_resolution + ymin

    return {
        "z": z, "xmin": xmin, "ymax": ymax,
        "gx": gx, "gy": gy, "res": grid_resolution, "epsg_out": epsg_out,
    }


def save_tiff(result, path, crs_wkt=None):
    """Save IDW result as GeoTIFF."""
    import rasterio
    from rasterio.transform import from_origin

    z = result["z"]
    trans = from_origin(result["xmin"], result["ymax"], result["res"], result["res"])
    epsg = result["epsg_out"]

    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=z.shape[0],
        width=z.shape[1],
        count=1,
        dtype=z.dtype,
        crs=f"EPSG:{epsg}",
        transform=trans,
        nodata=np.nan,
        tiled=True,
        compress="deflate",
    ) as dst:
        dst.write(z, 1)


if __name__ == "__main__":
    import pandas as pd, time

    df = pd.read_csv("data/all_equakes.csv", usecols=["longitude", "latitude", "mag"])
    data = df.dropna()[["longitude", "latitude", "mag"]].values.astype(np.float64)

    t0 = time.perf_counter()
    r = idw_interpolate(data, grid_resolution=5000, power=2.0, radius_factor=30.0)
    dt = time.perf_counter() - t0

    z = r["z"]
    print(f"Grid: {z.shape}")
    print(f"Time: {dt:.1f}s")
    print(f"Cells with data: {(~np.isnan(z)).sum():,}")
    print(f"Magnitude: min={np.nanmin(z):.2f} max={np.nanmax(z):.2f} mean={np.nanmean(z):.2f}")

    import os
    os.makedirs("outputs", exist_ok=True)
    save_tiff(r, "outputs/idw_5km.tiff")
    print("Saved outputs/idw_5km.tiff")