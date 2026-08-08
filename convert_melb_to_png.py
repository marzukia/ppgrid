import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

resolutions = ['10m', '25m', '50m', '100m', '250m', '500m']

# Create a red-to-green colormap with 10 discrete steps
colors = plt.cm.RdYlGn(np.linspace(0, 1, 10))
cmap = LinearSegmentedColormap.from_list("10step_rdg", colors, N=10)

for res in resolutions:
    tiff_path = f"examples/melb/{res}/value.tif"
    png_path = f"examples/melb/{res}/value.png"
    
    if not os.path.exists(tiff_path):
        print(f"Skipping {res}: {tiff_path} not found")
        continue

    with rasterio.open(tiff_path) as src:
        data = src.read(1)
        nodata = src.nodata
        
        # Mask nodata
        masked_data = np.ma.masked_equal(data, nodata)
        
        # Downsample if too large for PNG (e.g., > 2000px)
        if masked_data.shape[0] > 2000 or masked_data.shape[1] > 2000:
            factor = max(1, max(masked_data.shape) // 2000)
            masked_data = masked_data[::factor, ::factor]

        plt.figure(figsize=(8, 8))
        plt.imshow(masked_data, cmap=cmap, interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(png_path, dpi=100)
        plt.close()
        print(f"Generated {png_path}")
