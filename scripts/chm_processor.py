#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 00:32:51 2026

@author: simon
"""

from alsdb import ALSProvider
from alsdb.processing.chm import compute_chm, compute_all
import alsdb
alsdb.setup_logging()          # INFO by default


import time
start_time_ = time.time()

provider = ALSProvider(
    storage_type="local",
    uri="array_")


# CHM for the full array
compute_chm(
    provider, "output/chm.tif", resolution=1.0,
    bbox=(657430, 8900970, 659010, 8902550),
    year=2014,
    tile_size=500.0,    # 500×500 m sub-tiles → 3×3 = 9 tiles for 2.5 km²
    tile_buffer=50.0,   # 50 m overlap for accurate TIN at edges
    n_workers=6,        # parallel workers
)

print("--- %s seconds ---" % (time.time() - start_time_))

# # Restrict to one PNOA tile.
# compute_all(
#     provider,
#     output_dir="output/",
#     resolution=1.0,
#     #bbox=(308000, 4688000, 310000, 4690000),
# )


from alsdb.utils.viz_raster import plot_chm, plot_products

# Single product
plot_chm("output/chm.tif")

# # Three-panel overview
# fig = plot_products(
#     "output/dtm.tif",
#     "output/dsm.tif",
#     "output/chm.tif",
#     title="PNOA 2021 — tile 308/4690",
# )
# fig.savefig("output/products_overview.png", dpi=150, bbox_inches="tight")
