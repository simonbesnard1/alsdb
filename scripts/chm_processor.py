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
    bbox=(655000.0, 8901000.0, 656000.0, 8902000.0),
    year=2014,
    n_workers=6,        # parallel workers
)

print("--- %s seconds ---" % (time.time() - start_time_))


from alsdb.processing.gap import compute_gap_fraction

# Gap fraction only - no assumptions
compute_gap_fraction(
    provider=provider,
    output_path="output/gap.tif",
    bbox=(657430, 8900970, 659010, 8902550),
    resolution=10.0,
    year=2014,
)
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
