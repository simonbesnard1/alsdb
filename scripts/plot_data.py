#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 12:01:40 2026

@author: simon
"""

from alsdb import ALSProvider
from alsdb.utils.viz import plot_pointcloud_3d
from alsdb.utils.viz import plot_overview, plot_dsm, plot_rgb

# # Query
provider = ALSProvider(storage_type="local", uri="array_")
df = provider.query_tile(308, 4690)
df = provider.query_bbox(
    min_x=308_000,
    min_y=4_688_000,
    max_x=310_000,
    max_y=4_690_000,
)

# Matplotlib — static, good for export
fig = plot_pointcloud_3d(df, color_by="Z", max_points=500_000)
fig.savefig("cloud_3d.png", dpi=150, bbox_inches="tight")

# True colour
fig = plot_pointcloud_3d(df, color_by="RGB", max_points=500_000)
fig.savefig("cloud_3d_rgb.png", dpi=300, bbox_inches="tight")

# Classification with LAS colour palette
fig = plot_pointcloud_3d(df, color_by="Classification", max_points=500_000)
fig.savefig("cloud_3d_classification.png", dpi=300, bbox_inches="tight")

# Full 4-panel overview
fig = plot_overview(df, resolution=1.0)
fig.savefig("tile_308_4690.png", dpi=150, bbox_inches="tight")

# Individual panels
plot_dsm(df, resolution=1.0, hillshade=True)
plot_rgb(df, resolution=1.0)
