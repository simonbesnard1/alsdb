#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 13:58:19 2026

@author: simon
"""

from alsdb import ALSProvider
from alsdb.processing.trees import segment_trees
from alsdb.utils.viz_trees import plot_trees, plot_trees_3d
import alsdb
alsdb.setup_logging()          # INFO by default


provider = ALSProvider(storage_type="local", uri="array_")

points, trees = segment_trees(
    provider,
    bbox=(655000.0, 8901000.0, 655200.0, 8901200.0),  # 300 × 300 m
    year=2014,
    #tile_size=300.0,     # 300 m × 300 m sub-tiles (~11 tiles for 1 km²)
    #tile_buffer=30.0,    # 30 m buffer so edge trees are fully captured
    n_workers=4,
    voxel_size=0.5,
    min_height=3.0,
)

# 2-D crown map (circles or convex hulls when points provided)
plot_trees(trees, points=points, output_path="trees_2d.png")

# 3-D coloured point cloud
plot_trees_3d(points, trees, output_path="trees_3d.png")
