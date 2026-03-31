#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 13:58:19 2026

@author: simon
"""

from alsdb import ALSProvider
from alsdb.processing.trees import segment_trees
from alsdb.utils.viz_trees import plot_trees, plot_trees_3d

provider = ALSProvider(storage_type="local", uri="array_")


points, trees = segment_trees(
    provider,
    bbox=(655000.0, 8901000.0, 656000.0, 8902000.0),
    year=2014,
    min_height=3.0,
    voxel_size=0.5,
)

print(trees[["tree_id", "height", "crown_area", "n_points"]].head(10))
print(f"{len(trees)} trees detected")

# 2-D crown map (circles or convex hulls when points provided)
plot_trees(trees, points=points, output_path="trees_2d.png")

# 3-D coloured point cloud
plot_trees_3d(points, trees, output_path="trees_3d.png")
