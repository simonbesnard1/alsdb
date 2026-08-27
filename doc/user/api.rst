.. currentmodule:: alsdb

.. _api:

API Reference
=============

This page provides an auto-generated summary of alsDB's public API. For usage examples and conceptual background, refer to the :ref:`user` guide.

Core classes
============

.. autosummary::
   :toctree: generated/
   :recursive:

   ALSDatabase
   ALSProvider
   ALSTile

Storage
=======

.. autosummary::
   :toctree: generated/
   :recursive:

   storage.ALSZarrStore

TileDB backend
==============

.. autosummary::
   :toctree: generated/
   :recursive:

   TileDBProvider
   TileDBSchemaConfig
   create_schema

Tile and tile name
==================

.. autosummary::
   :toctree: generated/
   :recursive:

   Tile
   PNOATileName
   parse_tile_filename

Processing
==========

Canopy Height Model
-------------------

.. autosummary::
   :toctree: generated/
   :recursive:

   processing.chm.compute_chm
   processing.chm.compute_dtm
   processing.chm.compute_dsm
   processing.chm.compute_all

Gap fraction and LAI
---------------------

.. autosummary::
   :toctree: generated/
   :recursive:

   processing.gap.compute_gap_fraction

Structural metrics and biomass
-------------------------------

.. autosummary::
   :toctree: generated/
   :recursive:

   processing.biomass.compute_metrics
   processing.biomass.compute_biomass
   processing.biomass.naesset_model
   processing.biomass.calibrate_naesset
   processing.biomass.wrap_sklearn_model

Multi-temporal change detection
--------------------------------

.. autosummary::
   :toctree: generated/
   :recursive:

   processing.change.compute_change

Individual tree segmentation
-----------------------------

.. autosummary::
   :toctree: generated/
   :recursive:

   processing.trees.segment_trees

Waveform simulation
--------------------

.. autosummary::
   :toctree: generated/
   :recursive:

   processing.waveform.simulate_waveform
   processing.waveform.simulate_batch

Tiling utilities
-----------------

.. autosummary::
   :toctree: generated/
   :recursive:

   processing._tiling.tile_bboxes
   processing._tiling.run_tiled
   processing._tiling.array_crs
   processing._tiling.array_data_bbox
   processing._tiling.check_year_exists
   processing._tiling.check_bbox_overlap

Visualisation
=============

2-D point-cloud plots
----------------------

.. autosummary::
   :toctree: generated/
   :recursive:

   utils.viz.plot_overview
   utils.viz.plot_dsm
   utils.viz.plot_rgb
   utils.viz.plot_intensity
   utils.viz.plot_classification
   utils.viz.plot_waveform
   utils.viz.plot_rh_profile
   utils.viz.plot_waveforms_3d
   utils.viz.plot_pointcloud_3d

Gridded product plots
----------------------

.. autosummary::
   :toctree: generated/
   :recursive:

   utils.viz_raster.plot_chm
   utils.viz_raster.plot_dtm
   utils.viz_raster.plot_dsm
   utils.viz_raster.plot_agb
   utils.viz_raster.plot_gap
   utils.viz_raster.plot_lai
   utils.viz_raster.plot_metrics
   utils.viz_raster.plot_products
   utils.viz_raster.plot_products_agb

Utilities
=========

.. autosummary::
   :toctree: generated/
   :recursive:

   utils.constants.ALSProduct
   setup_logging
