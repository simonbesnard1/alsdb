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

   products.chm.compute_chm
   products.chm.compute_dtm
   products.chm.compute_dsm
   products.chm.compute_all

Gap fraction and LAI
---------------------

.. autosummary::
   :toctree: generated/
   :recursive:

   products.gap.compute_gap_fraction

Structural metrics and biomass
-------------------------------

.. autosummary::
   :toctree: generated/
   :recursive:

   products.biomass.compute_metrics
   products.biomass.compute_biomass
   products.biomass.naesset_model
   products.biomass.calibrate_naesset
   products.biomass.wrap_sklearn_model

Multi-temporal change detection
--------------------------------

.. autosummary::
   :toctree: generated/
   :recursive:

   products.change.compute_change

Waveform simulation
--------------------

.. autosummary::
   :toctree: generated/
   :recursive:

   products.waveform.simulate_waveform
   products.waveform.simulate_batch

Tiling utilities
-----------------

.. autosummary::
   :toctree: generated/
   :recursive:

   products._tiling.tile_bboxes
   products._tiling.run_tiled
   products._tiling.array_crs
   products._tiling.array_data_bbox
   products._tiling.check_year_exists
   products._tiling.check_bbox_overlap

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
