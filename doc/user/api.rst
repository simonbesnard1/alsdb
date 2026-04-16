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

   alsdb.storage.ALSZarrStore

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

   alsdb.processing.chm.compute_chm
   alsdb.processing.chm.compute_dtm
   alsdb.processing.chm.compute_dsm
   alsdb.processing.chm.compute_all

Gap fraction and LAI
---------------------

.. autosummary::
   :toctree: generated/
   :recursive:

   alsdb.processing.gap.compute_gap_fraction

Structural metrics and biomass
-------------------------------

.. autosummary::
   :toctree: generated/
   :recursive:

   alsdb.processing.biomass.compute_metrics
   alsdb.processing.biomass.compute_biomass
   alsdb.processing.biomass.naesset_model
   alsdb.processing.biomass.wrap_sklearn_model

Waveform simulation
--------------------

.. autosummary::
   :toctree: generated/
   :recursive:

   alsdb.processing.waveform.simulate_waveform
   alsdb.processing.waveform.simulate_batch

Tiling utilities
-----------------

.. autosummary::
   :toctree: generated/
   :recursive:

   alsdb.processing._tiling.tile_bboxes
   alsdb.processing._tiling.run_tiled
   alsdb.processing._tiling.array_crs
   alsdb.processing._tiling.array_data_bbox
   alsdb.processing._tiling.check_year_exists
   alsdb.processing._tiling.check_bbox_overlap

Visualisation
=============

2-D point-cloud plots
----------------------

.. autosummary::
   :toctree: generated/
   :recursive:

   alsdb.utils.viz.plot_overview
   alsdb.utils.viz.plot_dsm
   alsdb.utils.viz.plot_rgb
   alsdb.utils.viz.plot_intensity
   alsdb.utils.viz.plot_classification
   alsdb.utils.viz.plot_waveform
   alsdb.utils.viz.plot_rh_profile
   alsdb.utils.viz.plot_waveforms_3d
   alsdb.utils.viz.plot_pointcloud_3d

Gridded product plots
----------------------

.. autosummary::
   :toctree: generated/
   :recursive:

   alsdb.utils.viz_raster.plot_chm
   alsdb.utils.viz_raster.plot_dtm
   alsdb.utils.viz_raster.plot_dsm
   alsdb.utils.viz_raster.plot_agb
   alsdb.utils.viz_raster.plot_gap
   alsdb.utils.viz_raster.plot_lai
   alsdb.utils.viz_raster.plot_metrics
   alsdb.utils.viz_raster.plot_products
   alsdb.utils.viz_raster.plot_products_agb

Utilities
=========

.. autosummary::
   :toctree: generated/
   :recursive:

   alsdb.utils.constants.ALSProduct
   alsdb.setup_logging
