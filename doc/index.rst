.. _alsdb_docs_mainpage:

##################
alsdb Documentation
##################

.. toctree::
   :maxdepth: 1
   :hidden:

   Installation <user/installing>
   User Guide <user/index>
   Processing Pipeline <user/tiledb_database>
   API Reference <user/api>
   Examples <auto_examples/index>
   Discussions <https://github.com/simonbesnard1/alsdb/discussions>
   Development <user/contributing>

**alsdb** is an open-source Python package for processing Airborne Laser Scanning (ALS) point clouds at scale. It reads LAZ/LAS files via `PDAL <https://pdal.io>`_, stores them in a `TileDB <https://tiledb.com>`_ sparse array (locally or on S3), and provides a full pipeline for forest structure products: Canopy Height Model, DTM, DSM, gap fraction, LAI, LiDAR structural metrics, aboveground biomass, and GEDI-style waveform simulation. All gridded outputs are written directly to a `Zarr <https://zarr.dev>`_ v3 store — no GeoTIFF intermediates, no mosaic step.

.. grid:: 1 1 2 2
    :gutter: 2 3 4 4

    .. grid-item-card::
        :text-align: center

        **Getting Started**
        ^^^

        New to alsdb? Start here for a quick introduction to ingesting LAZ files and running your first processing pipeline.

        +++

        .. button-ref:: user/quick-overview
            :expand:
            :color: primary
            :click-parent:

            Explore Quick Overview

    .. grid-item-card::
        :text-align: center

        **User Guide**
        ^^^

        Dive into the User Guide for detailed explanations of the two-layer storage architecture, ingestion workflow, and all processing functions.

        +++

        .. button-ref:: user
            :expand:
            :color: primary
            :click-parent:

            Access User Guide

    .. grid-item-card::
        :text-align: center

        **API Reference**
        ^^^

        Auto-generated reference for all public classes and functions: ``ALSDatabase``, ``ALSProvider``, ``ALSZarrStore``, and all processing modules.

        +++

        .. button-ref:: user/api
            :expand:
            :color: primary
            :click-parent:

            Explore API Reference

    .. grid-item-card::
        :text-align: center

        **Contributor's Guide**
        ^^^

        Want to contribute to alsdb? This guide covers how to set up a development environment, run tests, and submit pull requests.

        +++

        .. button-ref:: user/contributing
            :expand:
            :color: primary
            :click-parent:

            View Contributor's Guide
