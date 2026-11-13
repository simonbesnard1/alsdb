.. _alsdb_docs_mainpage:

####################
alsDB Documentation
####################

.. toctree::
   :maxdepth: 1
   :hidden:

   Installation <user/installing>
   User Guide <user/index>
   Processing Pipeline <user/fundamentals.processing>
   Algorithms <user/algorithms>
   API Reference <user/api>
   Examples <auto_examples/index>
   Discussions <https://github.com/simonbesnard1/alsdb/discussions>
   Development <user/contributing>

**alsDB** is an open-source Python package for processing Airborne Laser Scanning (ALS) point clouds at scale. It reads LAZ/LAS files via `PDAL <https://pdal.io>`_, stores them in a `TileDB <https://tiledb.com>`_ sparse array (locally or on S3), and provides a full pipeline for forest structure products: Canopy Height Model, DTM, DSM, gap fraction, LAI, LiDAR structural metrics, aboveground biomass, and large-footprint waveform simulation. All gridded outputs are written directly to a `Zarr <https://zarr.dev>`_ v3 store.

.. grid:: 1 1 2 3
    :gutter: 2 3 4 4

    .. grid-item-card:: Getting Started
        :text-align: center

        New to alsDB? Run the whole workflow on a sample tile shipped with the package, in about ten seconds.

        .. button-ref:: user/quick-overview
            :expand:
            :color: primary
            :click-parent:

            Explore Quick Overview

    .. grid-item-card:: Algorithms
        :text-align: center

        Every method for ground classification, terrain, canopy height, metrics, biomass and waveforms, with its defaults, references and benchmark results.

        .. button-ref:: user/algorithms
            :expand:
            :color: primary
            :click-parent:

            See the Algorithms

    .. grid-item-card:: User Guide
        :text-align: center

        Detailed explanations of the two-layer storage architecture, ingestion workflow, and all processing functions.

        .. button-ref:: user
            :expand:
            :color: primary
            :click-parent:

            Access User Guide

    .. grid-item-card:: API Reference
        :text-align: center

        Auto-generated reference for all public classes and functions: ``ALSDatabase``, ``ALSProvider``, ``ALSZarrStore``, and all processing modules.

        .. button-ref:: user/api
            :expand:
            :color: primary
            :click-parent:

            Explore API Reference

    .. grid-item-card:: Contributor's Guide
        :text-align: center

        Want to contribute to alsDB? Set up a development environment, run the tests, and submit pull requests.

        .. button-ref:: user/contributing
            :expand:
            :color: primary
            :click-parent:

            View Contributor's Guide

