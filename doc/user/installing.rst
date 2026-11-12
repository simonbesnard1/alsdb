.. _installing:

Installation
============

.. important::

   `pdal` and `python-pdal` are **conda-only** packages and cannot be installed via pip. `pixi <https://pixi.sh>`_ is the only supported installation method as it resolves both conda-forge and PyPI dependencies in a single step.

Dependencies
------------

alsDB requires Python ≥ 3.11 and the following dependencies, all resolved automatically by pixi:

+-------------------+-----------------+----------------------------------------------+
| Dependency        | Minimum Version | Purpose                                      |
+===================+=================+==============================================+
| pdal              | 2.7             | LAZ/LAS reading, ground filters              |
+-------------------+-----------------+----------------------------------------------+
| python-pdal       | 3.4             | Python bindings for PDAL                     |
+-------------------+-----------------+----------------------------------------------+
| tiledb            | 0.35.2 (pinned) | Sparse point-cloud storage (local + S3)      |
+-------------------+-----------------+----------------------------------------------+
| zarr              | 3.0             | Zarr v3 store for gridded products           |
+-------------------+-----------------+----------------------------------------------+
| numpy             | 1.26            | Array operations, rasterisation              |
+-------------------+-----------------+----------------------------------------------+
| scipy             | 1.12            | ``binned_statistic_2d``, peak detection      |
+-------------------+-----------------+----------------------------------------------+
| pandas            | 2.0             | Query result format                          |
+-------------------+-----------------+----------------------------------------------+
| xarray            | 2024.1          | Dataset output, Zarr backend                 |
+-------------------+-----------------+----------------------------------------------+
| pyarrow           | 14.0            | Parquet I/O for waveform batch results       |
+-------------------+-----------------+----------------------------------------------+
| matplotlib        | 3.8             | Visualisation                                |
+-------------------+-----------------+----------------------------------------------+
| click             | 8.1             | CLI                                          |
+-------------------+-----------------+----------------------------------------------+
| rioxarray         | 0.15            | CRS attachment on ``to_dataset()`` output    |
+-------------------+-----------------+----------------------------------------------+
| rasterio          | 1.3             | TIN rasterisation (tin, pit-free CHMs)       |
+-------------------+-----------------+----------------------------------------------+
| pyproj            | 3.6             | CRS parsing                                  |
+-------------------+-----------------+----------------------------------------------+

Installation Instructions
-------------------------

Install via **pixi** (recommended):

.. code-block:: bash

    git clone https://github.com/simonbesnard1/alsdb.git
    cd alsdb
    pixi install          # resolves conda-forge + pip dependencies in one step
    pixi shell            # activate the environment

This installs all dependencies, including ``pdal`` and ``python-pdal`` from conda-forge, and installs alsdb itself as an editable package.

.. note::

   ``pip install alsdb`` will install the pure-Python dependencies but **will fail to import** without ``pdal`` present in the environment. Use pixi or set up a conda environment with ``pdal`` from conda-forge before using pip.

Optional environments
---------------------

Some methods need extra packages. Each lives in its own pixi environment; run a
script in it with ``pixi run -e <name> python script.py``.

.. list-table::
   :header-rows: 1
   :widths: 14 50 36

   * - Environment
     - Adds
     - Needed for
   * - ``spikefree``
     - the compiled backends (CGAL constrained Delaunay, PTD); build them once with
       ``pixi run -e spikefree build-spikefree``
     - ``method="spikefree"`` CHMs and ``ground_method="ptd"`` classification
   * - ``lasr``
     - R and lasR (``pixi run -e lasr install-lasr``)
     - the lasR reference in ``alsdb.benchmark``
   * - ``helios``
     - the HELIOS++ lidar simulator
     - simulated scenes in ``alsdb.benchmark``
   * - ``gedi``
     - gedidb
     - real GEDI shots (``alsdb.products.gedi_attenuation``)

Without pixi, create a conda environment with PDAL, rasterio and pyproj from
conda-forge, then install alsdb with pip:

.. code-block:: bash

    mamba create -n alsdb -c conda-forge python=3.13 pdal python-pdal rasterio pyproj
    mamba activate alsdb
    pip install -e .            # from the cloned repository

rasterio must come from conda-forge, not pip: alsdb passes rasters from PDAL to
rasterio through GDAL's in-memory files, which only works when both use the
same GDAL library (pip's rasterio bundles its own).

Development install
-------------------

The development environment includes pytest and pytest-cov for running the test suite:

.. code-block:: bash

    git clone https://github.com/simonbesnard1/alsdb.git
    cd alsdb
    pixi install
    pixi run pytest tests/ -v

To run only a subset of tests:

.. code-block:: bash

    pixi run pytest tests/test_zarr_store.py -v
    pixi run pytest tests/ -k "unit" -v

Verifying the install
---------------------

.. code-block:: bash

    pixi run python -c "import alsdb; print('alsdb', alsdb.__version__)"
    pixi run alsdb --help

Then run the :ref:`overview`: it uses a sample tile shipped with alsdb and takes
about ten seconds.
