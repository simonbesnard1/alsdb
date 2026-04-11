.. _installing:

Installation
============

.. important::

   `pdal` and `python-pdal` are **conda-only** packages and cannot be installed via pip. `pixi <https://pixi.sh>`_ is the only supported installation method as it resolves both conda-forge and PyPI dependencies in a single step.

Dependencies
------------

alsdb requires Python ≥ 3.11 and the following dependencies, all resolved automatically by pixi:

+-------------------+-----------------+----------------------------------------------+
| Dependency        | Minimum Version | Purpose                                      |
+===================+=================+==============================================+
| pdal              | 2.7             | LAZ/LAS reading, HAG filter                  |
+-------------------+-----------------+----------------------------------------------+
| python-pdal       | 3.4             | Python bindings for PDAL                     |
+-------------------+-----------------+----------------------------------------------+
| tiledb            | 0.28            | Sparse point-cloud storage (local + S3)      |
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

Installation Instructions
-------------------------

Install via **pixi** (recommended):

.. code-block:: bash

    git clone https://github.com/simonbesnard1/alsdb.git
    cd alsdb
    pixi install          # resolves conda-forge + pip dependencies in one step
    pixi shell            # activate the environment

This installs all dependencies — including ``pdal`` and ``python-pdal`` from conda-forge — and installs alsdb itself as an editable package.

.. note::

   ``pip install alsdb`` will install the pure-Python dependencies but **will fail to import** without ``pdal`` present in the environment. Use pixi or set up a conda environment with ``pdal`` from conda-forge before using pip.

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
