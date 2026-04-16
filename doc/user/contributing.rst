.. _devindex:

***********************
Contributing to alsDB
***********************

.. highlight:: shell

Overview
========

We welcome your skills and enthusiasm for the alsDB project! There are many ways to contribute beyond writing code: bug reports, documentation improvements, usage examples, and feature suggestions are all valuable.

All contributions are expected to follow our `Code of Conduct <https://github.com/simonbesnard1/alsdb/blob/main/CODE_OF_CONDUCT.md>`_.

Types of Contributions
-----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/simonbesnard1/alsdb/issues.

Please include:

* Your operating system name and version.
* The output of ``pixi run python -c "import alsdb; print(alsdb.__version__)"``
* A minimal reproducible example (ideally with synthetic data, not real LAZ files).
* The full traceback.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs tagged ``"bug"`` and ``"help wanted"``. These are open to anyone.

Implement Features
~~~~~~~~~~~~~~~~~~

Look for issues tagged ``"enhancement"`` and ``"help wanted"``. Before starting work on a large feature, open an issue first to discuss the design.

Write Documentation
~~~~~~~~~~~~~~~~~~~~

alsDB documentation lives in ``doc/``. Improvements to explanations, additional usage examples, and fixes to typos are always welcome.

Submit Feedback
~~~~~~~~~~~~~~~

Use the `GitHub Discussions <https://github.com/simonbesnard1/alsdb/discussions>`_ board for questions, ideas, and general feedback.

Development Setup
-----------------

1. Fork the repository on GitHub.

2. Clone your fork and install with pixi:

   .. code-block:: bash

       git clone https://github.com/YOUR_USERNAME/alsdb.git
       cd alsdb
       pixi install
       pixi shell

3. Create a feature branch:

   .. code-block:: bash

       git checkout -b feature/my-new-feature

4. Make your changes, add tests, and verify everything passes:

   .. code-block:: bash

       pixi run pytest tests/ -v --cov=alsdb

5. Check code style (Ruff):

   .. code-block:: bash

       pixi run ruff check alsdb/
       pixi run ruff format --check alsdb/

   To auto-format:

   .. code-block:: bash

       pixi run ruff format alsdb/

6. Commit and push to your fork, then open a pull request.

Code Style
----------

alsDB uses `Ruff <https://docs.astral.sh/ruff/>`_ for linting and formatting with a line length of 100 characters. All code must pass ``ruff check`` and ``ruff format --check`` before merging (enforced by CI).

Tests
-----

Tests live in ``tests/``. The test suite uses `pytest <https://docs.pytest.org>`_ and requires a real TileDB array fixture built from synthetic LAZ-like data (see ``tests/conftest.py``).

Run the full suite:

.. code-block:: bash

    pixi run pytest tests/ -v --cov=alsdb --cov-report=term-missing

Run only unit tests (no TileDB/PDAL):

.. code-block:: bash

    pixi run pytest tests/ -k "not provider and not compute" -v

Building the Documentation
--------------------------

.. code-block:: bash

    cd doc
    pixi run python -m sphinx -b html . _build/html
    # Open _build/html/index.html in a browser

Pull Request Guidelines
-----------------------

Before submitting a pull request:

1. Add tests for any new functionality.
2. Update docstrings (NumPy style) for any changed public functions.
3. Ensure ``pixi run pytest tests/`` passes with no failures.
4. Ensure ``ruff check alsdb/`` passes with no errors.
5. Update ``doc/`` if your change affects user-visible behaviour.

Pull requests are reviewed by Simon Besnard (``@simonbesnard1``). Response times are best-effort; please be patient.

License
-------

By contributing to alsDB you agree that your contributions will be licensed under the `EUPL-1.2 <https://opensource.org/licenses/EUPL-1.2>`_ license.
