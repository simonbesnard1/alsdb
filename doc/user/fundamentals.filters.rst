
.. for doctest:
    >>> import icesat2db as isdb

.. _fundamentals-filters:

#################
Quality Filtering
#################

This guide details the default quality filtering applied to ICESat-2 ATL08 data in the icesat2DB package during data ingestion.

Overview
--------

In icesat2DB, we have implemented a **default quality filtering routine** based on community-recommended practices. These filters are automatically applied during data ingestion to each 100 m land segment, reducing the dataset size while preserving scientifically reliable observations. This streamlines your analysis pipeline, saving both time and computational resources.

The filters rely on quality-related variables present in the raw ATL08 HDF5 files. While these variables are available in the raw data, they are **not pre-applied**, which can lead to the inclusion of low-quality segments if not filtered. icesat2DB automates this process, applying filters consistently across all six ICESat-2 beams to ensure high data integrity.

ATL08 Default Quality Filters
-----------------------------

The filters are defined in the :py:class:`icesat2db.ATL08Beam` class and are applied as a logical AND across all conditions — a segment is retained only if it passes every filter. Segments that fail any condition are dropped entirely.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Filter condition and rationale
   * - ``h_te_uncertainty``
     - Retained when value is below the HDF5 fill value (3.4028235×10³⁸) or above −999. Removes terrain height estimates flagged as invalid or missing.
   * - ``h_te_best_fit``
     - Retained when value is below the HDF5 fill value or above −999. Removes best-fit terrain elevation estimates flagged as invalid or missing.
   * - ``h_te_median``
     - Retained when value is below the HDF5 fill value or above −999. Removes median terrain height estimates flagged as invalid or missing.
   * - ``h_canopy``
     - Retained when value is below the HDF5 fill value (3.4028235×10³⁸). Removes canopy height estimates flagged as invalid.
   * - ``h_canopy_uncertainty``
     - Retained when value is below the HDF5 fill value. Removes segments where canopy height uncertainty is flagged as invalid.
   * - ``urban_flag``
     - Retained when ``urban_flag == 0`` (non-urban). Urban areas can introduce data artefacts due to complex surface returns and building interference.
   * - ``segment_watermask``
     - Retained when ``segment_watermask == 0`` (non-water). Water surfaces produce unreliable terrain and canopy retrievals.

Sub-segment Fill Handling
-------------------------

For sub-segment (20 m resolution) fields, invalid values are replaced with ``NaN`` rather than dropping the entire 100 m parent segment. This preserves the parent segment while marking individual 20 m values that failed the validity check (fill value ≥ 3.4028235×10³⁸ or ≤ −999) as missing.

The following fields are handled this way:

- ``h_te_best_fit_20m`` — best-fit terrain height at 20 m resolution
- ``h_canopy_20m`` — canopy height at 20 m resolution

.. note::

   Additional quality filters (e.g. on ``night_flag``, ``layer_flag``, ``segment_landcover``, ``n_ca_photons``, ``n_te_photons``) can be applied at query time via the :py:class:`icesat2db.IceSat2Provider` interface without re-ingesting the data. See :ref:`fundamentals-provider` for details.
