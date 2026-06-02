.. _fundamentals-filters:

####################################
LAS Attributes and Classification
####################################

This page describes the LAS attributes stored by alsDB and the classification scheme used throughout the processing pipeline.

LAS point attributes
--------------------

alsDB stores the following standard LAS attributes for each ingested point.

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Attribute
     - Type
     - Description
   * - ``Z``
     - float64
     - Ellipsoidal height above WGS84 or national vertical datum (m)
   * - ``Intensity``
     - uint16
     - Returned pulse intensity (instrument-dependent scale)
   * - ``ReturnNumber``
     - uint8
     - Which return this point represents (1 = first return)
   * - ``NumberOfReturns``
     - uint8
     - Total number of discrete returns for this pulse
   * - ``ScanDirectionFlag``
     - uint8
     - 0 = negative scan direction, 1 = positive scan direction
   * - ``EdgeOfFlightLine``
     - uint8
     - 1 if this point is at the edge of the flight line
   * - ``Classification``
     - uint8
     - ASPRS LAS classification code (see table below)
   * - ``ScanAngleRank``
     - float32
     - Scan angle in degrees relative to nadir (positive = right of flight)
   * - ``UserData``
     - uint8
     - User-defined byte
   * - ``PointSourceId``
     - uint16
     - Source file ID (useful for multi-file datasets)
   * - ``R``, ``G``, ``B``
     - uint16
     - RGB colour (0–65535), if present in the source file

ASPRS LAS classification codes
-------------------------------

The ``Classification`` attribute uses the ASPRS standard codes:

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Code
     - Meaning
   * - 0
     - Never classified
   * - 1
     - Unclassified
   * - 2
     - Ground
   * - 3
     - Low vegetation (height < 2 m)
   * - 4
     - Medium vegetation (2–5 m)
   * - 5
     - High vegetation (> 5 m)
   * - 6
     - Building
   * - 7
     - Low point (noise)
   * - 8
     - Model key-point
   * - 9
     - Water
   * - 10
     - Rail
   * - 11
     - Road surface
   * - 12–14
     - Reserved
   * - 15–17
     - Wire, transmission tower, connector wire
   * - 18
     - High noise
   * - 19+
     - Overlap / user-defined

Filtering during ingestion
--------------------------

Pass a list of class codes to ``ingest()`` or ``ingest_many()`` to store only specific point classes:

.. code-block:: python

    # Store only ground (2) and vegetation (3, 4, 5)
    db.ingest("tile.laz", classes=[2, 3, 4, 5])

    # Store everything (default is no filter)
    db.ingest("tile.laz")

Reducing the stored classes saves array space and speeds up both ingestion and later queries. For most forest structure applications, classes 2–5 are sufficient.

Role of classification in processing
--------------------------------------

The processing pipeline uses ``Classification`` and ``ReturnNumber`` directly:

**HAG computation (all processing functions)**
  ``filters.hag_delaunay`` uses ground points (``Classification == 2``) to build a Delaunay TIN, from which height-above-ground is interpolated for every other point.

**Gap fraction** (``alsdb.processing.gap``)
  The MacArthur–Wilson estimator uses first returns only (``ReturnNumber == 1``):

  .. math::

     P_\text{gap} = \frac{N_\text{ground, first}}{N_\text{ground, first} + N_\text{vegetation, first}}

  where vegetation = ``Classification`` in {3, 4, 5}.

**Structural metrics / biomass** (``alsdb.processing.biomass``)
  Vegetation points (``Classification`` ≥ 3, or ``HeightAboveGround`` above the minimum threshold) are used for height percentiles (h50, h75, h95, hmean) and canopy cover.

**CHM / DTM / DSM** (``alsdb.processing.chm``)
  DTM interpolates ground points (``Classification == 2``) using a Delaunay TIN
  (``dtm_method="tin"``, default), IDW (``"idw"``), or minimum-Z binning (``"min"``).
  DSM rasterises all first returns to the maximum ``Z``.
  CHM is computed from height-above-ground of vegetation first returns after HAG
  normalisation against the Delaunay TIN.

**Ground point outlier rejection** (all processing functions)
  Before the Delaunay TIN is built, ground points are screened for outliers using
  ``filters.elm`` (Extended Local Minimum — detects below-ground multipath returns)
  followed by ``filters.outlier`` (statistical — detects above-ground misclassified
  objects). Outlier ground points are reclassified to Class 1 (unclassified) so the
  TIN is never corrupted by bad ground returns. Vegetation classification is
  untouched by this step.

.. note::

   If a dataset uses non-standard class codes (e.g. some national campaigns use code 8 for ground), the HAG computation will be wrong. Verify the classification scheme of your input data and re-classify if necessary before ingestion.

.. note::

   ``ScanAngleRank`` (degrees, stored as float32) is available in the schema for
   future scan-angle normalisation of canopy-cover and density metrics.  Current
   metric computation does not apply this correction; apply a scan-angle filter
   (e.g. ``|ScanAngleRank| ≤ 10°``) during ingestion if your dataset combines
   flight lines with very different scan angles.
