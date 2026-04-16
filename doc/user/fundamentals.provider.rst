.. for doctest:
    >>> import alsdb

.. _fundamentals-provider:

#############
Data Provider
#############

:py:class:`alsdb.ALSProvider` is the interface for querying point clouds stored in the TileDB array. It supports spatial queries by bounding box and year, returning results as ``pandas.DataFrame`` or ``xarray.Dataset``.

Key capabilities
----------------

- **Spatial queries**: retrieve all points within a bounding box.
- **Temporal filtering**: restrict queries to a single survey year.
- **Attribute selection**: choose which LAS attributes to return.
- **Multi-year inspection**: list all years stored in the array.
- **xarray output**: get results as a labelled ``xarray.Dataset``.

Basic query example
-------------------

.. code-block:: python

    from alsdb import ALSProvider

    reader = ALSProvider(storage_type="local", uri="my_array")

    # All points in a bounding box. It returns pandas DataFrame
    df = reader.query_bbox(308_000, 4_688_000, 310_000, 4_690_000)

    print(df.columns.tolist())
    # ['X', 'Y', 'Z', 'Intensity', 'ReturnNumber', 'Classification',
    #  'R', 'G', 'B', 'HeightAboveGround', 'Year', ...]

    # Restrict to a single survey year
    df = reader.query_bbox(308_000, 4_688_000, 310_000, 4_690_000, year=2021)

    # Which years are available?
    print(reader.available_years())   # e.g. [2019, 2021, 2023]

Selecting attributes
--------------------

By default all stored attributes are returned. Pass ``attrs`` to restrict the output:

.. code-block:: python

    df = reader.query_bbox(
        308_000, 4_688_000, 310_000, 4_690_000,
        year=2021,
        attrs=["Z", "Classification", "HeightAboveGround"],
    )

To see what attributes are available:

.. code-block:: python

    print(reader.get_available_attributes())

xarray output
-------------

.. code-block:: python

    ds = reader.to_xarray(308_000, 4_688_000, 310_000, 4_690_000, year=2021)

    print(ds)
    # <xarray.Dataset>
    # Dimensions: (points: 1_482_310)
    # Coordinates:
    #   * X     (points) float64
    #   * Y     (points) float64
    # Data variables:
    #     Z                (points) float64
    #     Classification   (points) uint8
    #     ReturnNumber     (points) uint8
    #     ...

S3 provider
-----------

The API is identical for S3-backed arrays; only the constructor arguments change:

.. code-block:: python

    reader = ALSProvider(
        storage_type="s3",
        uri="s3://owner.bucket/als_array",
        url="https://s3.example.com",
        region="eu-central-1",
        credentials={
            "AccessKeyId": "...",
            "SecretAccessKey": "...",
        },
    )

    df = reader.query_bbox(308_000, 4_688_000, 310_000, 4_690_000, year=2021)

See :ref:`fundamentals-s3` for a full S3 setup walkthrough.

Output columns
--------------

The following columns are available depending on what was stored during ingestion:

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Column
     - Type
     - Description
   * - ``X``
     - float64
     - UTM easting (m)
   * - ``Y``
     - float64
     - UTM northing (m)
   * - ``Z``
     - float64
     - Ellipsoidal height (m)
   * - ``HeightAboveGround``
     - float32
     - Height above the Delaunay ground TIN (m); available if ingested with HAG filter
   * - ``Intensity``
     - uint16
     - Return pulse intensity
   * - ``ReturnNumber``
     - uint8
     - Return number (1 = first, 2 = second, …)
   * - ``NumberOfReturns``
     - uint8
     - Total number of returns for this pulse
   * - ``Classification``
     - uint8
     - LAS classification code (1=unclassified, 2=ground, 3=low veg, 4=medium veg, 5=high veg, …)
   * - ``R``, ``G``, ``B``
     - uint16
     - RGB colour values (0–65535)
   * - ``Year``
     - int16
     - Survey acquisition year

Performance notes
-----------------

- Queries scan only the TileDB fragments that overlap the requested ``X``/``Y``/``Year`` range. Fragment consolidation (see :ref:`fundamentals-processor`) significantly reduces per-query overhead for large arrays.
- For very large bounding boxes, consider using :ref:`fundamentals-tiledb-processing` (``run_tiled``) rather than a single ``query_bbox`` call.
- On S3, query performance depends on network bandwidth and TileDB S3 timeout settings.
