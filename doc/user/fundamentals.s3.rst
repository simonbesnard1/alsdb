.. _fundamentals-s3:

#############
S3 Storage
#############

Both the TileDB point-cloud array and the Zarr gridded store can be hosted on any S3-compatible object storage — AWS S3, Ceph/RadosGW, MinIO, or others. The API is identical to local storage; only the constructor arguments change.

TileDB array on S3
------------------

Pass ``storage_type="s3"`` and supply the endpoint URL and credentials as a dictionary:

.. code-block:: python

    from alsdb import ALSDatabase, ALSProvider

    s3_kwargs = dict(
        storage_type="s3",
        uri="s3://my-bucket/als_array",
        url="https://s3.example.com",       # endpoint URL (omit for AWS)
        region="eu-central-1",
        credentials={
            "AccessKeyId": "YOUR_ACCESS_KEY",
            "SecretAccessKey": "YOUR_SECRET_KEY",
        },
    )

    # Ingest
    db = ALSDatabase(**s3_kwargs)
    db.ingest("path/to/tile.laz")

    # Query
    reader = ALSProvider(**s3_kwargs)
    df = reader.query_bbox(308_000, 4_688_000, 310_000, 4_690_000, year=2021)

Using boto3 to load credentials
--------------------------------

If you manage credentials via AWS profiles or environment variables, use ``boto3`` to retrieve them:

.. code-block:: python

    import boto3

    session = boto3.Session(profile_name="my-profile")
    creds = session.get_credentials().get_frozen_credentials()

    db = ALSDatabase(
        storage_type="s3",
        uri="s3://my-bucket/als_array",
        url="https://s3.example.com",
        region="eu-central-1",
        credentials={
            "AccessKeyId": creds.access_key,
            "SecretAccessKey": creds.secret_key,
        },
    )

Zarr store on S3
----------------

``ALSZarrStore`` accepts an ``s3://`` URI directly. Pass ``storage_options`` with your S3 credentials:

.. code-block:: python

    from alsdb.storage import ALSZarrStore

    store = ALSZarrStore(
        "s3://my-bucket/forest.zarr",
        storage_options={
            "key": "YOUR_ACCESS_KEY",
            "secret": "YOUR_SECRET_KEY",
            "endpoint_url": "https://s3.example.com",
        },
    )

    # Use exactly like a local store
    from alsdb.processing.chm import compute_chm
    compute_chm(provider=reader, store=store, resolution=1.0, year=2021)

TileDB S3 performance tuning
------------------------------

For large-scale campaigns on S3, tune the TileDB S3 settings via the ``tiledb_config`` argument:

.. code-block:: python

    db = ALSDatabase(
        storage_type="s3",
        uri="s3://my-bucket/als_array",
        url="https://s3.example.com",
        region="eu-central-1",
        credentials={
            "AccessKeyId": "...",
            "SecretAccessKey": "...",
        },
        tiledb_config={
            "vfs.s3.connect_timeout_ms": "300000",
            "vfs.s3.request_timeout_ms": "600000",
            "vfs.s3.connect_max_tries": "10",
            "vfs.s3.multipart_part_size": "52428800",   # 50 MB parts
            "vfs.s3.backoff_scale": "2.0",
            "vfs.s3.backoff_max_ms": "120000",
        },
    )

Recommended settings for production S3 ingest:

- ``multipart_part_size``: 50–100 MB for large datasets.
- ``connect_max_tries``: 10 retries for transient S3 errors.
- ``request_timeout_ms``: 600 000 ms (10 min) for slow object stores.

Security considerations
-----------------------

- Never hardcode credentials in scripts. Use environment variables or an AWS credentials file.
- Restrict bucket policies so that only the ingest user has write access; read-only access can be wider.
- For public datasets, use anonymous access (``"aws.skip_credentials": "true"``).

.. code-block:: python

    # Anonymous read-only access (public bucket)
    reader = ALSProvider(
        storage_type="s3",
        uri="s3://public-bucket/als_array",
        tiledb_config={"vfs.s3.no_sign_request": "true"},
    )
