#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:21:32 2026

@author: simon
"""

from alsdb import ALSDatabase
import boto3
import alsdb

alsdb.setup_logging()  # INFO by default

# # Ingest a tile locally
db = ALSDatabase(storage_type="local", uri="array_")

# Skips already-ingested files automatically
db.ingest(
    "/home/simon/Documents/science/GFZ/projects/alsdb/data/example_als/brazil_als/RIB_A01_2014_laz_11.laz",
    overwrite=True,
)  # writes + records in manifest

# # Ingest thousands of files at once, auto-consolidates every 50
# db.ingest_many(sorted(Path("/home/simon/Documents/science/GFZ/projects/alsdb/data/example_als/brazil_als/").glob("*.laz")),
#                max_workers=2,
#                consolidate_every=50,
#                overwrite=True)

# Inspect what's been ingested
# db.list_ingested()

# Ingest to S3
session = boto3.Session(profile_name="icesat2db")
frozen = session.get_credentials()
credentials = {"AccessKeyId": frozen.access_key, "SecretAccessKey": frozen.secret_key}

db = ALSDatabase(
    storage_type="s3",
    uri="s3://dog.icesat2db.icesat2-atl08-v007/als_test",
    url="https://s3.gfz-potsdam.de",
    region="eu-central-1",
    credentials=credentials,
)
db.ingest(
    "/home/simon/Documents/science/GFZ/projects/alsdb/data/example_als/"
    "PNOA_2021_CYL-NW_308-4690_ORT-CLA-RGB.laz",
    overwrite=True,
)
