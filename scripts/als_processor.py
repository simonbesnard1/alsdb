#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:21:32 2026

@author: simon
"""
from alsdb import ALSDatabase
import boto3
from alsdb import ALSProvider
from alsdb.utils.viz import plot_overview, plot_dsm, plot_rgb

# Ingest a tile locally
db = ALSDatabase(storage_type="local", uri="array_")
db.ingest("/home/simon/Documents/science/GFZ/projects/alsdb/data/example_als/PNOA_2021_CYL-NW_308-4690_ORT-CLA-RGB.laz", 
          classification_filter=[2])

# Ingest to S3
session = boto3.Session(profile_name="alsdb")
frozen = session.get_credentials().get_frozen_credentials()
credentials = {
    "AccessKeyId": frozen.access_key,
    "SecretAccessKey": frozen.secret_key,
}

db = ALSDatabase(
    storage_type="s3",
    uri="dog-proj-3d-abc-qian-song.new-bucket-2f37f541/test",   # ← correct bucket name
    url="https://s3.gfz-potsdam.de",
    region="eu-central-1",
    credentials=credentials,
)
db.ingest(
    "/home/simon/Documents/science/GFZ/projects/alsdb/data/example_als/"
    "PNOA_2021_CYL-NW_308-4690_ORT-CLA-RGB.laz"
)

# Query
provider = ALSProvider(storage_type="s3", 
                       uri="dog-proj-3d-abc-qian-song.new-bucket-2f37f541/test",
                       url="https://s3.gfz-potsdam.de",
                       region="eu-central-1",
                       )
df = provider.query_tile(308, 4690)
ds = provider.to_xarray(308_000, 4_688_000, 310_000, 4_690_000)

# Full 4-panel overview
fig = plot_overview(df, resolution=1.0)
fig.savefig("tile_308_4690.png", dpi=150, bbox_inches="tight")

# Individual panels
plot_dsm(df, resolution=1.0, hillshade=True)
plot_rgb(df, resolution=1.0)
