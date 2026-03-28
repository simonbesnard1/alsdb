#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 00:32:51 2026

@author: simon
"""

from alsdb import ALSProvider
from alsdb.processing.chm import compute_chm, compute_all
import boto3


session = boto3.Session(profile_name="alsdb")
frozen = session.get_credentials().get_frozen_credentials()
credentials = {
    "AccessKeyId": frozen.access_key,
    "SecretAccessKey": frozen.secret_key,
}

provider = ALSProvider(
    storage_type="s3",
    uri="s3://dog-proj-3d-abc-qian-song.new-bucket-2f37f541/test",
    url="https://s3.gfz-potsdam.de",
    region="eu-central-1",
    credentials=credentials,
)

# CHM for the full array
compute_chm(provider, "output/chm.tif", resolution=1.0)

# Restrict to one PNOA tile.
compute_all(
    provider,
    output_dir="output/",
    resolution=1.0,
    bbox=(308000, 4688000, 310000, 4690000),
)
