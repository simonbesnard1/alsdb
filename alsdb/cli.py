from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from .config import S3Config, TileDBConfig
from .pipeline import ingest as _ingest
from .reader import get_native_bbox, read_metadata


@click.group()
@click.option("--verbose", "-v", is_flag=True)
def main(verbose: bool) -> None:
    """alsdb — ALS point cloud ingestion tool."""
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


@main.command("ingest")
@click.argument("laz_path", type=click.Path(exists=True, path_type=Path))
@click.argument("array_name")
@click.option("--s3-bucket", envvar="ALSDB_S3_BUCKET", default=None, help="S3 bucket name")
@click.option("--s3-prefix", envvar="ALSDB_S3_PREFIX", default="", help="Key prefix within bucket")
@click.option("--s3-region", envvar="ALSDB_S3_REGION", default="eu-central-1", show_default=True)
@click.option("--s3-endpoint", envvar="ALSDB_S3_ENDPOINT", default=None, help="Custom S3 endpoint (MinIO etc.)")
@click.option("--s3-access-key", envvar="ALSDB_S3_ACCESS_KEY", default=None)
@click.option("--s3-secret-key", envvar="ALSDB_S3_SECRET_KEY", default=None)
@click.option("--tile-extent", default=500.0, show_default=True, help="Spatial tile size in CRS units")
@click.option("--chunk-size", default=1_000_000, show_default=True, help="Points per write batch")
@click.option("--domain-min-x", default=100_000.0, show_default=True)
@click.option("--domain-max-x", default=900_000.0, show_default=True)
@click.option("--domain-min-y", default=3_000_000.0, show_default=True)
@click.option("--domain-max-y", default=9_999_900.0, show_default=True)
@click.option("--overwrite", is_flag=True, default=False, help="Delete existing array before writing")
def ingest_cmd(
    laz_path: Path,
    array_name: str,
    s3_bucket: Optional[str],
    s3_prefix: str,
    s3_region: str,
    s3_endpoint: Optional[str],
    s3_access_key: Optional[str],
    s3_secret_key: Optional[str],
    tile_extent: float,
    chunk_size: int,
    domain_min_x: float,
    domain_max_x: float,
    domain_min_y: float,
    domain_max_y: float,
    overwrite: bool,
) -> None:
    """Ingest LAZ_PATH into a TileDB array named ARRAY_NAME."""
    s3 = (
        S3Config(
            bucket=s3_bucket,
            prefix=s3_prefix,
            region=s3_region,
            endpoint_url=s3_endpoint,
            access_key_id=s3_access_key,
            secret_access_key=s3_secret_key,
        )
        if s3_bucket
        else None
    )
    tdb_cfg = TileDBConfig(
        tile_extent_x=tile_extent,
        tile_extent_y=tile_extent,
        domain_min_x=domain_min_x,
        domain_max_x=domain_max_x,
        domain_min_y=domain_min_y,
        domain_max_y=domain_max_y,
        chunk_size=chunk_size,
    )
    uri = _ingest(laz_path, array_name, tdb_cfg, s3, overwrite=overwrite)
    click.echo(uri)


@main.command("info")
@click.argument("laz_path", type=click.Path(exists=True, path_type=Path))
def info_cmd(laz_path: Path) -> None:
    """Print metadata and bounding box for a LAZ/LAS file."""
    import json
    meta = read_metadata(laz_path)
    click.echo(json.dumps(meta, indent=2))
