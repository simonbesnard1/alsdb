# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from alsdb.core.alsdatabase import ALSDatabase
from alsdb.core.classification import ClassificationConfig
from alsdb.utils.schema import TileDBSchemaConfig


@click.group()
@click.option("--verbose", "-v", is_flag=True)
def main(verbose: bool) -> None:
    """alsdb — ALS point cloud ingestion and query tool."""
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@main.command("ingest")
@click.argument("laz_path", type=click.Path(exists=True, path_type=Path))
@click.argument("array_uri")
@click.option(
    "--storage-type",
    default="local",
    show_default=True,
    type=click.Choice(["local", "s3"]),
    help="Storage backend.",
)
@click.option(
    "--s3-url",
    envvar="ALSDB_S3_URL",
    default=None,
    help="S3 endpoint URL (required for s3 storage).",
)
@click.option("--s3-region", envvar="ALSDB_S3_REGION", default="eu-central-1", show_default=True)
@click.option("--s3-access-key", envvar="ALSDB_S3_ACCESS_KEY", default=None)
@click.option("--s3-secret-key", envvar="ALSDB_S3_SECRET_KEY", default=None)
@click.option(
    "--tile-extent",
    default=500.0,
    show_default=True,
    help="Spatial tile size in CRS units (metres).",
)
@click.option("--chunk-size", default=1_000_000, show_default=True, help="Points per write batch.")
@click.option("--domain-min-x", default=100_000.0, show_default=True)
@click.option("--domain-max-x", default=900_000.0, show_default=True)
@click.option("--domain-min-y", default=3_000_000.0, show_default=True)
@click.option("--domain-max-y", default=9_999_900.0, show_default=True)
@click.option(
    "--classification",
    "-c",
    multiple=True,
    type=int,
    help="Keep only these classification codes (repeatable). E.g. -c 2 -c 5",
)
@click.option("--classify", is_flag=True, help="Classify ground and vegetation during ingestion.")
@click.option(
    "--ground-method",
    type=click.Choice(["csf", "pmf", "smrf"]),
    default="csf",
    show_default=True,
    help="Ground classifier used with --classify.",
)
@click.option(
    "--denoise", is_flag=True, help="Classify low and isolated noise before ground fitting."
)
@click.option(
    "--vegetation-thresholds",
    type=(float, float, float),
    default=(0.5, 2.0, 5.0),
    show_default=True,
    metavar="LOW MEDIUM HIGH",
    help="HAG thresholds in metres for vegetation classes 3, 4, and 5.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Delete existing array before writing.",
)
def ingest_cmd(
    laz_path: Path,
    array_uri: str,
    storage_type: str,
    s3_url: str | None,
    s3_region: str,
    s3_access_key: str | None,
    s3_secret_key: str | None,
    tile_extent: float,
    chunk_size: int,
    domain_min_x: float,
    domain_max_x: float,
    domain_min_y: float,
    domain_max_y: float,
    classification: tuple,
    classify: bool,
    ground_method: str,
    denoise: bool,
    vegetation_thresholds: tuple[float, float, float],
    overwrite: bool,
) -> None:
    """Ingest a PNOA LAZ tile into a TileDB array at ARRAY_URI."""
    credentials: dict[str, str] | None = None
    if s3_access_key and s3_secret_key:
        credentials = {"AccessKeyId": s3_access_key, "SecretAccessKey": s3_secret_key}

    schema_cfg = TileDBSchemaConfig(
        tile_extent_x=tile_extent,
        tile_extent_y=tile_extent,
        domain_min_x=domain_min_x,
        domain_max_x=domain_max_x,
        domain_min_y=domain_min_y,
        domain_max_y=domain_max_y,
        chunk_size=chunk_size,
    )

    db = ALSDatabase(
        storage_type=storage_type,
        uri=array_uri,
        schema_cfg=schema_cfg,
        url=s3_url,
        region=s3_region,
        credentials=credentials,
    )

    low, medium, high = vegetation_thresholds
    classification_config = ClassificationConfig(
        enabled=classify,
        ground_method=ground_method,
        denoise=denoise,
        vegetation_low=low,
        vegetation_medium=medium,
        vegetation_high=high,
    )
    db.ingest(
        laz_path,
        classification_filter=list(classification) if classification else None,
        classification_config=classification_config,
        overwrite=overwrite,
    )
    click.echo(f"Ingested {laz_path.name} → {array_uri}")


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


@main.command("info")
@click.argument("laz_path", type=click.Path(exists=True, path_type=Path))
def info_cmd(laz_path: Path) -> None:
    """Show filename metadata and bounding box for a PNOA LAZ tile."""
    from alsdb.tile.tile import Tile

    tile = Tile(laz_path)
    name = tile.name
    bbox = tile.native_bbox

    click.echo(f"File    : {laz_path.name}")
    click.echo(f"Year    : {name.year}")
    if hasattr(name, "region"):
        click.echo(f"Region  : {name.region}")
    if hasattr(name, "tile_x_km"):
        click.echo(f"Tile    : {name.tile_x_km} km E  /  {name.tile_y_km} km N")
    if hasattr(name, "product"):
        click.echo(f"Product : {name.product}")
    click.echo(f"BBox    : X [{bbox[0]:.0f} – {bbox[2]:.0f}]  Y [{bbox[1]:.0f} – {bbox[3]:.0f}]")
    click.echo(f"Points  : {tile.n_points:,}")
