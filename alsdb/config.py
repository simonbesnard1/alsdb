from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class S3Config(BaseModel):
    bucket: str
    prefix: str = ""
    region: str = "eu-central-1"
    endpoint_url: Optional[str] = None   # for MinIO / custom S3
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None


class TileDBConfig(BaseModel):
    tile_extent_x: float = 500.0
    tile_extent_y: float = 500.0
    # Domain covers ETRS89 / UTM Zone 30N (EPSG:25830)
    domain_min_x: float = 100_000.0
    domain_max_x: float = 900_000.0
    domain_min_y: float = 3_000_000.0
    domain_max_y: float = 9_999_900.0
    chunk_size: int = 1_000_000  # points per write chunk


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ALSDB_",
        env_nested_delimiter="__",
    )

    s3: Optional[S3Config] = None
    tiledb: TileDBConfig = Field(default_factory=TileDBConfig)
