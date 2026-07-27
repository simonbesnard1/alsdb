# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

import logging

import tiledb

logger = logging.getLogger(__name__)


def consolidate(uri: str, ctx: tiledb.Ctx | None = None) -> None:
    """
    Consolidate TileDB array fragments.

    Merging small fragments improves read performance after many incremental
    tile ingestions.

    Parameters
    ----------
    uri:
        URI of the TileDB array (local path or ``s3://`` URI).
    ctx:
        TileDB context.  Uses the default context if None.
    """
    logger.info("Consolidating %s", uri)
    cfg = tiledb.Config({"sm.consolidation.step_min_frags": "2"})
    tiledb.consolidate(uri, config=cfg, ctx=ctx)
    logger.info("Consolidation complete: %s", uri)


def vacuum(uri: str, ctx: tiledb.Ctx | None = None) -> None:
    """
    Vacuum a TileDB array, removing obsolete fragment files left after consolidation.

    Parameters
    ----------
    uri:
        URI of the TileDB array.
    ctx:
        TileDB context.  Uses the default context if None.
    """
    logger.info("Vacuuming %s", uri)
    tiledb.vacuum(uri, ctx=ctx)
    logger.info("Vacuum complete: %s", uri)
