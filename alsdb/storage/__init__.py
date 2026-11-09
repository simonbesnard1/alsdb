# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Stores for gridded ALS products."""

from alsdb.storage.zarr_store import ALSZarrStore

__all__ = ["ALSZarrStore"]
