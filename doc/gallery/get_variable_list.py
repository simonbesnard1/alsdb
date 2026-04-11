# SPDX-License-Identifier: EUPL-1.2
# Contact: felixd@gfz.de, besnard@gfz.de, urbazaev@gfz.de and amelia.holcomb@gmail.com.
# SPDX-FileCopyrightText: 2026 Felix Dombrowski
# SPDX-FileCopyrightText: 2026 Mikhail Urbazaev
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Amelia Holcomb
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences


"""
Retrieve List of Variables in TileDB Array
==========================================

This example demonstrates how to retrieve the list of variables stored in a TileDB array
using the `IceSat2Provider` class from the `icesat2db` package. The output is a `pandas.DataFrame`
containing variable names, descriptions, and metadata.

Prerequisites:
--------------
1. ICESat-2 data must be processed and stored in TileDB arrays.
2. Configure the TileDB storage backend (local or S3).

Steps:
------
1. Configure the TileDB backend (local or S3).
2. Initialize the `IceSat2Provider`.
3. Retrieve and display the list of available variables.
"""

import icesat2db as isdb

# Step 1: Configure the TileDB storage backend
storage_type = "local"  # Options: "local" or "s3"
local_path = "/path/to/processed/icesat2/data"  # Update with your local TileDB path
s3_bucket = None  # Set the S3 bucket name if using S3 storage

# Step 2: Initialize the IceSat2Provider
provider = isdb.IceSat2Provider(
    storage_type=storage_type,
    local_path=local_path,
    s3_bucket=s3_bucket,
)

# Step 3: Retrieve the list of available variables in the TileDB array
variables_df = provider.get_available_variables()

# Display the resulting DataFrame
print(variables_df)
