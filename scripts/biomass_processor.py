#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 09:31:34 2026

@author: simon
"""

from alsdb import ALSProvider
from alsdb.processing.biomass import compute_biomass, compute_metrics, naesset_model
import functools
import alsdb
alsdb.setup_logging()          # INFO by default

provider = ALSProvider(storage_type="local", uri="array_")

# All structural metrics as separate GeoTIFFs (good for calibration)
compute_metrics(provider, "output/metrics/", resolution=10.0,
                bbox=(655000.0, 8901000.0, 656000.0, 8902000.0))

# AGB with default Næsset model
compute_biomass(provider, "output/agb_10m.tif", resolution=10.0,
                bbox=(655000.0, 8901000.0, 656000.0, 8902000.0),
                n_workers=6)

compute_biomass(provider, "output/agb_100m.tif", resolution=100.0,
                bbox=(655000.0, 8901000.0, 656000.0, 8902000.0),
                n_workers=6)


# AGB with calibrated coefficients
my_model = functools.partial(naesset_model, a=1.2, b=2.1, c=0.6)
compute_biomass(provider, "output/agb.tif", resolution=10.0, model_fn=my_model)



from alsdb.utils.viz_raster import plot_agb
plot_agb("output/agb_100m.tif", cmap="YlGn", vmin=0, vmax=None) 
