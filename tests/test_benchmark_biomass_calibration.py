"""alsdb.benchmark.biomass_calibration: fit, cross-validation and screening on a known model."""

import numpy as np
import pandas as pd
import pytest

from alsdb.benchmark.biomass_calibration import calibrate


def test_calibrate_recovers_a_known_model_and_flags_mismatches():
    rng = np.random.default_rng(0)
    n = 600
    h95 = rng.uniform(3, 25, n)
    cc = rng.uniform(0.2, 1.0, n)
    agb = 2.0 * h95**1.2 * cc**0.8 * rng.lognormal(0, 0.05, n)
    table = pd.DataFrame({
        "x": rng.uniform(0, 60_000, n), "y": rng.uniform(0, 60_000, n),
        "h95": h95, "cc": cc, "AGBD": agb, "n_points": 1000,
        "hmax": h95 + 2, "Height_max": h95 + 2,
    })  # fmt: skip
    table.loc[:19, "Height_max"] += 15  # field far above the ALS canopy: flagged
    table.loc[20:29, "n_points"] = 50  # too few points: unusable
    t, results, settings = calibrate(table)
    assert settings["n_usable"] == n - 10 and settings["n_mismatch"] == 20
    for name in ("all", "screened"):
        r = results[name]
        # a and the h95 exponent trade off against each other: a gets a wider tolerance
        assert r["a"] == pytest.approx(2.0, rel=0.1)
        assert (r["b"], r["c"]) == pytest.approx((1.2, 0.8), rel=0.05)
        assert r["cv_r2"] > 0.95 and abs(r["cv_bias"]) < 1.0
    assert results["screened"]["n"] == n - 30
    assert t["cv_pred_all"].notna().sum() == n - 10
