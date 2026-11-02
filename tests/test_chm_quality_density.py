"""First-return TIN and pulse-density / scan-angle quality (Fischer et al. 2024)."""

import numpy as np
import pytest

from alsdb.processing._surface import (
    HIGH_SCAN_ANGLE,
    LOW_PULSE_DENSITY,
    CHMConfig,
    provenance_settings,
    pulse_density_grid,
    reconstruct_chm,
)


def _scene_with_angles():
    from tests.test_surface_pipeline import scene

    arr = scene()
    out = np.zeros(len(arr), dtype=[*arr.dtype.descr, ("ScanAngleRank", "f4")])
    for name in arr.dtype.names:
        out[name] = arr[name]
    out["ScanAngleRank"] = np.where(out["X"] < 5, -10.0, 25.0)
    return out


def test_first_return_tin_follows_the_canopy():
    result = reconstruct_chm(_scene_with_angles(), (0, 0, 10, 10), 1, CHMConfig(method="tin"))
    np.testing.assert_allclose(result["chm"], 10, atol=1e-5)


def test_pulse_density_grid_on_aligned_cells():
    pulses = np.zeros(50, dtype=[("X", "f8"), ("Y", "f8")])
    pulses["X"], pulses["Y"] = 101.0, 108.0
    density, x0, y1 = pulse_density_grid(pulses, (101.0, 101.0, 109.0, 109.0))
    assert (x0, y1) == (100.0, 110.0)
    assert density[0, 0] == pytest.approx(2.0) and density[1, 1] == 0


def test_density_and_scan_angle_layers_and_flags():
    result = reconstruct_chm(_scene_with_angles(), (0, 0, 10, 10), 1, CHMConfig(method="max"))
    density, angle = result["chm_pulse_density"], result["chm_scan_angle"]
    flags = result["chm_quality"].astype(np.uint16)
    # One canopy return per m2 in the scene, plus the 2 m ground lattice.
    assert np.nanmin(density) > 1.0 and np.nanmax(density) < 2.0
    np.testing.assert_allclose(angle[:, :5], 10)
    np.testing.assert_allclose(angle[:, 5:], 25)
    assert (flags[:, 5:] & HIGH_SCAN_ANGLE).all() and not (flags[:, :5] & HIGH_SCAN_ANGLE).any()
    # Below the 2 pulses/m2 default everywhere; flags disabled with None.
    assert (flags & LOW_PULSE_DENSITY).all()
    quiet = reconstruct_chm(
        _scene_with_angles(),
        (0, 0, 10, 10),
        1,
        CHMConfig(method="max", min_pulse_density=None, max_scan_angle=None),
    )["chm_quality"].astype(np.uint16)
    assert not (quiet & (LOW_PULSE_DENSITY | HIGH_SCAN_ANGLE)).any()


def test_new_settings_at_defaults_keep_the_provenance_fingerprint():
    settings = provenance_settings(CHMConfig(method="spikefree"))
    assert not {"spikefree_first_returns", "min_pulse_density", "max_scan_angle"} & set(settings)
    changed = provenance_settings(CHMConfig(method="spikefree", spikefree_first_returns=True))
    assert changed["spikefree_first_returns"] is True
