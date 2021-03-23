"""Test derivation of `rsus`."""
import iris
import numpy as np
import pytest

import esmvalcore.preprocessor._derive.rsus as rsus

from .test_shared import get_cube


@pytest.fixture
def cubes():
    rsds_name = 'surface_downwelling_shortwave_flux_in_air'
    rsns_name = 'surface_net_downward_shortwave_flux'
    rsds_cube = get_cube([[[100.]]],
                         air_pressure_coord=False,
                         standard_name=rsds_name)
    rsds_cube.attributes["positive"] = "down"
    rsns_cube = get_cube([[[50.0]]],
                         air_pressure_coord=False,
                         standard_name=rsns_name)
    rsns_cube.attributes["positive"] = "down"

    rsns_cube.coord("longitude").var_name = "lon"
    rsns_cube.coord("longitude").var_name = "lat"

    return iris.cube.CubeList([rsds_cube, rsns_cube])


def test_rsntcs_calculation(cubes):
    derived_var = rsus.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    np.testing.assert_allclose(out_cube.data,
                               np.array([[[50.0]]]))
    assert out_cube.attributes['positive'] == 'up'
