"""Test derivation of `rlus`."""
import iris
import numpy as np
import pytest

import esmvalcore.preprocessor._derive.rlus as rlus

from .test_shared import get_cube


@pytest.fixture
def cubes():
    rlds_name = 'surface_downwelling_longwave_flux_in_air'
    rlns_name = 'surface_net_downward_longwave_flux'
    rlds_cube = get_cube([[[100.]]],
                         air_pressure_coord=False,
                         standard_name=rlds_name)
    rlds_cube.attributes["positive"] = "down"
    rlns_cube = get_cube([[[50.0]]],
                         air_pressure_coord=False,
                         standard_name=rlns_name)
    rlns_cube.attributes["positive"] = "down"

    rlns_cube.coord("longitude").var_name = "lon"
    rlns_cube.coord("longitude").var_name = "lat"

    return iris.cube.CubeList([rlds_cube, rlns_cube])


def test_rlntcs_calculation(cubes):
    derived_var = rlus.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    np.testing.assert_allclose(out_cube.data,
                               np.array([[[50.0]]]))
    assert out_cube.attributes['positive'] == 'up'
