"""Test derivation of `rsntcs`."""
import iris
import numpy as np
import pytest

import esmvalcore.preprocessor._derive.rsntcs as rsntcs


@pytest.fixture
def cubes():
    rsdt_name = 'toa_incoming_shortwave_flux'
    rsutcs_name = 'toa_outgoing_shortwave_flux_assuming_clear_sky'
    rsdt_cube = iris.cube.Cube([[1.0, 2.0], [0.0, -2.0]],
                               standard_name=rsdt_name)
    rsutcs_cube = iris.cube.Cube([[5.0, -1.2], [0.8, -3.0]],
                                 standard_name=rsutcs_name)
    ta_cube = iris.cube.Cube([1.0], standard_name='air_temperature')
    return iris.cube.CubeList([rsdt_cube, rsutcs_cube, ta_cube])


def test_rsntcs_calculation(cubes):
    derived_var = rsntcs.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    np.testing.assert_allclose(out_cube.data,
                               np.array([[-4.0, 3.2], [-0.8, 1.0]]))
    assert out_cube.attributes['positive'] == 'down'
