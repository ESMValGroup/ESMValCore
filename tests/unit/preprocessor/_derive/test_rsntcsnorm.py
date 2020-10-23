"""Test derivation of `rsntcs`."""
import iris
import numpy as np
import pytest

import esmvalcore.preprocessor._derive.rsnstcsnorm as rsnstcsnorm


@pytest.fixture
def cubes():
    # names
    rsdscs_name = \
        'surface_downwelling_shortwave_flux_in_air_assuming_clear_sky'
    rsdt_name = 'toa_incoming_shortwave_flux'
    rsuscs_name = 'surface_upwelling_shortwave_flux_in_air_assuming_clear_sky'
    rsutcs_name = 'toa_outgoing_shortwave_flux_assuming_clear_sky'

    # cubes
    rsdscs_cube = iris.cube.Cube([[1.0, 2.0], [0.0, -2.0]],
                                 standard_name=rsdscs_name)
    rsdt_cube = iris.cube.Cube([[1.0, 2.0], [2.0, -2.0]],
                               standard_name=rsdt_name)
    rsuscs_cube = iris.cube.Cube([[1.0, 2.0], [0.0, -2.0]],
                                 standard_name=rsuscs_name)
    rsutcs_cube = iris.cube.Cube([[5.0, -1.2], [0.8, -3.0]],
                                 standard_name=rsutcs_name)
    return iris.cube.CubeList([rsdscs_cube, rsdt_cube,
                               rsuscs_cube, rsutcs_cube])


def test_rsntcs_calculation(cubes):
    # Actual derivation of rsnstcsnorm
    # rsnstcsnorm_cube = (((rsdt_cube - rsutcs_cube) -
    #                      (rsdscs_cube - rsuscs_cube)) / rsdt_cube) * 100.0
    derived_var = rsnstcsnorm.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    np.testing.assert_allclose(out_cube.data,
                               np.array([[-400.,  160.], [60., -50.0]]))
    assert out_cube.units == '%'
