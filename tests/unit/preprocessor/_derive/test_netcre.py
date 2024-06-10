"""Test derivation of `netcre`."""
import numpy as np
import pytest
from iris.cube import Cube, CubeList

import esmvalcore.preprocessor._derive.netcre as netcre


@pytest.fixture
def cubes():
    rlut_cube = Cube(
        3, standard_name='toa_outgoing_longwave_flux', units='W m-2'
    )
    rlutcs_cube = Cube(
        1,
        standard_name='toa_outgoing_longwave_flux_assuming_clear_sky',
        units='W m-2',
    )
    rsut_cube = Cube(
        3, standard_name='toa_outgoing_shortwave_flux', units='W m-2'
    )
    rsutcs_cube = Cube(
        1,
        standard_name='toa_outgoing_shortwave_flux_assuming_clear_sky',
        units='W m-2',
    )
    return CubeList([rlut_cube, rlutcs_cube, rsut_cube, rsutcs_cube])


def test_netcre_calculation(cubes):
    """Test calculation of `netcre`."""
    derived_var = netcre.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    np.testing.assert_equal(out_cube.data, -4)
    assert out_cube.units == 'W m-2'
    assert out_cube.attributes['positive'] == 'down'
