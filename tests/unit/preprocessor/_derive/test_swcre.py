"""Test derivation of `swcre`."""
import numpy as np
import pytest
from iris.cube import Cube, CubeList

import esmvalcore.preprocessor._derive.swcre as swcre


@pytest.fixture
def cubes():
    rsut_cube = Cube(
        3, standard_name='toa_outgoing_shortwave_flux', units='W m-2'
    )
    rsutcs_cube = Cube(
        1,
        standard_name='toa_outgoing_shortwave_flux_assuming_clear_sky',
        units='W m-2',
    )
    return CubeList([rsut_cube, rsutcs_cube])


def test_swcre_calculation(cubes):
    """Test calculation of `swcre`."""
    derived_var = swcre.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    np.testing.assert_equal(out_cube.data, -2)
    assert out_cube.units == 'W m-2'
    assert out_cube.attributes['positive'] == 'down'
