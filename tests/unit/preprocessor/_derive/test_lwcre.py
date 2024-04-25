"""Test derivation of `lwcre`."""
import numpy as np
import pytest
from iris.cube import Cube, CubeList

import esmvalcore.preprocessor._derive.lwcre as lwcre


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
    return CubeList([rlut_cube, rlutcs_cube])


def test_lwcre_calculation(cubes):
    """Test calculation of `lwcre`."""
    derived_var = lwcre.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    np.testing.assert_equal(out_cube.data, -2)
    assert out_cube.units == 'W m-2'
    assert out_cube.attributes['positive'] == 'down'
