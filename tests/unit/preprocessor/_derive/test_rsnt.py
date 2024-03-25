"""Test derivation of `rsnt`."""
import numpy as np
import pytest
from iris.cube import Cube, CubeList

import esmvalcore.preprocessor._derive.rsnt as rsnt


@pytest.fixture
def cubes():
    rsdt_cube = Cube(
        3, standard_name='toa_incoming_shortwave_flux', units='W m-2'
    )
    rsut_cube = Cube(
        1, standard_name='toa_outgoing_shortwave_flux', units='W m-2'
    )
    return CubeList([rsdt_cube, rsut_cube])


def test_rsnt_calculation(cubes):
    """Test calculation of `rsnt`."""
    derived_var = rsnt.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    np.testing.assert_equal(out_cube.data, 2)
    assert out_cube.units == 'W m-2'
    assert out_cube.attributes['positive'] == 'down'
