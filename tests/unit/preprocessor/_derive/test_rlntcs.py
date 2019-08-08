"""Test derivation of `rlntcs`."""
import iris
import numpy as np
import pytest

import esmvalcore.preprocessor._derive.rlntcs as rlntcs


@pytest.fixture
def cubes():
    std_name = 'toa_outgoing_longwave_flux_assuming_clear_sky'
    rlutcs_cube = iris.cube.Cube([[1.0, 2.0], [0.0, -2.0]],
                                 standard_name=std_name)
    ta_cube = iris.cube.Cube([1.0], standard_name='air_temperature')
    return iris.cube.CubeList([rlutcs_cube, ta_cube])


def test_rlntcs_calculation(cubes):
    derived_var = rlntcs.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    np.testing.assert_allclose(out_cube.data,
                               np.array([[-1.0, -2.0], [0.0, 2.0]]))
    assert out_cube.attributes['positive'] == 'down'
