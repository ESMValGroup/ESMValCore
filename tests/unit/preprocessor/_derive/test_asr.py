"""Test derivation of `asr`."""
import iris
import numpy as np
import pytest

import esmvalcore.preprocessor._derive.asr as asr


@pytest.fixture
def cubes():
    rsdt_name = 'toa_incoming_shortwave_flux'
    rsut_name = 'toa_outgoing_shortwave_flux'
    rsdt_cube = iris.cube.Cube([[1.0, 2.0], [0.0, -2.0]],
                               standard_name=rsdt_name)
    rsut_cube = iris.cube.Cube([[7.0, 0.0], [-1.0, 5.0]],
                               standard_name=rsut_name)
    ta_cube = iris.cube.Cube([1.0], standard_name='air_temperature')
    return iris.cube.CubeList([rsdt_cube, rsut_cube, ta_cube])


def test_asr_calculation(cubes):
    derived_var = asr.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    np.testing.assert_allclose(out_cube.data,
                               np.array([[-6.0, 2.0], [1.0, -7.0]]))
    assert out_cube.attributes['positive'] == 'down'
