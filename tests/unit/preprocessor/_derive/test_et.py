"""Test derivation of `et`."""
import iris
import numpy as np
import pytest
from cf_units import Unit

import esmvalcore.preprocessor._derive.et as et


@pytest.fixture
def cubes():
    hfls_cube = iris.cube.Cube([[1.0, 2.0], [0.0, -2.0]],
                               standard_name='surface_upward_latent_heat_flux')
    ta_cube = iris.cube.Cube([1.0], standard_name='air_temperature')
    return iris.cube.CubeList([hfls_cube, ta_cube])


def test_et_calculation(cubes):
    derived_var = et.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    np.testing.assert_allclose(
        out_cube.data, np.array([[0.03505071, 0.07010142], [0.0,
                                                            -0.07010142]]))
    assert out_cube.units == Unit('mm day-1')
