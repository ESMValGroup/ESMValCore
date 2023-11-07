"""Test derivation of ``xch4``."""
import iris
import numpy as np
import pytest

import esmvalcore.preprocessor._derive.xch4 as xch4

from .test_shared import get_cube


@pytest.fixture
def cubes():
    """Input cubes for derivation of ``xch4``."""
    xch4_cube = get_cube([[[[1.0]], [[2.0]]]], air_pressure_coord=True,
                         standard_name='mole_fraction_of_methane_in_air',
                         var_name='ch4', units='1e-3')
    hus_cube = get_cube([[[[0.2]], [[0.2]]]], air_pressure_coord=True,
                        standard_name='specific_humidity', var_name='hus',
                        units='%')
    zg_cube = get_cube([[[100.0]]], air_pressure_coord=False,
                       standard_name='geopotential_height', var_name='zg',

                       units='m')
    ps_cube = get_cube([[[100000.0]]], air_pressure_coord=False,
                       standard_name='surface_air_pressure', var_name='ps',
                       units='Pa')
    return iris.cube.CubeList([xch4_cube, hus_cube, zg_cube, ps_cube])


def test_xch4_calculate(cubes):
    """Test function ``calculate``."""
    derived_var = xch4.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    assert out_cube.shape == (1, 1, 1)
    assert out_cube.units == '1'
    assert out_cube.coords('time')
    assert out_cube.coords('air_pressure')
    assert out_cube.coords('latitude')
    assert out_cube.coords('longitude')
    np.testing.assert_allclose(out_cube.data, [[[1.85e-3]]])
    np.testing.assert_allclose(out_cube.coord('time').points, [0.0])
    np.testing.assert_allclose(out_cube.coord('air_pressure').points, 85000.0)
    np.testing.assert_allclose(out_cube.coord('air_pressure').bounds,
                               [[80000.0, 90000.0]])
    np.testing.assert_allclose(out_cube.coord('latitude').points, [45.0])
    np.testing.assert_allclose(out_cube.coord('longitude').points, [10.0])


def test_xch4_required():
    """Test function ``required``."""
    derived_var = xch4.DerivedVariable()
    output = derived_var.required(None)
    assert output == [
        {'short_name': 'ch4'},
        {'short_name': 'hus'},
        {'short_name': 'zg'},
        {'short_name': 'ps'},
    ]
