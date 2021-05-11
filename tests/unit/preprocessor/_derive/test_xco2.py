"""Test derivation of ``xco2``."""
import iris
import numpy as np
import pytest

import esmvalcore.preprocessor._derive.xco2 as xco2

from .test_shared import get_cube


@pytest.fixture
def cubes():
    """Input cubes for derivation of ``xco2``."""
    co2_cube = get_cube([[[[1.0]], [[2.0]]]], air_pressure_coord=True,
                        standard_name='mole_fraction_of_carbon_dioxide_in_air',
                        var_name='co2', units='1e-6')
    hus_cube = get_cube([[[[0.2]], [[0.2]]]], air_pressure_coord=True,
                        standard_name='specific_humidity', var_name='hus',
                        units='%')
    zg_cube = get_cube([[[100.0]]], air_pressure_coord=False,
                       standard_name='geopotential_height', var_name='zg',

                       units='m')
    ps_cube = get_cube([[[100000.0]]], air_pressure_coord=False,
                       standard_name='surface_air_pressure', var_name='ps',
                       units='Pa')
    return iris.cube.CubeList([co2_cube, hus_cube, zg_cube, ps_cube])


def test_xco2_calculate(cubes):
    """Test function ``calculate``."""
    derived_var = xco2.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    assert out_cube.shape == (1, 1, 1)
    assert out_cube.units == '1'
    assert out_cube.coords('time')
    assert out_cube.coords('air_pressure')
    assert out_cube.coords('latitude')
    assert out_cube.coords('longitude')
    np.testing.assert_allclose(out_cube.data, [[[1.85e-6]]])
    np.testing.assert_allclose(out_cube.coord('time').points, [0.0])
    np.testing.assert_allclose(out_cube.coord('air_pressure').points, 85000.0)
    np.testing.assert_allclose(out_cube.coord('air_pressure').bounds,
                               [[80000.0, 90000.0]])
    np.testing.assert_allclose(out_cube.coord('latitude').points, [45.0])
    np.testing.assert_allclose(out_cube.coord('longitude').points, [10.0])


def test_xco2_required():
    """Test function ``required``."""
    derived_var = xco2.DerivedVariable()
    output = derived_var.required(None)
    assert output == [
        {'short_name': 'co2'},
        {'short_name': 'hus'},
        {'short_name': 'zg'},
        {'short_name': 'ps'},
    ]
