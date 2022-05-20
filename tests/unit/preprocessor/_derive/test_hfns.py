"""Test derivation of ``hfns``."""
import numpy as np
import pytest
from iris.cube import CubeList

from esmvalcore.preprocessor._derive import hfns

from .test_shared import get_cube


@pytest.fixture
def cubes():
    """Input cubes for derivation of ``xch4``."""
    hfls_cube = get_cube([[[1.0]]], air_pressure_coord=False,
                         standard_name='surface_upward_latent_heat_flux',
                         var_name='hfls', units='W m-2')
    hfss_cube = get_cube([[[1.0]]], air_pressure_coord=False,
                         standard_name='surface_upward_sensible_heat_flux',
                         var_name='hfss', units='W m-2')
    return CubeList([hfls_cube, hfss_cube])


def test_hfns_calculate(cubes):
    """Test function ``calculate``."""
    derived_var = hfns.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    assert out_cube.shape == (1, 1, 1)
    assert out_cube.units == 'W m-2'
    assert out_cube.coords('time')
    assert out_cube.coords('latitude')
    assert out_cube.coords('longitude')
    np.testing.assert_allclose(out_cube.data, [[[2.0]]])
    np.testing.assert_allclose(out_cube.coord('time').points, [0.0])
    np.testing.assert_allclose(out_cube.coord('latitude').points, [45.0])
    np.testing.assert_allclose(out_cube.coord('longitude').points, [10.0])


def test_hfns_required():
    """Test function ``required``."""
    derived_var = hfns.DerivedVariable()
    output = derived_var.required(None)
    assert output == [
        {'short_name': 'hfls'},
        {'short_name': 'hfss'},
    ]
