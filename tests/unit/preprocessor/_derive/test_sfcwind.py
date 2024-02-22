"""Test derivation of ``sfcwind``."""
import numpy as np
import pytest
from iris.cube import CubeList

from esmvalcore.preprocessor._derive import sfcwind

from .test_shared import get_cube


@pytest.fixture
def cubes():
    """Input cubes for derivation of ``sfcwind``."""
    uas_cube = get_cube([[[3.0]]],
                        air_pressure_coord=False,
                        standard_name='eastward_wind',
                        var_name='uas',
                        units='m s-1')
    vas_cube = get_cube([[[4.0]]],
                        air_pressure_coord=False,
                        standard_name='northward_wind',
                        var_name='vas',
                        units='m s-1')
    return CubeList([uas_cube, vas_cube])


def test_sfcwind_calculate(cubes):
    """Test function ``calculate``."""
    derived_var = sfcwind.DerivedVariable()
    required_vars = derived_var.required("CMIP5")
    expected_required_vars = [
        {
            'short_name': 'uas'
        },
        {
            'short_name': 'vas'
        },
    ]
    assert required_vars == expected_required_vars
    out_cube = derived_var.calculate(cubes)
    assert out_cube.shape == (1, 1, 1)
    assert out_cube.units == 'm s-1'
    assert out_cube.coords('time')
    assert out_cube.coords('latitude')
    assert out_cube.coords('longitude')
    np.testing.assert_allclose(out_cube.data, [[[5.0]]])
    np.testing.assert_allclose(out_cube.coord('time').points, [0.0])
    np.testing.assert_allclose(out_cube.coord('latitude').points, [45.0])
    np.testing.assert_allclose(out_cube.coord('longitude').points, [10.0])
