"""Test derivation of `ohc`."""
import cf_units
import iris
import numpy as np
import pytest

import esmvalcore.preprocessor._derive.heatc as heatc


@pytest.fixture
def cubes():
    thetao_name = 'sea_water_potential_temperature'
    time_coord = iris.coords.DimCoord([0., 1., 2.],
                                      standard_name='time')
    thetao_cube = iris.cube.Cube([[[-272.15, -272.15 ], [-272.15, -272.15 ]],
                               [[-272.15, -272.15 ], [-272.15, -272.15 ]],
                               [[-272.15, -272.15 ], [-272.15, -272.15 ]]],
                              units='degC',
                              standard_name=thetao_name,
                              var_name='thetao',
                              dim_coords_and_dims=[(time_coord, 0)])
    return iris.cube.CubeList([thetao_cube])


def test_heatc_calculation(cubes):
    """Test function ``calculate`` for heatc."""
    derived_var = heatc.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    assert out_cube.units == cf_units.Unit('J m-3')
    expected = np.ones_like(cubes[0].data) * 1025 * 3985
    np.testing.assert_array_equal(out_cube.data, expected)

def test_heatc_required():
    """Test function ``required``."""
    derived_var = heatc.DerivedVariable()
    output = derived_var.required(None)
    assert output == [
        {'short_name': 'thetao', 'optional': 'true'},
    ]
