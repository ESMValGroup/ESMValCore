"""Test derivation of `ohc`."""
import cf_units
import iris
import numpy as np
import pytest

import esmvalcore.preprocessor._derive.siextent as siextent


@pytest.fixture
def cubes():
    sic_name = 'sea_ice_area_fraction'
    time_coord = iris.coords.DimCoord([0., 1., 2.],
                                      standard_name='time')
    sic_cube = iris.cube.Cube([[[20, 10], [10, 10]],
                               [[10, 10], [10, 10]],
                               [[10, 10], [10, 10]]],
                              units='%',
                              standard_name=sic_name,
                              var_name='sic',
                              dim_coords_and_dims=[(time_coord, 0)])
    return iris.cube.CubeList([sic_cube])


def test_siextent_calculation(cubes):
    """Test function ``calculate``."""
    derived_var = siextent.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    assert out_cube.units == cf_units.Unit('m2')
    out_data = out_cube.data
    expected = np.ma.ones_like(cubes[0].data)
    expected.mask = True
    expected[0][0][0] = 1.
    np.testing.assert_array_equal(out_data.mask, expected.mask)
    np.testing.assert_array_equal(out_data[0][0][0], expected[0][0][0])


def test_siextent_required():
    """Test function ``required``."""
    derived_var = siextent.DerivedVariable()
    output = derived_var.required(None)
    assert output == [
        {'short_name': 'sic'}
    ]
