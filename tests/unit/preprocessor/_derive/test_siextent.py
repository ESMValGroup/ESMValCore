"""Test derivation of `ohc`."""
import cf_units
import iris
import numpy as np
import pytest

import esmvalcore.preprocessor._derive.siextent as siextent
from esmvalcore.exceptions import RecipeError


@pytest.fixture
def cubes_sic():
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


@pytest.fixture
def cubes_siconca():
    sic_name = 'sea_ice_area_fraction'
    time_coord = iris.coords.DimCoord([0., 1., 2.],
                                      standard_name='time')
    sic_cube = iris.cube.Cube([[[20, 10], [10, 10]],
                               [[10, 10], [10, 10]],
                               [[10, 10], [10, 10]]],
                              units='%',
                              standard_name=sic_name,
                              var_name='siconca',
                              dim_coords_and_dims=[(time_coord, 0)])
    return iris.cube.CubeList([sic_cube])


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
    siconca_cube = iris.cube.Cube([[[20, 10], [10, 10]],
                                   [[10, 10], [10, 10]],
                                   [[10, 10], [10, 10]]],
                                  units='%',
                                  standard_name=sic_name,
                                  var_name='siconca',
                                  dim_coords_and_dims=[(time_coord, 0)])
    return iris.cube.CubeList([sic_cube, siconca_cube])


def test_siextent_calculation_sic(cubes_sic):
    """Test function ``calculate`` when sic is available."""
    derived_var = siextent.DerivedVariable()
    out_cube = derived_var.calculate(cubes_sic)
    assert out_cube.units == cf_units.Unit('m2')
    out_data = out_cube.data
    expected = np.ma.ones_like(cubes_sic[0].data)
    expected.mask = True
    expected[0][0][0] = 1.
    np.testing.assert_array_equal(out_data.mask, expected.mask)
    np.testing.assert_array_equal(out_data[0][0][0], expected[0][0][0])


def test_siextent_calculation_siconca(cubes_siconca):
    """Test function ``calculate`` when siconca is available."""
    derived_var = siextent.DerivedVariable()
    out_cube = derived_var.calculate(cubes_siconca)
    assert out_cube.units == cf_units.Unit('m2')
    out_data = out_cube.data
    expected = np.ma.ones_like(cubes_siconca[0].data)
    expected.mask = True
    expected[0][0][0] = 1.
    np.testing.assert_array_equal(out_data.mask, expected.mask)
    np.testing.assert_array_equal(out_data[0][0][0], expected[0][0][0])


def test_siextent_calculation(cubes):
    """Test function ``calculate`` when sic and siconca are available."""
    derived_var = siextent.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    assert out_cube.units == cf_units.Unit('m2')
    out_data = out_cube.data
    expected = np.ma.ones_like(cubes[0].data)
    expected.mask = True
    expected[0][0][0] = 1.
    np.testing.assert_array_equal(out_data.mask, expected.mask)
    np.testing.assert_array_equal(out_data[0][0][0], expected[0][0][0])


def test_siextent_no_data(cubes_sic):
    derived_var = siextent.DerivedVariable()
    cubes_sic[0].var_name = 'wrong'
    msg = ('Derivation of siextent failed due to missing variables '
           'sic and siconca.')
    with pytest.raises(RecipeError, match=msg):
        derived_var.calculate(cubes_sic)


def test_siextent_required():
    """Test function ``required``."""
    derived_var = siextent.DerivedVariable()
    output = derived_var.required(None)
    assert output == [
        {'short_name': 'sic', 'optional': 'true'},
        {'short_name': 'siconca', 'optional': 'true'}
    ]
