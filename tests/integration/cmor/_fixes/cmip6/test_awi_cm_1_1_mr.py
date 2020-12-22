"""Tests for the fixes of AWI-CM-1-1-MR."""
import iris
import pytest

from esmvalcore.cmor._fixes.cmip6.awi_cm_1_1_mr import AllVars
from esmvalcore.cmor.fix import Fix


@pytest.fixture
def cubes():
    correct_lat_coord = iris.coords.DimCoord([0.0],
                                             var_name='lat',
                                             standard_name='latitude',
                                             long_name='latitude')
    wrong_lat_coord = iris.coords.DimCoord([0.0],
                                           var_name='latitudeCoord',
                                           standard_name='latitude',
                                           long_name='Latitude')
    correct_lon_coord = iris.coords.DimCoord([0.0],
                                             var_name='lon',
                                             standard_name='longitude')
    correct_cube = iris.cube.Cube(
        [[10.0]],
        var_name='tas',
        dim_coords_and_dims=[(correct_lat_coord, 0), (correct_lon_coord, 1)],
    )
    wrong_cube = iris.cube.Cube(
        [[10.0]],
        var_name='ta',
        dim_coords_and_dims=[(wrong_lat_coord, 0), (correct_lon_coord, 1)],
    )
    return iris.cube.CubeList([correct_cube, wrong_cube])


def test_get_allvars_fix():
    fix = Fix.get_fixes('CMIP6', 'AWI-CM-1-1-MR', 'Amon', 'wrong_lat_lname')
    assert fix == [AllVars(None)]


def test_allvars_fix_metadata(cubes):
    fix = AllVars(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        try:
            lat_coord = cube.coord('latitude')
        except iris.exceptions.CoordinateNotFoundError:
            pass
        else:
            assert lat_coord.long_name == 'latitude'
