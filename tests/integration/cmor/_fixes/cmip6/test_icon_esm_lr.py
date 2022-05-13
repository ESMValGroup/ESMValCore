"""Tests for the fixes of ICON-ESM-LR."""
import pytest
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList

from esmvalcore.cmor._fixes.cmip6.icon_esm_lr import AllVars
from esmvalcore.cmor.fix import Fix


@pytest.fixture
def cubes():
    """Cubes to test fix."""
    correct_lat_coord = AuxCoord([0.0], var_name='lat',
                                 standard_name='latitude')
    wrong_lat_coord = AuxCoord([0.0], var_name='latitude',
                               standard_name='latitude')
    correct_lon_coord = AuxCoord([0.0], var_name='lon',
                                 standard_name='longitude')
    wrong_lon_coord = AuxCoord([0.0], var_name='longitude',
                               standard_name='longitude')
    correct_cube = Cube(
        [10.0],
        var_name='tas',
        aux_coords_and_dims=[(correct_lat_coord, 0), (correct_lon_coord, 0)],
    )
    wrong_cube = Cube(
        [10.0],
        var_name='pr',
        aux_coords_and_dims=[(wrong_lat_coord, 0), (wrong_lon_coord, 0)],
    )
    return CubeList([correct_cube, wrong_cube])


def test_get_allvars_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'ICON-ESM-LR', 'Amon', 'tas')
    assert fix == [AllVars(None)]


def test_allvars_fix_metadata_lat_lon(cubes):
    """Test ``fix_metadata`` for all variables."""
    fix = AllVars(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        lat_coord = cube.coord('latitude')
        lon_coord = cube.coord('longitude')
        assert lat_coord.var_name == 'lat'
        assert lon_coord.var_name == 'lon'


def test_allvars_fix_metadata_lat(cubes):
    """Test ``fix_metadata`` for all variables."""
    for cube in cubes:
        cube.remove_coord('longitude')
    fix = AllVars(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        lat_coord = cube.coord('latitude')
        assert lat_coord.var_name == 'lat'


def test_allvars_fix_metadata_lon(cubes):
    """Test ``fix_metadata`` for all variables."""
    for cube in cubes:
        cube.remove_coord('latitude')
    fix = AllVars(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        lon_coord = cube.coord('longitude')
        assert lon_coord.var_name == 'lon'


def test_allvars_fix_metadata_no_lat_lon(cubes):
    """Test ``fix_metadata`` for all variables."""
    for cube in cubes:
        cube.remove_coord('latitude')
        cube.remove_coord('longitude')
    fix = AllVars(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
