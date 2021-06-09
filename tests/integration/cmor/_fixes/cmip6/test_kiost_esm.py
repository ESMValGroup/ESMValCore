"""Test fixes for KIOST-ESM."""
import iris
import pytest
from cf_units import Unit

from esmvalcore.cmor._fixes.cmip6.kiost_esm import (
    SfcWind,
    Siconc,
    Tas,
)
from esmvalcore.cmor._fixes.common import SiconcFixScalarCoord
from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.cmor.table import get_var_info


@pytest.fixture
def sfcwind_cubes():
    correct_lat_coord = iris.coords.DimCoord([0.0],
                                             var_name='lat',
                                             standard_name='latitude')
    wrong_lat_coord = iris.coords.DimCoord([0.0],
                                           var_name='latitudeCoord',
                                           standard_name='latitude')
    correct_lon_coord = iris.coords.DimCoord([0.0],
                                             var_name='lon',
                                             standard_name='longitude')
    wrong_lon_coord = iris.coords.DimCoord([0.0],
                                           var_name='longitudeCoord',
                                           standard_name='longitude')
    correct_cube = iris.cube.Cube(
        [[10.0]],
        var_name='sfcWind',
        dim_coords_and_dims=[(correct_lat_coord, 0), (correct_lon_coord, 1)],
    )
    wrong_cube = iris.cube.Cube(
        [[10.0]],
        var_name='ta',
        dim_coords_and_dims=[(wrong_lat_coord, 0), (wrong_lon_coord, 1)],
        attributes={'parent_time_units': 'days since 0000-00-00 (noleap)'},
    )
    scalar_cube = iris.cube.Cube(0.0, var_name='ps')
    return iris.cube.CubeList([correct_cube, wrong_cube, scalar_cube])


@pytest.fixture
def tas_cubes():
    correct_lat_coord = iris.coords.DimCoord([0.0],
                                             var_name='lat',
                                             standard_name='latitude')
    wrong_lat_coord = iris.coords.DimCoord([0.0],
                                           var_name='latitudeCoord',
                                           standard_name='latitude')
    correct_lon_coord = iris.coords.DimCoord([0.0],
                                             var_name='lon',
                                             standard_name='longitude')
    wrong_lon_coord = iris.coords.DimCoord([0.0],
                                           var_name='longitudeCoord',
                                           standard_name='longitude')
    correct_cube = iris.cube.Cube(
        [[10.0]],
        var_name='tas',
        dim_coords_and_dims=[(correct_lat_coord, 0), (correct_lon_coord, 1)],
    )
    wrong_cube = iris.cube.Cube(
        [[10.0]],
        var_name='ta',
        dim_coords_and_dims=[(wrong_lat_coord, 0), (wrong_lon_coord, 1)],
        attributes={'parent_time_units': 'days since 0000-00-00 (noleap)'},
    )
    scalar_cube = iris.cube.Cube(0.0, var_name='ps')
    return iris.cube.CubeList([correct_cube, wrong_cube, scalar_cube])


def test_get_sfcwind_fix():
    fix = Fix.get_fixes('CMIP6', 'KIOST-ESM', 'Amon', 'sfcWind')
    assert fix == [SfcWind(None)]


def test_sfcwind_fix_metadata(sfcwind_cubes):
    for cube in sfcwind_cubes:
        with pytest.raises(iris.exceptions.CoordinateNotFoundError):
            cube.coord('height')
    height_coord = iris.coords.AuxCoord(10.0,
                                        var_name='height',
                                        standard_name='height',
                                        long_name='height',
                                        units=Unit('m'),
                                        attributes={'positive': 'up'})
    vardef = get_var_info('CMIP6', 'Amon', 'sfcWind')
    fix = SfcWind(vardef)

    # Check fix
    out_cubes = fix.fix_metadata(sfcwind_cubes)
    assert out_cubes[0].var_name == 'sfcWind'
    coord = out_cubes[0].coord('height')
    assert coord == height_coord

    # Check that height coordinate is not added twice
    out_cubes_2 = fix.fix_metadata(out_cubes)
    assert out_cubes_2[0].var_name == 'sfcWind'
    coord = out_cubes_2[0].coord('height')
    assert coord == height_coord


def test_get_siconc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'KIOST-ESM', 'SImon', 'siconc')
    assert fix == [Siconc(None)]


def test_siconc_fix():
    """Test fix for ``siconc``."""
    assert Siconc is SiconcFixScalarCoord


def test_get_tas_fix():
    fix = Fix.get_fixes('CMIP6', 'KIOST-ESM', 'Amon', 'tas')
    assert fix == [Tas(None)]


def test_tas_fix_metadata(tas_cubes):
    for cube in tas_cubes:
        with pytest.raises(iris.exceptions.CoordinateNotFoundError):
            cube.coord('height')
    height_coord = iris.coords.AuxCoord(2.0,
                                        var_name='height',
                                        standard_name='height',
                                        long_name='height',
                                        units=Unit('m'),
                                        attributes={'positive': 'up'})
    vardef = get_var_info('CMIP6', 'Amon', 'tas')
    fix = Tas(vardef)

    # Check fix
    out_cubes = fix.fix_metadata(tas_cubes)
    assert out_cubes[0].var_name == 'tas'
    coord = out_cubes[0].coord('height')
    assert coord == height_coord

    # Check that height coordinate is not added twice
    out_cubes_2 = fix.fix_metadata(out_cubes)
    assert out_cubes_2[0].var_name == 'tas'
    coord = out_cubes_2[0].coord('height')
    assert coord == height_coord
