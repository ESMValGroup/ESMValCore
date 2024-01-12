"""Test fixes for MPI-ESM1-2-HR."""
import iris
import pytest
from cf_units import Unit

from esmvalcore.cmor._fixes.cmip6.mpi_esm1_2_hr import (
    AllVars,
    Cl,
    Cli,
    Clw,
    SfcWind,
    Tas,
)
from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord
from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.cmor.table import get_var_info


def test_get_allvars_fix():
    """Test getting of fix."""
    fixes = Fix.get_fixes('CMIP6', 'MPI-ESM1-2-HR', 'Amon', 'tas')
    assert AllVars(None) in fixes
    assert len([fix for fix in fixes if isinstance(fix, AllVars)]) == 1


def test_allvars_r2i1p1f1():
    lat_coord1 = iris.coords.DimCoord(
        [-86.49036676628116],
        var_name='lat',
        standard_name='latitude',
        units='degrees',
    )
    lat_coord2 = iris.coords.DimCoord(
        [-86.49036676628118],
        var_name='lat',
        standard_name='latitude',
        units='degrees',
    )

    cube1 = iris.cube.Cube([0])
    cube1.attributes['variant_label'] = 'r2i1p1f1'
    cube1.add_dim_coord(lat_coord1, 0)

    cube2 = iris.cube.Cube([0])
    cube2.attributes['variant_label'] = 'r2i1p1f1'
    cube2.add_dim_coord(lat_coord2, 0)

    fix = AllVars(None)
    fixed_cubes = fix.fix_metadata([cube1, cube2])

    assert fixed_cubes[0].coord('latitude').points[0] == -86.49036676628
    assert fixed_cubes[1].coord('latitude').points[0] == -86.49036676628


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
    correct_cube = iris.cube.Cube([[10.0]], var_name='sfcWind',
                                  dim_coords_and_dims=[(correct_lat_coord, 0),
                                                       (correct_lon_coord, 1)]
                                  )
    wrong_cube = iris.cube.Cube([[10.0]],
                                var_name='ta',
                                dim_coords_and_dims=[(wrong_lat_coord, 0),
                                                     (wrong_lon_coord, 1)])
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
    correct_cube = iris.cube.Cube([[10.0]],
                                  var_name='tas',
                                  dim_coords_and_dims=[(correct_lat_coord, 0),
                                                       (correct_lon_coord, 1)])
    wrong_cube = iris.cube.Cube([[10.0]],
                                var_name='ta',
                                dim_coords_and_dims=[(wrong_lat_coord, 0),
                                                     (wrong_lon_coord, 1)])
    scalar_cube = iris.cube.Cube(0.0, var_name='ps')
    return iris.cube.CubeList([correct_cube, wrong_cube, scalar_cube])


def test_get_cl_fix():
    """Test getting of fix."""
    fixes = Fix.get_fixes('CMIP6', 'MPI-ESM1-2-HR', 'Amon', 'cl')
    assert Cl(None) in fixes


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is ClFixHybridPressureCoord


def test_get_cli_fix():
    """Test getting of fix."""
    fixes = Fix.get_fixes('CMIP6', 'MPI-ESM1-2-HR', 'Amon', 'cli')
    assert Cli(None) in fixes


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is ClFixHybridPressureCoord


def test_get_clw_fix():
    """Test getting of fix."""
    fixes = Fix.get_fixes('CMIP6', 'MPI-ESM1-2-HR', 'Amon', 'clw')
    assert Clw(None) in fixes


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is ClFixHybridPressureCoord


def test_get_sfcwind_fix():
    fixes = Fix.get_fixes('CMIP6', 'MPI_ESM1_2_HR', 'day', 'sfcWind')
    assert SfcWind(None) in fixes


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
    vardef = get_var_info('CMIP6', 'day', 'sfcWind')
    fix = SfcWind(vardef)

    out_cubes = fix.fix_metadata(sfcwind_cubes)
    assert out_cubes[0].var_name == 'sfcWind'
    coord = out_cubes[0].coord('height')
    assert coord == height_coord


def test_get_tas_fix():
    fixes = Fix.get_fixes('CMIP6', 'MPI_ESM1_2_HR', 'day', 'tas')
    assert Tas(None) in fixes


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
    vardef = get_var_info('CMIP6', 'day', 'tas')
    fix = Tas(vardef)

    out_cubes = fix.fix_metadata(tas_cubes)
    assert out_cubes[0].var_name == 'tas'
    coord = out_cubes[0].coord('height')
    assert coord == height_coord
