"""Tests for the fixes of NorESM2-LM."""

import iris
import numpy as np
import pytest

from esmvalcore.cmor._fixes.cmip6.noresm2_lm import (
    AllVars,
    Cl,
    Cli,
    Clw,
    Siconc,
)
from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'NorESM2-LM', 'Amon', 'cl')
    assert fix == [Cl(None), AllVars(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is ClFixHybridPressureCoord


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'NorESM2-LM', 'Amon', 'cli')
    assert fix == [Cli(None), AllVars(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is ClFixHybridPressureCoord


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'NorESM2-LM', 'Amon', 'clw')
    assert fix == [Clw(None), AllVars(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is ClFixHybridPressureCoord


@pytest.fixture
def siconc_cubes():
    """Sample cube."""
    time_coord = iris.coords.DimCoord([0.2], standard_name='time',
                                      var_name='time',
                                      units='days since 1850-01-01')
    lat_coord = iris.coords.DimCoord([30.0], standard_name='latitude',
                                     var_name='lat', units='degrees_north')
    lon_coord = iris.coords.DimCoord([30.0], standard_name='longitude',
                                     var_name='lon', units='degrees_east')
    coords_specs = [(time_coord, 0), (lat_coord, 1), (lon_coord, 2)]
    cube = iris.cube.Cube([[[22.0]]], standard_name='sea_ice_area_fraction',
                          var_name='siconc', units='%',
                          dim_coords_and_dims=coords_specs)
    return iris.cube.CubeList([cube])


@pytest.fixture
def cubes_bounds():
    """Correct and wrong cubes."""
    lat_coord = iris.coords.DimCoord([0.0],
                                     var_name='lat',
                                     standard_name='latitude')
    correct_lon_coord = iris.coords.DimCoord([0, 357.5],
                                             bounds=[[-1.25, 1.25],
                                                     [356.25, 358.75]],
                                             var_name='lon',
                                             standard_name='longitude')
    wrong_lon_coord = iris.coords.DimCoord([0, 357.5],
                                           bounds=[[0, 1.25], [356.25, 360]],
                                           var_name='lon',
                                           standard_name='longitude')
    correct_cube = iris.cube.Cube(
        [[10.0, 10.0]],
        var_name='tas',
        dim_coords_and_dims=[(lat_coord, 0), (correct_lon_coord, 1)],
    )
    wrong_cube = iris.cube.Cube(
        [[10.0, 10.0]],
        var_name='tas',
        dim_coords_and_dims=[(lat_coord, 0), (wrong_lon_coord, 1)],
    )
    return iris.cube.CubeList([correct_cube, wrong_cube])


def test_get_siconc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'NorESM2-LM', 'SImon', 'siconc')
    assert fix == [Siconc(None), AllVars(None)]


def test_allvars_fix_lon_bounds(cubes_bounds):
    """Test fixing of longitude bounds."""
    fix = AllVars(None)
    out_cubes = fix.fix_metadata(cubes_bounds)
    assert cubes_bounds is out_cubes
    for cube in out_cubes:
        try:
            lon_coord = cube.coord('longitude')
        except iris.exceptions.CoordinateNotFoundError:
            pass
        else:
            assert lon_coord.bounds[0][0] == -1.25
            assert lon_coord.bounds[-1][-1] == 358.75


def test_siconc_fix_metadata(siconc_cubes):
    """Test ``fix_metadata``."""
    for cube in siconc_cubes:
        cube.coord("latitude").bounds = [28.9956255, 32.3445677]
        cube.coord("longitude").bounds = [28.9956255, 32.3445677]

    # Raw cubes
    assert len(siconc_cubes) == 1
    siconc_cube = siconc_cubes[0]
    assert siconc_cube.var_name == "siconc"

    # Extract siconc cube
    siconc_cube = siconc_cubes.extract_cube('sea_ice_area_fraction')
    assert not siconc_cube.coords('typesi')

    # Apply fix
    vardef = get_var_info('CMIP6', 'SImon', 'siconc')
    fix = Siconc(vardef)
    fixed_cubes = fix.fix_metadata(siconc_cubes)
    assert len(fixed_cubes) == 1
    fixed_siconc_cube = fixed_cubes.extract_cube('sea_ice_area_fraction')
    fixed_lon = fixed_siconc_cube.coord('longitude')
    fixed_lat = fixed_siconc_cube.coord('latitude')
    assert fixed_lon.bounds is not None
    assert fixed_lat.bounds is not None
    np.testing.assert_equal(fixed_lon.bounds, [[28.9956, 32.3446]])
    np.testing.assert_equal(fixed_lat.bounds, [[28.9956, 32.3446]])
