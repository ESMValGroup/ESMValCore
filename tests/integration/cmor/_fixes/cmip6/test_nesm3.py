"""Test fixes for NESM3."""
import iris
import numpy as np
import pytest

from esmvalcore.cmor._fixes.cmip6.nesm3 import Cl, Cli, Clw, AllVars
from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord
from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.cmor.table import get_var_info


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'NESM3', 'Amon', 'cl')
    assert fix == [Cl(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is ClFixHybridPressureCoord


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'NESM3', 'Amon', 'cli')
    assert fix == [Cli(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is ClFixHybridPressureCoord


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'NESM3', 'Amon', 'clw')
    assert fix == [Clw(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is ClFixHybridPressureCoord


@pytest.fixture
def tos_cubes():
    """Sample cube."""
    time_coord = iris.coords.DimCoord([0.2], standard_name='time',
                                      var_name='time',
                                      units='days since 2015-01-01')
    lat_coord = iris.coords.DimCoord([1.0], standard_name='latitude',
                                     var_name='lat', units='degrees_north')
    lon_coord = iris.coords.DimCoord([1.0], standard_name='longitude',
                                     var_name='lon', units='degrees_east')
    coords_specs = [(time_coord, 0), (lat_coord, 1), (lon_coord, 2)]
    cube = iris.cube.Cube([[[22.0]]], standard_name='sea_surface_temperature',
                          var_name='tos', units='degC',
                          dim_coords_and_dims=coords_specs)
    correct_cube=cube.copy()
    wrong_cube=cube.copy()
    wrong_cube.attributes["branch_time_in_parent"] = 60000
    wrong_cube.attributes["branch_time_in_parent"] = 10000000
    return iris.cube.CubeList([correct_cube, wrong_cube])


def test_get_tos_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'NESM3', 'Omon', 'tos')
    assert fix == [AllVars(None)]


def test_tos_fix_metadata(tos_cubes):
    """Test ``fix_metadata``."""
    vardef = get_var_info('CMIP6', 'Omon', 'tos')
    fix = AllVars(vardef)
    out_cubes = fix.fix_metadata(tos_cubes)
    assert tos_cubes is out_cubes
    for cube in out_cubes:
        assert cube.attributes["branch_time_in_parent"] < 1000000
