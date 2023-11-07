"""Test fixes for MIROC6."""
import iris
import numpy as np
import pytest

from esmvalcore.cmor._fixes.cmip6.miroc6 import Cl, Cli, Clw, Tos
from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord
from esmvalcore.cmor._fixes.fix import Fix, GenericFix
from esmvalcore.cmor.table import get_var_info


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'MIROC6', 'Amon', 'cl')
    assert fix == [Cl(None), GenericFix(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is ClFixHybridPressureCoord


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'MIROC6', 'Amon', 'cli')
    assert fix == [Cli(None), GenericFix(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is ClFixHybridPressureCoord


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'MIROC6', 'Amon', 'clw')
    assert fix == [Clw(None), GenericFix(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is ClFixHybridPressureCoord


@pytest.fixture
def tos_cubes():
    """Sample cube."""
    time_coord = iris.coords.DimCoord([0.2],
                                      standard_name='time',
                                      var_name='time',
                                      units='days since 1850-01-01')
    lat_coord = iris.coords.DimCoord([23.0211550, 26.3700972],
                                     standard_name='latitude',
                                     var_name='lat',
                                     units='degrees_north')
    lon_coord = iris.coords.DimCoord([23.0211550, 26.3700972],
                                     standard_name='longitude',
                                     var_name='lon',
                                     units='degrees_east')
    coords_specs = [(time_coord, 0), (lat_coord, 1), (lon_coord, 2)]
    cube = iris.cube.Cube([[[22.0, 22.0], [22.0, 22.0]]],
                          standard_name='sea_surface_temperature',
                          var_name='tos',
                          units='degC',
                          dim_coords_and_dims=coords_specs)
    return iris.cube.CubeList([cube])


def test_get_tos_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'MIROC6', 'Omon', 'tos')
    assert fix == [Tos(None), GenericFix(None)]


def test_tos_fix_metadata(tos_cubes):
    """Test ``fix_metadata``."""
    vardef = get_var_info('CMIP6', 'Omon', 'tos')
    fix = Tos(vardef)
    fixed_cubes = fix.fix_metadata(tos_cubes)
    for fixed_tos_cube in fixed_cubes:
        fixed_lon = fixed_tos_cube.coord('longitude')
        fixed_lat = fixed_tos_cube.coord('latitude')
        assert fixed_lon.bounds is not None
        assert fixed_lat.bounds is not None
        assert fixed_lon.bounds.dtype == np.float64
        assert fixed_lon.points.dtype == np.float64
        assert fixed_lat.bounds.dtype == np.float64
        assert fixed_lat.points.dtype == np.float64
