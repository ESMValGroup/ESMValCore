"""Test fixes for CNRM-ESM2-1."""
import iris
import numpy as np
import pytest

from esmvalcore.cmor._fixes.cmip6.cnrm_cm6_1 import Cl as BaseCl
from esmvalcore.cmor._fixes.cmip6.cnrm_cm6_1 import Clcalipso as BaseClcalipso
from esmvalcore.cmor._fixes.cmip6.cnrm_cm6_1 import Cli as BaseCli
from esmvalcore.cmor._fixes.cmip6.cnrm_cm6_1 import Clw as BaseClw
from esmvalcore.cmor._fixes.cmip6.cnrm_esm2_1 import (Cl, Clcalipso,
                                                      Cli, Clw, Omon)
from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.cmor.table import get_var_info


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CNRM-ESM2-1', 'Amon', 'cl')
    assert fix == [Cl(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is BaseCl


def test_get_clcalipso_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CNRM-ESM2-1', 'Amon', 'clcalipso')
    assert fix == [Clcalipso(None)]


def test_clcalipso_fix():
    """Test fix for ``cl``."""
    assert Clcalipso is BaseClcalipso


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CNRM-ESM2-1', 'Amon', 'cli')
    assert fix == [Cli(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is BaseCli


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CNRM-ESM2-1', 'Amon', 'clw')
    assert fix == [Clw(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is BaseClw


@pytest.fixture
def thetao_cubes():
    """Cubes to test fixes for ``thetao``."""
    time_coord = iris.coords.DimCoord(
        [0.0004, 1.09776], var_name='time', standard_name='time',
        units='days since 1850-01-01 00:00:00')
    lat_coord = iris.coords.DimCoord(
        [0.0, 1.0], var_name='lat', standard_name='latitude', units='degrees')
    lon_coord = iris.coords.DimCoord(
        [0.0, 1.0], var_name='lon', standard_name='longitude', units='degrees')
    lev_coord = iris.coords.DimCoord(
        [5.0, 10.0], bounds=[[2.5, 7.5], [7.5, 12.5]],
        var_name='lev', standard_name=None, units='m',
        attributes={'positive': 'up'})
    coord_specs = [
        (time_coord, 0),
        (lev_coord, 1),
        (lat_coord, 2),
        (lon_coord, 3),
    ]
    thetao_cube = iris.cube.Cube(
        np.ones((2, 2, 2, 2)),
        var_name='thetao',
        dim_coords_and_dims=coord_specs,
    )
    return iris.cube.CubeList([thetao_cube])


def test_get_thetao_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CNRM-ESM2-1', 'Omon', 'thetao')
    assert fix == [Omon(None)]


def test_thetao_fix_metadata(thetao_cubes):
    """Test ``fix_metadata`` for ``thetao``."""
    vardef = get_var_info('CMIP6', 'Omon', 'thetao')
    fix = Omon(vardef)
    out_cubes = fix.fix_metadata(thetao_cubes)
    assert out_cubes is thetao_cubes
    assert len(out_cubes) == 1
    out_cube = out_cubes[0]

    # Check metadata of depth coordinate
    depth_coord = out_cube.coord('depth')
    assert depth_coord.standard_name == 'depth'
    assert depth_coord.var_name == 'lev'
    assert depth_coord.long_name == 'ocean depth coordinate'
    assert depth_coord.units == 'm'
    assert depth_coord.attributes == {'positive': 'down'}
