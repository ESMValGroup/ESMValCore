"""Tests for the fixes of CESM2-WACCM."""
import os
import sys
import unittest.mock

import iris
import numpy as np
import pytest

from esmvalcore.cmor._fixes.cmip6.cesm2 import Cl as BaseCl
from esmvalcore.cmor._fixes.cmip6.cesm2 import Fgco2 as BaseFgco2
from esmvalcore.cmor._fixes.cmip6.cesm2 import Tas as BaseTas
from esmvalcore.cmor._fixes.cmip6.cesm2_waccm import (
    Cl,
    Cli,
    Clw,
    Fgco2,
    Omon,
    Siconc,
    Tas,
)
from esmvalcore.cmor._fixes.common import SiconcFixScalarCoord
from esmvalcore.cmor.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CESM2-WACCM', 'Amon', 'cl')
    assert fix == [Cl(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert issubclass(Cl, BaseCl)


@pytest.mark.skipif(sys.version_info < (3, 7, 6),
                    reason="requires python3.7.6 or newer")
@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip6.cesm2.Fix.get_fixed_filepath',
    autospec=True)
def test_cl_fix_file(mock_get_filepath, tmp_path, test_data_path):
    """Test ``fix_file`` for ``cl``."""
    nc_path = test_data_path / 'cesm2_waccm_cl.nc'
    mock_get_filepath.return_value = os.path.join(tmp_path,
                                                  'fixed_cesm2_waccm_cl.nc')
    fix = Cl(None)
    fixed_file = fix.fix_file(str(nc_path), tmp_path)
    mock_get_filepath.assert_called_once_with(tmp_path, str(nc_path))
    fixed_cube = iris.load_cube(fixed_file)
    lev_coord = fixed_cube.coord(var_name='lev')
    a_coord = fixed_cube.coord(var_name='a')
    b_coord = fixed_cube.coord(var_name='b')
    assert lev_coord.standard_name == (
        'atmosphere_hybrid_sigma_pressure_coordinate')
    assert lev_coord.units == '1'
    np.testing.assert_allclose(a_coord.points, [1.0, 2.0])
    np.testing.assert_allclose(a_coord.bounds, [[0.0, 1.5], [1.5, 3.0]])
    np.testing.assert_allclose(b_coord.points, [0.0, 1.0])
    np.testing.assert_allclose(b_coord.bounds, [[-1.0, 0.5], [0.5, 2.0]])


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CESM2-WACCM', 'Amon', 'cli')
    assert fix == [Cli(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is Cl


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CESM2-WACCM', 'Amon', 'clw')
    assert fix == [Clw(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is Cl


def test_get_fgco2_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CESM2-WACCM', 'Omon', 'fgco2')
    assert fix == [Fgco2(None), Omon(None)]


def test_fgco2_fix():
    """Test fix for ``fgco2``."""
    assert Fgco2 is BaseFgco2


def test_get_siconc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CESM2-WACCM', 'SImon', 'siconc')
    assert fix == [Siconc(None)]


def test_siconc_fix():
    """Test fix for ``siconc``."""
    assert Siconc is SiconcFixScalarCoord


@pytest.fixture
def tas_cubes():
    """Cubes to test fixes for ``tas``."""
    ta_cube = iris.cube.Cube([1.0], var_name='ta')
    tas_cube = iris.cube.Cube([3.0], var_name='tas')
    return iris.cube.CubeList([ta_cube, tas_cube])


def test_get_tas_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CESM2-WACCM', 'Amon', 'tas')
    assert fix == [Tas(None)]


def test_tas_fix():
    """Test fix for ``tas``."""
    assert Tas is BaseTas
