"""Tests for the fixes of CESM2-WACCM."""
import os
import sys
import unittest.mock

import iris
import numpy as np
import pytest
from netCDF4 import Dataset

from esmvalcore.cmor._fixes.cmip6.cesm2 import Tas as BaseTas
from esmvalcore.cmor._fixes.cmip6.cesm2_waccm import Cl, Cli, Clw, Tas
from esmvalcore.cmor.fix import Fix


@pytest.fixture
def cl_file(tmp_path):
    """Create netcdf file with similar issues as ``cl``."""
    nc_path = os.path.join(tmp_path, 'cesm2_waccm_cl.nc')
    dataset = Dataset(nc_path, mode='w')
    dataset.createDimension('lev', size=2)
    dataset.createDimension('bnds', size=2)

    # Dimensional variables
    dataset.createVariable('lev', np.float64, dimensions=('lev',))
    dataset.createVariable('lev_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.variables['lev'][:] = [1.0, 2.0]
    dataset.variables['lev'].bounds = 'lev_bnds'
    dataset.variables['lev'].units = '1'
    dataset.variables['lev_bnds'][:] = [[0.5, 1.5], [1.5, 3.0]]
    dataset.variables['lev_bnds'].standard_name = (
        'atmosphere_hybrid_sigma_pressure_coordinate')
    dataset.variables['lev_bnds'].units = '1'
    dataset.variables['lev_bnds'].formula_terms = (
        'p0: p0 a: a_bnds b: b_bnds ps: ps')

    # Coordinates for derivation of pressure coordinate
    dataset.createVariable('a', np.float64, dimensions=('lev',))
    dataset.createVariable('a_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.createVariable('b', np.float64, dimensions=('lev',))
    dataset.createVariable('b_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.variables['a'][:] = [1.0, 2.0]
    dataset.variables['a'].bounds = 'a_bnds'
    dataset.variables['a_bnds'][:] = [[1.5, 0.0], [3.0, 1.5]]
    dataset.variables['b'][:] = [0.0, 1.0]
    dataset.variables['b'].bounds = 'b_bnds'
    dataset.variables['b_bnds'][:] = [[0.5, -1.0], [2.0, 0.5]]

    dataset.close()
    return nc_path


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CESM2-WACCM', 'Amon', 'cl')
    assert fix == [Cl(None)]


@pytest.mark.skipif(sys.version_info < (3, 7, 6),
                    reason="requires python3.7.6 or newer")
@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip6.cesm2.Fix.get_fixed_filepath',
    autospec=True)
def test_cl_fix_file(mock_get_filepath, cl_file, tmp_path):
    """Test ``fix_file`` for ``cl``."""
    mock_get_filepath.return_value = os.path.join(tmp_path,
                                                  'fixed_cesm2_waccm_cl.nc')
    fix = Cl(None)
    fixed_file = fix.fix_file(cl_file, tmp_path)
    mock_get_filepath.assert_called_once_with(tmp_path, cl_file)
    fixed_dataset = Dataset(fixed_file, mode='r')
    assert fixed_dataset.variables['lev'].standard_name == (
        'atmosphere_hybrid_sigma_pressure_coordinate')
    assert fixed_dataset.variables['lev'].formula_terms == (
        'p0: p0 a: a b: b ps: ps')
    assert fixed_dataset.variables['lev'].units == '1'
    np.testing.assert_allclose(fixed_dataset.variables['a'][:], [1.0, 2.0])
    np.testing.assert_allclose(fixed_dataset.variables['b'][:], [0.0, 1.0])
    np.testing.assert_allclose(fixed_dataset.variables['a_bnds'][:],
                               [[0.0, 1.5], [1.5, 3.0]])
    np.testing.assert_allclose(fixed_dataset.variables['b_bnds'][:],
                               [[-1.0, 0.5], [0.5, 2.0]])


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
