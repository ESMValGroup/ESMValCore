"""Test fixes for SAM0-UNICON."""
import iris
import numpy as np
import pytest

from esmvalcore.cmor._fixes.cmip6.sam0_unicon import Cl, Cli, Clw, Nbp
from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord
from esmvalcore.cmor._fixes.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'SAM0-UNICON', 'Amon', 'cl')
    assert fix == [Cl(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is ClFixHybridPressureCoord


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'SAM0-UNICON', 'Amon', 'cli')
    assert fix == [Cli(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is ClFixHybridPressureCoord


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'SAM0-UNICON', 'Amon', 'clw')
    assert fix == [Clw(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is ClFixHybridPressureCoord


def test_get_nbp_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'SAM0-UNICON', 'Lmon', 'nbp')
    assert fix == [Nbp(None)]


@pytest.fixture
def nbp_cube():
    """``nbp`` cube."""
    cube = iris.cube.Cube(
        [1.0],
        var_name='nbp',
        standard_name='surface_net_downward_mass_flux_of_carbon_dioxide'
        '_expressed_as_carbon_due_to_all_land_processes',
        units='kg m-2 s-1',
    )
    return cube


def test_nbp_fix_data(nbp_cube):
    """Test ``fix_data`` for ``nbp``."""
    fix = Nbp(None)
    out_cube = fix.fix_data(nbp_cube)
    np.testing.assert_allclose(out_cube.data, [-1.0])
