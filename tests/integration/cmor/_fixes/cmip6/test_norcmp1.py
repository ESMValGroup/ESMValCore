"""Tests for the fixes of NorCPM1."""
import iris
import numpy as np
import pytest

from esmvalcore.cmor._fixes.cmip6.norcpm1 import Nbp
from esmvalcore.cmor.fix import Fix


def test_get_nbp_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'NorCPM1', 'Lmon', 'nbp')
    assert fix == [Nbp(None)]


@pytest.fixture
def nbp_cube():
    """``nbp`` cube."""
    cube = iris.cube.Cube(
        [1.0],
        var_name='nbp',
        standard_name='surface_net_downward_mass_flux_of_carbon_dioxide_'
        'expressed_as_carbon_due_to_all_land_processes',
        units='kg m-2 s-1',
    )
    return cube


def test_nbp_fix_data(nbp_cube):
    """Test ``fix_data`` for ``nbp``."""
    fix = Nbp(None)
    out_cube = fix.fix_data(nbp_cube)
    np.testing.assert_allclose(out_cube.data, [29. / 44.])
