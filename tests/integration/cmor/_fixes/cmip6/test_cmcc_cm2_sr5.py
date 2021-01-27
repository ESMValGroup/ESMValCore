"""Tests for the fixes of CMCC-CM2-SR5."""
from unittest import mock

import iris
import pytest

from esmvalcore.cmor._fixes.cmip6.cmcc_cm2_sr5 import Cl
from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CMCC-CM2-SR5', 'Amon', 'cl')
    assert fix == [Cl(None)]


@pytest.fixture
def cl_cubes():
    """``cl`` cubes."""
    ps_coord = iris.coords.AuxCoord([0.0], var_name='ps',
                                    standard_name='air_pressure')
    cube = iris.cube.Cube(
        [1.0],
        var_name='cl',
        standard_name='cloud_area_fraction_in_atmosphere_layer',
        units='%',
        aux_coords_and_dims=[(ps_coord, 0)],
    )
    return iris.cube.CubeList([cube])


def test_cl_fix():
    """Test fix for ``cl``."""
    assert issubclass(Cl, ClFixHybridPressureCoord)


@mock.patch(
    'esmvalcore.cmor._fixes.cmip6.cmcc_cm2_sr5.ClFixHybridPressureCoord.'
    'fix_metadata', autospec=True)
def test_cl_fix_metadata(mock_base_fix_metadata, cl_cubes):
    """Test ``fix_metadata`` for ``cl``."""
    mock_base_fix_metadata.side_effect = lambda x, y: y
    vardef = get_var_info('CMIP6', 'Amon', 'cl')
    fix = Cl(vardef)
    assert cl_cubes[0].coord(var_name='ps').standard_name == 'air_pressure'
    out_cube = fix.fix_metadata(cl_cubes)[0]
    assert out_cube.coord(var_name='ps').standard_name is None
