"""Tests for the fixes of IPSL-CM6."""
import iris
import pytest

from esmvalcore.cmor._fixes.ipslcm.ipsl_cm6 import Tas
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


@pytest.fixture
def test_get_tas_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('IPSLCM', 'IPSL-CM6', 'Amon', 'tas')
    assert fix == [Tas(None)]


@pytest.fixture
def cubes():
    """``tas`` cube."""

    cube = iris.cube.Cube(
        [200.0],  # chilly, isn't it ?
        var_name='tas',
        standard_name='air_temperature',
        units='K',
    )
    return iris.cube.CubeList([cube])


def test_tas_fix_metadata(cubes):
    """Test ``fix_metadata`` for ``tas``."""
    vardef = get_var_info('CMIP6', 'Amon', 'tas')
    fix = Tas(vardef)
    out_cubes = fix.fix_metadata(cubes)
    out_cube = fix.get_cube_from_list(out_cubes, 'tas')
    assert any([coord.standard_name == 'height'
                for coord in out_cube.aux_coords])
