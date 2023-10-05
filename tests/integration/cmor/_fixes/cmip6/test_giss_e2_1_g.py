"""Test fixes for GISS-E2-1-G."""
import dask.array as da
import numpy as np
from iris.cube import Cube

from esmvalcore.cmor._fixes.cmip6.giss_e2_1_g import Cl, Cli, Clw
from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord
from esmvalcore.cmor._fixes.fix import Fix, GenericFix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'GISS-E2-1-G', 'Amon', 'cl')
    assert fix == [Cl(None), GenericFix(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is ClFixHybridPressureCoord


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'GISS-E2-1-G', 'Amon', 'cli')
    assert fix == [Cli(None), GenericFix(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is ClFixHybridPressureCoord


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'GISS-E2-1-G', 'Amon', 'clw')
    assert fix == [Clw(None), GenericFix(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is ClFixHybridPressureCoord


def test_tos_fix():
    fix = Fix.get_fixes('CMIP6', 'GISS-E2-1-G', 'Omon', 'tos')[0]
    cube = Cube(
        da.array([274], dtype=np.float32),
        var_name='tos',
        units='degC',
    )
    result, = fix.fix_metadata([cube])
    assert 0. < result.data < 1.
    assert result.units == 'degC'
