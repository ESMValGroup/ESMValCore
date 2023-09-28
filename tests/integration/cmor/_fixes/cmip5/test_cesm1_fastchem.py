"""Tests for CESM1-FASTCHEM fixes."""
from esmvalcore.cmor._fixes.cmip5.cesm1_cam5 import Cl as BaseCl
from esmvalcore.cmor._fixes.cmip5.cesm1_fastchem import Cl
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP5', 'CESM1-FASTCHEM', 'Amon', 'cl')
    assert fix == [Cl(None), GenericFix(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is BaseCl
