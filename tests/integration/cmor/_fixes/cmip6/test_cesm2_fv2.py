"""Tests for the fixes of CESM2-FV2."""
import unittest

from esmvalcore.cmor._fixes.cmip6.cesm2_fv2 import Cl, Cli, Clw, Tas
from esmvalcore.cmor.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CESM2-FV2', 'Amon', 'cl')
    assert fix == [Cl(None)]


@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip6.cesm2_fv2.BaseCl.fix_metadata',
    autospec=True)
def test_cl_fix_metadata(mock_base_fix_metadata):
    """Test ``fix_metadata`` for ``cl``."""
    fix = Cl(None)
    fix.fix_metadata('cubes')
    mock_base_fix_metadata.assert_called_once_with(fix, 'cubes')


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CESM2-FV2', 'Amon', 'cl')
    assert fix == [Cl(None)]


@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip6.cesm2_fv2.Cl.fix_metadata',
    autospec=True)
def test_cli_fix_metadata(mock_base_fix_metadata):
    """Test ``fix_metadata`` for ``cli``."""
    fix = Cli(None)
    fix.fix_metadata('cubes')
    mock_base_fix_metadata.assert_called_once_with(fix, 'cubes')


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CESM2-FV2', 'Amon', 'clw')
    assert fix == [Clw(None)]


@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip6.cesm2_fv2.Cl.fix_metadata',
    autospec=True)
def test_clw_fix_metadata(mock_base_fix_metadata):
    """Test ``fix_metadata`` for ``clw``."""
    fix = Clw(None)
    fix.fix_metadata('cubes')
    mock_base_fix_metadata.assert_called_once_with(fix, 'cubes')


def test_get_tas_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CESM2-FV2', 'Amon', 'tas')
    assert fix == [Tas(None)]


@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip6.cesm2_fv2.BaseTas.fix_metadata',
    autospec=True)
def test_tas_fix_metadata(mock_base_fix_metadata):
    """Test ``fix_metadata`` for ``tas``."""
    fix = Tas(None)
    fix.fix_metadata('cubes')
    mock_base_fix_metadata.assert_called_once_with(fix, 'cubes')
