"""Test fixes for CNRM-ESM2-1."""
import unittest

from esmvalcore.cmor._fixes.cmip6.cnrm_esm2_1 import Cl, Clcalipso, Cli, Clw
from esmvalcore.cmor._fixes.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CNRM-ESM2-1', 'Amon', 'cl')
    assert fix == [Cl(None)]


@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip6.cnrm_esm2_1.BaseCl.fix_metadata',
    autospec=True)
def test_cl_fix_metadata(mock_base_fix_metadata):
    """Test ``fix_metadata`` for ``cl``."""
    fix = Cl(None)
    fix.fix_metadata('cubes')
    mock_base_fix_metadata.assert_called_once_with(fix, 'cubes')


def test_get_clcalipso_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CNRM-ESM2-1', 'Amon', 'clcalipso')
    assert fix == [Clcalipso(None)]


@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip6.cnrm_esm2_1.BaseClcalipso.fix_metadata',
    autospec=True)
def test_clcalipso_fix_metadata(mock_base_fix_metadata):
    """Test ``fix_metadata`` for ``clcalipso`."""
    fix = Clcalipso(None)
    fix.fix_metadata('cubes')
    mock_base_fix_metadata.assert_called_once_with(fix, 'cubes')


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CNRM-ESM2-1', 'Amon', 'cli')
    assert fix == [Cli(None)]


@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip6.cnrm_esm2_1.Cl.fix_metadata',
    autospec=True)
def test_cli_fix_metadata(mock_base_fix_metadata):
    """Test ``fix_metadata`` for ``cli``."""
    fix = Cli(None)
    fix.fix_metadata('cubes')
    mock_base_fix_metadata.assert_called_once_with(fix, 'cubes')


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CNRM-ESM2-1', 'Amon', 'clw')
    assert fix == [Clw(None)]


@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip6.cnrm_esm2_1.Cl.fix_metadata',
    autospec=True)
def test_clw_fix_metadata(mock_base_fix_metadata):
    """Test ``fix_metadata`` for ``clw``."""
    fix = Clw(None)
    fix.fix_metadata('cubes')
    mock_base_fix_metadata.assert_called_once_with(fix, 'cubes')
