"""Test fixes for CCSM4."""
import unittest

import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube

from esmvalcore.cmor._fixes.cmip5.bnu_esm import Cl as BaseCl
from esmvalcore.cmor._fixes.cmip5.ccsm4 import AllVars, Cl, Csoil, So
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


def test_get_cl_fix():
    """Test getting of fix."""
    fixes = Fix.get_fixes('CMIP5', 'CCSM4', 'Amon', 'cl')
    assert Cl(None) in fixes


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is BaseCl


class TestCsoil(unittest.TestCase):
    """Test cSoil fixes."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0, 1.e33], var_name='cSoil', units='kg m-2')
        self.fix = Csoil(None)

    def test_get(self):
        """Test fix get."""
        assert Csoil(None) in Fix.get_fixes('CMIP5', 'CCSM4', 'Lmon', 'cSoil')

    def test_fix_data(self):
        """Test data fix."""
        cube = self.fix.fix_data(self.cube)
        expected = np.ma.masked_array([1.0, 1.0], [False, True])
        self.assertTrue(np.all(cube.data == expected))


class TestsAllVars(unittest.TestCase):
    """Test for rlut fixes."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0, 2.0], var_name='rlut')
        self.cube.add_dim_coord(
            DimCoord([0.50001, 1.499999],
                     standard_name='latitude',
                     bounds=[
                         [0.00001, 0.999999],
                         [1.00001, 1.999999],
                     ]), 0)
        self.fix = AllVars(None)

    def test_get(self):
        """Test fix get."""
        self.assertListEqual(Fix.get_fixes('CMIP5', 'CCSM4', 'Amon', 'rlut'),
                             [AllVars(None)])

    def test_fix_metadata(self):
        """Check that latitudes values are rounded."""
        cube = self.fix.fix_metadata([self.cube])[0]

        latitude = cube.coord('latitude')
        self.assertTrue(np.all(latitude.points == np.array([0.5000, 1.5000])))
        self.assertTrue(
            np.all(latitude.bounds == np.array([[0.0000, 1.0000],
                                                [1.0000, 2.0000]])))


class TestSo(unittest.TestCase):
    """Tests for so fixes."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0, 2.0], var_name='so', units='1.0')
        self.vardef = get_var_info('CMIP5', 'Omon', self.cube.var_name)
        self.fix = So(self.vardef)

    def test_get(self):
        """Test fix get."""
        assert So(self.vardef) in Fix.get_fixes('CMIP5', 'CCSM4', 'Amon', 'so')

    def test_fix_metadata(self):
        """Checks that units are changed to the correct value."""
        cube = self.fix.fix_metadata([self.cube])[0]
        self.assertEqual('1e3', cube.units.origin)
