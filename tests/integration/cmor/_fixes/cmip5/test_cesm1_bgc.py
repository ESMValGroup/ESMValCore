"""Tests for CESM1-BGC fixes."""
import unittest

import numpy as np
from cf_units import Unit
from iris.cube import Cube

from esmvalcore.cmor._fixes.cmip5.cesm1_bgc import Co2, Gpp, Nbp
from esmvalcore.cmor.fix import Fix


class TestCo2(unittest.TestCase):
    """Tests for co2."""
    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='co2', units='J')
        self.fix = Co2()

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(Fix.get_fixes('CMIP5', 'CESM1-BGC', 'co2'),
                             [Co2()])

    def test_fix_data(self):
        """Test fix to set units correctly."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 28.966 / 44.0)
        self.assertEqual(cube.units, Unit('J'))


class TestGpp(unittest.TestCase):
    """Tests for gpp."""
    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0, 1.0e33, 2.0])
        self.fix = Gpp()

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(Fix.get_fixes('CMIP5', 'CESM1-BGC', 'gpp'),
                             [Gpp()])

    def test_fix_data(self):
        """Test fix to set missing values correctly."""
        cube = self.fix.fix_data(self.cube)
        np.testing.assert_allclose(cube.data[0], 1.0)
        np.testing.assert_allclose(cube.data[2], 2.0)
        assert not np.ma.is_masked(cube.data[0])
        assert np.ma.is_masked(cube.data[1])
        assert not np.ma.is_masked(cube.data[2])


class TestNbp(TestGpp):
    """Tests for nbp."""
    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0, 1.0e33, 2.0])
        self.fix = Nbp()

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(Fix.get_fixes('CMIP5', 'CESM1-BGC', 'nbp'),
                             [Nbp()])
