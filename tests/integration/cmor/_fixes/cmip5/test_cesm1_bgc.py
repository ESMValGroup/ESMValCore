"""Tests for CESM1-BGC fixes."""
import unittest

import numpy as np
from iris.cube import Cube

from esmvalcore.cmor._fixes.cmip5.cesm1_bgc import Gpp, Nbp
from esmvalcore.cmor.fix import Fix


class TestGpp(unittest.TestCase):
    """Tests for gpp."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0, 1.0e33, 2.0])
        self.fix = Gpp(None)

    def test_get(self):
        """Test fix get."""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'CESM1-BGC', 'Amon', 'gpp'), [Gpp(None)])

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
        self.fix = Nbp(None)

    def test_get(self):
        """Test fix get."""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'CESM1-BGC', 'Amon', 'nbp'), [Nbp(None)])
