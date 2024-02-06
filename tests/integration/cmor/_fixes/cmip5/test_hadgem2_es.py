"""Test HADGEM2-ES fixes."""
import unittest

import dask.array as da
import iris.coords
import iris.cube
import numpy as np

from esmvalcore.cmor._fixes.cmip5.hadgem2_es import O2, AllVars, Cl
from esmvalcore.cmor._fixes.common import ClFixHybridHeightCoord
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor.fix import Fix
from tests import assert_array_equal


class TestAllVars(unittest.TestCase):
    """Test allvars fixes."""

    def test_get(self):
        """Test fix get."""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'HADGEM2-ES', 'Amon', 'tas'),
            [AllVars(None), GenericFix(None)])

    @staticmethod
    def test_clip_latitude():
        cube = iris.cube.Cube(
            da.arange(2, dtype=np.float32),
            aux_coords_and_dims=[
                (
                    iris.coords.AuxCoord(
                        da.asarray([90., 91.]),
                        bounds=da.asarray([[89.5, 90.5], [90.5, 91.5]]),
                        standard_name='latitude',
                    ),
                    0,
                ),
            ],
        )
        fix = AllVars(None)
        cubes = fix.fix_metadata([cube])
        assert len(cubes) == 1
        coord = cubes[0].coord('latitude')
        assert coord.has_lazy_points()
        assert coord.has_lazy_bounds()
        assert_array_equal(coord.points, np.array([90., 90]))
        assert_array_equal(coord.bounds, np.array([[89.5, 90.], [90., 90.]]))


class TestO2(unittest.TestCase):
    """Test o2 fixes."""

    def test_get(self):
        """Test fix get."""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'HADGEM2-ES', 'Amon', 'o2'),
            [O2(None), AllVars(None), GenericFix(None)])


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP5', 'HadGEM2-ES', 'Amon', 'cl')
    assert fix == [Cl(None), AllVars(None), GenericFix(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is ClFixHybridHeightCoord
