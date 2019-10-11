import unittest

import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube
from iris.util import monotonic
from cf_units import Unit

from esmvalcore.cmor._fixes.cmip6.bcc_csm2_mr import allvars


class BrokenDimCoord(DimCoord):
    """DimCoord does not allow to specifiy a non-monotonic coordinate,
       which is exactly what we need for the tests. Workaround: define
       new class with checks disabled"""
    def _new_points_requirements(self, points):
        pass

    def _new_bounds_requirements(self, bounds):
        pass

    """is_monotonic has to be (re)defined here because the function
       inherited from DimCoord always returns true"""
    def is_monotonic(self):
        if self.points is not None:
            if not monotonic(self.points, strict=True):
                return False

        if self.has_bounds():
            for b_index in range(self.nbounds):
                if not monotonic(self.bounds[..., b_index], strict=True):
                    return False

        return True


class TestAll(unittest.TestCase):
    """Test for allvars fix"""

    def setUp(self):
        """Prepare tests"""
        self.cube = Cube([1.0, 2.0, 3.0, 4.0, 5.0], var_name='var')
        self.cube.add_dim_coord(
            BrokenDimCoord([15.5, 45., 74.5, 0.0, 0.0],
                           standard_name='time',
                           bounds=[
                                   [0.0, 31.0],
                                   [31.0, 59.0],
                                   [59.0, 90.0],
                                   [0.0, 0.0],
                                   [0.0, 0.0],
                           ],
                           units=Unit('days since 1850-01-01 00:00:00', calendar='365_day')
                          ), 0)

        self.fix = allvars()

    def test_fix_metadata(self):
        """Check that latitudes values are rounded"""
        cube = self.fix.fix_metadata([self.cube])[0]
        time = cube.coord('time')
        self.assertTrue(time.is_monotonic())

