"""Test EC-EARTH fixes."""
import unittest

import numpy as np
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from esmvalcore.cmor._fixes.cmip5.ec_earth import Areacello, Sftlf, Sic, Tas
from esmvalcore.cmor.fix import Fix


class TestSic(unittest.TestCase):
    """Test sic fixes."""
    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='sic', units='J')
        self.fix = Sic(None)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(Fix.get_fixes('CMIP5', 'EC-EARTH', 'Amon', 'sic'),
                             [Sic(None)])

    def test_fix_data(self):
        """Test data fix."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 100)
        self.assertEqual(cube.units, Unit('J'))


class TestSftlf(unittest.TestCase):
    """Test sftlf fixes."""
    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='sftlf', units='J')
        self.fix = Sftlf(None)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'EC-EARTH', 'Amon', 'sftlf'), [Sftlf(None)])

    def test_fix_data(self):
        """Test data fix."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 100)
        self.assertEqual(cube.units, Unit('J'))


class TestTas(unittest.TestCase):
    """Test tas fixes."""
    def setUp(self):
        """Prepare tests."""

        height_coord = DimCoord(2.,
                                standard_name='height',
                                long_name='height',
                                var_name='height',
                                units='m',
                                bounds=None,
                                attributes={'positive': 'up'})

        time_coord = DimCoord(
            1.,
            standard_name='time',
            var_name='time',
            units=Unit('days since 2070-01-01 00:00:00', calendar='gregorian'),
        )

        self.height_coord = height_coord

        self.cube_without = CubeList([Cube([3.0], var_name='tas')])
        self.cube_without[0].add_aux_coord(time_coord, 0)

        self.cube_with = CubeList([Cube([3.0], var_name='tas')])
        self.cube_with[0].add_aux_coord(height_coord, ())
        self.cube_with[0].add_aux_coord(time_coord, 0)
        self.cube_with[0].coord('time').long_name = 'time'

        self.fix = Tas(None)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(Fix.get_fixes('CMIP5', 'EC-EARTH', 'Amon', 'tas'),
                             [Tas(None)])

    def test_tas_fix_metadata(self):
        """Test metadata fix."""

        out_cube_without = self.fix.fix_metadata(self.cube_without)

        # make sure this does not raise an error
        out_cube_with = self.fix.fix_metadata(self.cube_with)

        coord = out_cube_without[0].coord('height')
        assert coord == self.height_coord

        coord = out_cube_without[0].coord('time')
        assert coord.long_name == "time"

        coord = out_cube_with[0].coord('height')
        assert coord == self.height_coord

        coord = out_cube_with[0].coord('time')
        assert coord.long_name == "time"


class TestAreacello(unittest.TestCase):
    """Test areacello fixes."""
    def setUp(self):
        """Prepare tests."""

        latitude = Cube(
            np.ones((2, 2)),
            standard_name='latitude',
            long_name='latitude',
            var_name='lat',
            units='degrees_north',
        )

        longitude = Cube(
            np.ones((2, 2)),
            standard_name='longitude',
            long_name='longitude',
            var_name='lon',
            units='degrees_north',
        )

        self.cubes = CubeList([
            Cube(
                np.ones((2, 2)),
                var_name='areacello',
                long_name='Areas of grid cell',
            ), latitude, longitude
        ])

        self.fix = Areacello(None)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'EC-EARTH', 'Omon', 'areacello'),
            [Areacello(None)],
        )

    def test_areacello_fix_metadata(self):
        """Test metadata fix."""

        out_cube = self.fix.fix_metadata(self.cubes)
        assert len(out_cube) == 1

        out_cube[0].coord('latitude')
        out_cube[0].coord('longitude')
