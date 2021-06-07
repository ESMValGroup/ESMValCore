"""Test EC-EARTH fixes."""
import unittest

import numpy as np

from cf_units import Unit
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError
from esmvalcore.cmor._fixes.cmip5.ec_earth import (
    Areacello,
    Pr,
    Sftlf,
    Sic,
    Tas,
    )
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


class TestPr(unittest.TestCase):
    """Test pr fixes."""
    def setUp(self):
        """Prepare tests."""

        wrong_time_coord = AuxCoord(
            points=[1.0, 2.0, 1.0, 2.0, 3.0],
            var_name='time',
            standard_name='time',
            units='days since 1850-01-01',
            )

        correct_time_coord = AuxCoord(
            points=[1.0, 2.0, 3.0],
            var_name='time',
            standard_name='time',
            units='days since 1850-01-01',
            )

        lat_coord = DimCoord(
            [0.0],
            standard_name='latitude',
            var_name='lat',
            )

        lon_coord = DimCoord(
            [0.0],
            standard_name='longitude',
            var_name='lon',
            )

        self.time_coord = correct_time_coord
        self.wrong_cube = CubeList([Cube(np.ones((5, 1, 1)),
                                         var_name='pr',
                                         units='kg m-2 s-1')])
        self.wrong_cube[0].add_aux_coord(wrong_time_coord, 0)
        self.wrong_cube[0].add_dim_coord(lat_coord, 1)
        self.wrong_cube[0].add_dim_coord(lon_coord, 2)
        self.correct_cube = CubeList([Cube(np.ones(3),
                                           var_name='pr',
                                           units='kg m-2 s-1')])
        self.correct_cube[0].add_aux_coord(correct_time_coord, 0)

        self.fix = Pr(None)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'EC-EARTH', 'Amon', 'pr'),
            [Pr(None)],
        )

    def test_pr_fix_metadata(self):
        """Test metadata fix."""

        out_wrong_cube = self.fix.fix_metadata(self.wrong_cube)
        out_correct_cube = self.fix.fix_metadata(self.correct_cube)

        time = out_wrong_cube[0].coord('time')
        assert time == self.time_coord

        time = out_correct_cube[0].coord('time')
        assert time == self.time_coord

    def test_pr_fix_metadata_no_time(self):
        """Test metadata fix with no time coord."""
        self.correct_cube[0].remove_coord('time')
        out_correct_cube = self.fix.fix_metadata(self.correct_cube)
        with self.assertRaises(CoordinateNotFoundError):
            out_correct_cube[0].coord('time')
