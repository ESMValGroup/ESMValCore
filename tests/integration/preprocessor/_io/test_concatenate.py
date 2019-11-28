"""Integration tests for :func:`esmvalcore.preprocessor._io.concatenate`."""

import unittest

from cf_units import Unit
import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube
import iris.exceptions as Exc

from esmvalcore.preprocessor import _io


class TestConcatenate(unittest.TestCase):
    """Tests for :func:`esmvalcore.preprocessor._io.concatenate`."""

    def setUp(self):
        """Start tests."""
        self._model_coord = DimCoord(
            [1., 2.], var_name='time', standard_name='time',
            units='days since 1950-01-01'
        )
        self.raw_cubes = []
        self._add_cube([1., 2.], [1., 2.])
        self._add_cube([3., 4.], [3., 4.])
        self._add_cube([5., 6.], [5., 6.])

    def _add_cube(self, data, coord):
        self.raw_cubes.append(
            Cube(data,
                 var_name='sample',
                 dim_coords_and_dims=((self._model_coord.copy(coord), 0), )
                 )
        )

    def test_concatenate(self):
        """Test concatenation of two cubes."""
        concatenated = _io.concatenate(self.raw_cubes)
        self.assertTrue((concatenated.coord('time').points == np.array(
            [1, 2, 3, 4, 5, 6])).all())

    def test_concatenate_with_overlap(self):
        """Test concatenation of time overalapping cubes"""
        self._add_cube([6.5, 7.5], [6., 7.])
        concatenated = _io.concatenate(self.raw_cubes)
        self.assertTrue(np.allclose(
            concatenated.coord('time').points,
            np.array([1., 2., 3., 4., 5., 6., 7.])
        ))
        self.assertTrue(np.allclose(
            concatenated.data,
            np.array([1., 2., 3., 4., 5., 6.5, 7.5])
        ))

    def test_concatenate_with_overlap_2(self):
        """Test a more generic case."""
        self._add_cube([65., 75.], [3., 200.])
        self._add_cube([65., 75.], [1000., 7000.])
        concatenated = _io.concatenate(self.raw_cubes)
        self.assertTrue(np.allclose(
            concatenated.coord('time').points,
            np.array([1., 2., 3., 4., 5., 6., 1000., 7000.])
        ))

    def test_concatenate_with_overlap_same_start(self):
        """Test a more generic case."""
        cube1 = self.raw_cubes[0]
        raw_cubes = [cube1, ]
        time_coord = DimCoord(
            [1., 7.], var_name='time', standard_name='time',
            units='days since 1950-01-01'
        )
        raw_cubes.append(
            Cube([33., 55.],
                 var_name='sample',
                 dim_coords_and_dims=((time_coord, 0), )
                 )
        )
        concatenated = _io.concatenate(raw_cubes)
        self.assertTrue(np.allclose(
            concatenated.coord('time').points,
            np.array([1., 7.])
        ))
        raw_cubes.reverse()
        concatenated = _io.concatenate(raw_cubes)
        self.assertTrue(np.allclose(
            concatenated.coord('time').points,
            np.array([1., 7.])
        ))

    def test_concatenate_with_iris_exception(self):
        """Test a more generic case."""
        time_coord_1 = DimCoord(
            [1.5, 5., 7.], var_name='time', standard_name='time',
            units='days since 1950-01-01')
        cube1 = Cube([33., 55., 77.],
                     var_name='sample',
                     dim_coords_and_dims=((time_coord_1, 0), ))
        time_coord_2 = DimCoord(
            [1., 5., 7.], var_name='time', standard_name='time',
            units='days since 1950-01-01')
        cube2 = Cube([33., 55., 77.],
                     var_name='sample',
                     dim_coords_and_dims=((time_coord_2, 0), ))
        cubes_single_ovlp = [cube2, cube1]
        with self.assertRaises(Exc.ConcatenateError):
            _io.concatenate(cubes_single_ovlp)

    def test_concatenate_with_order(self):
        """Test a more generic case."""
        time_coord_1 = DimCoord(
            [1.5, 2., 5., 7.], var_name='time', standard_name='time',
            units='days since 1950-01-01')
        cube1 = Cube([33., 44., 55., 77.],
                     var_name='sample',
                     dim_coords_and_dims=((time_coord_1, 0), ))
        time_coord_2 = DimCoord(
            [1., 2., 5., 7., 100.], var_name='time', standard_name='time',
            units='days since 1950-01-01')
        cube2 = Cube([33., 44., 55., 77., 1000.],
                     var_name='sample',
                     dim_coords_and_dims=((time_coord_2, 0), ))
        cubes_ordered = [cube2, cube1]
        concatenated = _io.concatenate(cubes_ordered)
        self.assertTrue(np.allclose(
            concatenated.coord('time').points,
            np.array([1., 2., 5., 7., 100.])
        ))
        cubes_reverse = [cube1, cube2]
        concatenated = _io.concatenate(cubes_reverse)
        self.assertTrue(np.allclose(
            concatenated.coord('time').points,
            np.array([1., 2., 5., 7., 100.])
        ))

    def test_fail_on_calendar_concatenate_with_overlap(self):
        """Test fail of concatenation with overlap."""
        time_coord = DimCoord(
            [3., 7000.], var_name='time', standard_name='time',
            units=Unit('days since 1950-01-01', calendar='360_day')
        )
        self.raw_cubes.append(
            Cube([33., 55.],
                 var_name='sample',
                 dim_coords_and_dims=((time_coord, 0), )
                 )
        )
        with self.assertRaises(TypeError):
            _io.concatenate(self.raw_cubes)

    def test_fail_on_units_concatenate_with_overlap(self):
        """Test fail of concatenation with overlap."""
        time_coord_1 = DimCoord(
            [3., 7000.], var_name='time', standard_name='time',
            units=Unit('days since 1950-01-01', calendar='360_day')
        )
        time_coord_2 = DimCoord(
            [3., 9000.], var_name='time', standard_name='time',
            units=Unit('days since 1950-01-01', calendar='360_day')
        )
        time_coord_3 = DimCoord(
            [3., 9000.], var_name='time', standard_name='time',
            units=Unit('days since 1850-01-01', calendar='360_day')
        )
        raw_cubes = []
        raw_cubes.append(
            Cube([33., 55.],
                 var_name='sample',
                 dim_coords_and_dims=((time_coord_1, 0), )
                 )
        )
        raw_cubes.append(
            Cube([33., 55.],
                 var_name='sample',
                 dim_coords_and_dims=((time_coord_2, 0), )
                 )
        )
        raw_cubes.append(
            Cube([33., 55.],
                 var_name='sample',
                 dim_coords_and_dims=((time_coord_3, 0), )
                 )
        )
        with self.assertRaises(ValueError):
            _io.concatenate(raw_cubes)

    def test_fail_metadata_differs(self):
        """Test exception raised if two cubes have different metadata."""
        self.raw_cubes[0].units = 'm'
        with self.assertRaises(ValueError):
            _io.concatenate(self.raw_cubes)

    def test_fix_attributes(self):
        """Test fixing attributes for concatenation."""
        identical_attrs = {
            'int': 42,
            'float': 3.1415,
            'bool': True,
            'str': 'Hello, world',
            'list': [1, 1, 2, 3, 5, 8, 13],
            'tuple': (1, 2, 3, 4, 5),
            'dict': {
                1: 'one',
                2: 'two',
                3: 'three'
            },
            'nparray': np.arange(42),
        }
        differing_attrs = [
            {
                'new_int': 0,
                'new_str': 'hello',
                'new_nparray': np.arange(3),
                'mix': np.arange(2),
            },
            {
                'new_int': 1,
                'new_str': 'world',
                'new_list': [1, 1, 2],
                'new_tuple': (0, 1),
                'new_dict': {
                    0: 'zero',
                },
                'mix': {
                    1: 'one',
                },
            },
            {
                'new_str': '!',
                'new_list': [1, 1, 2, 3],
                'new_tuple': (1, 2, 3),
                'new_dict': {
                    0: 'zeroo',
                    1: 'one',
                },
                'new_nparray': np.arange(2),
                'mix': False,
            },
        ]
        resulting_attrs = {
            'new_int': '0;1',
            'new_str': 'hello;world;!',
            'new_nparray': '[0 1 2];[0 1]',
            'new_list': '[1, 1, 2];[1, 1, 2, 3]',
            'new_tuple': '(0, 1);(1, 2, 3)',
            'new_dict': "{0: 'zero'};{0: 'zeroo', 1: 'one'}",
            'mix': "[0 1];{1: 'one'};False",
        }
        resulting_attrs.update(identical_attrs)

        for idx in range(3):
            self.raw_cubes[idx].attributes = identical_attrs
            self.raw_cubes[idx].attributes.update(differing_attrs[idx])
        _io._fix_cube_attributes(self.raw_cubes)  # noqa
        for cube in self.raw_cubes:
            self.assertTrue(cube.attributes == resulting_attrs)
