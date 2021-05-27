"""Unit tests for the :func:`esmvalcore.preprocessor._other` module."""

import unittest

import dask.array as da
import numpy as np
import iris.coord_categorisation
import pytest
from iris.coords import DimCoord

from cf_units import Unit
from iris.cube import Cube
from numpy.testing import assert_array_equal

from esmvalcore.preprocessor._other import clip, fix_cubes_endianness


class TestOther(unittest.TestCase):
    """Test class for _other."""
    def test_clip(self):
        """Test clip function."""
        cube = Cube(np.array([-10, 0, 10]))
        cube.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(3),
                standard_name='time',
                units=Unit('days since 1950-01-01 00:00:00',
                           calendar='gregorian'),
            ),
            0,
        )
        # Cube needs to be copied, since it is modified in-place and test cube
        # should not change.
        assert_array_equal(clip(cube.copy(), 0, None).data,
                           np.array([0, 0, 10]))
        assert_array_equal(clip(cube.copy(), None, 0).data,
                           np.array([-10, 0, 0]))
        assert_array_equal(clip(cube.copy(), -1, 2).data,
                           np.array([-1, 0, 2]))
        # Masked cube TODO
        # No parameters specified
        with self.assertRaises(ValueError):
            clip(cube, None, None)
        # Maximum lower than minimum
        with self.assertRaises(ValueError):
            clip(cube, 10, 8)

    @pytest.mark.parametrize("lazy", (True, False))
    def test_fix_cubes_endianness(self, lazy=True):

        def make_cube(data, big_endian=False):
            dtype = ">f8" if big_endian else "<f8"
            data = np.array(data, dtype=dtype)
            if lazy:
                data = da.from_array(data)
            # We reuse the same array for the coords to simplify
            coords = data.copy()
            ocube = Cube(
                data,
                var_name='sample',
                dim_coords_and_dims=(
                    (
                        DimCoord(
                            coords,
                            var_name='time',
                            standard_name='time',
                            units='days since 1950-01-01'
                        ),
                        0
                    ),
                )
            )
            return ocube

        big_endian_cube = make_cube([7., 8.], big_endian=True)
        little_endian_cube = make_cube([7., 8.], big_endian=False)
        test_cubes = [make_cube(vals) for vals in [(1, 2), (3, 4), (5, 6)]]
        test_cubes += [big_endian_cube]
        expected_cubes = [c.copy() for c in test_cubes[:-1]] + [little_endian_cube]
        actual_cubes = fix_cubes_endianness(test_cubes)
        self.assertEqual(actual_cubes, expected_cubes)


if __name__ == '__main__':
    unittest.main()
