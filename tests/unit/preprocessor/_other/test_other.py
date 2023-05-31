"""Unit tests for the :func:`esmvalcore.preprocessor._other` module."""

import unittest

import dask.array as da
import iris.coord_categorisation
import iris.coords
import numpy as np
from cf_units import Unit
from iris.cube import Cube
from numpy.testing import assert_array_equal

from esmvalcore.preprocessor import PreprocessorFile
from esmvalcore.preprocessor._other import (
    _group_products,
    clip,
    get_array_module,
)


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


def test_group_products_string_list():
    products = [
        PreprocessorFile(
            filename='A_B.nc',
            attributes={
                'project': 'A',
                'dataset': 'B',
            },
        ),
        PreprocessorFile(
            filename='A_C.nc',
            attributes={
                'project': 'A',
                'dataset': 'C',
            }
        ),
    ]
    grouped_by_string = _group_products(products, 'project')
    grouped_by_list = _group_products(products, ['project'])

    assert grouped_by_list == grouped_by_string


def test_get_array_module_da():

    npx = get_array_module(da.array([1, 2]))
    assert npx is da


def test_get_array_module_np():

    npx = get_array_module(np.array([1, 2]))
    assert npx is np


def test_get_array_module_mixed():

    npx = get_array_module(da.array([1]), np.array([1]))
    assert npx is da


if __name__ == '__main__':
    unittest.main()
