"""Unit tests for the :func:`esmvalcore.preprocessor._other` module."""

import unittest

import dask.array as da
import iris.coord_categorisation
import iris.coords
import numpy as np
import pytest
from cf_units import Unit
from iris.coords import CellMethod
from iris.cube import Cube
from numpy.testing import assert_array_equal

from esmvalcore.preprocessor import PreprocessorFile
from esmvalcore.preprocessor._other import (
    _group_products,
    clip,
    get_array_module,
    histogram,
)
from tests.unit.preprocessor._compare_with_refs.test_compare_with_refs import (
    get_3d_cube,
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


@pytest.fixture
def cube():
    """Regular cube."""
    cube_data = np.ma.masked_inside(
        np.arange(8.0, dtype=np.float32).reshape(2, 2, 2), 1.5, 3.5
    )
    cube = get_3d_cube(
        cube_data, standard_name='air_temperature', var_name='tas', units='K'
    )
    return cube


@pytest.mark.parametrize('lazy', [False, True])
def test_histogram_defaults(cube, lazy):
    """Test `histogram`."""
    if lazy:
        cube.data = cube.lazy_data()
    input_cube = cube.copy()

    result = histogram(input_cube)

    assert input_cube == cube
    assert result.shape == (10,)
    if lazy:
        assert result.has_lazy_data()
    else:
        assert not result.has_lazy_data()
    np.testing.assert_allclose(
        result.data, [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    )
    assert result.dtype == np.float32
    assert result.var_name == 'histogram_tas'
    assert result.long_name == 'Histogram'
    assert result.standard_name == 'air_temperature'
    assert result.units == '1'
    assert result.cell_methods == (
        CellMethod('histogram', ('time', 'latitude', 'longitude')),
    )
    assert result.coords('Histogram Bin')
    bin_coord = result.coord('Histogram Bin')
    bin_coord.shape == (10,)
    bin_coord.dtype == np.float64
    bin_coord.bounds_dtype == np.float64
    np.testing.assert_allclose(
        bin_coord.points,
        [0.35, 1.05, 1.75, 2.45, 3.15, 3.85, 4.55, 5.25, 5.95, 6.65],
    )
    np.testing.assert_allclose(
        bin_coord.bounds,
        [
            [0.0, 0.7],
            [0.7, 1.4],
            [1.4, 2.1],
            [2.1, 2.8],
            [2.8, 3.5],
            [3.5, 4.2],
            [4.2, 4.9],
            [4.9, 5.6],
            [5.6, 6.3],
            [6.3, 7.0],
        ],
    )
    assert bin_coord.var_name == 'bin'
    assert bin_coord.long_name == 'Histogram Bin'
    assert bin_coord.standard_name is None
    assert bin_coord.units == 'K'
    assert bin_coord.attributes == {}


if __name__ == '__main__':
    unittest.main()
