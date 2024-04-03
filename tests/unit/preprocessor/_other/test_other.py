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
    cube_data = np.swapaxes(cube_data, 0, -1)
    cube = get_3d_cube(
        cube_data, standard_name='air_temperature', var_name='tas', units='K'
    )
    return cube


def assert_metadata(cube, normalization=None):
    """Assert correct metadata."""
    assert cube.standard_name is None
    assert cube.var_name == 'histogram_tas'
    assert cube.long_name == 'Histogram'
    if normalization == 'integral':
        assert cube.units == 'K-1'
    else:
        assert cube.units == '1'
    if normalization is None:
        assert cube.attributes == {'normalization': 'none'}
    else:
        assert cube.attributes == {'normalization': normalization}
    assert cube.coords('air_temperature')
    bin_coord = cube.coord('air_temperature')
    assert bin_coord.standard_name == 'air_temperature'
    assert bin_coord.var_name == 'tas'
    assert bin_coord.long_name is None
    assert bin_coord.units == 'K'
    assert bin_coord.attributes == {}


@pytest.mark.parametrize('lazy', [False, True])
def test_histogram_defaults(cube, lazy):
    """Test `histogram`."""
    if lazy:
        cube.data = cube.lazy_data()
    input_cube = cube.copy()

    result = histogram(input_cube)

    assert input_cube == cube
    assert_metadata(result)
    assert result.cell_methods == (
        CellMethod('histogram', ('time', 'latitude', 'longitude')),
    )
    assert result.shape == (10,)
    if lazy:
        assert result.has_lazy_data()
    else:
        assert not result.has_lazy_data()
    assert result.dtype == np.float32
    np.testing.assert_allclose(
        result.data, [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    )
    np.testing.assert_allclose(result.data.mask, [False] * 10)
    bin_coord = result.coord('air_temperature')
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


@pytest.mark.parametrize('normalization', [None, 'sum', 'integral'])
@pytest.mark.parametrize('weights', [False, None])
@pytest.mark.parametrize('lazy', [False, True])
def test_histogram_over_time(cube, lazy, weights, normalization):
    """Test `histogram`."""
    if lazy:
        cube.data = cube.lazy_data()
    input_cube = cube.copy()

    result = histogram(
        input_cube,
        coords=['time'],
        bins=[4.5, 6.5, 8.5, 10.5],
        bin_range=(4.5, 10.5),
        weights=weights,
        normalization=normalization,
    )

    assert input_cube == cube
    assert_metadata(result, normalization=normalization)
    assert result.coord('latitude') == input_cube.coord('latitude')
    assert result.coord('longitude') == input_cube.coord('longitude')
    assert result.cell_methods == (CellMethod('histogram', ('time',)),)
    assert result.shape == (2, 2, 3)
    if lazy:
        assert result.has_lazy_data()
    else:
        assert not result.has_lazy_data()
    assert result.dtype == np.float32
    if normalization == 'integral':
        expected_data = np.ma.masked_invalid([
            [[np.nan, np.nan, np.nan], [0.5, 0.0, 0.0]],
            [[np.nan, np.nan, np.nan], [0.25, 0.25, 0.0]],
        ])
    elif normalization == 'sum':
        expected_data = np.ma.masked_invalid([
            [[np.nan, np.nan, np.nan], [1.0, 0.0, 0.0]],
            [[np.nan, np.nan, np.nan], [0.5, 0.5, 0.0]],
        ])
    else:
        expected_data = np.ma.masked_invalid([
            [[np.nan, np.nan, np.nan], [1.0, 0.0, 0.0]],
            [[np.nan, np.nan, np.nan], [1.0, 1.0, 0.0]],
        ])
    np.testing.assert_allclose(result.data, expected_data)
    np.testing.assert_allclose(result.data.mask, expected_data.mask)
    bin_coord = result.coord('air_temperature')
    bin_coord.shape == (10,)
    bin_coord.dtype == np.float64
    bin_coord.bounds_dtype == np.float64
    np.testing.assert_allclose(bin_coord.points, [5.5, 7.5, 9.5])
    np.testing.assert_allclose(
        bin_coord.bounds, [[4.5, 6.5], [6.5, 8.5], [8.5, 10.5]],
    )


@pytest.mark.parametrize('normalization', [None, 'sum', 'integral'])
@pytest.mark.parametrize('lazy', [False, True])
def test_histogram_fully_masked(cube, lazy, normalization):
    """Test `histogram`."""
    cube.data = np.ma.masked_all((2, 2, 2), dtype=np.float32)
    if lazy:
        cube.data = cube.lazy_data()

    result = histogram(cube, bin_range=(0, 10), normalization=normalization)

    assert_metadata(result, normalization=normalization)
    assert result.cell_methods == (
        CellMethod('histogram', ('time', 'latitude', 'longitude')),
    )
    assert result.shape == (10,)
    if lazy:
        assert result.has_lazy_data()
    else:
        assert not result.has_lazy_data()
    assert result.dtype == np.float32
    np.testing.assert_allclose(result.data, np.ma.masked_all(10,))
    np.testing.assert_equal(result.data.mask, [True] * 10)
    bin_coord = result.coord('air_temperature')
    bin_coord.shape == (10,)
    bin_coord.dtype == np.float64
    bin_coord.bounds_dtype == np.float64
    np.testing.assert_allclose(
        bin_coord.points,
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
    )
    np.testing.assert_allclose(
        bin_coord.bounds,
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
            [9.0, 10.0],
        ],
    )


@pytest.mark.parametrize('normalization', [None, 'sum', 'integral'])
@pytest.mark.parametrize('weights', [True])
@pytest.mark.parametrize('lazy', [False, True])
def test_histogram_weights(cube, lazy, weights, normalization):
    """Test `histogram`."""
    if lazy:
        cube.data = cube.lazy_data()
    input_cube = cube.copy()

    result = histogram(
        input_cube,
        coords=['time', 'longitude'],
        bins=[0.0, 2.0, 4.0, 8.0],
        weights=weights,
        normalization=normalization,
    )

    assert input_cube == cube
    assert_metadata(result, normalization=normalization)
    assert result.coord('latitude') == input_cube.coord('latitude')
    assert result.cell_methods == (
        CellMethod('histogram', ('time', 'longitude')),
    )
    assert result.shape == (2, 3)
    if lazy:
        assert result.has_lazy_data()
    else:
        assert not result.has_lazy_data()
    assert result.dtype == np.float32
    print(result.data)
    if normalization == 'integral':
        expected_data = np.ma.masked_invalid(
            [[0.25, 0.0, 0.125], [0.0, 0.0, 0.25]]
        )
    elif normalization == 'sum':
        expected_data = np.ma.masked_invalid(
            [[0.5, 0.0, 0.5], [0.0, 0.0, 1.0]]
        )
    else:
        expected_data = np.ma.masked_invalid(
            [[8.0, 0.0, 8.0], [0.0, 0.0, 8.0]]
        )
    np.testing.assert_allclose(result.data, expected_data)
    np.testing.assert_allclose(result.data.mask, expected_data.mask)
    bin_coord = result.coord('air_temperature')
    bin_coord.shape == (10,)
    bin_coord.dtype == np.float64
    bin_coord.bounds_dtype == np.float64
    np.testing.assert_allclose(bin_coord.points, [1.0, 3.0, 6.0])
    np.testing.assert_allclose(
        bin_coord.bounds, [[0.0, 2.0], [2.0, 4.0], [4.0, 8.0]],
    )


@pytest.mark.parametrize('lazy', [False, True])
def test_histogram_fully_masked_no_bin_range(cube, lazy):
    """Test `histogram`."""
    cube.data = np.ma.masked_all((2, 2, 2), dtype=np.float32)
    if lazy:
        cube.data = cube.lazy_data()

    msg = (
        r"Cannot calculate histogram for bin_range=\(masked, masked\) \(or "
        r"for fully masked data when `bin_range` is not given\)"
    )
    with pytest.raises(ValueError, match=msg):
        histogram(cube)


if __name__ == '__main__':
    unittest.main()
