"""Unit tests for :func:`esmvalcore.preprocessor.mask_multimodel`."""

from unittest import mock

import iris
import numpy as np
import pytest

from esmvalcore.preprocessor import PreprocessorFile as PreprocessorFileBase
from esmvalcore.preprocessor._mask import (
    _get_shape,
    _multimodel_mask_cubes,
    _multimodel_mask_products,
    mask_multimodel,
)


class PreprocessorFile(mock.Mock):
    """Mocked PreprocessorFile."""

    def __init__(self, cubes, filename, **kwargs):
        """Initialize with cubes."""
        super().__init__(spec=PreprocessorFileBase, **kwargs)
        self.filename = filename
        self.cubes = cubes
        self.mock_ancestors = set()
        self.wasderivedfrom = mock.Mock(side_effect=self.mock_ancestors.add)


def assert_array_equal(array_1, array_2):
    """Assert that (masked) array 1 equals (masked) array 2."""
    if np.ma.is_masked(array_1) or np.ma.is_masked(array_2):
        np.testing.assert_array_equal(np.ma.getmaskarray(array_1),
                                      np.ma.getmaskarray(array_2))
        mask = np.ma.getmaskarray(array_1)
        np.testing.assert_array_equal(array_1[~mask], array_2[~mask])
    else:
        np.testing.assert_array_equal(array_1, array_2)


def _get_cube(ndim):
    """Create stock cube."""
    time_coord = iris.coords.DimCoord([1], var_name='time',
                                      standard_name='time',
                                      units='days since 1850-01-01')
    lev_coord = iris.coords.DimCoord([10, 5], var_name='plev',
                                     standard_name='air_pressure', units='hPa')
    lat_coord = iris.coords.DimCoord([1], var_name='lat',
                                     standard_name='latitude', units='degrees')
    lon_coord = iris.coords.DimCoord([0, 1], var_name='lon',
                                     standard_name='longitude',
                                     units='degrees')
    x_coord = iris.coords.DimCoord([-1], var_name='x',
                                   long_name='Arbitrary coordinate',
                                   units='no unit')

    if ndim == 0:
        cube_data = 42
        coord_spec = []
    elif ndim == 1:
        cube_data = [0]
        coord_spec = [(time_coord, 0)]
    elif ndim == 2:
        cube_data = np.arange(1 * 2).reshape(1, 2)
        coord_spec = [(time_coord, 0), (lev_coord, 1)]
    elif ndim == 3:
        cube_data = np.arange(1 * 1 * 2).reshape(1, 1, 2)
        coord_spec = [(time_coord, 0), (lat_coord, 1), (lon_coord, 2)]
    elif ndim == 4:
        cube_data = np.arange(1 * 2 * 1 * 2).reshape(1, 2, 1, 2)
        coord_spec = [(time_coord, 0), (lev_coord, 1), (lat_coord, 2),
                      (lon_coord, 3)]
    elif ndim == 5:
        cube_data = np.arange(1 * 2 * 1 * 2 * 1).reshape(1, 2, 1, 2, 1)
        coord_spec = [(time_coord, 0), (lev_coord, 1), (lat_coord, 2),
                      (lon_coord, 3), (x_coord, 4)]
    else:
        assert False, f"Invalid ndim: {ndim}"

    cube = iris.cube.Cube(cube_data, var_name='y', long_name='Y',
                          dim_coords_and_dims=coord_spec)
    return cube


@pytest.fixture
def cube_0d():
    """0D cube."""
    return _get_cube(0)


@pytest.fixture
def cube_1d():
    """1D cube."""
    return _get_cube(1)


@pytest.fixture
def cube_2d():
    """2D cube."""
    return _get_cube(2)


@pytest.fixture
def cube_3d():
    """3D cube."""
    return _get_cube(3)


@pytest.fixture
def cube_4d():
    """4D cube."""
    return _get_cube(4)


@pytest.fixture
def cube_5d():
    """5D cube."""
    return _get_cube(5)


def test_get_shape(cube_0d, cube_5d):
    """Test ``_get_shape``."""
    cubes = iris.cube.CubeList([cube_0d])
    assert _get_shape(cubes) == ()

    cubes = iris.cube.CubeList([cube_0d, cube_0d])
    assert _get_shape(cubes) == ()

    cubes = iris.cube.CubeList([cube_5d])
    assert _get_shape(cubes) == (1, 2, 1, 2, 1)

    cubes = iris.cube.CubeList([cube_5d, cube_5d])
    assert _get_shape(cubes) == (1, 2, 1, 2, 1)

    cubes = iris.cube.CubeList([cube_0d, cube_5d])
    msg = 'Expected cubes with identical shapes, got shapes'
    with pytest.raises(ValueError, match=msg):
        _get_shape(cubes)


def test_multimodel_mask_cubes_0d(cube_0d):
    """Test ``_multimodel_mask_cubes`` with 0D cubes."""
    cubes = iris.cube.CubeList([cube_0d, cube_0d])
    out_cubes = _multimodel_mask_cubes(cubes, ())
    assert out_cubes is cubes
    for cube in out_cubes:
        assert_array_equal(cube.data, 42)

    m_array = np.ma.masked_equal(33, 33)
    cube_masked = cube_0d.copy(m_array)
    cubes = iris.cube.CubeList([cube_0d, cube_masked])
    out_cubes = _multimodel_mask_cubes(cubes, ())
    assert out_cubes is cubes
    assert_array_equal(out_cubes[0].data, m_array)
    assert_array_equal(out_cubes[1].data, m_array)


def test_multimodel_mask_cubes_3d(cube_3d):
    """Test ``_multimodel_mask_cubes`` with 3D cubes."""
    cubes = iris.cube.CubeList([cube_3d, cube_3d])
    out_cubes = _multimodel_mask_cubes(cubes, (1, 1, 2))
    assert out_cubes is cubes
    for cube in out_cubes:
        assert_array_equal(cube.data, [[[0, 1]]])

    m_array = np.ma.masked_equal([[[42, 33]]], 33)
    cube_masked = cube_3d.copy(m_array)
    cubes = iris.cube.CubeList([cube_3d, cube_masked])
    out_cubes = _multimodel_mask_cubes(cubes, (1, 1, 2))
    assert out_cubes is cubes
    assert_array_equal(out_cubes[0].data, np.ma.masked_equal([[[0, 33]]], 33))
    assert_array_equal(out_cubes[1].data, m_array)


def test_multimodel_mask_products_1d(cube_1d):
    """Test ``_multimodel_mask_products`` with 1D cubes."""
    products = [
        PreprocessorFile(iris.cube.CubeList([cube_1d]), 'A'),
        PreprocessorFile(iris.cube.CubeList([cube_1d, cube_1d]), 'B'),
    ]
    out_products = _multimodel_mask_products(products, (1,))
    assert out_products == products
    assert out_products[0].filename == 'A'
    assert out_products[0].cubes == iris.cube.CubeList([cube_1d])
    assert out_products[1].filename == 'B'
    assert out_products[1].cubes == iris.cube.CubeList([cube_1d, cube_1d])
    for product in out_products:
        product.wasderivedfrom.assert_not_called()
        assert product.mock_ancestors == set()

    m_array = np.ma.masked_equal([33], 33)
    cube_masked = cube_1d.copy(m_array)
    prod_a = PreprocessorFile(iris.cube.CubeList([cube_1d]), 'A')
    prod_b = PreprocessorFile(iris.cube.CubeList([cube_masked]), 'B')
    products = [prod_a, prod_b]
    out_products = _multimodel_mask_products(products, (1,))
    assert out_products == products
    assert out_products[0].filename == 'A'
    assert_array_equal(out_products[0].cubes[0].data, m_array)
    assert out_products[1].filename == 'B'
    assert out_products[1].cubes == iris.cube.CubeList([cube_masked])
    out_products[0].wasderivedfrom.assert_called_once_with(prod_b)
    assert out_products[0].mock_ancestors == {prod_b}
    out_products[1].wasderivedfrom.assert_not_called()
    assert out_products[1].mock_ancestors == set()


def test_multimodel_mask_products_5d(cube_5d):
    """Test ``_multimodel_mask_products`` with 5D cubes."""
    products = [
        PreprocessorFile(iris.cube.CubeList([cube_5d]), 'A'),
        PreprocessorFile(iris.cube.CubeList([cube_5d, cube_5d]), 'B'),
    ]
    out_products = _multimodel_mask_products(products, (1, 2, 1, 2, 1))
    assert out_products == products
    assert out_products[0].filename == 'A'
    assert out_products[0].cubes == iris.cube.CubeList([cube_5d])
    assert out_products[1].filename == 'B'
    assert out_products[1].cubes == iris.cube.CubeList([cube_5d, cube_5d])
    for product in out_products:
        product.wasderivedfrom.assert_not_called()
        assert product.mock_ancestors == set()

    m_array_1 = np.ma.masked_equal([[[[[33], [1]]], [[[33], [2]]]]], 33)
    m_array_2 = np.ma.masked_equal([[[[[1], [1]]], [[[3], [33]]]]], 33)
    cube_masked_1 = cube_5d.copy(m_array_1)
    cube_masked_2 = cube_5d.copy(m_array_2)
    prod_a = PreprocessorFile(iris.cube.CubeList([cube_5d]), 'A')
    prod_b = PreprocessorFile(iris.cube.CubeList([cube_masked_1]), 'B')
    prod_c = PreprocessorFile(iris.cube.CubeList([cube_masked_2]), 'C')
    products = [prod_a, prod_b, prod_c]
    out_products = _multimodel_mask_products(products, (1, 2, 1, 2, 1))
    expected_data = np.ma.masked_equal([[[[[33], [1]]], [[[33], [33]]]]], 33)
    assert out_products == products
    assert out_products[0].filename == 'A'
    assert out_products[1].filename == 'B'
    assert out_products[2].filename == 'C'
    for product in out_products:
        assert len(product.cubes) == 1
        assert_array_equal(product.cubes[0].data, expected_data)
    assert out_products[0].wasderivedfrom.call_count == 2
    assert out_products[0].mock_ancestors == {prod_b, prod_c}
    out_products[1].wasderivedfrom.assert_called_once_with(prod_c)
    assert out_products[1].mock_ancestors == {prod_c}
    out_products[2].wasderivedfrom.assert_called_once_with(prod_b)
    assert out_products[2].mock_ancestors == {prod_b}


def test_mask_multimodel_fail(cube_1d, cube_2d):
    """Test ``mask_multimodel`` expected fail."""
    cubes = iris.cube.CubeList([cube_1d, cube_2d])
    msg = 'Expected cubes with identical shapes, got shapes'
    with pytest.raises(ValueError, match=msg):
        mask_multimodel(cubes)

    products = [
        cube_1d,
        PreprocessorFile(iris.cube.CubeList([cube_1d]), 'A'),
    ]
    msg = 'Input type for mask_multimodel not understood.'
    with pytest.raises(TypeError, match=msg):
        mask_multimodel(products)
    with pytest.raises(TypeError, match=msg):
        mask_multimodel([1, 2, 3])


def test_mask_multimodel_empty():
    """Test ``mask_multimodel`` with empty input."""
    out_products = mask_multimodel([])
    assert out_products == []

    out_cubes = mask_multimodel(iris.cube.CubeList([]))
    assert out_cubes == iris.cube.CubeList([])

    products = [
        PreprocessorFile(iris.cube.CubeList([]), 'A'),
        PreprocessorFile(iris.cube.CubeList([]), 'B'),
    ]
    out_products = mask_multimodel(products)
    assert out_products is products


def test_mask_multimodel(cube_2d, cube_4d):
    """Test ``mask_multimodel``."""
    m_array = np.ma.masked_equal([[42, 33]], 33)
    cube_masked = cube_2d.copy(m_array)
    cubes = iris.cube.CubeList([cube_2d, cube_masked])
    out_cubes = mask_multimodel(cubes)
    assert out_cubes is cubes
    assert_array_equal(out_cubes[0].data, np.ma.masked_equal([[0, 33]], 33))
    assert_array_equal(out_cubes[1].data, m_array)

    m_array_1 = np.ma.masked_equal([[[[33, 1]], [[33, 3]]]], 33)
    m_array_2 = np.ma.masked_equal([[[[1, 33]], [[3, 3]]]], 33)
    cube_masked_1 = cube_4d.copy(m_array_1)
    cube_masked_2 = cube_4d.copy(m_array_2)
    prod_a = PreprocessorFile(iris.cube.CubeList([cube_4d]), 'A')
    prod_b = PreprocessorFile(iris.cube.CubeList([cube_masked_1]), 'B')
    prod_c = PreprocessorFile(iris.cube.CubeList([cube_masked_2]), 'C')
    products = [prod_a, prod_b, prod_c]
    out_products = mask_multimodel(products)
    expected_data = np.ma.masked_equal([[[[33, 33]], [[33, 3]]]], 33)
    assert out_products == products
    assert out_products[0].filename == 'A'
    assert out_products[1].filename == 'B'
    assert out_products[2].filename == 'C'
    for product in out_products:
        assert len(product.cubes) == 1
        assert_array_equal(product.cubes[0].data, expected_data)
    assert out_products[0].wasderivedfrom.call_count == 2
    assert out_products[0].mock_ancestors == {prod_b, prod_c}
    out_products[1].wasderivedfrom.assert_called_once_with(prod_c)
    assert out_products[1].mock_ancestors == {prod_c}
    out_products[2].wasderivedfrom.assert_called_once_with(prod_b)
    assert out_products[2].mock_ancestors == {prod_b}
