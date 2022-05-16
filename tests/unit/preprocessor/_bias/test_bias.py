"""Unit tests for :mod:`esmvalcore.preprocessor._bias`."""

import iris
import iris.cube
import numpy as np
import pytest
from cf_units import Unit

from esmvalcore.preprocessor._bias import bias
from tests import PreprocessorFile


def assert_array_equal(array_1, array_2):
    """Assert that (masked) array 1 equals (masked) array 2."""
    if np.ma.is_masked(array_1) or np.ma.is_masked(array_2):
        np.testing.assert_array_equal(np.ma.getmaskarray(array_1),
                                      np.ma.getmaskarray(array_2))
        mask = np.ma.getmaskarray(array_1)
        np.testing.assert_array_equal(array_1[~mask], array_2[~mask])
    else:
        np.testing.assert_array_equal(array_1, array_2)


def products_set_to_dict(products):
    """Convert :obj:`set` of products to :obj:`dict`."""
    new_dict = {}
    for product in products:
        new_dict[product.filename] = product
    return new_dict


def get_3d_cube(data, **cube_kwargs):
    """Create 3D cube."""
    time_units = Unit('days since 1850-01-01 00:00:00')
    times = iris.coords.DimCoord([0.0, 1.0], standard_name='time',
                                 var_name='time', long_name='time',
                                 units=time_units)
    lats = iris.coords.DimCoord([0.0, 10.0], standard_name='latitude',
                                var_name='lat', long_name='latitude',
                                units='degrees_north')
    lons = iris.coords.DimCoord([20.0, 30.0], standard_name='longitude',
                                var_name='lon', long_name='longitude',
                                units='degrees_east')
    coord_specs = [(times, 0), (lats, 1), (lons, 2)]
    cube = iris.cube.Cube(data.astype('float32'),
                          dim_coords_and_dims=coord_specs, **cube_kwargs)
    return cube


@pytest.fixture
def regular_cubes():
    """Regular cube."""
    cube_data = np.arange(8.0).reshape(2, 2, 2)
    cube = get_3d_cube(cube_data, standard_name='air_temperature',
                       var_name='tas', units='K')
    return iris.cube.CubeList([cube])


@pytest.fixture
def ref_cubes():
    """Reference cube."""
    cube_data = np.full((2, 2, 2), 2.0)
    cube_data[1, 1, 1] = 4.0
    cube = get_3d_cube(cube_data, standard_name='air_temperature',
                       var_name='tas', units='K')
    return iris.cube.CubeList([cube])


def test_no_reference_for_bias(regular_cubes, ref_cubes):
    """Test fail when no reference_for_bias is given."""
    products = {
        PreprocessorFile(regular_cubes, 'A', {}),
        PreprocessorFile(regular_cubes, 'B', {}),
        PreprocessorFile(ref_cubes, 'REF', {}),
    }
    msg = "Expected exactly 1 dataset with 'reference_for_bias: true', found 0"
    with pytest.raises(ValueError, match=msg):
        bias(products)


def test_two_reference_for_bias(regular_cubes, ref_cubes):
    """Test fail when two reference_for_bias is given."""
    products = {
        PreprocessorFile(regular_cubes, 'A', {'reference_for_bias': False}),
        PreprocessorFile(ref_cubes, 'REF1', {'reference_for_bias': True}),
        PreprocessorFile(ref_cubes, 'REF2', {'reference_for_bias': True}),
    }
    msg = "Expected exactly 1 dataset with 'reference_for_bias: true', found 2"
    with pytest.raises(ValueError, match=msg):
        bias(products)


def test_invalid_bias_type(regular_cubes, ref_cubes):
    """Test fail when invalid bias_type is given."""
    products = {
        PreprocessorFile(regular_cubes, 'A', {}),
        PreprocessorFile(regular_cubes, 'B', {}),
        PreprocessorFile(ref_cubes, 'REF', {'reference_for_bias': True}),
    }
    msg = (r"Expected one of \['absolute', 'relative'\] for bias_type, got "
           r"'invalid_bias_type'")
    with pytest.raises(ValueError, match=msg):
        bias(products, 'invalid_bias_type')


def test_absolute_bias(regular_cubes, ref_cubes):
    """Test calculation of absolute bias."""
    ref_product = PreprocessorFile(ref_cubes, 'REF',
                                   {'reference_for_bias': True})
    products = {
        PreprocessorFile(regular_cubes, 'A', {'dataset': 'a'}),
        PreprocessorFile(regular_cubes, 'B', {'dataset': 'b'}),
        ref_product,
    }
    out_products = bias(products)
    out_dict = products_set_to_dict(out_products)
    assert len(out_dict) == 2

    product_a = out_dict['A']
    assert product_a.filename == 'A'
    assert product_a.attributes == {'units': 'K', 'dataset': 'a'}
    assert len(product_a.cubes) == 1
    out_cube = product_a.cubes[0]
    expected_data = [[[-2.0, -1.0],
                      [0.0, 1.0]],
                     [[2.0, 3.0],
                      [4.0, 3.0]]]
    assert_array_equal(out_cube.data, expected_data)
    assert out_cube.var_name == 'tas'
    assert out_cube.standard_name == 'air_temperature'
    assert out_cube.units == 'K'
    assert out_cube.dim_coords == regular_cubes[0].dim_coords
    assert out_cube.aux_coords == regular_cubes[0].aux_coords
    product_a.wasderivedfrom.assert_called_once()
    assert product_a.mock_ancestors == {ref_product}

    product_b = out_dict['B']
    assert product_b.filename == 'B'
    assert product_b.attributes == {'units': 'K', 'dataset': 'b'}
    assert len(product_b.cubes) == 1
    out_cube = product_b.cubes[0]
    expected_data = [[[-2.0, -1.0],
                      [0.0, 1.0]],
                     [[2.0, 3.0],
                      [4.0, 3.0]]]
    assert_array_equal(out_cube.data, expected_data)
    assert out_cube.var_name == 'tas'
    assert out_cube.standard_name == 'air_temperature'
    assert out_cube.units == 'K'
    assert out_cube.dim_coords == regular_cubes[0].dim_coords
    assert out_cube.aux_coords == regular_cubes[0].aux_coords
    product_b.wasderivedfrom.assert_called_once()
    assert product_b.mock_ancestors == {ref_product}


def test_relative_bias(regular_cubes, ref_cubes):
    """Test calculation of relative bias."""
    ref_product = PreprocessorFile(ref_cubes, 'REF',
                                   {'reference_for_bias': True})
    products = {
        PreprocessorFile(regular_cubes, 'A', {'dataset': 'a'}),
        PreprocessorFile(regular_cubes, 'B', {'dataset': 'b'}),
        ref_product,
    }
    out_products = bias(products, 'relative')
    out_dict = products_set_to_dict(out_products)
    assert len(out_dict) == 2

    product_a = out_dict['A']
    assert product_a.filename == 'A'
    assert product_a.attributes == {'units': '1', 'dataset': 'a'}
    assert len(product_a.cubes) == 1
    out_cube = product_a.cubes[0]
    expected_data = [[[-1.0, -0.5],
                      [0.0, 0.5]],
                     [[1.0, 1.5],
                      [2.0, 0.75]]]
    assert_array_equal(out_cube.data, expected_data)
    assert out_cube.var_name == 'tas'
    assert out_cube.standard_name == 'air_temperature'
    assert out_cube.units == '1'
    assert out_cube.dim_coords == regular_cubes[0].dim_coords
    assert out_cube.aux_coords == regular_cubes[0].aux_coords
    product_a.wasderivedfrom.assert_called_once()
    assert product_a.mock_ancestors == {ref_product}

    product_b = out_dict['B']
    assert product_b.filename == 'B'
    assert product_b.attributes == {'units': '1', 'dataset': 'b'}
    assert len(product_b.cubes) == 1
    out_cube = product_b.cubes[0]
    expected_data = [[[-1.0, -0.5],
                      [0.0, 0.5]],
                     [[1.0, 1.5],
                      [2.0, 0.75]]]
    assert_array_equal(out_cube.data, expected_data)
    assert out_cube.var_name == 'tas'
    assert out_cube.standard_name == 'air_temperature'
    assert out_cube.units == '1'
    assert out_cube.dim_coords == regular_cubes[0].dim_coords
    assert out_cube.aux_coords == regular_cubes[0].aux_coords
    product_b.wasderivedfrom.assert_called_once()
    assert product_b.mock_ancestors == {ref_product}


def test_denominator_mask_threshold(regular_cubes, ref_cubes):
    """Test denominator_mask_threshold argument."""
    ref_product = PreprocessorFile(ref_cubes, 'REF',
                                   {'reference_for_bias': True})
    products = {
        PreprocessorFile(regular_cubes, 'A', {'dataset': 'a'}),
        ref_product,
    }
    out_products = bias(products, 'relative', denominator_mask_threshold=3.0)
    out_dict = products_set_to_dict(out_products)
    assert len(out_dict) == 1

    product_a = out_dict['A']
    assert product_a.filename == 'A'
    assert product_a.attributes == {'units': '1', 'dataset': 'a'}
    assert len(product_a.cubes) == 1
    out_cube = product_a.cubes[0]
    expected_data = np.ma.masked_equal([[[42.0, 42.0],
                                         [42.0, 42.0]],
                                        [[42.0, 42.0],
                                         [42.0, 0.75]]], 42.0)
    assert_array_equal(out_cube.data, expected_data)
    assert out_cube.var_name == 'tas'
    assert out_cube.standard_name == 'air_temperature'
    assert out_cube.units == '1'
    assert out_cube.dim_coords == regular_cubes[0].dim_coords
    assert out_cube.aux_coords == regular_cubes[0].aux_coords
    product_a.wasderivedfrom.assert_called_once()
    assert product_a.mock_ancestors == {ref_product}


def test_keep_reference_dataset(regular_cubes, ref_cubes):
    """Test denominator_mask_threshold argument."""
    products = {
        PreprocessorFile(regular_cubes, 'A', {'dataset': 'a'}),
        PreprocessorFile(ref_cubes, 'REF', {'reference_for_bias': True})
    }
    out_products = bias(products, keep_reference_dataset=True)
    out_dict = products_set_to_dict(out_products)
    assert len(out_dict) == 2

    product_ref = out_dict['REF']
    assert product_ref.filename == 'REF'
    assert product_ref.attributes == {'reference_for_bias': True}
    assert len(product_ref.cubes) == 1
    out_cube = product_ref.cubes[0]
    expected_data = [[[2.0, 2.0],
                      [2.0, 2.0]],
                     [[2.0, 2.0],
                      [2.0, 4.0]]]
    assert_array_equal(out_cube.data, expected_data)
    assert out_cube.var_name == 'tas'
    assert out_cube.standard_name == 'air_temperature'
    assert out_cube.units == 'K'
    assert out_cube.dim_coords == ref_cubes[0].dim_coords
    assert out_cube.aux_coords == ref_cubes[0].aux_coords
