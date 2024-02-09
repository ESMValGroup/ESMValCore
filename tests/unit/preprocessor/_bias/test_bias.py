"""Unit tests for :mod:`esmvalcore.preprocessor._bias`."""

import iris
import numpy as np
import pytest
from cf_units import Unit
from iris.cube import Cube, CubeList

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
    cube = Cube(data.astype('float32'),
                dim_coords_and_dims=coord_specs, **cube_kwargs)
    return cube


@pytest.fixture
def regular_cubes():
    """Regular cube."""
    cube_data = np.arange(8.0).reshape(2, 2, 2)
    cube = get_3d_cube(cube_data, standard_name='air_temperature',
                       var_name='tas', units='K')
    return CubeList([cube])


@pytest.fixture
def ref_cubes():
    """Reference cube."""
    cube_data = np.full((2, 2, 2), 2.0)
    cube_data[1, 1, 1] = 4.0
    cube = get_3d_cube(cube_data, standard_name='air_temperature',
                       var_name='tas', units='K')
    return CubeList([cube])


TEST_BIAS = [
    ('absolute', [[[-2.0, -1.0], [0.0, 1.0]], [[2.0, 3.0], [4.0, 3.0]]], 'K'),
    ('relative', [[[-1.0, -0.5], [0.0, 0.5]], [[1.0, 1.5], [2.0, 0.75]]], '1'),
]


@pytest.mark.parametrize('bias_type,data,units', TEST_BIAS)
def test_bias_products(regular_cubes, ref_cubes, bias_type, data, units):
    """Test calculation of bias with products."""
    ref_product = PreprocessorFile(ref_cubes, 'REF',
                                   {'reference_for_bias': True})
    products = {
        PreprocessorFile(regular_cubes, 'A', {'dataset': 'a'}),
        PreprocessorFile(regular_cubes, 'B', {'dataset': 'b'}),
        ref_product,
    }
    out_products = bias(products, bias_type=bias_type)

    assert isinstance(out_products, set)
    out_dict = products_set_to_dict(out_products)
    assert len(out_dict) == 2

    product_a = out_dict['A']
    assert product_a.filename == 'A'
    assert product_a.attributes == {'units': units, 'dataset': 'a'}
    assert len(product_a.cubes) == 1
    out_cube = product_a.cubes[0]
    assert_array_equal(out_cube.data, data)
    assert out_cube.var_name == 'tas'
    assert out_cube.standard_name == 'air_temperature'
    assert out_cube.units == units
    assert out_cube.dim_coords == regular_cubes[0].dim_coords
    assert out_cube.aux_coords == regular_cubes[0].aux_coords
    product_a.wasderivedfrom.assert_called_once()
    assert product_a.mock_ancestors == {ref_product}

    product_b = out_dict['B']
    assert product_b.filename == 'B'
    assert product_b.attributes == {'units': units, 'dataset': 'b'}
    assert len(product_b.cubes) == 1
    out_cube = product_b.cubes[0]
    assert_array_equal(out_cube.data, data)
    assert out_cube.var_name == 'tas'
    assert out_cube.standard_name == 'air_temperature'
    assert out_cube.units == units
    assert out_cube.dim_coords == regular_cubes[0].dim_coords
    assert out_cube.aux_coords == regular_cubes[0].aux_coords
    product_b.wasderivedfrom.assert_called_once()
    assert product_b.mock_ancestors == {ref_product}


@pytest.mark.parametrize('bias_type,data,units', TEST_BIAS)
def test_bias_cubes(regular_cubes, ref_cubes, bias_type, data, units):
    """Test calculation of bias with cubes."""
    ref_cube = ref_cubes[0]
    out_cubes = bias(regular_cubes, ref_cube, bias_type=bias_type)

    assert isinstance(out_cubes, CubeList)
    assert len(out_cubes) == 1
    out_cube = out_cubes[0]

    assert_array_equal(out_cube.data, data)
    assert out_cube.var_name == 'tas'
    assert out_cube.standard_name == 'air_temperature'
    assert out_cube.units == units
    assert out_cube.dim_coords == regular_cubes[0].dim_coords
    assert out_cube.aux_coords == regular_cubes[0].aux_coords


TEST_BIAS_BROADCASTABLE = [
    ('absolute', [[[-2.0, -1.0], [0.0, 1.0]], [[2.0, 3.0], [4.0, 5.0]]], 'K'),
    ('relative', [[[-1.0, -0.5], [0.0, 0.5]], [[1.0, 1.5], [2.0, 2.5]]], '1'),
]


@pytest.mark.parametrize('bias_type,data,units', TEST_BIAS_BROADCASTABLE)
def test_bias_cubes_broadcastable(
    regular_cubes, ref_cubes, bias_type, data, units
):
    """Test calculation of bias with cubes."""
    ref_cube = ref_cubes[0][0]  # only select one time step
    out_cubes = bias(regular_cubes, ref_cube, bias_type=bias_type)

    assert isinstance(out_cubes, CubeList)
    assert len(out_cubes) == 1
    out_cube = out_cubes[0]

    assert_array_equal(out_cube.data, data)
    assert out_cube.var_name == 'tas'
    assert out_cube.standard_name == 'air_temperature'
    assert out_cube.units == units
    assert out_cube.dim_coords == regular_cubes[0].dim_coords
    assert out_cube.aux_coords == regular_cubes[0].aux_coords


def test_denominator_mask_threshold_products(regular_cubes, ref_cubes):
    """Test denominator_mask_threshold argument with products."""
    ref_product = PreprocessorFile(ref_cubes, 'REF',
                                   {'reference_for_bias': True})
    products = {
        PreprocessorFile(regular_cubes, 'A', {'dataset': 'a'}),
        ref_product,
    }
    out_products = bias(
        products, bias_type='relative', denominator_mask_threshold=3.0
    )

    assert isinstance(out_products, set)
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


def test_denominator_mask_threshold_cubes(regular_cubes, ref_cubes):
    """Test denominator_mask_threshold argument with cubes."""
    ref_cube = ref_cubes[0]
    out_cubes = bias(
        regular_cubes,
        ref_cube,
        bias_type='relative',
        denominator_mask_threshold=3.0,
    )

    assert isinstance(out_cubes, CubeList)
    assert len(out_cubes) == 1
    out_cube = out_cubes[0]

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


@pytest.mark.parametrize('bias_type', ['absolute', 'relative'])
def test_keep_reference_dataset(regular_cubes, ref_cubes, bias_type):
    """Test denominator_mask_threshold argument."""
    products = {
        PreprocessorFile(regular_cubes, 'A', {'dataset': 'a'}),
        PreprocessorFile(ref_cubes, 'REF', {'reference_for_bias': True})
    }
    out_products = bias(
        products, bias_type=bias_type, keep_reference_dataset=True
    )

    assert isinstance(out_products, set)
    out_dict = products_set_to_dict(out_products)
    assert len(out_dict) == 2

    product_ref = out_dict['REF']
    assert product_ref.filename == 'REF'
    assert product_ref.attributes == {'reference_for_bias': True}
    assert len(product_ref.cubes) == 1
    out_cube = product_ref.cubes[0]
    expected_data = [[[2.0, 2.0], [2.0, 2.0]], [[2.0, 2.0], [2.0, 4.0]]]
    assert_array_equal(out_cube.data, expected_data)
    assert out_cube.var_name == 'tas'
    assert out_cube.standard_name == 'air_temperature'
    assert out_cube.units == 'K'
    assert out_cube.dim_coords == ref_cubes[0].dim_coords
    assert out_cube.aux_coords == ref_cubes[0].aux_coords


@pytest.mark.parametrize('bias_type,data,units', TEST_BIAS)
@pytest.mark.parametrize('keep_ref', [True, False])
def test_bias_products_and_ref_cube(
    regular_cubes, ref_cubes, keep_ref, bias_type, data, units
):
    """Test calculation of bias with products and ref_cube given."""
    ref_cube = ref_cubes[0]
    products = set([PreprocessorFile(regular_cubes, 'A', {'dataset': 'a'})])

    out_products = bias(
        products,
        ref_cube=ref_cube,
        bias_type=bias_type,
        keep_reference_dataset=keep_ref,
    )

    assert isinstance(out_products, set)
    out_dict = products_set_to_dict(out_products)
    assert len(out_dict) == 1

    product_a = out_dict['A']
    assert product_a.filename == 'A'
    assert product_a.attributes == {'units': units, 'dataset': 'a'}
    assert len(product_a.cubes) == 1
    out_cube = product_a.cubes[0]
    assert_array_equal(out_cube.data, data)
    assert out_cube.var_name == 'tas'
    assert out_cube.standard_name == 'air_temperature'
    assert out_cube.units == units
    assert out_cube.dim_coords == regular_cubes[0].dim_coords
    assert out_cube.aux_coords == regular_cubes[0].aux_coords
    product_a.wasderivedfrom.assert_not_called()
    assert product_a.mock_ancestors == set()


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


def test_two_references_for_bias(regular_cubes, ref_cubes):
    """Test fail when two reference_for_bias products is given."""
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
        bias(products, bias_type='invalid_bias_type')


def test_ref_cube_non_cubes(regular_cubes):
    """Test ref_cube=None with with cubes."""
    msg = (
        "A list of Cubes is given to this preprocessor; please specify a "
        "`ref_cube`"
    )
    with pytest.raises(ValueError, match=msg):
        bias(regular_cubes)
