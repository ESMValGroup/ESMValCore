"""Unit tests for `esmvalcore.preprocessor._shared`."""
import inspect
import warnings

import dask.array as da
import iris.analysis
import numpy as np
import pytest
from cf_units import Unit
from iris.coords import AuxCoord
from iris.cube import Cube

from esmvalcore.exceptions import ESMValCoreDeprecationWarning
from esmvalcore.preprocessor import PreprocessorFile
from esmvalcore.preprocessor._shared import (
    _get_area_weights,
    _group_products,
    aggregator_accept_weights,
    get_array_module,
    get_iris_aggregator,
    preserve_float_dtype,
    try_adding_calculated_cell_area,
)


@pytest.mark.parametrize('operator', ['gmean', 'GmEaN', 'GMEAN'])
def test_get_iris_aggregator_gmean(operator):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator)
    assert agg == iris.analysis.GMEAN
    assert agg_kwargs == {}


@pytest.mark.parametrize('operator', ['hmean', 'hMeAn', 'HMEAN'])
def test_get_iris_aggregator_hmean(operator):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator)
    assert agg == iris.analysis.HMEAN
    assert agg_kwargs == {}


@pytest.mark.parametrize('operator', ['max', 'mAx', 'MAX'])
def test_get_iris_aggregator_max(operator):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator)
    assert agg == iris.analysis.MAX
    assert agg_kwargs == {}


@pytest.mark.parametrize('kwargs', [{}, {'weights': True}, {'weights': False}])
@pytest.mark.parametrize('operator', ['mean', 'mEaN', 'MEAN'])
def test_get_iris_aggregator_mean(operator, kwargs):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    assert agg == iris.analysis.MEAN
    assert agg_kwargs == kwargs


@pytest.mark.parametrize('kwargs', [{}, {'weights': True}])
@pytest.mark.parametrize('operator', ['median', 'mEdIaN', 'MEDIAN'])
def test_get_iris_aggregator_median(operator, kwargs):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    assert agg == iris.analysis.MEDIAN
    assert agg_kwargs == kwargs


@pytest.mark.parametrize('operator', ['min', 'MiN', 'MIN'])
def test_get_iris_aggregator_min(operator):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator)
    assert agg == iris.analysis.MIN
    assert agg_kwargs == {}


@pytest.mark.parametrize('operator', ['peak', 'pEaK', 'PEAK'])
def test_get_iris_aggregator_peak(operator):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator)
    assert agg == iris.analysis.PEAK
    assert agg_kwargs == {}


@pytest.mark.parametrize('kwargs', [{'percent': 80.0, 'alphap': 0.5}])
@pytest.mark.parametrize('operator', ['percentile', 'PERCENTILE'])
def test_get_iris_aggregator_percentile(operator, kwargs):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    assert agg == iris.analysis.PERCENTILE
    assert agg_kwargs == kwargs


@pytest.mark.parametrize('kwargs', [{}, {'alphap': 0.5}])
@pytest.mark.parametrize('operator', ['p10', 'P10.5'])
def test_get_iris_aggregator_pxxyy(operator, kwargs):
    """Test ``get_iris_aggregator``."""
    with pytest.warns(ESMValCoreDeprecationWarning):
        (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    assert agg == iris.analysis.PERCENTILE
    assert agg_kwargs == {'percent': float(operator[1:]), **kwargs}


@pytest.mark.parametrize('kwargs', [{}, {'weights': True}])
@pytest.mark.parametrize('operator', ['rms', 'rMs', 'RMS'])
def test_get_iris_aggregator_rms(operator, kwargs):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    assert agg == iris.analysis.RMS
    assert agg_kwargs == kwargs


@pytest.mark.parametrize('kwargs', [{}, {'ddof': 1}])
@pytest.mark.parametrize('operator', ['std', 'STD', 'std_dev', 'STD_DEV'])
def test_get_iris_aggregator_std(operator, kwargs):
    """Test ``get_iris_aggregator``."""
    if operator.lower() == 'std':
        with pytest.warns(ESMValCoreDeprecationWarning):
            (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter(
                'error', category=ESMValCoreDeprecationWarning
            )
            (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    assert agg == iris.analysis.STD_DEV
    assert agg_kwargs == kwargs


@pytest.mark.parametrize('kwargs', [{}, {'weights': True}])
@pytest.mark.parametrize('operator', ['sum', 'SuM', 'SUM'])
def test_get_iris_aggregator_sum(operator, kwargs):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    assert agg == iris.analysis.SUM
    assert agg_kwargs == kwargs


@pytest.mark.parametrize('operator', ['variance', 'vArIaNcE', 'VARIANCE'])
def test_get_iris_aggregator_variance(operator):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator)
    assert agg == iris.analysis.VARIANCE
    assert agg_kwargs == {}


@pytest.mark.parametrize('kwargs', [{'percent': 10, 'weights': True}])
@pytest.mark.parametrize('operator', ['wpercentile', 'WPERCENTILE'])
def test_get_iris_aggregator_wpercentile(operator, kwargs):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    assert agg == iris.analysis.WPERCENTILE
    assert agg_kwargs == kwargs


@pytest.mark.parametrize('operator', ['invalid', 'iNvAliD', 'INVALID'])
def test_get_iris_aggregator_invalid_operator_fail(operator):
    """Test ``get_iris_aggregator``."""
    with pytest.raises(ValueError):
        get_iris_aggregator(operator)


@pytest.mark.parametrize('operator', ['mean', 'mEaN', 'MEAN'])
def test_get_iris_aggregator_no_aggregator_fail(operator, monkeypatch):
    """Test ``get_iris_aggregator``."""
    monkeypatch.setattr(iris.analysis, 'MEAN', 1)
    with pytest.raises(ValueError):
        get_iris_aggregator(operator)


def test_get_iris_aggregator_invalid_kwarg():
    """Test ``get_iris_aggregator``."""
    with pytest.raises(ValueError):
        get_iris_aggregator('max', invalid_kwarg=1)


def test_get_iris_aggregator_missing_kwarg():
    """Test ``get_iris_aggregator``."""
    with pytest.raises(ValueError):
        get_iris_aggregator('percentile')


@pytest.mark.parametrize(
    'aggregator,result',
    [
        (iris.analysis.MEAN, True),
        (iris.analysis.SUM, True),
        (iris.analysis.RMS, True),
        (iris.analysis.WPERCENTILE, True),
        (iris.analysis.MAX, False),
        (iris.analysis.MIN, False),
        (iris.analysis.PERCENTILE, False),
    ],
)
def test_aggregator_accept_weights(aggregator, result):
    """Test ``aggregator_accept_weights``."""
    assert aggregator_accept_weights(aggregator) == result


@preserve_float_dtype
def _dummy_func(obj, arg, kwarg=2.0):
    """Dummy function to test `preserve_float_dtype`."""
    obj = obj * arg * kwarg
    if isinstance(obj, Cube):
        obj.data = obj.core_data().astype(np.float64)
    else:
        obj = obj.astype(np.float64)
    return obj


@pytest.mark.parametrize(
    'data,dtype',
    [
        (np.array([1.0], dtype=np.float64), np.float64),
        (np.array([1.0], dtype=np.float32), np.float32),
        (np.array([1], dtype=np.int64), np.float64),
        (np.array([1], dtype=np.int32), np.float64),
        (da.array([1.0], dtype=np.float64), np.float64),
        (da.array([1.0], dtype=np.float32), np.float32),
        (da.array([1], dtype=np.int64), np.float64),
        (da.array([1], dtype=np.int32), np.float64),
        (Cube(np.array([1.0], dtype=np.float64)), np.float64),
        (Cube(np.array([1.0], dtype=np.float32)), np.float32),
        (Cube(np.array([1], dtype=np.int64)), np.float64),
        (Cube(np.array([1], dtype=np.int32)), np.float64),
        (Cube(da.array([1.0], dtype=np.float64)), np.float64),
        (Cube(da.array([1.0], dtype=np.float32)), np.float32),
        (Cube(da.array([1], dtype=np.int64)), np.float64),
        (Cube(da.array([1], dtype=np.int32)), np.float64),
    ],
)
def test_preserve_float_dtype(data, dtype):
    """Test `preserve_float_dtype`."""
    input_data = data.copy()

    result = _dummy_func(input_data, 2.0)

    assert input_data.dtype == data.dtype
    assert result.dtype == dtype
    assert isinstance(result, type(data))
    if isinstance(data, Cube):
        assert result.has_lazy_data() == data.has_lazy_data()

    assert _dummy_func.__name__ == '_dummy_func'
    signature = inspect.signature(_dummy_func)
    assert list(signature.parameters) == ['obj', 'arg', 'kwarg']


def test_get_array_module_da():
    npx = get_array_module(da.array([1, 2]))
    assert npx is da


def test_get_array_module_np():
    npx = get_array_module(np.array([1, 2]))
    assert npx is np


def test_get_array_module_mixed():
    npx = get_array_module(da.array([1]), np.array([1]))
    assert npx is da


def _create_sample_full_cube():
    cube = Cube(np.zeros((4, 180, 360)), var_name='co2', units='J')
    cube.add_dim_coord(
        iris.coords.DimCoord(
            np.array([10., 40., 70., 110.]),
            standard_name='time',
            units=Unit('days since 1950-01-01 00:00:00', calendar='gregorian'),
        ),
        0,
    )
    cube.add_dim_coord(
        iris.coords.DimCoord(
            np.arange(-90., 90., 1.),
            standard_name='latitude',
            units='degrees',
        ),
        1,
    )
    cube.add_dim_coord(
        iris.coords.DimCoord(
            np.arange(0., 360., 1.),
            standard_name='longitude',
            units='degrees',
        ),
        2,
    )

    cube.coord("time").guess_bounds()
    cube.coord("longitude").guess_bounds()
    cube.coord("latitude").guess_bounds()

    return cube


@pytest.mark.parametrize('lazy', [True, False])
def test_get_area_weights(lazy):
    """Test _get_area_weights."""
    cube = _create_sample_full_cube()
    if lazy:
        cube.data = cube.lazy_data()
    weights = _get_area_weights(cube)
    if lazy:
        assert isinstance(weights, da.Array)
        assert weights.chunks == cube.lazy_data().chunks
    else:
        assert isinstance(weights, np.ndarray)
    np.testing.assert_allclose(
        weights, iris.analysis.cartography.area_weights(cube)
    )


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


def test_try_adding_calculated_cell_area():
    """Test ``try_adding_calculated_cell_area``."""
    cube = _create_sample_full_cube()
    cube.coord('latitude').rename('grid_latitude')
    cube.coord('longitude').rename('grid_longitude')
    lat = AuxCoord(np.zeros((180, 360)), standard_name='latitude')
    lon = AuxCoord(np.zeros((180, 360)), standard_name='longitude')
    cube.add_aux_coord(lat, (1, 2))
    cube.add_aux_coord(lon, (1, 2))

    try_adding_calculated_cell_area(cube)

    assert cube.cell_measures('cell_area')
