"""Unit tests for `esmvalcore.preprocessor._shared`."""

import inspect
import warnings

import dask.array as da
import iris.analysis
import numpy as np
import pytest
from cf_units import Unit
from iris.aux_factory import HybridPressureFactory
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube

from esmvalcore.preprocessor import PreprocessorFile
from esmvalcore.preprocessor._shared import (
    _compute_area_weights,
    _group_products,
    _rechunk_aux_factory_dependencies,
    aggregator_accept_weights,
    apply_mask,
    get_array_module,
    get_coord_weights,
    get_iris_aggregator,
    preserve_float_dtype,
    try_adding_calculated_cell_area,
)
from tests import assert_array_equal
from tests.unit.preprocessor._time.test_time import (
    _make_cube,
    get_1d_time,
    get_lon_coord,
)


@pytest.mark.parametrize("operator", ["gmean", "GmEaN", "GMEAN"])
def test_get_iris_aggregator_gmean(operator):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator)
    assert agg == iris.analysis.GMEAN
    assert agg_kwargs == {}


@pytest.mark.parametrize("operator", ["hmean", "hMeAn", "HMEAN"])
def test_get_iris_aggregator_hmean(operator):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator)
    assert agg == iris.analysis.HMEAN
    assert agg_kwargs == {}


@pytest.mark.parametrize("operator", ["max", "mAx", "MAX"])
def test_get_iris_aggregator_max(operator):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator)
    assert agg == iris.analysis.MAX
    assert agg_kwargs == {}


@pytest.mark.parametrize("kwargs", [{}, {"weights": True}, {"weights": False}])
@pytest.mark.parametrize("operator", ["mean", "mEaN", "MEAN"])
def test_get_iris_aggregator_mean(operator, kwargs):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    assert agg == iris.analysis.MEAN
    assert agg_kwargs == kwargs


@pytest.mark.parametrize("operator", ["median", "mEdIaN", "MEDIAN"])
def test_get_iris_aggregator_median(operator):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator)
    assert agg == iris.analysis.MEDIAN
    assert agg_kwargs == {}


@pytest.mark.parametrize("operator", ["min", "MiN", "MIN"])
def test_get_iris_aggregator_min(operator):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator)
    assert agg == iris.analysis.MIN
    assert agg_kwargs == {}


@pytest.mark.parametrize("operator", ["peak", "pEaK", "PEAK"])
def test_get_iris_aggregator_peak(operator):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator)
    assert agg == iris.analysis.PEAK
    assert agg_kwargs == {}


@pytest.mark.parametrize("kwargs", [{"percent": 80.0, "alphap": 0.5}])
@pytest.mark.parametrize("operator", ["percentile", "PERCENTILE"])
def test_get_iris_aggregator_percentile(operator, kwargs):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    assert agg == iris.analysis.PERCENTILE
    assert agg_kwargs == kwargs


@pytest.mark.parametrize("kwargs", [{}, {"weights": True}])
@pytest.mark.parametrize("operator", ["rms", "rMs", "RMS"])
def test_get_iris_aggregator_rms(operator, kwargs):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    assert agg == iris.analysis.RMS
    assert agg_kwargs == kwargs


@pytest.mark.parametrize("kwargs", [{}, {"ddof": 1}])
@pytest.mark.parametrize("operator", ["std_dev", "STD_DEV"])
def test_get_iris_aggregator_std(operator, kwargs):
    """Test ``get_iris_aggregator``."""
    with warnings.catch_warnings():
        (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    assert agg == iris.analysis.STD_DEV
    assert agg_kwargs == kwargs


@pytest.mark.parametrize("kwargs", [{}, {"weights": True}])
@pytest.mark.parametrize("operator", ["sum", "SuM", "SUM"])
def test_get_iris_aggregator_sum(operator, kwargs):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    assert agg == iris.analysis.SUM
    assert agg_kwargs == kwargs


@pytest.mark.parametrize("operator", ["variance", "vArIaNcE", "VARIANCE"])
def test_get_iris_aggregator_variance(operator):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator)
    assert agg == iris.analysis.VARIANCE
    assert agg_kwargs == {}


@pytest.mark.parametrize("kwargs", [{"percent": 10, "weights": True}])
@pytest.mark.parametrize("operator", ["wpercentile", "WPERCENTILE"])
def test_get_iris_aggregator_wpercentile(operator, kwargs):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    assert agg == iris.analysis.WPERCENTILE
    assert agg_kwargs == kwargs


@pytest.mark.parametrize("operator", ["invalid", "iNvAliD", "INVALID"])
def test_get_iris_aggregator_invalid_operator_fail(operator):
    """Test ``get_iris_aggregator``."""
    with pytest.raises(ValueError):
        get_iris_aggregator(operator)


@pytest.mark.parametrize("operator", ["mean", "mEaN", "MEAN"])
def test_get_iris_aggregator_no_aggregator_fail(operator, monkeypatch):
    """Test ``get_iris_aggregator``."""
    monkeypatch.setattr(iris.analysis, "MEAN", 1)
    with pytest.raises(ValueError):
        get_iris_aggregator(operator)


def test_get_iris_aggregator_invalid_kwarg():
    """Test ``get_iris_aggregator``."""
    with pytest.raises(ValueError):
        get_iris_aggregator("max", invalid_kwarg=1)


def test_get_iris_aggregator_missing_kwarg():
    """Test ``get_iris_aggregator``."""
    with pytest.raises(ValueError):
        get_iris_aggregator("percentile")


def test_get_iris_aggregator_no_weights_allowed():
    """Test ``get_iris_aggregator``."""
    operator = "median"
    kwargs = {"weights": True}
    with pytest.raises(ValueError):
        get_iris_aggregator(operator, **kwargs)


@pytest.mark.parametrize(
    ("aggregator", "result"),
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
    """Compute something to test `preserve_float_dtype`."""
    obj = obj * arg * kwarg
    if isinstance(obj, Cube):
        obj.data = obj.core_data().astype(np.float64)
    else:
        obj = obj.astype(np.float64)
    return obj


TEST_PRESERVE_FLOAT_TYPE = [
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
]


@pytest.mark.parametrize(("data", "dtype"), TEST_PRESERVE_FLOAT_TYPE)
def test_preserve_float_dtype(data, dtype):
    """Test `preserve_float_dtype`."""
    input_data = data.copy()

    result = _dummy_func(input_data, 2.0)

    assert input_data.dtype == data.dtype
    assert result.dtype == dtype
    assert isinstance(result, type(data))
    if isinstance(data, Cube):
        assert result.has_lazy_data() == data.has_lazy_data()

    assert _dummy_func.__name__ == "_dummy_func"
    signature = inspect.signature(_dummy_func)
    assert list(signature.parameters) == ["obj", "arg", "kwarg"]


@pytest.mark.parametrize(("data", "dtype"), TEST_PRESERVE_FLOAT_TYPE)
def test_preserve_float_dtype_kwargs_only(data, dtype):
    """Test `preserve_float_dtype`."""
    input_data = data.copy()

    result = _dummy_func(arg=2.0, obj=input_data, kwarg=2.0)

    assert input_data.dtype == data.dtype
    assert result.dtype == dtype
    assert isinstance(result, type(data))
    if isinstance(data, Cube):
        assert result.has_lazy_data() == data.has_lazy_data()

    assert _dummy_func.__name__ == "_dummy_func"
    signature = inspect.signature(_dummy_func)
    assert list(signature.parameters) == ["obj", "arg", "kwarg"]


def test_preserve_float_dtype_invalid_args():
    """Test `preserve_float_dtype`."""
    msg = r"missing 2 required positional arguments: 'obj' and 'arg'"
    with pytest.raises(TypeError, match=msg):
        _dummy_func()


def test_preserve_float_dtype_invalid_kwarg():
    """Test `preserve_float_dtype`."""
    msg = r"got an unexpected keyword argument 'data'"
    with pytest.raises(TypeError, match=msg):
        _dummy_func(np.array(1), 2.0, data=3.0)


def test_preserve_float_dtype_invalid_func():
    """Test `preserve_float_dtype`."""
    msg = (
        r"Cannot preserve float dtype during function '<lambda>', function "
        r"takes no arguments"
    )
    with pytest.raises(TypeError, match=msg):
        preserve_float_dtype(lambda: None)


def test_preserve_float_dtype_first_arg_no_dtype():
    """Test `preserve_float_dtype`."""

    @preserve_float_dtype
    def func(obj):
        return obj * np.array(1)

    msg = (
        r"Cannot preserve float dtype during function 'func', the function's "
        r"first argument of type"
    )
    with pytest.raises(TypeError, match=msg):
        func(1.0)


def test_preserve_float_dtype_return_value_no_dtype():
    """Test `preserve_float_dtype`."""

    @preserve_float_dtype
    def func(_):
        return 1

    msg = (
        r"Cannot preserve float dtype during function 'func', the function's "
        r"first argument of type"
    )
    with pytest.raises(TypeError, match=msg):
        func(np.array(1.0))


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
    cube = Cube(np.zeros((4, 180, 360)), var_name="co2", units="J")
    cube.add_dim_coord(
        DimCoord(
            np.array([10.0, 40.0, 70.0, 110.0]),
            standard_name="time",
            units=Unit("days since 1950-01-01 00:00:00", calendar="gregorian"),
        ),
        0,
    )
    cube.add_dim_coord(
        DimCoord(
            np.arange(-90.0, 90.0, 1.0),
            standard_name="latitude",
            units="degrees",
        ),
        1,
    )
    cube.add_dim_coord(
        DimCoord(
            np.arange(0.0, 360.0, 1.0),
            standard_name="longitude",
            units="degrees",
        ),
        2,
    )

    cube.coord("time").guess_bounds()
    cube.coord("longitude").guess_bounds()
    cube.coord("latitude").guess_bounds()

    return cube


@pytest.mark.parametrize("lazy", [True, False])
def test_compute_area_weights(lazy):
    """Test _compute_area_weights."""
    cube = _create_sample_full_cube()
    if lazy:
        cube.data = cube.lazy_data().rechunk((2, 180, 360))
    weights = _compute_area_weights(cube)
    if lazy:
        assert isinstance(weights, da.Array)
        assert weights.chunks == cube.lazy_data().chunks
    else:
        assert isinstance(weights, np.ndarray)
    np.testing.assert_allclose(
        weights,
        iris.analysis.cartography.area_weights(cube),
    )


def test_group_products_string_list():
    products = [
        PreprocessorFile(
            filename="A_B.nc",
            attributes={
                "project": "A",
                "dataset": "B",
            },
        ),
        PreprocessorFile(
            filename="A_C.nc",
            attributes={
                "project": "A",
                "dataset": "C",
            },
        ),
    ]
    grouped_by_string = _group_products(products, "project")
    grouped_by_list = _group_products(products, ["project"])

    assert grouped_by_list == grouped_by_string


def test_try_adding_calculated_cell_area():
    """Test ``try_adding_calculated_cell_area``."""
    cube = _create_sample_full_cube()
    cube.coord("latitude").rename("grid_latitude")
    cube.coord("longitude").rename("grid_longitude")
    lat = AuxCoord(np.zeros((180, 360)), standard_name="latitude")
    lon = AuxCoord(np.zeros((180, 360)), standard_name="longitude")
    cube.add_aux_coord(lat, (1, 2))
    cube.add_aux_coord(lon, (1, 2))

    try_adding_calculated_cell_area(cube)

    assert cube.cell_measures("cell_area")


@pytest.mark.parametrize(
    ("mask", "array", "dim_map", "expected"),
    [
        (
            np.arange(2),
            da.arange(2),
            (0,),
            da.ma.masked_array(np.arange(2), np.arange(2)),
        ),
        (
            da.arange(2),
            np.arange(2),
            (0,),
            da.ma.masked_array(np.arange(2), np.arange(2)),
        ),
        (
            np.ma.masked_array(np.arange(2), mask=[1, 0]),
            da.arange(2),
            (0,),
            da.ma.masked_array(np.arange(2), np.ones(2)),
        ),
        (
            np.ones((2, 5)),
            da.zeros((2, 3, 5), chunks=(1, 2, 3)),
            (0, 2),
            da.ma.masked_array(
                da.zeros((2, 3, 5), da.ones(2, 3, 5), chunks=(1, 2, 3)),
            ),
        ),
        (
            np.arange(2),
            np.ones((3, 2)),
            (1,),
            np.ma.masked_array(np.ones((3, 2)), mask=[[0, 1], [0, 1], [0, 1]]),
        ),
    ],
)
def test_apply_mask(mask, array, dim_map, expected):
    result = apply_mask(mask, array, dim_map)
    assert isinstance(result, type(expected))
    if isinstance(expected, da.Array):
        assert result.chunks == expected.chunks
    assert_array_equal(result, expected)


def test_rechunk_aux_factory_dependencies():
    delta = AuxCoord(
        points=np.array([0.0, 1.0, 2.0], dtype=np.float64),
        bounds=np.array(
            [[-0.5, 0.5], [0.5, 1.5], [1.5, 2.5]],
            dtype=np.float64,
        ),
        long_name="level_pressure",
        units="Pa",
    )
    sigma = AuxCoord(
        np.array([1.0, 0.9, 0.8], dtype=np.float64),
        long_name="sigma",
        units="1",
    )
    surface_air_pressure = AuxCoord(
        np.arange(4).astype(np.float64).reshape(2, 2),
        long_name="surface_air_pressure",
        units="Pa",
    )
    factory = HybridPressureFactory(
        delta=delta,
        sigma=sigma,
        surface_air_pressure=surface_air_pressure,
    )

    cube = Cube(
        da.asarray(
            np.arange(3 * 2 * 2).astype(np.float32).reshape(3, 2, 2),
            chunks=(1, 2, 2),
        ),
    )
    cube.add_aux_coord(delta, 0)
    cube.add_aux_coord(sigma, 0)
    cube.add_aux_coord(surface_air_pressure, [1, 2])
    cube.add_aux_factory(factory)

    result = _rechunk_aux_factory_dependencies(cube, "air_pressure")

    # Check that the 'air_pressure' coordinate of the resulting cube has been
    # rechunked:
    assert result.coord("air_pressure").core_points().chunks == (
        (1, 1, 1),
        (2,),
        (2,),
    )
    # Check that the original cube has not been modified:
    assert cube.coord("air_pressure").core_points().chunks == (
        (3,),
        (2,),
        (2,),
    )


def get_0d_time():
    """Get 0D time coordinate."""
    return AuxCoord(
        15.0,
        bounds=[0.0, 30.0],
        standard_name="time",
        units="days since 1850-01-01 00:00:00",
    )


@pytest.mark.parametrize("lazy", [True, False])
def test_get_coord_weights_time(lazy):
    """Test ``get_coord_weights`` for complex cube."""
    cube = _make_cube()
    if lazy:
        cube.data = cube.lazy_data().rechunk((1, 1, 1, 3))
    weights = get_coord_weights(cube, "time")
    assert weights.shape == (2,)
    if lazy:
        assert isinstance(weights, da.Array)
        assert weights.chunks == ((1, 1),)
    else:
        assert isinstance(weights, np.ndarray)
    np.testing.assert_allclose(weights, [15.0, 30.0])


@pytest.mark.parametrize("lazy", [True, False])
def test_get_coord_weights_time_broadcast(lazy):
    """Test ``get_coord_weights`` for complex cube."""
    cube = _make_cube()
    if lazy:
        cube.data = cube.lazy_data().rechunk((1, 1, 1, 3))
    weights = get_coord_weights(cube, "time", broadcast=True)
    assert weights.shape == (2, 1, 1, 3)
    if lazy:
        assert isinstance(weights, da.Array)
        assert weights.chunks == ((1, 1), (1,), (1,), (3,))
    else:
        assert isinstance(weights, np.ndarray)
    expected_data = [[[[15.0, 15.0, 15.0]]], [[[30.0, 30.0, 30.0]]]]
    np.testing.assert_allclose(weights, expected_data)


@pytest.mark.parametrize("lazy", [True, False])
def test_get_coord_weights_plev(lazy):
    """Test ``get_coord_weights`` for complex cube."""
    cube = _make_cube()
    if lazy:
        cube.data = cube.lazy_data().rechunk((1, 1, 1, 3))
    weights = get_coord_weights(cube, "air_pressure")
    assert weights.shape == (1,)
    if lazy:
        assert isinstance(weights, da.Array)
        assert weights.chunks == ((1,),)
    else:
        assert isinstance(weights, np.ndarray)
    np.testing.assert_allclose(weights, [2.5])


@pytest.mark.parametrize("lazy", [True, False])
def test_get_coord_weights_lat(lazy):
    """Test ``get_coord_weights`` for complex cube."""
    cube = _make_cube()
    if lazy:
        cube.data = cube.lazy_data().rechunk((1, 1, 1, 3))
    weights = get_coord_weights(cube, "latitude")
    assert weights.shape == (1,)
    if lazy:
        assert isinstance(weights, da.Array)
        assert weights.chunks == ((1,),)
    else:
        assert isinstance(weights, np.ndarray)
    np.testing.assert_allclose(weights, [1.0])


@pytest.mark.parametrize("lazy", [True, False])
def test_get_coord_weights_lon(lazy):
    """Test ``get_coord_weights`` for complex cube."""
    cube = _make_cube()
    if lazy:
        cube.data = cube.lazy_data().rechunk((1, 1, 1, 3))
    weights = get_coord_weights(cube, "longitude")
    assert weights.shape == (3,)
    if lazy:
        assert isinstance(weights, da.Array)
        assert weights.chunks == ((3,),)
    else:
        assert isinstance(weights, np.ndarray)
    np.testing.assert_allclose(weights, [1.0, 1.0, 1.0])


@pytest.mark.parametrize("lazy", [True, False])
def test_get_coord_weights_0d_time(lazy):
    """Test ``get_coord_weights`` for 0D time coordinate."""
    time = get_0d_time()
    cube = Cube(0.0, var_name="x", units="K", aux_coords_and_dims=[(time, ())])
    if lazy:
        cube.data = cube.lazy_data()
    weights = get_coord_weights(cube, "time")
    assert weights.shape == (1,)
    if lazy:
        assert isinstance(weights, da.Array)
    else:
        assert isinstance(weights, np.ndarray)
    np.testing.assert_allclose(weights, [30.0])


@pytest.mark.parametrize("lazy", [True, False])
def test_get_coord_weights_0d_time_1d_lon(lazy):
    """Test ``get_coord_weights`` for 0D time and 1D longitude coordinate."""
    time = get_0d_time()
    lons = get_lon_coord()
    cube = Cube(
        [0.0, 0.0, 0.0],
        var_name="x",
        units="K",
        aux_coords_and_dims=[(time, ())],
        dim_coords_and_dims=[(lons, 0)],
    )
    if lazy:
        cube.data = cube.lazy_data()
    weights = get_coord_weights(cube, "time")
    assert weights.shape == (1,)
    if lazy:
        assert isinstance(weights, da.Array)
    else:
        assert isinstance(weights, np.ndarray)
    np.testing.assert_allclose(weights, [30.0])


@pytest.mark.parametrize("lazy", [True, False])
def test_get_coord_weights_0d_time_1d_lon_broadcast(lazy):
    """Test ``get_coord_weights`` for 0D time and 1D longitude coordinate."""
    time = get_0d_time()
    lons = get_lon_coord()
    cube = Cube(
        [0.0, 0.0, 0.0],
        var_name="x",
        units="K",
        aux_coords_and_dims=[(time, ())],
        dim_coords_and_dims=[(lons, 0)],
    )
    if lazy:
        cube.data = cube.lazy_data()
    weights = get_coord_weights(cube, "time", broadcast=True)
    assert weights.shape == (3,)
    if lazy:
        assert isinstance(weights, da.Array)
    else:
        assert isinstance(weights, np.ndarray)
    np.testing.assert_allclose(weights, [30.0, 30.0, 30.0])


@pytest.mark.parametrize("lazy", [True, False])
def test_get_coord_weights_1d_time(lazy):
    """Test ``get_coord_weights`` for 1D time coordinate."""
    time = get_1d_time()
    cube = Cube(
        [0.0, 1.0],
        var_name="x",
        units="K",
        dim_coords_and_dims=[(time, 0)],
    )
    if lazy:
        cube.data = cube.lazy_data()
    weights = get_coord_weights(cube, "time")
    assert weights.shape == (2,)
    if lazy:
        assert isinstance(weights, da.Array)
    else:
        assert isinstance(weights, np.ndarray)
    np.testing.assert_allclose(weights, [15.0, 30.0])


@pytest.mark.parametrize("lazy", [True, False])
def test_get_coord_weights_1d_time_1d_lon(lazy):
    """Test ``get_coord_weights`` for 1D time and 1D longitude coordinate."""
    time = get_1d_time()
    lons = get_lon_coord()
    cube = Cube(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        var_name="x",
        units="K",
        dim_coords_and_dims=[(time, 0), (lons, 1)],
    )
    if lazy:
        cube.data = cube.lazy_data()
    weights = get_coord_weights(cube, "time")
    assert weights.shape == (2,)
    if lazy:
        assert isinstance(weights, da.Array)
    else:
        assert isinstance(weights, np.ndarray)
    np.testing.assert_allclose(weights, [15.0, 30.0])


@pytest.mark.parametrize("lazy", [True, False])
def test_get_coord_weights_2d_time(lazy):
    """Test ``get_coord_weights`` for 2D time coordinate."""
    time = AuxCoord(
        [[20.0, 45.0]],
        standard_name="time",
        bounds=[[[15.0, 30.0], [30.0, 60.0]]],
        units=Unit("days since 1950-01-01", calendar="gregorian"),
    )
    cube = Cube(
        [[0.0, 1.0]],
        var_name="x",
        units="K",
        aux_coords_and_dims=[(time, (0, 1))],
    )
    if lazy:
        cube.data = cube.lazy_data()
    weights = get_coord_weights(cube, "time")
    assert weights.shape == (1, 2)
    if lazy:
        assert isinstance(weights, da.Array)
    else:
        assert isinstance(weights, np.ndarray)
    np.testing.assert_allclose(weights, [[15.0, 30.0]])


def test_get_coord_weights_no_bounds_fail():
    """Test ``get_coord_weights``."""
    cube = _make_cube()
    cube.coord("time").bounds = None
    msg = r"Cannot calculate weights for coordinate 'time' without bounds"
    with pytest.raises(ValueError, match=msg):
        get_coord_weights(cube, "time")


def test_get_coord_weights_triangular_bound_fail():
    """Test ``get_coord_weights``."""
    cube = _make_cube()
    cube.coord("latitude").bounds = [[1.0, 2.0, 3.0]]
    msg = (
        r"Cannot calculate weights for coordinate 'latitude' with 3 bounds "
        r"per point, expected 2 bounds per point"
    )
    with pytest.raises(ValueError, match=msg):
        get_coord_weights(cube, "latitude")
