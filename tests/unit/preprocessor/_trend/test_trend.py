"""Unit tests for :mod:`esmvalcore.preprocessor._trend`."""
import dask.array as da
import iris
import iris.coord_categorisation
import numpy as np
import pytest
from cf_units import Unit

from esmvalcore.preprocessor._trend import linear_trend, linear_trend_stderr


def assert_masked_array_equal(arr_1, arr_2):
    """Check equality of two masked arrays."""
    arr_1 = np.ma.array(arr_1)
    arr_2 = np.ma.array(arr_2)
    mask_1 = np.ma.getmaskarray(arr_1)
    mask_2 = np.ma.getmaskarray(arr_2)
    np.testing.assert_allclose(mask_1, mask_2)
    data_1 = arr_1.filled(np.nan)
    data_2 = arr_2.filled(np.nan)
    np.testing.assert_allclose(data_1, data_2)


def get_cube(times=None, time_units=None):
    """Create cube."""
    lats = iris.coords.DimCoord([0.0, 20.0], standard_name='latitude',
                                units='m')
    lons = iris.coords.DimCoord([500.0, 600.0], standard_name='longitude',
                                units='m')
    aux_coord = iris.coords.AuxCoord([0.0, 0.0], var_name='aux')
    if times is None:
        cube = iris.cube.Cube([[1.0, 2.0], [3.0, 4.0]], var_name='x',
                              long_name='X', units='kg',
                              dim_coords_and_dims=[(lats, 0), (lons, 1)],
                              aux_coords_and_dims=[(aux_coord, 0)])
        return cube
    if time_units is None:
        time_units = Unit('days since 1850-01-01 00:00:00')
    times = iris.coords.DimCoord(times, standard_name='time', units=time_units)
    cube_data = np.arange(4 * times.shape[0]).reshape(times.shape[0], 2, 2)
    cube = iris.cube.Cube(cube_data.astype('float32'), var_name='x',
                          long_name='X', units='kg',
                          dim_coords_and_dims=[(times, 0), (lats, 1),
                                               (lons, 2)],
                          aux_coords_and_dims=[(aux_coord, 1)])

    return cube


@pytest.fixture
def cube_no_time():
    """Cube with no time dimension."""
    return get_cube()


@pytest.fixture
def cube_1_time():
    """Cube with single time point."""
    return get_cube(times=[0.0])


@pytest.fixture
def cube_3_time():
    """Cube with three time points."""
    return get_cube(times=[0.0, 1.0, 2.0])


@pytest.fixture
def cube_3_time_years():
    """Cube with three years."""
    return get_cube(times=[0.0, 1.0, 2.0], time_units='year')


def test_linear_trend_coord_not_found(cube_no_time):
    """Test calculation of linear trend when dimension is not available."""
    with pytest.raises(iris.exceptions.CoordinateNotFoundError) as err:
        linear_trend(cube_no_time)
    assert 'time' in str(err.value)
    with pytest.raises(iris.exceptions.CoordinateNotFoundError) as err:
        linear_trend(cube_no_time, coordinate='time')
    assert 'time' in str(err.value)
    with pytest.raises(iris.exceptions.CoordinateNotFoundError) as err:
        linear_trend(cube_no_time, coordinate='aux')
    assert 'aux' in str(err.value)


def test_linear_trend_1_time(cube_1_time):
    """Test calculation of linear trend with single time point."""
    cube_trend = linear_trend(cube_1_time)
    assert cube_trend.shape == (2, 2)
    assert_masked_array_equal(cube_trend.data,
                              np.ma.masked_equal([[0.0, 0.0], [0.0, 0.0]],
                                                 0.0))
    assert not cube_trend.coords('time', dim_coords=True)
    assert cube_trend.coords('latitude', dim_coords=True)
    assert cube_trend.coords('longitude', dim_coords=True)
    assert cube_trend.units == 'kg day-1'
    assert (iris.coords.CellMethod('trend', coords=('time',)) in
            cube_trend.cell_methods)


def test_linear_trend_3_time(cube_3_time):
    """Test calculation of linear trend with three time points."""
    cube_3_time.data[0, 0, 0] = 1.0
    cube_trend = linear_trend(cube_3_time)
    assert cube_trend.shape == (2, 2)
    assert_masked_array_equal(cube_trend.data, [[3.5, 4.0], [4.0, 4.0]])
    assert not cube_trend.coords('time', dim_coords=True)
    assert cube_trend.coords('latitude', dim_coords=True)
    assert cube_trend.coords('longitude', dim_coords=True)
    assert cube_trend.units == 'kg day-1'
    assert (iris.coords.CellMethod('trend', coords=('time',)) in
            cube_trend.cell_methods)


def test_linear_trend_3_time_lazy(cube_3_time):
    """Test lazy calculation of linear trend with three time points."""
    cube_3_time.data = -2.0 * da.arange(3 * 2 * 2).reshape(3, 2, 2)
    assert cube_3_time.has_lazy_data()
    cube_trend = linear_trend(cube_3_time)
    assert cube_trend.shape == (2, 2)
    assert_masked_array_equal(cube_trend.data, [[-8.0, -8.0], [-8.0, -8.0]])
    assert not cube_trend.coords('time', dim_coords=True)
    assert cube_trend.coords('latitude', dim_coords=True)
    assert cube_trend.coords('longitude', dim_coords=True)
    assert cube_trend.units == 'kg day-1'
    assert (iris.coords.CellMethod('trend', coords=('time',)) in
            cube_trend.cell_methods)


def test_linear_trend_3_time_no_metadata(cube_3_time):
    """Test calculation of trend with three time points and no metadata."""
    cube_3_time.units = None
    cube_trend = linear_trend(cube_3_time)
    assert cube_trend.shape == (2, 2)
    assert_masked_array_equal(cube_trend.data, [[4.0, 4.0], [4.0, 4.0]])
    assert cube_trend.units == Unit('unknown')
    assert (iris.coords.CellMethod('trend', coords=('time',)) in
            cube_trend.cell_methods)

    # Cube with unknown units
    cube_3_time.units = Unit('unknown')
    cube_trend = linear_trend(cube_3_time)
    assert cube_trend.shape == (2, 2)
    assert_masked_array_equal(cube_trend.data, [[4.0, 4.0], [4.0, 4.0]])
    assert cube_trend.units == Unit('unknown')
    assert (iris.coords.CellMethod('trend', coords=('time',)) in
            cube_trend.cell_methods)

    # Cube with no units
    cube_3_time.units = Unit('no unit')
    cube_trend = linear_trend(cube_3_time)
    assert cube_trend.shape == (2, 2)
    assert_masked_array_equal(cube_trend.data, [[4.0, 4.0], [4.0, 4.0]])
    assert cube_trend.units == Unit('no unit')
    assert (iris.coords.CellMethod('trend', coords=('time',)) in
            cube_trend.cell_methods)

    # Time with unknown units
    cube_3_time.units = 'kg'
    cube_3_time.coord('time').units = Unit('unknown')
    cube_trend = linear_trend(cube_3_time)
    assert cube_trend.shape == (2, 2)
    assert_masked_array_equal(cube_trend.data, [[4.0, 4.0], [4.0, 4.0]])
    assert cube_trend.units == Unit('unknown')
    assert (iris.coords.CellMethod('trend', coords=('time',)) in
            cube_trend.cell_methods)

    # Time with no units
    cube_3_time.coord('time').units = Unit('no unit')
    cube_trend = linear_trend(cube_3_time)
    assert cube_trend.shape == (2, 2)
    assert_masked_array_equal(cube_trend.data, [[4.0, 4.0], [4.0, 4.0]])
    assert cube_trend.units == Unit('kg')
    assert (iris.coords.CellMethod('trend', coords=('time',)) in
            cube_trend.cell_methods)


def test_linear_trend_3_time_years(cube_3_time_years):
    """Test calculation of linear trend with three years."""
    cube_trend = linear_trend(cube_3_time_years)
    assert cube_trend.shape == (2, 2)
    assert_masked_array_equal(cube_trend.data, [[4.0, 4.0], [4.0, 4.0]])
    assert cube_trend.units == 'kg yr-1'
    assert (iris.coords.CellMethod('trend', coords=('time',)) in
            cube_trend.cell_methods)


def test_linear_trend_latitude(cube_3_time):
    """Test calculation of linear trend along latitude coordinate."""
    cube_3_time.data[0, 0, 0] = np.nan
    cube_3_time.data = np.ma.masked_invalid(cube_3_time.data)
    cube_trend = linear_trend(cube_3_time, coordinate='latitude')
    assert cube_trend.shape == (3, 2)
    assert_masked_array_equal(cube_trend.data, np.ma.masked_invalid(
        [[np.nan, 0.1], [0.1, 0.1], [0.1, 0.1]]))
    assert cube_trend.coords('time', dim_coords=True)
    assert not cube_trend.coords('latitude', dim_coords=True)
    assert cube_trend.coords('longitude', dim_coords=True)
    assert cube_trend.units == 'kg m-1'
    assert (iris.coords.CellMethod('trend', coords=('latitude',)) in
            cube_trend.cell_methods)


def test_linear_trend_longitude(cube_3_time):
    """Test calculation of linear trend along longitude coordinate."""
    cube_3_time.data[1, 0, 0] = np.nan
    cube_3_time.data = np.ma.masked_invalid(cube_3_time.data)
    cube_trend = linear_trend(cube_3_time, coordinate='longitude')
    assert cube_trend.shape == (3, 2)
    assert_masked_array_equal(cube_trend.data, np.ma.masked_invalid(
        [[0.01, 0.01], [np.nan, 0.01], [0.01, 0.01]]))
    assert cube_trend.coords('time', dim_coords=True)
    assert cube_trend.coords('latitude', dim_coords=True)
    assert not cube_trend.coords('longitude', dim_coords=True)
    assert cube_trend.units == 'kg m-1'
    assert (iris.coords.CellMethod('trend', coords=('longitude',)) in
            cube_trend.cell_methods)


def test_linear_trend_stderr_coord_not_found(cube_no_time):
    """Test calculation of trend stderr when dimension is not available."""
    with pytest.raises(iris.exceptions.CoordinateNotFoundError) as err:
        linear_trend_stderr(cube_no_time)
    assert 'time' in str(err.value)
    with pytest.raises(iris.exceptions.CoordinateNotFoundError) as err:
        linear_trend_stderr(cube_no_time, coordinate='time')
    assert 'time' in str(err.value)
    with pytest.raises(iris.exceptions.CoordinateNotFoundError) as err:
        linear_trend_stderr(cube_no_time, coordinate='aux')
    assert 'aux' in str(err.value)


def test_linear_trend_stderr_1_time(cube_1_time):
    """Test calculation of trend stderr with single time point."""
    cube_stderr = linear_trend_stderr(cube_1_time)
    assert cube_stderr.shape == (2, 2)
    assert_masked_array_equal(cube_stderr.data,
                              np.ma.masked_equal([[0.0, 0.0], [0.0, 0.0]],
                                                 0.0))
    assert not cube_stderr.coords('time', dim_coords=True)
    assert cube_stderr.coords('latitude', dim_coords=True)
    assert cube_stderr.coords('longitude', dim_coords=True)
    assert cube_stderr.units == 'kg day-1'
    assert (iris.coords.CellMethod('trend_stderr', coords=('time',)) in
            cube_stderr.cell_methods)


def test_linear_trend_stderr_3_time(cube_3_time):
    """Test calculation of trend stderr with three time points."""
    cube_3_time.data[0, 0, 0] = 1.0
    cube_stderr = linear_trend_stderr(cube_3_time)
    assert cube_stderr.shape == (2, 2)
    assert_masked_array_equal(cube_stderr.data,
                              [[0.28867513459482086, 0.0], [0.0, 0.0]])
    assert not cube_stderr.coords('time', dim_coords=True)
    assert cube_stderr.coords('latitude', dim_coords=True)
    assert cube_stderr.coords('longitude', dim_coords=True)
    assert cube_stderr.units == 'kg day-1'
    assert (iris.coords.CellMethod('trend_stderr', coords=('time',)) in
            cube_stderr.cell_methods)


def test_linear_trend_stderr_3_time_lazy(cube_3_time):
    """Test lazy calculation of trend stderr with three time points."""
    cube_3_time.data = da.array([[[1.0, 1.0], [2.0, 3.0]],
                                 [[4.0, 5.0], [6.0, 7.0]],
                                 [[8.0, 9.0], [10.0, 11.0]]])
    assert cube_3_time.has_lazy_data()
    cube_stderr = linear_trend_stderr(cube_3_time)
    assert cube_stderr.shape == (2, 2)
    assert_masked_array_equal(cube_stderr.data,
                              [[0.28867513459482086, 0.0], [0.0, 0.0]])
    assert not cube_stderr.coords('time', dim_coords=True)
    assert cube_stderr.coords('latitude', dim_coords=True)
    assert cube_stderr.coords('longitude', dim_coords=True)
    assert cube_stderr.units == 'kg day-1'
    assert (iris.coords.CellMethod('trend_stderr', coords=('time',)) in
            cube_stderr.cell_methods)


def test_linear_trend_stderr_3_time_no_metadata(cube_3_time):
    """Test calculation of trend stderr with no metadata."""
    cube_3_time.units = None
    cube_stderr = linear_trend_stderr(cube_3_time)
    assert cube_stderr.shape == (2, 2)
    assert_masked_array_equal(cube_stderr.data, [[0.0, 0.0], [0.0, 0.0]])
    assert cube_stderr.units == Unit('unknown')
    assert (iris.coords.CellMethod('trend_stderr', coords=('time',)) in
            cube_stderr.cell_methods)

    # Cube with unknown units
    cube_3_time.units = Unit('unknown')
    cube_stderr = linear_trend_stderr(cube_3_time)
    assert cube_stderr.shape == (2, 2)
    assert_masked_array_equal(cube_stderr.data, [[0.0, 0.0], [0.0, 0.0]])
    assert cube_stderr.units == Unit('unknown')
    assert (iris.coords.CellMethod('trend_stderr', coords=('time',)) in
            cube_stderr.cell_methods)

    # Cube with no units
    cube_3_time.units = Unit('no unit')
    cube_stderr = linear_trend_stderr(cube_3_time)
    assert cube_stderr.shape == (2, 2)
    assert_masked_array_equal(cube_stderr.data, [[0.0, 0.0], [0.0, 0.0]])
    assert cube_stderr.units == Unit('no unit')
    assert (iris.coords.CellMethod('trend_stderr', coords=('time',)) in
            cube_stderr.cell_methods)

    # Time with unknown units
    cube_3_time.units = 'kg'
    cube_3_time.coord('time').units = Unit('unknown')
    cube_stderr = linear_trend_stderr(cube_3_time)
    assert cube_stderr.shape == (2, 2)
    assert_masked_array_equal(cube_stderr.data, [[0.0, 0.0], [0.0, 0.0]])
    assert cube_stderr.units == Unit('unknown')
    assert (iris.coords.CellMethod('trend_stderr', coords=('time',)) in
            cube_stderr.cell_methods)

    # Time with no units
    cube_3_time.coord('time').units = Unit('no unit')
    cube_stderr = linear_trend_stderr(cube_3_time)
    assert cube_stderr.shape == (2, 2)
    assert_masked_array_equal(cube_stderr.data, [[0.0, 0.0], [0.0, 0.0]])
    assert cube_stderr.units == Unit('kg')
    assert (iris.coords.CellMethod('trend_stderr', coords=('time',)) in
            cube_stderr.cell_methods)


def test_linear_trend_stderr_3_time_years(cube_3_time_years):
    """Test calculation of trend stderr with three years."""
    cube_3_time_years.data[1, 1, 1] = 1.0
    cube_stderr = linear_trend_stderr(cube_3_time_years)
    assert cube_stderr.shape == (2, 2)
    assert_masked_array_equal(cube_stderr.data,
                              [[0.0, 0.0], [0.0, 3.464101615137754]])
    assert cube_stderr.units == 'kg yr-1'
    assert (iris.coords.CellMethod('trend_stderr', coords=('time',)) in
            cube_stderr.cell_methods)


def test_linear_trend_stderr_latitude(cube_3_time):
    """Test calculation of trend stderr along latitude coordinate."""
    cube_3_time.data[0, 0, 0] = np.nan
    cube_3_time.data = np.ma.masked_invalid(cube_3_time.data)
    cube_stderr = linear_trend_stderr(cube_3_time, coordinate='latitude')
    assert cube_stderr.shape == (3, 2)
    assert_masked_array_equal(cube_stderr.data, np.ma.masked_invalid(
        [[np.nan, 0.0], [0.0, 0.0], [0.0, 0.0]]))
    assert cube_stderr.coords('time', dim_coords=True)
    assert not cube_stderr.coords('latitude', dim_coords=True)
    assert cube_stderr.coords('longitude', dim_coords=True)
    assert cube_stderr.units == 'kg m-1'
    assert (iris.coords.CellMethod('trend_stderr', coords=('latitude',)) in
            cube_stderr.cell_methods)


def test_linear_trend_stderr_longitude(cube_3_time):
    """Test calculation of trend stderr along longitude coordinate."""
    cube_3_time.data[1, 0, 0] = np.nan
    cube_3_time.data = np.ma.masked_invalid(cube_3_time.data)
    cube_stderr = linear_trend_stderr(cube_3_time, coordinate='longitude')
    assert cube_stderr.shape == (3, 2)
    assert_masked_array_equal(cube_stderr.data, np.ma.masked_invalid(
        [[0.0, 0.0], [np.nan, 0.0], [0.0, 0.0]]))
    assert cube_stderr.coords('time', dim_coords=True)
    assert cube_stderr.coords('latitude', dim_coords=True)
    assert not cube_stderr.coords('longitude', dim_coords=True)
    assert cube_stderr.units == 'kg m-1'
    assert (iris.coords.CellMethod('trend_stderr', coords=('longitude',)) in
            cube_stderr.cell_methods)
