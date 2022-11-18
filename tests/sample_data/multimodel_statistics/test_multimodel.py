"""Test using sample data for :func:`esmvalcore.preprocessor._multimodel`."""

import pickle
import platform
from itertools import groupby
from pathlib import Path
from typing import Optional

import cf_units
import iris
import numpy as np
import pytest

from esmvalcore.preprocessor import extract_time
from esmvalcore.preprocessor._multimodel import multi_model_statistics

esmvaltool_sample_data = pytest.importorskip("esmvaltool_sample_data")

# Increase this number anytime you change the cached input data to the tests.
TEST_REVISION = 1

CALENDAR_PARAMS = (
    pytest.param(
        '360_day',
        marks=pytest.mark.skip(
            reason='Cannot calculate statistics with single cube in list')),
    '365_day',
    'standard' if cf_units.__version__ >= '3.1' else 'gregorian',
    pytest.param(
        'proleptic_gregorian',
        marks=pytest.mark.xfail(
            raises=iris.exceptions.MergeError,
            reason='https://github.com/ESMValGroup/ESMValCore/issues/956')),
    pytest.param(
        'julian',
        marks=pytest.mark.skip(
            reason='Cannot calculate statistics with single cube in list')),
)

SPAN_PARAMS = ('overlap', 'full')


def assert_array_almost_equal(this, other):
    """Assert that array `this` almost equals array `other`."""
    if np.ma.isMaskedArray(this) or np.ma.isMaskedArray(other):
        np.testing.assert_array_equal(this.mask, other.mask)

    np.testing.assert_allclose(this, other)


def assert_coords_equal(this: list, other: list):
    """Assert coords list `this` equals coords list `other`."""
    for this_coord, other_coord in zip(this, other):
        np.testing.assert_equal(this_coord.points, other_coord.points)
        assert this_coord.var_name == other_coord.var_name
        assert this_coord.standard_name == other_coord.standard_name
        assert this_coord.units == other_coord.units


def assert_metadata_equal(this, other):
    """Assert metadata `this` are equal to metadata `other`."""
    assert this.standard_name == other.standard_name
    assert this.long_name == other.long_name
    assert this.var_name == other.var_name
    assert this.units == other.units


def fix_metadata(cubes):
    """Fix metadata."""
    for cube in cubes:
        cube.coord('air_pressure').bounds = None


def preprocess_data(cubes, time_slice: Optional[dict] = None):
    """Regrid the data to the first cube and optional time-slicing."""
    # Increase TEST_REVISION anytime you make changes to this function.
    if time_slice:
        cubes = [extract_time(cube, **time_slice) for cube in cubes]

    first_cube = cubes[0]

    # regrid to first cube
    regrid_kwargs = {
        'grid': first_cube,
        'scheme': iris.analysis.Nearest(),
    }

    cubes = [cube.regrid(**regrid_kwargs) for cube in cubes]

    return cubes


def get_cache_key(value):
    """Get a cache key that is hopefully unique enough for unpickling.

    If this doesn't avoid problems with unpickling the cached data,
    manually clean the pytest cache with the command `pytest --cache-
    clear`.
    """
    py_version = platform.python_version()
    return (f'{value}_iris-{iris.__version__}_'
            f'numpy-{np.__version__}_python-{py_version}'
            f'rev-{TEST_REVISION}')


@pytest.fixture(scope="module")
def timeseries_cubes_month(request):
    """Load representative timeseries data."""
    # cache the cubes to save about 30-60 seconds on repeat use
    cache_key = get_cache_key("sample_data/monthly")
    data = request.config.cache.get(cache_key, None)

    if data:
        cubes = pickle.loads(data.encode('latin1'))
    else:
        # Increase TEST_REVISION anytime you make changes here.
        time_slice = {
            'start_year': 1985,
            'end_year': 1987,
            'start_month': 12,
            'end_month': 2,
            'start_day': 1,
            'end_day': 1,
        }
        cubes = esmvaltool_sample_data.load_timeseries_cubes(mip_table='Amon')
        cubes = preprocess_data(cubes, time_slice=time_slice)

        # cubes are not serializable via json, so we must go via pickle
        request.config.cache.set(cache_key,
                                 pickle.dumps(cubes).decode('latin1'))

    fix_metadata(cubes)

    return cubes


@pytest.fixture(scope="module")
def timeseries_cubes_day(request):
    """Load representative timeseries data grouped by calendar."""
    # cache the cubes to save about 30-60 seconds on repeat use
    cache_key = get_cache_key("sample_data/daily")
    data = request.config.cache.get(cache_key, None)

    if data:
        cubes = pickle.loads(data.encode('latin1'))

    else:
        # Increase TEST_REVISION anytime you make changes here.
        time_slice = {
            'start_year': 2001,
            'end_year': 2002,
            'start_month': 12,
            'end_month': 2,
            'start_day': 1,
            'end_day': 1,
        }
        cubes = esmvaltool_sample_data.load_timeseries_cubes(mip_table='day')
        cubes = preprocess_data(cubes, time_slice=time_slice)

        # cubes are not serializable via json, so we must go via pickle
        request.config.cache.set(cache_key,
                                 pickle.dumps(cubes).decode('latin1'))

    fix_metadata(cubes)

    def calendar(cube):
        return cube.coord('time').units.calendar

    # groupby requires sorted list
    grouped = groupby(sorted(cubes, key=calendar), key=calendar)

    cube_dict = {key: list(group) for key, group in grouped}

    return cube_dict


def multimodel_test(cubes, statistic, span):
    """Run multimodel test with some simple checks."""
    statistics = [statistic]

    result = multi_model_statistics(products=cubes,
                                    statistics=statistics,
                                    span=span)
    assert isinstance(result, dict)
    assert statistic in result

    return result


def multimodel_regression_test(cubes, span, name):
    """Run multimodel regression test.

    This test will fail if the input data or multimodel code changed. To
    update the data for the regression test, remove the corresponding
    `.nc` files in this directory and re-run the tests. The tests will
    fail the first time with a RuntimeError, because the reference data
    are being written.
    """
    statistic = 'mean'
    result = multimodel_test(cubes, statistic=statistic, span=span)
    result_cube = result[statistic]

    filename = Path(__file__).with_name(f'{name}-{span}-{statistic}.nc')
    if filename.exists():
        reference_cube = iris.load_cube(str(filename))

        assert_array_almost_equal(result_cube.data, reference_cube.data)
        assert_metadata_equal(result_cube.metadata, reference_cube.metadata)
        assert_coords_equal(result_cube.coords(), reference_cube.coords())

    else:
        # The test will fail if no regression data are available.
        iris.save(result_cube, filename)
        raise RuntimeError(f'Wrote reference data to {filename.absolute()}')


@pytest.mark.xfail(
    raises=iris.exceptions.MergeError,
    reason='https://github.com/ESMValGroup/ESMValCore/issues/956')
@pytest.mark.use_sample_data
@pytest.mark.parametrize('span', SPAN_PARAMS)
def test_multimodel_regression_month(timeseries_cubes_month, span):
    """Test statistic."""
    cubes = timeseries_cubes_month
    name = 'timeseries_monthly'
    multimodel_regression_test(
        name=name,
        span=span,
        cubes=cubes,
    )


@pytest.mark.use_sample_data
@pytest.mark.parametrize('calendar', CALENDAR_PARAMS)
@pytest.mark.parametrize('span', SPAN_PARAMS)
def test_multimodel_regression_day(timeseries_cubes_day, span, calendar):
    """Test statistic."""
    cubes = timeseries_cubes_day[calendar]
    name = f'timeseries_daily_{calendar}'
    multimodel_regression_test(
        name=name,
        span=span,
        cubes=cubes,
    )


@pytest.mark.use_sample_data
def test_multimodel_no_vertical_dimension(timeseries_cubes_month):
    """Test statistic without vertical dimension using monthly data."""
    span = 'full'
    cubes = timeseries_cubes_month
    cubes = [cube[:, 0] for cube in cubes]
    multimodel_test(cubes, span=span, statistic='mean')


@pytest.mark.use_sample_data
@pytest.mark.xfail(
    raises=iris.exceptions.MergeError,
    reason='https://github.com/ESMValGroup/ESMValCore/issues/956')
# @pytest.mark.xfail(
#     raises=iris.exceptions.CoordinateNotFoundError,
#     reason='https://github.com/ESMValGroup/ESMValCore/issues/891')
def test_multimodel_no_horizontal_dimension(timeseries_cubes_month):
    """Test statistic without horizontal dimension using monthly data."""
    span = 'full'
    cubes = timeseries_cubes_month
    cubes = [cube[:, :, 0, 0] for cube in cubes]
    # Coordinate not found error
    # iris.exceptions.CoordinateNotFoundError:
    # 'Expected to find exactly 1 depth coordinate, but found none.'
    multimodel_test(cubes, span=span, statistic='mean')


@pytest.mark.use_sample_data
def test_multimodel_only_time_dimension(timeseries_cubes_month):
    """Test statistic without only the time dimension using monthly data."""
    cubes = timeseries_cubes_month
    span = 'full'
    cubes = [cube[:, 0, 0, 0] for cube in cubes]
    multimodel_test(cubes, span=span, statistic='mean')


@pytest.mark.use_sample_data
@pytest.mark.xfail(
    raises=ValueError,
    reason='https://github.com/ESMValGroup/ESMValCore/issues/890')
def test_multimodel_no_time_dimension(timeseries_cubes_month):
    """Test statistic without time dimension using monthly data."""
    span = 'full'
    cubes = timeseries_cubes_month
    cubes = [cube[0] for cube in cubes]
    # ValueError: Cannot guess bounds for a coordinate of length 1.
    multimodel_test(cubes, span=span, statistic='mean')
