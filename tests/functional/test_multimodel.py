"""Functional test for :func:`esmvalcore.preprocessor._multimodel`."""

import pickle
from itertools import groupby
from pathlib import Path

import iris
import numpy as np
import pytest

from esmvalcore.preprocessor import extract_time, multi_model_statistics

esmvaltool_sample_data = pytest.importorskip("esmvaltool_sample_data")

CALENDAR_PARAMS = (
    pytest.param(
        '360_day',
        marks=pytest.mark.skip(
            reason='Cannot calculate statistics with single cube in list')),
    '365_day',
    'gregorian',
    'proleptic_gregorian',
    pytest.param(
        'julian',
        marks=pytest.mark.skip(
            reason='Cannot calculate statistics with single cube in list')),
)

SPAN_PARAMS = ('overlap', 'full')


def assert_array_equal(a, b):
    """Assert that array a equals array b."""
    np.testing.assert_array_equal(a, b)
    if np.ma.isMaskedArray(a) or np.ma.isMaskedArray(b):
        np.testing.assert_array_equal(a.mask, b.mask)


def preprocess_data(cubes, time_slice: dict = None):
    """Regrid the data to the first cube and optional time-slicing."""
    if time_slice:
        cubes = [extract_time(cube, **time_slice) for cube in cubes]

    first_cube = cubes[0]

    # regrid to first cube
    regrid_kwargs = {
        'grid': first_cube,
        'scheme': iris.analysis.Linear(),
    }

    cubes = [cube.regrid(**regrid_kwargs) for cube in cubes]

    return cubes


@pytest.fixture(scope="module")
def timeseries_cubes_month(request):
    """Representative timeseries data."""

    # cache the cubes to save about 30-60 seconds on repeat use
    data = request.config.cache.get("functional/monthly", None)

    if data:
        cubes = pickle.loads(data.encode('latin1'))
    else:
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
        request.config.cache.set("functional/monthly",
                                 pickle.dumps(cubes).decode('latin1'))

    return cubes


@pytest.fixture(scope="module")
def timeseries_cubes_day(request):
    """Representative timeseries data grouped by calendar."""

    # cache the cubes to save about 30-60 seconds on repeat use
    data = request.config.cache.get("functional/daily", None)

    if data:
        cubes = pickle.loads(data.encode('latin1'))

    else:
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
        request.config.cache.set("functional/daily",
                                 pickle.dumps(cubes).decode('latin1'))

    def calendar(cube):
        return cube.coord('time').units.calendar

    # groupby requires sorted list
    grouped = groupby(sorted(cubes, key=calendar), key=calendar)

    cube_dict = {key: list(group) for key, group in grouped}

    return cube_dict


def multimodel_test(cubes, span, statistic):
    statistics = [statistic]

    output = multi_model_statistics(cubes, span=span, statistics=statistics)
    assert isinstance(output, dict)
    assert statistic in output

    return output


def multimodel_regression_test(cubes, span, name):
    statistic = 'mean'
    output = multimodel_test(cubes, span=span, statistic=statistic)
    this_cube = output[statistic]

    # NOTE for the regression test
    # The following test will fail if the data are changed or if the
    # multimodel code changes significantly. To update the data for the
    # regression test, remove the corresponding `.nc` files.
    filename = Path(__file__).with_name(f'{name}-{span}-{statistic}.nc')
    if filename.exists():
        other_cube = iris.load(str(filename))[0]
        assert_array_equal(this_cube.data, other_cube.data)

        # Compare coords
        for this_coord, other_coord in zip(this_cube.coords(),
                                           other_cube.coords()):
            assert this_coord == other_coord

        # remove Conventions which are added by Iris on save
        other_cube.attributes.pop('Conventions', None)

        assert other_cube.metadata == this_cube.metadata

    else:
        iris.save(this_cube, filename)
        raise RuntimeError(f'Wrote file {filename.absolute()}')


@pytest.mark.functional
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


@pytest.mark.functional
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


@pytest.mark.functional
def test_multimodel_no_vertical_dimension(timeseries_cubes_month):
    """Test statistic without vertical dimension using monthly data."""
    span = 'full'
    cubes = timeseries_cubes_month
    cubes = [cube[:, 0] for cube in cubes]
    multimodel_test(cubes, span=span, statistic='mean')


@pytest.mark.functional
@pytest.mark.xfail('iris.exceptions.CoordinateNotFoundError')
def test_multimodel_no_horizontal_dimension(timeseries_cubes_month):
    """Test statistic without horizontal dimension using monthly data."""
    span = 'full'
    cubes = timeseries_cubes_month
    cubes = [cube[:, :, 0, 0] for cube in cubes]
    # Coordinate not found error
    # iris.exceptions.CoordinateNotFoundError:
    # 'Expected to find exactly 1 depth coordinate, but found none.'
    multimodel_test(cubes, span=span, statistic='mean')


@pytest.mark.functional
def test_multimodel_only_time_dimension(timeseries_cubes_month):
    """Test statistic without only the time dimension using monthly data."""
    cubes = timeseries_cubes_month
    span = 'full'
    cubes = [cube[:, 0, 0, 0] for cube in cubes]
    multimodel_test(cubes, span=span, statistic='mean')


@pytest.mark.functional
@pytest.mark.xfail('ValueError')
def test_multimodel_no_time_dimension(timeseries_cubes_month):
    """Test statistic without time dimension using monthly data."""
    span = 'full'
    cubes = timeseries_cubes_month
    cubes = [cube[0] for cube in cubes]
    # ValueError: Cannot guess bounds for a coordinate of length 1.
    multimodel_test(cubes, span=span, statistic='mean')
