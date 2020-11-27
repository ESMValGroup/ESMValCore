"""Functional test for :func:`esmvalcore.preprocessor._multimodel`."""

from itertools import groupby
from pathlib import Path

import iris
import numpy as np
import pytest

from esmvalcore.preprocessor import extract_time, multi_model_statistics

esmvaltool_sample_data = pytest.importorskip("esmvaltool_sample_data")

CACHE_FILES = True


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
def timeseries_cubes_month():
    """Representative timeseries data."""

    if CACHE_FILES:
        filename = Path(__file__).with_name('monthly.nc')
        if filename.exists():
            return iris.load(str(filename))

    time_slice = {
        'start_year': 1985,
        'end_year': 1987,
        'start_month': 12,
        'end_month': 1,
        'start_day': 1,
        'end_day': 30,
    }
    cubes = esmvaltool_sample_data.load_timeseries_cubes(mip_table='Amon')
    cubes = preprocess_data(cubes, time_slice=time_slice)

    if CACHE_FILES:
        iris.save(cubes, filename)

    return cubes


@pytest.fixture(scope="module")
def timeseries_cubes_day():
    """Representative timeseries data."""

    if CACHE_FILES:
        filename = Path(__file__).with_name('daily.nc')
        if filename.exists():
            return iris.load(str(filename))

    time_slice = {
        'start_year': 2001,
        'end_year': 2002,
        'start_month': 12,
        'end_month': 1,
        'start_day': 1,
        'end_day': 30,
    }
    cubes = esmvaltool_sample_data.load_timeseries_cubes(mip_table='day')
    cubes = preprocess_data(cubes, time_slice=time_slice)

    if CACHE_FILES:
        iris.save(cubes, filename)

    return cubes


def multimodel_test(cubes, span, statistic):
    statistics = [statistic]

    output = multi_model_statistics(cubes, span=span, statistics=statistics)
    assert isinstance(output, dict)
    assert statistic in output

    return output


def multimodel_regression_test(cubes, span, name):
    statistic = 'mean'
    output = multimodel_test(cubes, span=span, statistic=statistic)
    output_cube = output[statistic]

    # NOTE for the regression test
    # The following test will fail if the data are changed or if the
    # multimodel code changes significantly. To update the data for the
    # regression test, remove the corresponding `.nc` files.
    filename = Path(__file__).with_name(f'{name}-{span}-{statistic}.nc')
    if filename.exists():
        expected_cube = iris.load(str(filename))[0]
        assert np.allclose(output_cube.data, expected_cube.data)
    else:
        iris.save(output_cube, filename)
        raise RuntimeError(f'Wrote file {filename.absolute()}')


@pytest.mark.functional
@pytest.mark.parametrize('span', (
    'overlap',
    'full',
))
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
@pytest.mark.parametrize('span', (
    'overlap',
    'full',
))
def test_multimodel_regression_day(timeseries_cubes_day, span):
    """Test statistic."""
    cubes = timeseries_cubes_day

    def calendar(cube):
        return cube.coord('time').units.calendar

    # groupby requires sorted list
    grouped = groupby(sorted(cubes, key=calendar), key=calendar)

    for key, cube_group in grouped:
        cube_group = list(cube_group)

        # skip groups with 1 member
        if len(cube_group) <= 1:
            print('skipping', key)
            continue

        name = f'timeseries_daily_{key}'
        multimodel_regression_test(
            name=name,
            span=span,
            cubes=cube_group,
        )


@pytest.mark.functional
@pytest.mark.parametrize('span', ('overlap', 'full'))
def test_multimodel_month_no_vertical_dimension(timeseries_cubes_month, span):
    """Test statistic without vertical dimension."""
    cubes = timeseries_cubes_month
    cubes = [cube[0:50, 0] for cube in cubes]
    multimodel_test(cubes, span=span, statistic='mean')


@pytest.mark.functional
@pytest.mark.xfail('iris.exceptions.CoordinateNotFoundError')
@pytest.mark.parametrize('span', ('overlap', 'full'))
def test_multimodel_month_no_horizontal_dimension(timeseries_cubes_month,
                                                  span):
    """Test statistic without horizontal dimension."""
    cubes = timeseries_cubes_month
    cubes = [cube[:, :, 0, 0] for cube in cubes]
    # Coordinate not found error
    # iris.exceptions.CoordinateNotFoundError:
    # 'Expected to find exactly 1 depth coordinate, but found none.'
    multimodel_test(cubes, span=span, statistic='mean')


@pytest.mark.functional
@pytest.mark.parametrize('span', ('overlap', 'full'))
def test_multimodel_month_only_time_dimension(timeseries_cubes_month, span):
    """Test statistic without only the time dimension."""
    cubes = timeseries_cubes_month
    cubes = [cube[:, 0, 0, 0] for cube in cubes]
    multimodel_test(cubes, span=span, statistic='mean')


@pytest.mark.functional
@pytest.mark.xfail(
    'ValueError: Cannot guess bounds for a coordinate of length 1.')
@pytest.mark.parametrize('span', ('overlap', 'full'))
def test_multimodel_month_no_time_dimension(timeseries_cubes_month, span):
    """Test statistic without time dimension."""
    cubes = timeseries_cubes_month
    cubes = [cube[0] for cube in cubes]
    # ValueError: Cannot guess bounds for a coordinate of length 1.
    multimodel_test(cubes, span=span, statistic='mean')
