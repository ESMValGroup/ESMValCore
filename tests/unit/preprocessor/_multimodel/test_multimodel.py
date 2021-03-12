"""Unit test for :func:`esmvalcore.preprocessor._multimodel`"""

from datetime import datetime

import iris
import numpy as np
import pytest
from cf_units import Unit
from iris.cube import Cube

import esmvalcore.preprocessor._multimodel as mm
from esmvalcore.preprocessor import multi_model_statistics

SPAN_OPTIONS = ('overlap', 'full')

FREQUENCY_OPTIONS = ('daily', 'monthly', 'yearly')  # hourly

CALENDAR_OPTIONS = ('360_day', '365_day', 'gregorian', 'proleptic_gregorian',
                    'julian')


def assert_array_allclose(this, other):
    """Assert that array `this` is close to array `other`."""
    if np.ma.isMaskedArray(this) or np.ma.isMaskedArray(other):
        np.testing.assert_array_equal(this.mask, other.mask)

    np.testing.assert_allclose(this, other)


def timecoord(frequency,
              calendar='gregorian',
              offset='days since 1850-01-01',
              num=3):
    """Return a time coordinate with the given time points and calendar."""

    time_points = range(1, num + 1)

    if frequency == 'hourly':
        dates = [datetime(1850, 1, 1, i, 0, 0) for i in time_points]
    if frequency == 'daily':
        dates = [datetime(1850, 1, i, 0, 0, 0) for i in time_points]
    elif frequency == 'monthly':
        dates = [datetime(1850, i, 15, 0, 0, 0) for i in time_points]
    elif frequency == 'yearly':
        dates = [datetime(1850, 7, i, 0, 0, 0) for i in time_points]

    unit = Unit(offset, calendar=calendar)
    points = unit.date2num(dates)
    return iris.coords.DimCoord(points, standard_name='time', units=unit)


def generate_cube_from_dates(
    dates,
    calendar='gregorian',
    offset='days since 1850-01-01',
    fill_val=1,
    len_data=3,
    var_name=None,
):
    """Generate test cube from list of dates / frequency specification.

    Parameters
    ----------
    calendar : str or list
        Date frequency: 'hourly' / 'daily' / 'monthly' / 'yearly' or
        list of datetimes.
    offset : str
        Offset to use
    fill_val : int
        Value to fill the data with
    len_data : int
        Number of data / time points
    var_name : str
        Name of the data variable

    Returns
    -------
    iris.cube.Cube
    """
    if isinstance(dates, str):
        time = timecoord(dates, calendar, offset=offset, num=len_data)
    else:
        unit = Unit(offset, calendar=calendar)
        time = iris.coords.DimCoord(unit.date2num(dates),
                                    standard_name='time',
                                    units=unit)

    return Cube((fill_val, ) * len_data,
                dim_coords_and_dims=[(time, 0)],
                var_name=var_name)


def get_cubes_for_validation_test(frequency):
    """Set up cubes used for testing multimodel statistics."""

    # Simple 1d cube with standard time cord
    cube1 = generate_cube_from_dates(frequency)

    # Cube with masked data
    cube2 = cube1.copy()
    cube2.data = np.ma.array([5, 5, 5], mask=[True, False, False])

    # Cube with deviating time coord
    cube3 = generate_cube_from_dates(frequency,
                                     calendar='360_day',
                                     offset='days since 1950-01-01',
                                     len_data=2,
                                     fill_val=9)

    return [cube1, cube2, cube3]


VALIDATION_DATA_SUCCESS = (
    ('full', 'mean', (5, 5, 3)),
    pytest.param(
        'full',
        'std', (5.656854249492381, 4, 2.8284271247461903),
        marks=pytest.mark.xfail(
            raises=AssertionError,
            reason='https://github.com/ESMValGroup/ESMValCore/issues/1024')),
    ('full', 'min', (1, 1, 1)),
    ('full', 'max', (9, 9, 5)),
    ('full', 'median', (5, 5, 3)),
    ('full', 'p50', (5, 5, 3)),
    ('full', 'p99.5', (8.96, 8.96, 4.98)),
    ('overlap', 'mean', (5, 5)),
    pytest.param(
        'full',
        'std', (5.656854249492381, 4),
        marks=pytest.mark.xfail(
            raises=AssertionError,
            reason='https://github.com/ESMValGroup/ESMValCore/issues/1024')),
    ('overlap', 'min', (1, 1)),
    ('overlap', 'max', (9, 9)),
    ('overlap', 'median', (5, 5)),
    ('overlap', 'p50', (5, 5)),
    ('overlap', 'p99.5', (8.96, 8.96)),
    # test multiple statistics
    ('overlap', ('min', 'max'), ((1, 1), (9, 9))),
    ('full', ('min', 'max'), ((1, 1, 1), (9, 9, 5))),
)


@pytest.mark.parametrize('frequency', FREQUENCY_OPTIONS)
@pytest.mark.parametrize('span, statistics, expected', VALIDATION_DATA_SUCCESS)
def test_multimodel_statistics(frequency, span, statistics, expected):
    """High level test for multicube statistics function.

    - Should work for multiple data frequencies
    - Should be able to deal with multiple statistics
    - Should work for both span arguments
    - Should deal correctly with different mask options
    - Return type should be a dict with all requested statistics as keys
    """
    cubes = get_cubes_for_validation_test(frequency)

    if isinstance(statistics, str):
        statistics = (statistics, )
        expected = (expected, )

    result = multi_model_statistics(cubes, span, statistics)

    assert isinstance(result, dict)
    assert set(result.keys()) == set(statistics)

    for i, statistic in enumerate(statistics):
        result_cube = result[statistic]
        expected_data = np.ma.array(expected[i], mask=False)
        assert_array_allclose(result_cube.data, expected_data)


@pytest.mark.parametrize('calendar1, calendar2, expected', (
    ('360_day', '360_day', '360_day'),
    ('365_day', '365_day', '365_day'),
    ('365_day', '360_day', 'gregorian'),
    ('360_day', '365_day', 'gregorian'),
    ('gregorian', '365_day', 'gregorian'),
    ('proleptic_gregorian', 'julian', 'gregorian'),
    ('julian', '365_day', 'gregorian'),
))
def test_get_consistent_time_unit(calendar1, calendar2, expected):
    """Test same calendar returned or default if calendars differ.

    Expected behaviour: If the calendars are the same, return that one.
    If the calendars are not the same, return 'gregorian'.
    """
    cubes = (
        generate_cube_from_dates('monthly', calendar=calendar1),
        generate_cube_from_dates('monthly', calendar=calendar2),
    )

    result = mm._get_consistent_time_unit(cubes)
    assert result.calendar == expected


@pytest.mark.parametrize('span', SPAN_OPTIONS)
def test_edge_case_different_time_offsets(span):
    cubes = (
        generate_cube_from_dates('monthly',
                                 '360_day',
                                 offset='days since 1888-01-01'),
        generate_cube_from_dates('monthly',
                                 '360_day',
                                 offset='days since 1899-01-01'),
    )

    statistic = 'min'
    statistics = (statistic, )

    result = multi_model_statistics(cubes, span, statistics)

    result_cube = result[statistic]

    time_coord = result_cube.coord('time')

    assert time_coord.units.calendar == 'gregorian'
    assert time_coord.units.origin == 'days since 1850-01-01'

    desired = np.array((14., 45., 73.))
    np.testing.assert_array_equal(time_coord.points, desired)

    # input cubes are updated in-place
    for cube in cubes:
        np.testing.assert_array_equal(cube.coord('time').points, desired)


def generate_cubes_with_non_overlapping_timecoords():
    """Generate sample data where time coords do not overlap."""
    time_points = range(1, 4)
    dates1 = [datetime(1850, i, 15, 0, 0, 0) for i in time_points]
    dates2 = [datetime(1950, i, 15, 0, 0, 0) for i in time_points]

    return (
        generate_cube_from_dates(dates1),
        generate_cube_from_dates(dates2),
    )


@pytest.mark.xfail(reason='Multimodel statistics returns the original cubes.')
def test_edge_case_time_no_overlap_fail():
    """Test case when time coords do not overlap using span='overlap'.

    Expected behaviour: `multi_model_statistics` should fail if time
    points are not overlapping.
    """
    cubes = generate_cubes_with_non_overlapping_timecoords()

    statistic = 'min'
    statistics = (statistic, )

    with pytest.raises(ValueError):
        _ = multi_model_statistics(cubes, 'overlap', statistics)


def test_edge_case_time_no_overlap_success():
    """Test case when time coords do not overlap using span='full'.

    Expected behaviour: `multi_model_statistics` should use all
    available time points.
    """
    cubes = generate_cubes_with_non_overlapping_timecoords()

    statistic = 'min'
    statistics = (statistic, )

    result = multi_model_statistics(cubes, 'full', statistics)
    result_cube = result[statistic]

    assert result_cube.coord('time').shape == (6, )


@pytest.mark.parametrize('span', SPAN_OPTIONS)
def test_edge_case_time_not_in_middle_of_months(span):
    """Test case when time coords are not on 15th for monthly data.

    Expected behaviour: `multi_model_statistics` will set all dates to
    the 15th.
    """
    time_points = range(1, 4)
    dates1 = [datetime(1850, i, 12, 0, 0, 0) for i in time_points]
    dates2 = [datetime(1850, i, 25, 0, 0, 0) for i in time_points]

    cubes = (
        generate_cube_from_dates(dates1),
        generate_cube_from_dates(dates2),
    )

    statistic = 'min'
    statistics = (statistic, )

    result = multi_model_statistics(cubes, span, statistics)
    result_cube = result[statistic]

    time_coord = result_cube.coord('time')

    desired = np.array((14., 45., 73.))
    np.testing.assert_array_equal(time_coord.points, desired)

    # input cubes are updated in-place
    for cube in cubes:
        np.testing.assert_array_equal(cube.coord('time').points, desired)


@pytest.mark.parametrize('span', SPAN_OPTIONS)
def test_edge_case_sub_daily_data_fail(span):
    """Test case when cubes with sub-daily time coords are passed."""
    cube = generate_cube_from_dates('hourly')
    cubes = (cube, cube)

    statistic = 'min'
    statistics = (statistic, )

    with pytest.raises(ValueError):
        _ = multi_model_statistics(cubes, span, statistics)


def test_unify_time_coordinates():
    """Test set common calendar."""
    cube1 = generate_cube_from_dates('monthly',
                                     calendar='360_day',
                                     offset='days since 1850-01-01')
    cube2 = generate_cube_from_dates('monthly',
                                     calendar='gregorian',
                                     offset='days since 1943-05-16')

    mm._unify_time_coordinates([cube1, cube2])

    assert cube1.coord('time') == cube2.coord('time')


class PreprocessorFile:
    """Mockup to test output of multimodel."""
    def __init__(self, cube=None):
        if cube:
            self.cubes = [cube]

    def wasderivedfrom(self, product):
        pass


def test_return_products():
    """Check that the right product set is returned."""
    cube1 = generate_cube_from_dates('monthly', fill_val=1)
    cube2 = generate_cube_from_dates('monthly', fill_val=9)

    input1 = PreprocessorFile(cube1)
    input2 = PreprocessorFile(cube2)

    products = set([input1, input2])

    output = PreprocessorFile()
    output_products = {'mean': output}

    kwargs = {
        'statistics': ['mean'],
        'span': 'full',
        'output_products': output_products
    }

    result1 = mm._multiproduct_statistics(products,
                                          keep_input_datasets=True,
                                          **kwargs)
    result2 = mm._multiproduct_statistics(products,
                                          keep_input_datasets=False,
                                          **kwargs)

    assert result1 == set([input1, input2, output])
    assert result2 == set([output])

    result3 = mm.multi_model_statistics(products, **kwargs)
    result4 = mm.multi_model_statistics(products,
                                        keep_input_datasets=False,
                                        **kwargs)

    assert result3 == result1
    assert result4 == result2
