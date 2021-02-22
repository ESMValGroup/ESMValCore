"""Unit test for :func:`esmvalcore.preprocessor._multimodel`

test_align(cubes, span)
    --> pass multiple cubes with different time coords
    --> check that the returned cubes have consistent shapes and calendars
    --> check that if a cube is extended, the extended points are masked (not NaN!)

test_combine(cubes, dim='new_dim')
    --> pass multiple combinations of cubes
        - if cubes have the same shape, check that they are combined along a new dimension
        - if they have inconsistent shapes, check that iris raises an error
        - if they have inconsistent variable names, they should not be combined

test_compute(cube, statistic, dim='new_dim')
    --> make one big cube with a dimension called 'new dim'
        - call with multiple different statistics
        - check that the resulting data (computed statistics) is correct
        - check that the output has a correct variable name
        - check that the 'new_dim' dimension is removed again
        - what happens if some of the input data is masked or NaN?
        - test with COUNT statistics whether masked points are treated as expected.
"""

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


def timecoord(frequency, calendar='gregorian', offset='days since 1850-01-01'):
    """Return a time coordinate with the given time points and calendar."""
    if frequency == 'hourly':
        dates = [datetime(1850, 1, 1, i, 0, 0) for i in [1, 2, 3]]
    if frequency == 'daily':
        dates = [datetime(1850, 1, i, 0, 0, 0) for i in [1, 2, 3]]
    elif frequency == 'monthly':
        dates = [datetime(1850, i, 15, 0, 0, 0) for i in [1, 2, 3]]
    elif frequency == 'yearly':
        dates = [datetime(1850, 7, i, 0, 0, 0) for i in [1, 2, 3]]

    unit = Unit(offset, calendar=calendar)
    points = unit.date2num(dates)
    return iris.coords.DimCoord(points, standard_name='time', units=unit)


def get_cubes(frequency):
    """Set up cubes used for testing multimodel statistics."""

    # Simple 1d cube with standard time cord
    time = timecoord(frequency)
    cube1 = Cube([1, 1, 1], dim_coords_and_dims=[(time, 0)])

    # Cube with masked data
    cube2 = cube1.copy()
    cube2.data = np.ma.array([5, 5, 5], mask=[True, False, False])

    # Cube with deviating time coord
    time = timecoord(frequency,
                     calendar='360_day',
                     offset='days since 1950-01-01')[:2]
    cube3 = Cube([9, 9], dim_coords_and_dims=[(time, 0)])
    return [cube1, cube2, cube3]


@pytest.mark.parametrize('frequency', FREQUENCY_OPTIONS)
@pytest.mark.parametrize('span', SPAN_OPTIONS)
# @pytest.mark.parametrize('stats', STATISTICS_OPTIONS)
def test_multimodel_statistics(span, frequency):
    """High level test for multicube statistics function.

    - Should work for multiple data frequencies
    - Should be able to deal with multiple statistics
    - Should work for both span arguments
    - Should deal correctly with different mask options
    - Return type should be a dict with all requested statistics as keys
    """
    cubes = get_cubes(frequency)
    verification_data = {
        # For span=overlap, take the first 2 items.
        # Span = full --> statistic computed on [1, 1, 1], [-, 5, 5], [9, 9, -]
        # Span = overlap --> statistic computed on [1, 1], [-, 5], [9, 9]
        'mean': [5, 5, 3],
        'std': [5.656854249492381, 4, 2.8284271247461903],
        'std_dev': [5.656854249492381, 4, 2.8284271247461903],
        'min': [1, 1, 1],
        'max': [9, 9, 5],
        'median': [5, 5, 3],
        'p50': [5, 5, 3],
        'p99.5': [8.96, 8.96, 4.98],
    }

    statistics = verification_data.keys()
    results = multi_model_statistics(cubes, span, statistics)

    assert isinstance(results, dict)
    assert results.keys() == statistics

    for statistic, result in results.items():
        expected = np.ma.array(verification_data[statistic], mask=False)
        if span == 'overlap':
            expected = expected[:2]
        np.testing.assert_array_equal(result.data.mask, expected.mask)
        np.testing.assert_array_almost_equal(result.data, expected.data)


def test_get_consistent_time_unit():
    """Test same calendar returned or default if calendars differ."""

    time1 = timecoord('monthly', '360_day')
    cube1 = Cube([1, 1, 1], dim_coords_and_dims=[(time1, 0)])
    time2 = timecoord('monthly', '365_day')
    cube2 = Cube([
        1,
        1,
        1,
    ], dim_coords_and_dims=[(time2, 0)])

    result1 = mm._get_consistent_time_unit([cube1, cube1])
    result2 = mm._get_consistent_time_unit([cube1, cube2])
    assert result1.calendar == '360_day'
    assert result2.calendar == 'gregorian'


def test_unify_time_coordinates():
    """Test whether the time coordinates are made consistent."""

    # # Check that monthly data have midpoints at 15th day
    # cube1 = Cube([1, 1, 1], )

    # hourly = {
    #     'input1': timecoord([datetime(1850, 1, 1, i, 0, 0) for i in [1, 2, 3]],
    #               calendar='standard'),
    #     'input2' timecoord([datetime(1850, 1, 1, i, 0, 0) for i in [1, 2, 3]]),
    #               calendar='gregorian'),
    #     'output': timecoord([datetime(1850, 1, 1, i, 0, 0) for i in [1, 2, 3]],
    #               calendar='gregorian')
    # }

    # daily = ([datetime(1850, 1, i, 0, 0, 0) for i in [1, 2, 3]],
    #          [datetime(1850, 1, i, 12, 0, 0) for i in [1, 2, 3]])
    # monthly = ([datetime(1850, i, 1, 0, 0, 0) for i in [1, 2, 3]],
    #            [datetime(1850, i, 15, 0, 0, 0) for i in [1, 2, 3]])
    # yearly = ([datetime(1850+i, 1, 7, 0, 0, 0) for i in [1, 2, 3]],
    #           [datetime(1850+i, 1, 1, 0, 0, 0) for i in [1, 2, 3]])

    # time_sets = [hourly, daily, monthly, yearly]
    # calendars_sets = [
    #     ('standard', 'gregorian'),
    #     ('360_day', '360_day'),
    #     ('365_day', 'proleptic_gregorian'),
    #     ('standard', 'standard'),
    # ]

    # for (time1, time2), (calendar1, calendar2) in zip(time_sets, calendar_sets):
    #     cube1 = [Cube([1, 1, 1], dim_coords_and_dims=[(timecoord(time1, calendar1), 0)])
    #     cube2 = [Cube([1, 1, 1], dim_coords_and_dims=[(timecoord(time2, calendar2), 0)])
    #     cubes = mm._unify_time_coordinates([cube1, cube2])

    # --> pass multiple cubes with all kinds of different calendars
    #     - Check that output cubes all have the same calendar
    #     - check that the dates in the output correspond to the dates in the input
    #     - do this for different time frequencies
    #     - check warning/error for (sub)daily data


def test_resolve_span():
    """Check that resolve_span returns the correct union/intersection."""
    span1 = [1, 2, 3]
    span2 = [2, 3, 4]
    span3 = [3, 4, 5]
    span4 = [4, 5, 6]

    assert all(mm._resolve_span([span1, span2], span='overlap') == [2, 3])
    assert all(mm._resolve_span([span1, span2], span='full') == [1, 2, 3, 4])

    assert all(mm._resolve_span([span1, span2, span3], span='overlap') == [3])
    assert all(
        mm._resolve_span([span1, span2, span3], span='full') ==
        [1, 2, 3, 4, 5])

    with pytest.raises(ValueError):
        mm._resolve_span([span1, span4], span='overlap')


# test edge cases

# different time offsets in calendar
# different calendars
# no overlap
# statistic without kwargs
# time points not in middle of months
# fail for sub-daily data
#
