"""tests for multimodel preprocessor."""

from datetime import datetime

import iris
import numpy as np
import pytest
from cf_units import Unit
from iris.cube import Cube

from esmvalcore.preprocessor import multi_model_statistics

# import esmvalcore.preprocessor._multimodel as mm

SPAN_OPTIONS = ('overlap', 'full')

FREQUENCY_OPTIONS = ('daily', 'monthly', 'yearly')  # hourly


def timecoord(dates, calendar='gregorian'):
    """Return a time coordinate with the given time points and calendar."""
    unit = Unit('days since 1850-01-01', calendar=calendar)
    points = unit.date2num(dates)
    return iris.coords.DimCoord(points, standard_name='time', units=unit)


# lons = iris.coords.DimCoord([0,],
#                             standard_name='longitude', units='degrees_east')
# lats = iris.coords.DimCoord([0,],
#                             standard_name='latitude', units='degrees_north')


def get_cubes(frequency):
    """Set up cubes used for testing multimodel statistics."""
    if frequency == 'hourly':
        dates = [datetime(1850, 1, 1, i, 0, 0) for i in range(1, 4)]
    if frequency == 'daily':
        dates = [datetime(1850, 1, i, 0, 0, 0) for i in range(1, 4)]
    elif frequency == 'monthly':
        dates = [datetime(1850, i, 15, 0, 0, 0) for i in range(1, 4)]
    elif frequency == 'yearly':
        dates = [datetime(1850, 7, i, 0, 0, 0) for i in range(1, 4)]

    cube1 = Cube([1, 1, 1], dim_coords_and_dims=[(timecoord(dates), 0)])
    cube2 = cube1.copy()
    cube2.data = np.ma.array([5, 5, 5], mask=[True, False, False])
    cube3 = Cube([9, 9], dim_coords_and_dims=[(timecoord(dates[:2]), 0)])
    return [cube1, cube2, cube3]


@pytest.mark.parametrize('frequency', FREQUENCY_OPTIONS)
@pytest.mark.parametrize('span', SPAN_OPTIONS)
# @pytest.mark.parametrize('stats', STATISTICS_OPTIONS)
def test_multimodel_statistics(span, frequency):
    """High level test for multicube statistics function."""
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
