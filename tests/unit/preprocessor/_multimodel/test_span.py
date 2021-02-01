"""tests for multimodel preprocessor."""

import iris
import numpy as np
import pytest
from cf_units import Unit

# from esmvalcore.preprocessor._multimodel import multi_model_statistics
import esmvalcore.preprocessor._multimodel as mm

SPAN_OPTIONS = ('overlap', 'full')

FREQUENCY_OPTIONS = ('daily', 'monthly')

STATISTICS_OPTIONS = ('mean', 'std', 'std_dev', 'min', 'max', 'median',
                      'count', 'p50', 'p99.5')

EXPECTED = {
    'overlap': {
        'mean': [5, 5],
        'std': [5.656854249492381, 5.656854249492381],
        'std_dev': [5.656854249492381, 5.656854249492381],
        'min': [1, 1],
        'max': [9, 9],
        'median': [5, 5]
    },
    'full': {
        'mean': [5, 5, 9],
        'std': [5.656854249492381, 5.656854249492381, 0],
        'std_dev': [5.656854249492381, 5.656854249492381, 0],
        'min': [1, 1, 9],
        'max': [9, 9, 9],
        'median': [5, 5, 9]
    }
}


def timecoord(days=[1, 2], calendar='gregorian'):
    """Return a standard time coordinate with the given days as time points."""
    return iris.coords.DimCoord(days,
                                standard_name='time',
                                units=Unit('days since 1850-01-01',
                                           calendar=calendar))


def cubes(frequency):
    """Set up cubes used for testing multimodel statistics."""
    if frequency == 'daily':
        points1 = [1, 2]
        points2 = [1, 2, 3]
    elif frequency == 'monthly':
        points1 = [14, 45]
        points2 = [14, 45, 74]
    cube1 = iris.cube.Cube([1, 1],
                           dim_coords_and_dims=[(timecoord(points1), 0)])
    cube2 = iris.cube.Cube([9, 9, 9],
                           dim_coords_and_dims=[(timecoord(points2), 0)])
    return cube1, cube2


@pytest.mark.parametrize('frequency', FREQUENCY_OPTIONS)
@pytest.mark.parametrize('span', SPAN_OPTIONS)
# @pytest.mark.parametrize('stats', STATISTICS_OPTIONS)
def test_mean(span, frequency):
    """overlap between cube 1 and 2."""
    cube1, cube2 = cubes(frequency)

    result = mm.multi_model_statistics([cube1, cube2],
                                       span=span,
                                       statistics=['mean'])
    result = result['mean']
    expected = np.array(EXPECTED[span]['mean'])
    assert np.all(result.data == expected.data)
