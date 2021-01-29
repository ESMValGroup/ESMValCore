import pytest
import iris
from cf_units import Unit
import numpy as np

# from esmvalcore.preprocessor._multimodel import multi_model_statistics
import esmvalcore.preprocessor._multimodel as mm

SPAN_OPTIONS = ('overlap', 'full')

STATISTICS_OPTIONS = ('mean', 'std', 'std_dev', 'min', 'max', 'median')

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
        'std_dev':
        [5.656854249492381, 5.656854249492381, 0],
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


@pytest.fixture
def cubes_time():
    """Set up cubes used for testing multimodel statistics"""
    cube1 = iris.cube.Cube([1, 1],
                           dim_coords_and_dims=[(timecoord([1, 2]), 0)])
    cube2 = iris.cube.Cube([9, 9, 9],
                           dim_coords_and_dims=[(timecoord([1, 2, 3]), 0)])
    return ([cube1, cube2])


@pytest.mark.parametrize('span', SPAN_OPTIONS)
@pytest.mark.parametrize('stats', STATISTICS_OPTIONS)
def test_mean(cubes_time, span, stats):
    '''overlap between cube 1 and 2'''
    result = mm.multi_model_statistics([cubes_time[0], cubes_time[1]],
                                       span=span,
                                       statistics=[stats])
    result = result[stats]
    expected = np.array(EXPECTED[span][stats])
    assert np.all(result.data == expected.data)
