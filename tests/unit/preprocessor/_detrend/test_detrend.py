"""Unit tests for the :func:`esmvalcore.preprocessor._detrend` module."""

import unittest

import iris
import iris.coords
from iris.cube import Cube
import numpy as np
import pytest
from cf_units import Unit

from numpy.testing import assert_array_almost_equal

from esmvalcore.preprocessor._detrend import detrend


def _create_sample_cube():
    cube = Cube(
        np.array((np.arange(1, 25), np.arange(25, 49))),
        var_name='co2',
        units='J'
    )
    cube.add_dim_coord(
        iris.coords.DimCoord(
            np.arange(15., 720., 30.),
            standard_name='time',
            units=Unit('days since 1950-01-01 00:00:00', calendar='gregorian'),
        ),
        1,
    )
    cube.add_dim_coord(
        iris.coords.DimCoord(
            np.arange(1, 3),
            standard_name='latitude',
        ),
        0,
    )
    return cube


@pytest.mark.parametrize('method', ['linear', 'constant'])
def test_decadal_average(method):
    """Test for decadal average."""
    cube = _create_sample_cube()

    result = detrend(cube, 'time', method)
    if method == 'linear':
        expected = np.zeros([2, 24])
    else:
        expected = np.array(
            (np.arange(1, 25) - 12.5, np.arange(25, 49) - 36.5)
        )
    assert_array_almost_equal(result.data, expected)


if __name__ == '__main__':
    unittest.main()
