"""Unit tests for the :func:`esmvalcore.preprocessor._other` module."""

import unittest
import pytest
import tests

import numpy as np
from numpy.testing import assert_array_equal

from cf_units import Unit
import iris
import iris.coord_categorisation
import iris.coords
from iris.cube import Cube
from tests import Test

from esmvalcore.preprocessor._other import (
    clip
)


class TestClip(unittest.TestCase):

    def test_clip(self):
        cube = Cube(np.array([-10,0,10]), var_name='co2', units='J')
        cube.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(3),
                standard_name='time',
                units=Unit('days since 1950-01-01 00:00:00', calendar='gregorian'),
            ),
            0,
        )
        assert_array_equal(clip(cube, 0, None).data,np.array([0,0,10]))
        assert_array_equal(clip(cube, 0, 2).data,np.array([0,0,2]))
        with self.assertRaises(ValueError):
            clip(cube, None, None)
        with self.assertRaises(ValueError):
            clip(cube, 10, 8)


if __name__ == '__main__':
    unittest.main()
