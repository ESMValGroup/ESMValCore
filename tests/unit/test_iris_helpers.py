"""Tests for :mod:`esmvalcore.iris_helpers`."""
import datetime

import iris
import numpy as np
import pytest
from cf_units import Unit

from esmvalcore.iris_helpers import date2num, var_name_constraint


@pytest.fixture
def cubes():
    """Test cubes."""
    cubes = iris.cube.CubeList([
        iris.cube.Cube(0.0, var_name='a', long_name='a'),
        iris.cube.Cube(0.0, var_name='a', long_name='b'),
        iris.cube.Cube(0.0, var_name='c', long_name='d'),
    ])
    return cubes


@pytest.fixture
def units():
    return Unit('days since 0001-01-01', calendar='proleptic_gregorian')


@pytest.mark.parametrize("date, dtype, expected", [
    (datetime.datetime(1, 1, 1), np.float64, 0.0),
    (datetime.datetime(1, 1, 1), int, 0.0),
    (datetime.datetime(1, 1, 2, 12), np.float64, 1.5),
])
def test_date2num_scalar(date, dtype, expected, units):
    num = date2num(date, units, dtype=dtype)
    assert num == expected
    assert num.dtype == dtype


def test_var_name_constraint(cubes):
    """Test :func:`esmvalcore.iris_helpers.var_name_constraint`."""
    out_cubes = cubes.extract(var_name_constraint('a'))
    assert out_cubes == iris.cube.CubeList([
        iris.cube.Cube(0.0, var_name='a', long_name='a'),
        iris.cube.Cube(0.0, var_name='a', long_name='b'),
    ])
    out_cubes = cubes.extract(var_name_constraint('b'))
    assert out_cubes == iris.cube.CubeList([])
    out_cubes = cubes.extract(var_name_constraint('c'))
    assert out_cubes == iris.cube.CubeList([
        iris.cube.Cube(0.0, var_name='c', long_name='d'),
    ])
    with pytest.raises(iris.exceptions.ConstraintMismatchError):
        cubes.extract_cube(var_name_constraint('a'))
    with pytest.raises(iris.exceptions.ConstraintMismatchError):
        cubes.extract_cube(var_name_constraint('b'))
    out_cube = cubes.extract_cube(var_name_constraint('c'))
    assert out_cube == iris.cube.Cube(0.0, var_name='c', long_name='d')
