"""Tests for :mod:`esmvalcore.iris_helpers`."""
import iris
import pytest

from esmvalcore.iris_helpers import var_name_constraint


@pytest.fixture
def cubes():
    """Test cubes."""
    cubes = iris.cube.CubeList([
        iris.cube.Cube(0.0, var_name='a', long_name='a'),
        iris.cube.Cube(0.0, var_name='a', long_name='b'),
        iris.cube.Cube(0.0, var_name='c', long_name='d'),
    ])
    return cubes


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
        cubes.extract_strict(var_name_constraint('a'))
    with pytest.raises(iris.exceptions.ConstraintMismatchError):
        cubes.extract_strict(var_name_constraint('b'))
    out_cube = cubes.extract_strict(var_name_constraint('c'))
    assert out_cube == iris.cube.Cube(0.0, var_name='c', long_name='d')
