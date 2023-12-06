"""Tests for :mod:`esmvalcore.iris_helpers`."""
import datetime
from copy import deepcopy
from itertools import permutations
from unittest import mock

import numpy as np
import pytest
from cf_units import Unit
from iris.coords import (
    AncillaryVariable,
    AuxCoord,
    CellMeasure,
    CellMethod,
    DimCoord,
)
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateMultiDimError

from esmvalcore.iris_helpers import (
    add_leading_dim_to_cube,
    date2num,
    merge_cube_attributes,
)


@pytest.fixture
def cubes():
    """Test cubes."""
    cubes = CubeList([
        Cube(0.0, var_name='a', long_name='a'),
        Cube(0.0, var_name='a', long_name='b'),
        Cube(0.0, var_name='c', long_name='d'),
    ])
    return cubes


@pytest.fixture
def units():
    return Unit('days since 0001-01-01', calendar='proleptic_gregorian')


def test_add_leading_dim_to_cube():
    """Test :func:`esmvalcore.iris_helpers.add_leading_dim_to_cube`."""
    lat_coord = DimCoord(
        [0.0, 1.0, 2.0],
        var_name='lat',
        standard_name='latitude',
        long_name='latitude',
        units='degrees_north',
    )
    lon_coord = DimCoord(
        [0.0, 1.0],
        var_name='lon',
        standard_name='longitude',
        long_name='longitude',
        units='degrees_east',
    )
    height_coord = AuxCoord(
        [2.0, 1.0],
        var_name='height',
        standard_name='height',
        long_name='height',
        units='m',
        attributes={'positive': 'up'},
    )
    land_mask = AncillaryVariable(
        [0.5, 0.2],
        var_name='sftlf',
        standard_name=None,
        long_name='Land fraction',
        units='1',
    )
    cell_area = CellMeasure(
        [1.0, 2.0],
        var_name='areacella',
        standard_name='cell_area',
        long_name='Cell Area',
        units='m2',
        measure='area',
    )
    cube = Cube(
        [1, 42],
        var_name='ta',
        standard_name='air_temperature',
        long_name='Air Temperature',
        units='K',
        attributes={'model_name': 'ESM'},
        cell_methods=[CellMethod('mean', coords='time')],
        aux_coords_and_dims=[(height_coord, 0)],
        dim_coords_and_dims=[(lon_coord, 0)],
        ancillary_variables_and_dims=[(land_mask, 0)],
        cell_measures_and_dims=[(cell_area, 0)],
    )

    new_cube = add_leading_dim_to_cube(cube, lat_coord)

    np.testing.assert_equal(new_cube.data, [[1, 42], [1, 42], [1, 42]])

    assert new_cube.var_name == 'ta'
    assert new_cube.standard_name == 'air_temperature'
    assert new_cube.long_name == 'Air Temperature'
    assert new_cube.units == 'K'
    assert new_cube.cell_methods == (CellMethod('mean', coords='time'),)
    assert new_cube.attributes == {'model_name': 'ESM'}

    assert new_cube.coords(lat_coord, dim_coords=True)
    assert new_cube.coords(lon_coord, dim_coords=True)
    assert new_cube.coords(height_coord, dim_coords=False)
    assert new_cube.coord_dims(lat_coord) == (0,)
    assert new_cube.coord_dims(lon_coord) == (1,)
    assert new_cube.coord_dims(height_coord) == (1,)

    assert new_cube.ancillary_variables(land_mask)
    assert new_cube.cell_measures(cell_area)
    assert new_cube.ancillary_variable_dims(land_mask) == (1,)
    assert new_cube.cell_measure_dims(cell_area) == (1,)


def test_add_leading_dim_to_cube_non_1d():
    """Test :func:`esmvalcore.iris_helpers.add_leading_dim_to_cube`."""
    coord_2d = AuxCoord([[0, 1], [2, 3]], var_name='coord_2d')
    msg = "Multi-dimensional coordinate not supported: 'coord_2d'"
    with pytest.raises(CoordinateMultiDimError, match=msg):
        add_leading_dim_to_cube(mock.sentinel.cube, coord_2d)


@pytest.mark.parametrize("date, dtype, expected", [
    (datetime.datetime(1, 1, 1), np.float64, 0.0),
    (datetime.datetime(1, 1, 1), int, 0.0),
    (datetime.datetime(1, 1, 2, 12), np.float64, 1.5),
])
def test_date2num_scalar(date, dtype, expected, units):
    num = date2num(date, units, dtype=dtype)
    assert num == expected
    assert num.dtype == dtype


def assert_attribues_equal(attrs_1: dict, attrs_2: dict) -> None:
    """Check attributes using :func:`numpy.testing.assert_array_equal`."""
    assert len(attrs_1) == len(attrs_2)
    for (attr, val) in attrs_1.items():
        assert attr in attrs_2
        np.testing.assert_array_equal(attrs_2[attr], val)


def make_cube_with_attrs(index):
    """Make cube that contains different types of attributes."""
    attributes = {
        # Identical attribute values across cubes
        'int': 42,
        'float': 3.1415,
        'bool': True,
        'str': 'Hello, world',
        'list': [1, 1, 2, 3, 5, 8, 13],
        'tuple': (1, 2, 3, 4, 5),
        'nparray': np.arange(42),

        # Differing attribute values across cubes
        'diff_int': index,
        'diff_str': 'abc'[index],
        'diff_nparray': np.arange(index),
        'mix': np.arange(3) if index == 0 else index,
        'diff_list': [index, index],
        'diff_tuple': (index, index),

        # Differing attribute keys across cubes
        str(index + 1000): index,
        str(index % 2 + 100): index,
        str(index % 2): index % 2,
    }
    return Cube(0.0, attributes=attributes)


CUBES = [make_cube_with_attrs(i) for i in range(3)]


# Test all permutations of CUBES to test that results do not depend on order
@pytest.mark.parametrize("cubes", list(permutations(CUBES)))
def test_merge_cube_attributes(cubes):
    """Test `merge_cube_attributes`."""
    expected_attributes = {
        'int': 42,
        'float': 3.1415,
        'bool': True,
        'str': 'Hello, world',
        'list': [1, 1, 2, 3, 5, 8, 13],
        'tuple': (1, 2, 3, 4, 5),
        'nparray': np.arange(42),
        'diff_int': '0 1 2',
        'diff_str': 'a b c',
        'diff_nparray': '[0 1] [0] []',
        'mix': '1 2 [0 1 2]',
        'diff_list': '[0, 0] [1, 1] [2, 2]',
        'diff_tuple': '(0, 0) (1, 1) (2, 2)',
        '1000': 0,
        '1001': 1,
        '1002': 2,
        '100': '0 2',
        '101': 1,
        '0': 0,
        '1': 1,
    }
    cubes = deepcopy(cubes)
    merge_cube_attributes(cubes)
    assert len(cubes) == 3
    for cube in cubes:
        assert_attribues_equal(cube.attributes, expected_attributes)


def test_merge_cube_attributes_0_cubes():
    """Test `merge_cube_attributes` with 0 cubes."""
    merge_cube_attributes([])


def test_merge_cube_attributes_1_cube():
    """Test `merge_cube_attributes` with 1 cube."""
    cubes = CubeList([deepcopy(CUBES[0])])
    expected_attributes = deepcopy(cubes[0].attributes)
    merge_cube_attributes(cubes)
    assert len(cubes) == 1
    assert_attribues_equal(cubes[0].attributes, expected_attributes)
