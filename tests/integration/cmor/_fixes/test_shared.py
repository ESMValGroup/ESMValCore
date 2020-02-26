"""Tests for shared functions for fixes."""
import iris
import pytest
from cf_units import Unit

from esmvalcore.cmor._fixes.shared import (add_scalar_depth_coord,
                                           add_scalar_height_coord,
                                           add_scalar_typeland_coord,
                                           add_scalar_typesea_coord,
                                           round_coordinates)

DIM_COORD = iris.coords.DimCoord([3.141592],
                                 bounds=[[1.23, 4.567891011]],
                                 standard_name='latitude')
CUBE_1 = iris.cube.Cube([1.0], standard_name='air_temperature')
CUBE_2 = iris.cube.Cube([3.0], dim_coords_and_dims=[(DIM_COORD, 0)])
TEST_ADD_SCALAR_COORD = [
    (CUBE_1.copy(), None),
    (CUBE_1.copy(), -5.0),
    (CUBE_2.copy(), None),
    (CUBE_2.copy(), 100.0),
]


@pytest.mark.parametrize('cube_in,depth', TEST_ADD_SCALAR_COORD)
def test_add_scalar_depth_coord(cube_in, depth):
    cube_in = cube_in.copy()
    if depth is None:
        depth = 0.0
    depth_coord = iris.coords.AuxCoord(depth,
                                       var_name='depth',
                                       standard_name='depth',
                                       long_name='depth',
                                       units=Unit('m'),
                                       attributes={'positive': 'down'})
    with pytest.raises(iris.exceptions.CoordinateNotFoundError):
        cube_in.coord('depth')
    if depth == 0.0:
        cube_out = add_scalar_depth_coord(cube_in)
    else:
        cube_out = add_scalar_depth_coord(cube_in, depth)
    assert cube_out is cube_in
    coord = cube_in.coord('depth')
    assert coord == depth_coord


@pytest.mark.parametrize('cube_in,height', TEST_ADD_SCALAR_COORD)
def test_add_scalar_height_coord(cube_in, height):
    cube_in = cube_in.copy()
    if height is None:
        height = 2.0
    height_coord = iris.coords.AuxCoord(height,
                                        var_name='height',
                                        standard_name='height',
                                        long_name='height',
                                        units=Unit('m'),
                                        attributes={'positive': 'up'})
    with pytest.raises(iris.exceptions.CoordinateNotFoundError):
        cube_in.coord('height')
    if height == 2.0:
        cube_out = add_scalar_height_coord(cube_in)
    else:
        cube_out = add_scalar_height_coord(cube_in, height)
    assert cube_out is cube_in
    coord = cube_in.coord('height')
    assert coord == height_coord


@pytest.mark.parametrize('cube_in,typeland', TEST_ADD_SCALAR_COORD)
def test_add_scalar_typeland_coord(cube_in, typeland):
    cube_in = cube_in.copy()
    if typeland is None:
        typeland = 'default'
    typeland_coord = iris.coords.AuxCoord(typeland,
                                          var_name='type',
                                          standard_name='area_type',
                                          long_name='Land area type',
                                          units=Unit('no unit'))
    with pytest.raises(iris.exceptions.CoordinateNotFoundError):
        cube_in.coord('area_type')
    if typeland == 'default':
        cube_out = add_scalar_typeland_coord(cube_in)
    else:
        cube_out = add_scalar_typeland_coord(cube_in, typeland)
    assert cube_out is cube_in
    coord = cube_in.coord('area_type')
    assert coord == typeland_coord


@pytest.mark.parametrize('cube_in,typesea', TEST_ADD_SCALAR_COORD)
def test_add_scalar_typesea_coord(cube_in, typesea):
    cube_in = cube_in.copy()
    if typesea is None:
        typesea = 'default'
    typesea_coord = iris.coords.AuxCoord(typesea,
                                         var_name='type',
                                         standard_name='area_type',
                                         long_name='Ocean area type',
                                         units=Unit('no unit'))
    with pytest.raises(iris.exceptions.CoordinateNotFoundError):
        cube_in.coord('area_type')
    if typesea == 'default':
        cube_out = add_scalar_typesea_coord(cube_in)
    else:
        cube_out = add_scalar_typesea_coord(cube_in, typesea)
    assert cube_out is cube_in
    coord = cube_in.coord('area_type')
    assert coord == typesea_coord


DIM_COORD_NB = iris.coords.DimCoord([3.1415], standard_name='latitude')
CUBE_3 = iris.cube.Cube([5.0], dim_coords_and_dims=[(DIM_COORD_NB, 0)])
COORD_3_DEC = DIM_COORD.copy([3.142], [[1.23, 4.568]])
COORD_5_DEC = DIM_COORD.copy([3.14159], [[1.23, 4.56789]])
COORD_NB_3_DEC = DIM_COORD_NB.copy([3.142])
TEST_ROUND = [
    (iris.cube.CubeList([CUBE_2]), None, [COORD_5_DEC]),
    (iris.cube.CubeList([CUBE_3]), None, [DIM_COORD_NB]),
    (iris.cube.CubeList([CUBE_1, CUBE_2]), 3, [None, COORD_3_DEC]),
    (iris.cube.CubeList([CUBE_2, CUBE_2]), 3, [COORD_3_DEC, COORD_3_DEC]),
    (iris.cube.CubeList([CUBE_2, CUBE_3]), 3, [COORD_3_DEC, COORD_NB_3_DEC]),
]


@pytest.mark.parametrize('cubes_in,decimals,out', TEST_ROUND)
def test_round_coordinate(cubes_in, decimals, out):
    kwargs = {} if decimals is None else {'decimals': decimals}
    cubes_out = round_coordinates(cubes_in, **kwargs)
    assert cubes_out is cubes_in
    for (idx, cube) in enumerate(cubes_out):
        coords = cube.coords(dim_coords=True)
        if out[idx] is None:
            assert not coords
        else:
            assert coords[0] == out[idx]
