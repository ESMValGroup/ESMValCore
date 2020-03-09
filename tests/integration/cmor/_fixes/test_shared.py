"""Tests for shared functions for fixes."""
import numpy as np
import iris
import pytest
from cf_units import Unit

from esmvalcore.cmor._fixes.shared import (add_scalar_depth_coord,
                                           add_scalar_height_coord,
                                           add_scalar_typeland_coord,
                                           add_scalar_typesea_coord,
                                           add_sigma_factory,
                                           round_coordinates,
                                           cube_to_aux_coord)


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
    cube_out_2 = add_scalar_depth_coord(cube_out)
    assert cube_out_2 is cube_out
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
    cube_out_2 = add_scalar_height_coord(cube_out)
    assert cube_out_2 is cube_out
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
    cube_out_2 = add_scalar_typeland_coord(cube_out)
    assert cube_out_2 is cube_out
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
    cube_out_2 = add_scalar_typesea_coord(cube_out)
    assert cube_out_2 is cube_out
    coord = cube_in.coord('area_type')
    assert coord == typesea_coord


PS_COORD = iris.coords.AuxCoord([[[101000.0]]], var_name='ps', units='Pa')
PTOP_COORD = iris.coords.AuxCoord(1000.0, var_name='ptop', units='Pa')
LEV_COORD = iris.coords.AuxCoord([0.5], bounds=[[0.2, 0.8]], var_name='lev',
                                 units='1',
                                 standard_name='atmosphere_sigma_coordinate')
P_COORD_HYBRID = iris.coords.AuxCoord([[[[51000.0]]]],
                                      bounds=[[[[[21000.0, 81000.0]]]]],
                                      standard_name='air_pressure', units='Pa')
CUBE_HYBRID = iris.cube.Cube([[[[1.0]]]], var_name='x',
                             aux_coords_and_dims=[(PS_COORD, (0, 2, 3)),
                                                  (PTOP_COORD, ()),
                                                  (LEV_COORD, 1)])


TEST_ADD_SIGMA_FACTORY = [
    (CUBE_HYBRID.copy(), P_COORD_HYBRID.copy()),
    (iris.cube.Cube(0.0), None),
]


@pytest.mark.parametrize('cube,output', TEST_ADD_SIGMA_FACTORY)
def test_add_sigma_factory(cube, output):
    """Test adding of factory for ``atmosphere_sigma_coordinate``."""
    if output is None:
        with pytest.raises(ValueError) as err:
            add_sigma_factory(cube)
        msg = ("Cannot add 'air_pressure' coordinate, "
               "'atmosphere_sigma_coordinate' coordinate not available")
        assert str(err.value) == msg
        return
    assert not cube.coords('air_pressure')
    add_sigma_factory(cube)
    air_pressure_coord = cube.coord('air_pressure')
    assert air_pressure_coord == output


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


def test_cube_to_aux_coord():
    cube = iris.cube.Cube(
        np.ones((2, 2)),
        standard_name='longitude',
        long_name='longitude',
        var_name='lon',
        units='degrees_north',
    )
    coord = cube_to_aux_coord(cube)
    assert coord.var_name == cube.var_name
    assert coord.standard_name == cube.standard_name
    assert coord.long_name == cube.long_name
    assert coord.units == cube.units
    assert np.all(coord.points == cube.data)


def test_round_coordinates_single_coord():
    """Test rounding of specified coordinate"""
    coords, bounds = [10.0001], [[9.0001, 11.0001]]
    latcoord = iris.coords.DimCoord(coords.copy(), bounds=bounds.copy(),
                                    standard_name='latitude')
    loncoord = iris.coords.DimCoord(coords.copy(), bounds=bounds.copy(),
                                    standard_name='longitude')
    cube = iris.cube.Cube([[1.0]], standard_name='air_temperature',
                          dim_coords_and_dims=[(latcoord, 0), (loncoord, 1)])
    cubes = iris.cube.CubeList([cube])

    out = round_coordinates(cubes, decimals=3, coord_names=['latitude'])
    assert out is cubes
    assert cubes[0].coord('longitude') is out[0].coord('longitude')
    np.testing.assert_allclose(out[0].coord('latitude').points, [10])
    np.testing.assert_allclose(out[0].coord('latitude').bounds, [[9, 11]])
