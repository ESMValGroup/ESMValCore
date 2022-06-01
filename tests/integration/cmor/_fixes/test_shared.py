"""Tests for shared functions for fixes."""
import iris
import iris.coords
import iris.cube
import numpy as np
import pytest
from cf_units import Unit
from iris import NameConstraint

from esmvalcore.cmor._fixes.shared import (
    add_altitude_from_plev,
    add_aux_coords_from_cubes,
    add_plev_from_altitude,
    add_scalar_depth_coord,
    add_scalar_height_coord,
    add_scalar_lambda550nm_coord,
    add_scalar_typeland_coord,
    add_scalar_typesea_coord,
    add_scalar_typesi_coord,
    cube_to_aux_coord,
    fix_bounds,
    fix_ocean_depth_coord,
    get_altitude_to_pressure_func,
    get_bounds_cube,
    get_pressure_to_altitude_func,
    round_coordinates,
)


@pytest.mark.sequential
def test_altitude_to_pressure_func():
    """Test altitude to pressure function."""
    func = get_altitude_to_pressure_func()
    assert callable(func)
    np.testing.assert_allclose(func(-6000.0), 196968.01058487315)
    np.testing.assert_allclose(func(-5000.0), 177687.0)
    np.testing.assert_allclose(func(0.0), 101325.0)
    np.testing.assert_allclose(func(50.0), 100725.54298598564)
    np.testing.assert_allclose(func(80000.0), 0.88628)
    np.testing.assert_allclose(func(90000.0), 0.1576523580997673)
    np.testing.assert_allclose(func(np.array([0.0, 100.0])),
                               [101325.0, 100129.0])


@pytest.mark.sequential
def test_pressure_to_altitude_func():
    """Test pressure to altitude function."""
    func = get_pressure_to_altitude_func()
    assert callable(func)
    np.testing.assert_allclose(func(200000.0), -6166.332306480035)
    np.testing.assert_allclose(func(177687.0), -5000.0)
    np.testing.assert_allclose(func(101325.0), 0.0, atol=1.0e-7)
    np.testing.assert_allclose(func(1000.0), 31054.63120206961)
    np.testing.assert_allclose(func(75.9448), 50000)
    np.testing.assert_allclose(func(0.1), 91607.36011892557)
    np.testing.assert_allclose(func(np.array([101325.0, 177687.0])),
                               [0.0, -5000.0], atol=1.0e-7)


TEST_ADD_AUX_COORDS_FROM_CUBES = [
    ({}, 1),
    ({'x': ()}, 0),
    ({'x': 1, 'a': ()}, 0),
    ({'a': ()}, 1),
    ({'a': (), 'b': 1}, 1),
    ({'a': (), 'b': 1}, 1),
    ({'c': 1}, 2),
    ({'a': (), 'b': 1, 'c': 1}, 2),
    ({'d': (0, 1)}, 1),
    ({'a': (), 'b': 1, 'd': (0, 1)}, 1),
]


@pytest.mark.sequential
@pytest.mark.parametrize('coord_dict,output', TEST_ADD_AUX_COORDS_FROM_CUBES)
def test_add_aux_coords_from_cubes(coord_dict, output):
    """Test extraction of auxiliary coordinates from cubes."""
    cube = iris.cube.Cube([[0.0]])
    cubes = iris.cube.CubeList([
        iris.cube.Cube(0.0, var_name='a'),
        iris.cube.Cube([0.0], var_name='b'),
        iris.cube.Cube([0.0], var_name='c'),
        iris.cube.Cube([0.0], var_name='c'),
        iris.cube.Cube([[0.0]], var_name='d'),
    ])
    if output == 1:
        add_aux_coords_from_cubes(cube, cubes, coord_dict)
        for (coord_name, coord_dims) in coord_dict.items():
            coord = cube.coord(var_name=coord_name)
            if len(cube.coord_dims(coord)) == 1:
                assert cube.coord_dims(coord)[0] == coord_dims
            else:
                assert cube.coord_dims(coord) == coord_dims
            points = np.full(coord.shape, 0.0)
            assert coord.points == points
            assert not cubes.extract(NameConstraint(var_name=coord_name))
        assert len(cubes) == 5 - len(coord_dict)
        return
    with pytest.raises(ValueError) as err:
        add_aux_coords_from_cubes(cube, cubes, coord_dict)
    if output == 0:
        assert "Expected exactly one coordinate cube 'x'" in str(err.value)
        assert "got 0" in str(err.value)
    else:
        assert "Expected exactly one coordinate cube 'c'" in str(err.value)
        assert "got 2" in str(err.value)


ALT_COORD = iris.coords.AuxCoord([0.0], bounds=[[-100.0, 500.0]],
                                 standard_name='altitude', units='m')
ALT_COORD_NB = iris.coords.AuxCoord([0.0], standard_name='altitude', units='m')
ALT_COORD_KM = iris.coords.AuxCoord([0.0], bounds=[[-0.1, 0.5]],
                                    var_name='alt', long_name='altitude',
                                    standard_name='altitude', units='km')
P_COORD = iris.coords.AuxCoord([101325.0], bounds=[[102532.0, 95460.8]],
                               standard_name='air_pressure', units='Pa')
P_COORD_NB = iris.coords.AuxCoord([101325.0], standard_name='air_pressure',
                                  units='Pa')
CUBE_ALT = iris.cube.Cube([1.0], var_name='x',
                          aux_coords_and_dims=[(ALT_COORD, 0)])
CUBE_ALT_NB = iris.cube.Cube([1.0], var_name='x',
                             aux_coords_and_dims=[(ALT_COORD_NB, 0)])
CUBE_ALT_KM = iris.cube.Cube([1.0], var_name='x',
                             aux_coords_and_dims=[(ALT_COORD_KM, 0)])


TEST_ADD_PLEV_FROM_ALTITUDE = [
    (CUBE_ALT.copy(), P_COORD.copy()),
    (CUBE_ALT_NB.copy(), P_COORD_NB.copy()),
    (CUBE_ALT_KM.copy(), P_COORD.copy()),
    (iris.cube.Cube(0.0), None),
]


@pytest.mark.sequential
@pytest.mark.parametrize('cube,output', TEST_ADD_PLEV_FROM_ALTITUDE)
def test_add_plev_from_altitude(cube, output):
    """Test adding of pressure level coordinate."""
    if output is None:
        with pytest.raises(ValueError) as err:
            add_plev_from_altitude(cube)
        msg = ("Cannot add 'air_pressure' coordinate, 'altitude' coordinate "
               "not available")
        assert str(err.value) == msg
        return
    assert not cube.coords('air_pressure')
    add_plev_from_altitude(cube)
    air_pressure_coord = cube.coord('air_pressure')
    assert air_pressure_coord == output
    assert cube.coords('altitude')


P_COORD_HPA = iris.coords.AuxCoord([1013.25], bounds=[[1025.32, 954.60]],
                                   var_name='plev',
                                   standard_name='air_pressure',
                                   long_name='pressure', units='hPa')
CUBE_PLEV = iris.cube.Cube([1.0], var_name='x',
                           aux_coords_and_dims=[(P_COORD, 0)])
CUBE_PLEV_NB = iris.cube.Cube([1.0], var_name='x',
                              aux_coords_and_dims=[(P_COORD_NB, 0)])
CUBE_PLEV_HPA = iris.cube.Cube([1.0], var_name='x',
                               aux_coords_and_dims=[(P_COORD_HPA, 0)])


TEST_ADD_ALTITUDE_FROM_PLEV = [
    (CUBE_PLEV.copy(), ALT_COORD.copy()),
    (CUBE_PLEV_NB.copy(), ALT_COORD_NB.copy()),
    (CUBE_PLEV_HPA.copy(), ALT_COORD.copy()),
    (iris.cube.Cube(0.0), None),
]


@pytest.mark.sequential
@pytest.mark.parametrize('cube,output', TEST_ADD_ALTITUDE_FROM_PLEV)
def test_add_altitude_from_plev(cube, output):
    """Test adding of altitude coordinate."""
    if output is None:
        with pytest.raises(ValueError) as err:
            add_altitude_from_plev(cube)
        msg = ("Cannot add 'altitude' coordinate, 'air_pressure' coordinate "
               "not available")
        assert str(err.value) == msg
        return
    assert not cube.coords('altitude')
    add_altitude_from_plev(cube)
    altitude_coord = cube.coord('altitude')
    metadata_list = [
        'var_name',
        'standard_name',
        'long_name',
        'units',
        'attributes',
    ]
    for attr in metadata_list:
        assert getattr(altitude_coord, attr) == getattr(output, attr)
    np.testing.assert_allclose(altitude_coord.points, output.points, atol=1e-7)
    if output.bounds is None:
        assert altitude_coord.bounds is None
    else:
        np.testing.assert_allclose(altitude_coord.bounds, output.bounds,
                                   rtol=1e-3)
    assert cube.coords('air_pressure')


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
TEST_ADD_SCALAR_COORD_NO_VALS = [CUBE_1.copy(), CUBE_2.copy()]


@pytest.mark.sequential
@pytest.mark.parametrize('cube_in,depth', TEST_ADD_SCALAR_COORD)
def test_add_scalar_depth_coord(cube_in, depth):
    """Test adding of scalar depth coordinate."""
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


@pytest.mark.sequential
@pytest.mark.parametrize('cube_in,height', TEST_ADD_SCALAR_COORD)
def test_add_scalar_height_coord(cube_in, height):
    """Test adding of scalar height coordinate."""
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


@pytest.mark.sequential
@pytest.mark.parametrize('cube_in', TEST_ADD_SCALAR_COORD_NO_VALS)
def test_add_scalar_lambda550nm_coord(cube_in):
    """Test adding of scalar lambda550nm coordinate."""
    cube_in = cube_in.copy()
    lambda550nm_coord = iris.coords.AuxCoord(
        550.0,
        var_name='wavelength',
        standard_name='radiation_wavelength',
        long_name='Radiation Wavelength 550 nanometers',
        units='nm',
    )
    with pytest.raises(iris.exceptions.CoordinateNotFoundError):
        cube_in.coord('radiation_wavelength')
    cube_out = add_scalar_lambda550nm_coord(cube_in)
    assert cube_out is cube_in
    coord = cube_in.coord('radiation_wavelength')
    assert coord == lambda550nm_coord
    cube_out_2 = add_scalar_lambda550nm_coord(cube_out)
    assert cube_out_2 is cube_out
    coord = cube_in.coord('radiation_wavelength')
    assert coord == lambda550nm_coord


@pytest.mark.sequential
@pytest.mark.parametrize('cube_in,typeland', TEST_ADD_SCALAR_COORD)
def test_add_scalar_typeland_coord(cube_in, typeland):
    """Test adding of scalar typeland coordinate."""
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


@pytest.mark.sequential
@pytest.mark.parametrize('cube_in,typesea', TEST_ADD_SCALAR_COORD)
def test_add_scalar_typesea_coord(cube_in, typesea):
    """Test adding of scalar typesea coordinate."""
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


@pytest.mark.sequential
@pytest.mark.parametrize('cube_in,typesi', TEST_ADD_SCALAR_COORD)
def test_add_scalar_typesi_coord(cube_in, typesi):
    """Test adding of scalar typesi coordinate."""
    cube_in = cube_in.copy()
    if typesi is None:
        typesi = 'sea_ice'
    typesi_coord = iris.coords.AuxCoord(typesi,
                                        var_name='type',
                                        standard_name='area_type',
                                        long_name='Sea Ice area type',
                                        units=Unit('no unit'))
    with pytest.raises(iris.exceptions.CoordinateNotFoundError):
        cube_in.coord('area_type')
    if typesi == 'sea_ice':
        cube_out = add_scalar_typesi_coord(cube_in)
    else:
        cube_out = add_scalar_typesi_coord(cube_in, typesi)
    assert cube_out is cube_in
    coord = cube_in.coord('area_type')
    assert coord == typesi_coord
    cube_out_2 = add_scalar_typesi_coord(cube_out)
    assert cube_out_2 is cube_out
    coord = cube_in.coord('area_type')
    assert coord == typesi_coord


@pytest.mark.sequential
def test_cube_to_aux_coord():
    """Test converting cube to auxiliary coordinate."""
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


TEST_GET_BOUNDS_CUBE = [
    ('x', None),
    ('a', iris.cube.Cube(0.0, var_name='a_bnds')),
    ('b', iris.cube.Cube([0.0], var_name='b_bounds')),
    ('c', False),
    ('d', iris.cube.Cube([[0.0]], var_name='d_bnds')),
    ('e', False),
]


@pytest.mark.sequential
@pytest.mark.parametrize('coord_name,output', TEST_GET_BOUNDS_CUBE)
def test_get_bounds_cube(coord_name, output):
    """Test retrieving of bounds cube from list of cubes."""
    cubes = iris.cube.CubeList([
        iris.cube.Cube(0.0, var_name='a_bnds'),
        iris.cube.Cube([0.0], var_name='b_bounds'),
        iris.cube.Cube([0.0], var_name='c_bnds'),
        iris.cube.Cube([0.0], var_name='c_bnds'),
        iris.cube.Cube([[0.0]], var_name='d_bnds'),
        iris.cube.Cube([[0.0]], var_name='d_bounds'),
        iris.cube.Cube([[0.0]], var_name='e_bounds'),
        iris.cube.Cube([[0.0]], var_name='e_bounds'),
    ])
    if output is None:
        with pytest.raises(ValueError) as err:
            get_bounds_cube(cubes, coord_name)
        msg = "No bounds for coordinate variable 'x' available in"
        assert msg in str(err.value)
        return
    if not isinstance(output, iris.cube.Cube):
        with pytest.raises(ValueError) as err:
            get_bounds_cube(cubes, coord_name)
        msg = f"Multiple cubes with var_name '{coord_name}"
        assert msg in str(err.value)
        return
    bounds_cube = get_bounds_cube(cubes, coord_name)
    assert bounds_cube == output


TEST_FIX_BOUNDS = [
    ([], [None, [[-3.0, 4.0]]]),
    (['a'], [[[1.0, 2.0]], [[-3.0, 4.0]]]),
    (['b'], [None, [[-3.0, 4.0]]]),
    (['a', 'b'], [[[1.0, 2.0]], [[-3.0, 4.0]]]),
]


@pytest.mark.sequential
@pytest.mark.parametrize('var_names,output', TEST_FIX_BOUNDS)
def test_fix_bounds(var_names, output):
    """Test retrieving of bounds cube from list of cubes."""
    a_coord = iris.coords.AuxCoord(1.5, var_name='a')
    b_coord = iris.coords.AuxCoord(1.5, bounds=[-3.0, 4.0], var_name='b')
    cube = iris.cube.Cube(
        0.0,
        aux_coords_and_dims=[(a_coord, ()), (b_coord, ())],
        var_name='x',
    )
    cubes = iris.cube.CubeList([
        iris.cube.Cube([1.0, 2.0], var_name='a_bnds'),
        iris.cube.Cube([1.0, 2.0], var_name='b_bounds'),
        iris.cube.Cube([1000.0, 2000.0], var_name='c_bounds'),
    ])
    assert cube.coord(var_name='a').bounds is None
    fix_bounds(cube, cubes, var_names)
    if output[0] is None:
        assert cube.coord(var_name='a').bounds is None
    else:
        np.testing.assert_allclose(cube.coord(var_name='a').bounds, output[0])
    np.testing.assert_allclose(cube.coord(var_name='b').bounds, output[1])


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


@pytest.mark.sequential
@pytest.mark.parametrize('cubes_in,decimals,out', TEST_ROUND)
def test_round_coordinate(cubes_in, decimals, out):
    """Test rounding of coordinates."""
    kwargs = {} if decimals is None else {'decimals': decimals}
    cubes_out = round_coordinates(cubes_in, **kwargs)
    assert cubes_out is cubes_in
    for (idx, cube) in enumerate(cubes_out):
        coords = cube.coords(dim_coords=True)
        if out[idx] is None:
            assert not coords
        else:
            assert coords[0] == out[idx]


@pytest.mark.sequential
def test_round_coordinates_single_coord():
    """Test rounding of specified coordinate."""
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


def test_fix_ocean_depth_coord():
    """Test `fix_ocean_depth_coord`."""
    z_coord = iris.coords.DimCoord(0.0, var_name='alt',
                                   attributes={'positive': 'up'})
    cube = iris.cube.Cube([0.0], var_name='x',
                          dim_coords_and_dims=[(z_coord, 0)])
    fix_ocean_depth_coord(cube)
    depth_coord = cube.coord('depth')
    assert depth_coord.standard_name == 'depth'
    assert depth_coord.var_name == 'lev'
    assert depth_coord.units == 'm'
    assert depth_coord.long_name == 'ocean depth coordinate'
    assert depth_coord.attributes == {'positive': 'down'}
