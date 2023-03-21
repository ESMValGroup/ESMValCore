"""Tests for general CORDEX fixes."""
import cordex as cx
import iris
import numpy as np
import pytest
from cf_units import Unit
from iris.fileformats.pp import EARTH_RADIUS

from esmvalcore.cmor._fixes.cordex.cordex_fixes import (
    AllVars,
    CLMcomCCLM4817,
    MOHCHadREM3GA705,
    TimeLongName,
)
from esmvalcore.cmor.table import CMOR_TABLES
from esmvalcore.exceptions import RecipeError

SPAN = (5650 * 1000)


@pytest.fixture
def cubes():
    correct_time_coord = iris.coords.DimCoord([0.0],
                                              var_name='time',
                                              standard_name='time',
                                              long_name='time')
    wrong_time_coord = iris.coords.DimCoord([0.0],
                                            var_name='time',
                                            standard_name='time',
                                            long_name='wrong')
    correct_lat_coord = iris.coords.DimCoord([0.0, 1.0],
                                             var_name='lat',
                                             standard_name='latitude',
                                             long_name='latitude')
    wrong_lat_coord = iris.coords.DimCoord([0.0, 1.0],
                                           var_name='latitudeCoord',
                                           standard_name='latitude',
                                           long_name='latitude')
    correct_lon_coord = iris.coords.DimCoord([0.0],
                                             var_name='lon',
                                             standard_name='longitude',
                                             long_name='longitude')
    wrong_lon_coord = iris.coords.DimCoord([0.0],
                                           var_name='longitudeCoord',
                                           standard_name='longitude',
                                           long_name='longitude')
    correct_cube = iris.cube.Cube(
        [[[10.0], [10.0]]],
        var_name='tas',
        dim_coords_and_dims=[(correct_time_coord, 0), (correct_lat_coord, 1),
                             (correct_lon_coord, 2)],
    )
    wrong_cube = iris.cube.Cube(
        [[[10.0], [10.0]]],
        var_name='tas',
        dim_coords_and_dims=[(wrong_time_coord, 0), (wrong_lat_coord, 1),
                             (wrong_lon_coord, 2)],
    )
    return iris.cube.CubeList([correct_cube, wrong_cube])


@pytest.fixture
def cordex_cubes():
    coord_system = iris.coord_systems.RotatedGeogCS(
        grid_north_pole_latitude=39.25,
        grid_north_pole_longitude=-162,
    )
    time = iris.coords.DimCoord(np.arange(0, 3),
                                var_name='time',
                                standard_name='time')

    rlat = iris.coords.DimCoord(
        np.arange(0, 412),
        var_name='rlat',
        standard_name='grid_latitude',
        coord_system=coord_system,
    )
    rlon = iris.coords.DimCoord(
        np.arange(0, 424),
        var_name='rlon',
        standard_name='grid_longitude',
        coord_system=coord_system,
    )
    lat = iris.coords.AuxCoord(np.ones((412, 424)),
                               var_name='lat',
                               standard_name='latitude')
    lon = iris.coords.AuxCoord(np.ones((412, 424)),
                               var_name='lon',
                               standard_name='longitude')

    cube = iris.cube.Cube(np.ones((3, 412, 424)),
                          var_name='tas',
                          dim_coords_and_dims=[(time, 0), (rlat, 1),
                                               (rlon, 2)],
                          aux_coords_and_dims=[(lat, (1, 2)), (lon, (1, 2))])
    return iris.cube.CubeList([cube])


@pytest.fixture
def cordex_lambert_cubes():
    coord_system = iris.coord_systems.LambertConformal(
        central_lat=49.5,
        central_lon=10.5,
        false_easting=0.0,
        false_northing=0.0,
        secant_latitudes=(49.5, ))
    geog_system = iris.coord_systems.GeogCS(EARTH_RADIUS).as_cartopy_crs()
    fix = AllVars(vardef=None,
                  extra_facets={
                      'domain': 'EUR-11',
                      'dataset': 'DATASET',
                      'driver': 'DRIVER'
                  })

    time = iris.coords.DimCoord(np.arange(0, 3),
                                var_name='time',
                                standard_name='time')

    ax_pts = np.linspace(-SPAN // 2, SPAN // 2, 453)
    proj_x = iris.coords.DimCoord(ax_pts,
                                  var_name='x',
                                  standard_name='projection_x_coordinate',
                                  coord_system=coord_system,
                                  units='m')
    proj_y = iris.coords.DimCoord(ax_pts,
                                  var_name='y',
                                  standard_name='projection_y_coordinate',
                                  coord_system=coord_system,
                                  units='m')

    gx, gy = np.meshgrid(ax_pts, ax_pts)
    lonlat = fix._transform_points(crs_from=coord_system.as_cartopy_crs(),
                                   crs_to=geog_system,
                                   x_data=gx,
                                   y_data=gy)

    lon = iris.coords.AuxCoord(lonlat[..., 0],
                               var_name='lon',
                               standard_name='longitude')
    lat = iris.coords.AuxCoord(lonlat[..., 1],
                               var_name='lat',
                               standard_name='latitude')

    cube = iris.cube.Cube(np.ones((3, 453, 453)),
                          var_name='tas',
                          dim_coords_and_dims=[(time, 0), (proj_x, 1),
                                               (proj_y, 2)],
                          aux_coords_and_dims=[(lat, (1, 2)), (lon, (1, 2))])
    return iris.cube.CubeList([cube])


@pytest.mark.parametrize('coord, var_name, long_name', [
    ('time', 'time', 'time'),
    ('latitude', 'lat', 'latitude'),
    ('longitude', 'lon', 'longitude'),
])
def test_mohchadrem3ga705_fix_metadata(cubes, coord, var_name, long_name):
    fix = MOHCHadREM3GA705(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        assert cube.coord(standard_name=coord).var_name == var_name
        assert cube.coord(standard_name=coord).long_name == long_name


def test_timelongname_fix_metadata(cubes):
    fix = TimeLongName(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        assert cube.coord('time').long_name == 'time'


def test_clmcomcclm4817_fix_metadata(cubes):
    cubes[0].coord('time').units = Unit('days since 1850-1-1 00:00:00',
                                        calendar='proleptic_gregorian')
    cubes[1].coord('time').units = Unit('days since 1850-1-1 00:00:00',
                                        calendar='standard')
    for coord in cubes[1].coords():
        coord.points = coord.core_points().astype('>f8', casting='same_kind')
    lat = cubes[1].coord('latitude')
    lat.guess_bounds()
    lat.bounds = lat.core_bounds().astype('>f4', casting='same_kind')

    fix = CLMcomCCLM4817(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        assert cube.coord('time').units == Unit('days since 1850-1-1 00:00:00',
                                                calendar='proleptic_gregorian')
        for coord in cube.coords():
            assert coord.points.dtype == np.float64


def test_rotated_grid_fix(cordex_cubes):
    fix = AllVars(vardef=None,
                  extra_facets={
                      'domain': 'EUR-11',
                      'dataset': 'DATASET',
                      'driver': 'DRIVER'
                  })
    domain = cx.cordex_domain('EUR-11', add_vertices=True)
    for cube in cordex_cubes:
        for coord in ['rlat', 'rlon', 'lat', 'lon']:
            cube_coord = cube.coord(var_name=coord)
            cube_coord.points = domain[coord].data + 1e-6
    out_cubes = fix.fix_metadata(cordex_cubes)
    assert cordex_cubes is out_cubes
    for out_cube in out_cubes:
        for coord in ['rlat', 'rlon', 'lat', 'lon']:
            cube_coord = out_cube.coord(var_name=coord)
            domain_coord = domain[coord].data
            np.testing.assert_array_equal(cube_coord.points, domain_coord)


def test_rotated_grid_fix_error(cordex_cubes):
    fix = AllVars(vardef=None,
                  extra_facets={
                      'domain': 'EUR-11',
                      'dataset': 'DATASET',
                      'driver': 'DRIVER'
                  })
    msg = ("Differences between the original grid and the "
           "standardised grid are above 10e-4 degrees.")
    with pytest.raises(RecipeError) as exc:
        fix.fix_metadata(cordex_cubes)
    assert msg == exc.value.message


def test_lambert_grid_fix(cordex_lambert_cubes):
    cmor_table = CMOR_TABLES["CORDEX"]
    mip = "mon"
    short_name = "tas"
    vardef = cmor_table.get_variable(mip, short_name)
    fix = AllVars(vardef=vardef,
                  extra_facets={
                      'domain': 'EUR-11',
                      'dataset': 'DATASET',
                      'driver': 'DRIVER'
                  })

    # Mess up proj coords.
    cube = cordex_lambert_cubes[0]
    proj_x = cube.coord(var_name="x")
    proj_y = cube.coord(var_name="y")
    proj_x = proj_x.copy(proj_x.points + SPAN // 2)
    proj_y = proj_x.copy(proj_x.points + SPAN // 2)
    cube.replace_coord(proj_x)
    cube.replace_coord(proj_y)

    out_cubes = fix.fix_metadata(cordex_lambert_cubes)
    assert cordex_lambert_cubes is out_cubes
    for out_cube, input_cube in zip(out_cubes, cordex_lambert_cubes):
        for coord in ['x', 'y', 'lat', 'lon']:
            np.testing.assert_array_equal(
                out_cube.coord(var_name=coord).points,
                input_cube.coord(var_name=coord).points)


def test_lambert_grid_fix_error(cordex_lambert_cubes):
    cmor_table = CMOR_TABLES["CORDEX"]
    mip = "mon"
    short_name = "tas"
    vardef = cmor_table.get_variable(mip, short_name)
    fix = AllVars(vardef=vardef,
                  extra_facets={
                      'domain': 'EUR-11',
                      'dataset': 'DATASET',
                      'driver': 'DRIVER'
                  })
    # Mess up both.
    cube = cordex_lambert_cubes[0]

    proj_x = cube.coord(var_name="x")
    proj_y = cube.coord(var_name="y")
    proj_x = proj_x.copy(proj_x.points + SPAN // 2)
    proj_y = proj_x.copy(proj_x.points + SPAN // 2)
    cube.replace_coord(proj_x)
    cube.replace_coord(proj_y)

    lon = cube.coord(var_name="lon")
    lat = cube.coord(var_name="lat")
    lon = lon.copy(lon.points * 2)
    lat = lat.copy(lat.points * 2)
    cube.replace_coord(lon)
    cube.replace_coord(lat)

    msg = ('Differences between standard geographical domain '
           'and the domain of the cube are above 15%')
    with pytest.raises(RecipeError) as exc:
        fix.fix_metadata(cordex_lambert_cubes)
    assert msg == exc.value.message


def test_wrong_coord_system(cubes):
    fix = AllVars(vardef=None,
                  extra_facets={
                      'domain': 'EUR-11',
                      'dataset': 'DATASET',
                      'driver': 'DRIVER'
                  })
    for cube in cubes:
        cube.coord_system = iris.coord_systems.AlbersEqualArea
    msg = ("Coordinate system albers_conical_equal_area not supported in "
           "CORDEX datasets. Must be rotated_latitude_longitude "
           "or lambert_conformal_conic.")
    with pytest.raises(RecipeError) as exc:
        fix.fix_metadata(cubes)
    assert msg == exc.value.message
