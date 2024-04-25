"""Tests for the ICON on-the-fly CMORizer."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from unittest import mock

import iris
import numpy as np
import pytest
from cf_units import Unit
from iris import NameConstraint
from iris.coords import AuxCoord, CellMethod, DimCoord
from iris.cube import Cube, CubeList

import esmvalcore.cmor._fixes.icon.icon
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor._fixes.icon._base_fixes import IconFix
from esmvalcore.cmor._fixes.icon.icon import (
    AllVars,
    Clwvi,
    Hfls,
    Hfss,
    Rtmt,
    Rtnt,
)
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import CoordinateInfo, get_var_info
from esmvalcore.config import CFG
from esmvalcore.config._config import get_extra_facets
from esmvalcore.dataset import Dataset

TEST_GRID_FILE_URI = (
    'https://github.com/ESMValGroup/ESMValCore/raw/main/tests/integration/'
    'cmor/_fixes/test_data/icon_grid.nc'
)
TEST_GRID_FILE_NAME = 'icon_grid.nc'


@pytest.fixture(autouse=True)
def tmp_cache_dir(monkeypatch, tmp_path):
    """Use temporary path as cache directory for all tests in this module."""
    monkeypatch.setattr(IconFix, 'CACHE_DIR', tmp_path)


# Note: test_data_path is defined in tests/integration/cmor/_fixes/conftest.py


@pytest.fixture
def cubes_2d(test_data_path):
    """2D sample cubes."""
    nc_path = test_data_path / 'icon_2d.nc'
    return iris.load(str(nc_path))


@pytest.fixture
def cubes_3d(test_data_path):
    """3D sample cubes."""
    nc_path = test_data_path / 'icon_3d.nc'
    return iris.load(str(nc_path))


@pytest.fixture
def cubes_grid(test_data_path):
    """Grid description sample cubes."""
    nc_path = test_data_path / 'icon_grid.nc'
    return iris.load(str(nc_path))


@pytest.fixture
def cubes_regular_grid():
    """Cube with regular grid."""
    time_coord = DimCoord([0], var_name='time', standard_name='time',
                          units='days since 1850-01-01')
    lat_coord = DimCoord([0.0, 1.0], var_name='lat', standard_name='latitude',
                         long_name='latitude', units='degrees_north')
    lon_coord = DimCoord([-1.0, 1.0], var_name='lon',
                         standard_name='longitude', long_name='longitude',
                         units='degrees_east')
    cube = Cube([[[0.0, 1.0], [2.0, 3.0]]], var_name='tas', units='K',
                dim_coords_and_dims=[(time_coord, 0),
                                     (lat_coord, 1),
                                     (lon_coord, 2)])
    return CubeList([cube])


@pytest.fixture
def cubes_2d_lat_lon_grid():
    """Cube with 2D latitude and longitude."""
    time_coord = DimCoord([0], var_name='time', standard_name='time',
                          units='days since 1850-01-01')
    lat_coord = AuxCoord([[0.0, 0.0], [1.0, 1.0]], var_name='lat',
                         standard_name='latitude', long_name='latitude',
                         units='degrees_north')
    lon_coord = AuxCoord([[0.0, 1.0], [0.0, 1.0]], var_name='lon',
                         standard_name='longitude', long_name='longitude',
                         units='degrees_east')
    cube = Cube([[[0.0, 1.0], [2.0, 3.0]]], var_name='tas', units='K',
                dim_coords_and_dims=[(time_coord, 0)],
                aux_coords_and_dims=[(lat_coord, (1, 2)),
                                     (lon_coord, (1, 2))])
    return CubeList([cube])


@pytest.fixture
def simple_unstructured_cube():
    """Simple cube with unstructured grid."""
    time_coord = DimCoord([0], var_name='time', standard_name='time',
                          units='days since 1850-01-01')
    height_coord = DimCoord([0, 1, 2], var_name='height')
    lat_coord = AuxCoord([0.0, 1.0], var_name='lat', standard_name='latitude',
                         long_name='latitude', units='degrees_north')
    lon_coord = AuxCoord([0.0, 1.0], var_name='lon',
                         standard_name='longitude', long_name='longitude',
                         units='degrees_east')
    cube = Cube([[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]], var_name='ta',
                units='K',
                dim_coords_and_dims=[(time_coord, 0), (height_coord, 1)],
                aux_coords_and_dims=[(lat_coord, 2), (lon_coord, 2)])
    return cube


def _get_fix(mip, short_name, fix_name, session=None):
    """Load a fix from esmvalcore.cmor._fixes.icon.icon."""
    dataset = Dataset(
        project='ICON',
        dataset='ICON',
        mip=mip,
        short_name=short_name,
    )
    extra_facets = get_extra_facets(dataset, ())
    extra_facets['frequency'] = 'mon'
    extra_facets['exp'] = 'amip'
    vardef = get_var_info(project='ICON', mip=mip, short_name=short_name)
    cls = getattr(esmvalcore.cmor._fixes.icon.icon, fix_name)
    fix = cls(vardef, extra_facets=extra_facets, session=session)
    return fix


def get_fix(mip, short_name, session=None):
    """Load a variable fix from esmvalcore.cmor._fixes.icon.icon."""
    fix_name = short_name[0].upper() + short_name[1:]
    return _get_fix(mip, short_name, fix_name, session=session)


def get_allvars_fix(mip, short_name, session=None):
    """Load the AllVars fix from esmvalcore.cmor._fixes.icon.icon."""
    return _get_fix(mip, short_name, 'AllVars', session=session)


def fix_metadata(cubes, mip, short_name, session=None):
    """Fix metadata of cubes."""
    fix = get_fix(mip, short_name, session=session)
    cubes = fix.fix_metadata(cubes)
    fix = get_allvars_fix(mip, short_name, session=session)
    cubes = fix.fix_metadata(cubes)
    return cubes


def fix_data(cube, mip, short_name, session=None):
    """Fix data of cube."""
    fix = get_fix(mip, short_name, session=session)
    cube = fix.fix_data(cube)
    fix = get_allvars_fix(mip, short_name, session=session)
    cube = fix.fix_data(cube)
    return cube


def check_ta_metadata(cubes):
    """Check ta metadata."""
    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == 'ta'
    assert cube.standard_name == 'air_temperature'
    assert cube.long_name == 'Air Temperature'
    assert cube.units == 'K'
    assert 'positive' not in cube.attributes
    return cube


def check_tas_metadata(cubes):
    """Check tas metadata."""
    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == 'tas'
    assert cube.standard_name == 'air_temperature'
    assert cube.long_name == 'Near-Surface Air Temperature'
    assert cube.units == 'K'
    assert 'positive' not in cube.attributes
    return cube


def check_siconc_metadata(cubes, var_name, long_name):
    """Check tas metadata."""
    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == var_name
    assert cube.standard_name == 'sea_ice_area_fraction'
    assert cube.long_name == long_name
    assert cube.units == '%'
    assert 'positive' not in cube.attributes
    return cube


def check_time(cube):
    """Check time coordinate of cube."""
    assert cube.coords('time', dim_coords=True)
    time = cube.coord('time', dim_coords=True)
    assert time.var_name == 'time'
    assert time.standard_name == 'time'
    assert time.long_name == 'time'
    assert time.units == Unit('days since 1850-01-01',
                              calendar='proleptic_gregorian')
    np.testing.assert_allclose(time.points, [54770.5])
    np.testing.assert_allclose(time.bounds, [[54755.0, 54786.0]])
    assert time.attributes == {}


def check_model_level_metadata(cube):
    """Check metadata of model_level coordinate."""
    assert cube.coords('model level number', dim_coords=True)
    height = cube.coord('model level number', dim_coords=True)
    assert height.var_name == 'model_level'
    assert height.standard_name is None
    assert height.long_name == 'model level number'
    assert height.units == 'no unit'
    assert height.attributes == {'positive': 'up'}
    return height


def check_air_pressure_metadata(cube):
    """Check metadata of air_pressure coordinate."""
    assert cube.coords('air_pressure', dim_coords=False)
    plev = cube.coord('air_pressure', dim_coords=False)
    assert plev.var_name == 'plev'
    assert plev.standard_name == 'air_pressure'
    assert plev.long_name == 'pressure'
    assert plev.units == 'Pa'
    assert plev.attributes == {'positive': 'down'}
    return plev


def check_height(cube, plev_has_bounds=True):
    """Check height coordinate of cube."""
    height = check_model_level_metadata(cube)
    np.testing.assert_array_equal(height.points, np.arange(47))
    assert height.bounds is None

    plev = check_air_pressure_metadata(cube)
    assert cube.coord_dims('air_pressure') == (0, 1, 2)

    np.testing.assert_allclose(
        plev.points[0, :4, 0],
        [100566.234, 99652.07, 97995.77, 95686.08],
    )
    if plev_has_bounds:
        np.testing.assert_allclose(
            plev.bounds[0, :4, 0],
            [[100825.04, 100308.09],
             [100308.09, 99000.336],
             [99000.336, 97001.42],
             [97001.42, 94388.59]],
        )
    else:
        assert plev.bounds is None


def check_heightxm(cube, height_value):
    """Check scalar heightxm coordinate of cube."""
    assert cube.coords('height')
    height = cube.coord('height')
    assert height.var_name == 'height'
    assert height.standard_name == 'height'
    assert height.long_name == 'height'
    assert height.units == 'm'
    assert height.attributes == {'positive': 'up'}
    np.testing.assert_allclose(height.points, [height_value])
    assert height.bounds is None


def check_lat(cube):
    """Check latitude coordinate of cube."""
    assert cube.coords('latitude', dim_coords=False)
    lat = cube.coord('latitude', dim_coords=False)
    assert lat.var_name == 'lat'
    assert lat.standard_name == 'latitude'
    assert lat.long_name == 'latitude'
    assert lat.units == 'degrees_north'
    assert lat.attributes == {}
    np.testing.assert_allclose(
        lat.points,
        [-45.0, -45.0, -45.0, -45.0, 45.0, 45.0, 45.0, 45.0],
        rtol=1e-5
    )
    np.testing.assert_allclose(
        lat.bounds,
        [
            [-90.0, 0.0, 0.0],
            [-90.0, 0.0, 0.0],
            [-90.0, 0.0, 0.0],
            [-90.0, 0.0, 0.0],
            [0.0, 0.0, 90.0],
            [0.0, 0.0, 90.0],
            [0.0, 0.0, 90.0],
            [0.0, 0.0, 90.0],
        ],
        rtol=1e-5
    )
    return lat


def check_lon(cube):
    """Check longitude coordinate of cube."""
    assert cube.coords('longitude', dim_coords=False)
    lon = cube.coord('longitude', dim_coords=False)
    assert lon.var_name == 'lon'
    assert lon.standard_name == 'longitude'
    assert lon.long_name == 'longitude'
    assert lon.units == 'degrees_east'
    assert lon.attributes == {}
    np.testing.assert_allclose(
        lon.points,
        [225.0, 315.0, 45.0, 135.0, 225.0, 315.0, 45.0, 135.0],
        rtol=1e-5
    )
    np.testing.assert_allclose(
        lon.bounds,
        [
            [0.0, 270.0, 180.0],
            [0.0, 0.0, 270.0],
            [0.0, 90.0, 0.0],
            [0.0, 180.0, 90.0],
            [180.0, 270.0, 0.0],
            [270.0, 0.0, 0.0],
            [0.0, 90.0, 0.0],
            [90.0, 180.0, 0.0],
        ],
        rtol=1e-5
    )
    return lon


def check_lat_lon(cube):
    """Check latitude, longitude and mesh of cube."""
    lat = check_lat(cube)
    lon = check_lon(cube)

    # Check that latitude and longitude are mesh coordinates
    assert cube.coords('latitude', mesh_coords=True)
    assert cube.coords('longitude', mesh_coords=True)

    # Check dimensional coordinate describing the mesh
    assert cube.coords('first spatial index for variables stored on an '
                       'unstructured grid', dim_coords=True)
    i_coord = cube.coord('first spatial index for variables stored on an '
                         'unstructured grid', dim_coords=True)
    assert i_coord.var_name == 'i'
    assert i_coord.standard_name is None
    assert i_coord.long_name == ('first spatial index for variables stored on '
                                 'an unstructured grid')
    assert i_coord.units == '1'
    np.testing.assert_allclose(i_coord.points, [0, 1, 2, 3, 4, 5, 6, 7])
    assert i_coord.bounds is None

    assert len(cube.coord_dims(lat)) == 1
    assert cube.coord_dims(lat) == cube.coord_dims(lon)
    assert cube.coord_dims(lat) == cube.coord_dims(i_coord)

    # Check the mesh itself
    assert cube.location == 'face'
    mesh = cube.mesh
    check_mesh(mesh)


def check_mesh(mesh):
    """Check the mesh."""
    assert mesh is not None
    assert mesh.var_name is None
    assert mesh.standard_name is None
    assert mesh.long_name is None
    assert mesh.units == 'unknown'
    assert mesh.attributes == {}
    assert mesh.cf_role == 'mesh_topology'
    assert mesh.topology_dimension == 2

    # Check face coordinates
    assert len(mesh.coords(include_faces=True)) == 2

    mesh_face_lat = mesh.coord(include_faces=True, axis='y')
    assert mesh_face_lat.var_name == 'lat'
    assert mesh_face_lat.standard_name == 'latitude'
    assert mesh_face_lat.long_name == 'latitude'
    assert mesh_face_lat.units == 'degrees_north'
    assert mesh_face_lat.attributes == {}
    np.testing.assert_allclose(
        mesh_face_lat.points,
        [-45.0, -45.0, -45.0, -45.0, 45.0, 45.0, 45.0, 45.0],
        rtol=1e-5
    )
    np.testing.assert_allclose(
        mesh_face_lat.bounds,
        [
            [-90.0, 0.0, 0.0],
            [-90.0, 0.0, 0.0],
            [-90.0, 0.0, 0.0],
            [-90.0, 0.0, 0.0],
            [0.0, 0.0, 90.0],
            [0.0, 0.0, 90.0],
            [0.0, 0.0, 90.0],
            [0.0, 0.0, 90.0],
        ],
        rtol=1e-5
    )

    mesh_face_lon = mesh.coord(include_faces=True, axis='x')
    assert mesh_face_lon.var_name == 'lon'
    assert mesh_face_lon.standard_name == 'longitude'
    assert mesh_face_lon.long_name == 'longitude'
    assert mesh_face_lon.units == 'degrees_east'
    assert mesh_face_lon.attributes == {}
    np.testing.assert_allclose(
        mesh_face_lon.points,
        [225.0, 315.0, 45.0, 135.0, 225.0, 315.0, 45.0, 135.0],
        rtol=1e-5
    )
    np.testing.assert_allclose(
        mesh_face_lon.bounds,
        [
            [0.0, 270.0, 180.0],
            [0.0, 0.0, 270.0],
            [0.0, 90.0, 0.0],
            [0.0, 180.0, 90.0],
            [180.0, 270.0, 0.0],
            [270.0, 0.0, 0.0],
            [0.0, 90.0, 0.0],
            [90.0, 180.0, 0.0],
        ],
        rtol=1e-5
    )

    # Check node coordinates
    assert len(mesh.coords(include_nodes=True)) == 2

    mesh_node_lat = mesh.coord(include_nodes=True, axis='y')
    assert mesh_node_lat.var_name == 'nlat'
    assert mesh_node_lat.standard_name == 'latitude'
    assert mesh_node_lat.long_name == 'node latitude'
    assert mesh_node_lat.units == 'degrees_north'
    assert mesh_node_lat.attributes == {}
    np.testing.assert_allclose(
        mesh_node_lat.points,
        [-90.0, 0.0, 0.0, 0.0, 0.0, 90.0],
        rtol=1e-5
    )
    assert mesh_node_lat.bounds is None

    mesh_node_lon = mesh.coord(include_nodes=True, axis='x')
    assert mesh_node_lon.var_name == 'nlon'
    assert mesh_node_lon.standard_name == 'longitude'
    assert mesh_node_lon.long_name == 'node longitude'
    assert mesh_node_lon.units == 'degrees_east'
    assert mesh_node_lon.attributes == {}
    np.testing.assert_allclose(
        mesh_node_lon.points,
        [0.0, 180.0, 270.0, 0.0, 90, 0.0],
        rtol=1e-5
    )
    assert mesh_node_lon.bounds is None

    # Check connectivity
    assert len(mesh.connectivities()) == 1
    conn = mesh.connectivity()
    assert conn.var_name is None
    assert conn.standard_name is None
    assert conn.long_name is None
    assert conn.units == 'unknown'
    assert conn.attributes == {}
    assert conn.cf_role == 'face_node_connectivity'
    assert conn.start_index == 1
    assert conn.location_axis == 0
    assert conn.shape == (8, 3)
    np.testing.assert_array_equal(
        conn.indices,
        [[1, 3, 2],
         [1, 4, 3],
         [1, 5, 4],
         [1, 2, 5],
         [2, 3, 6],
         [3, 4, 6],
         [4, 5, 6],
         [5, 2, 6]],
    )


def check_typesi(cube):
    """Check scalar typesi coordinate of cube."""
    assert cube.coords('area_type')
    typesi = cube.coord('area_type')
    assert typesi.var_name == 'type'
    assert typesi.standard_name == 'area_type'
    assert typesi.long_name == 'Sea Ice area type'
    assert typesi.units.is_no_unit()
    np.testing.assert_array_equal(typesi.points, ['sea_ice'])
    assert typesi.bounds is None


# Test areacella and areacello (for extra_facets, and grid_latitude and
# grid_longitude coordinates)


def test_get_areacella_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'fx', 'areacella')
    assert fix == [AllVars(None), GenericFix(None)]


def test_areacella_fix(cubes_grid):
    """Test fix."""
    fix = get_allvars_fix('fx', 'areacella')
    fixed_cubes = fix.fix_metadata(cubes_grid)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'areacella'
    assert cube.standard_name == 'cell_area'
    assert cube.long_name == 'Grid-Cell Area for Atmospheric Grid Variables'
    assert cube.units == 'm2'
    assert 'positive' not in cube.attributes

    check_lat_lon(cube)


def test_get_areacello_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'Ofx', 'areacello')
    assert fix == [AllVars(None), GenericFix(None)]


def test_areacello_fix(cubes_grid):
    """Test fix."""
    fix = get_allvars_fix('Ofx', 'areacello')
    fixed_cubes = fix.fix_metadata(cubes_grid)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'areacello'
    assert cube.standard_name == 'cell_area'
    assert cube.long_name == 'Grid-Cell Area for Ocean Variables'
    assert cube.units == 'm2'
    assert 'positive' not in cube.attributes

    check_lat_lon(cube)


# Test clwvi (for extra fix)


def test_get_clwvi_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'Amon', 'clwvi')
    assert fix == [Clwvi(None), AllVars(None), GenericFix(None)]


def test_clwvi_fix(cubes_regular_grid):
    """Test fix."""
    cubes = CubeList([
        cubes_regular_grid[0].copy(),
        cubes_regular_grid[0].copy()
    ])
    cubes[0].var_name = 'cllvi'
    cubes[1].var_name = 'clivi'
    cubes[0].units = '1e3 kg m-2'
    cubes[1].units = '1e3 kg m-2'

    fixed_cubes = fix_metadata(cubes, 'Amon', 'clwvi')

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'clwvi'
    assert cube.standard_name == ('atmosphere_mass_content_of_cloud_'
                                  'condensed_water')
    assert cube.long_name == 'Condensed Water Path'
    assert cube.units == 'kg m-2'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [[[0.0, 2000.0], [4000.0, 6000.0]]])


# Test lwp (for extra_facets)


def test_get_lwp_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'AERmon', 'lwp')
    assert fix == [AllVars(None), GenericFix(None)]


def test_lwp_fix(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('AERmon', 'lwp')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'lwp'
    assert cube.standard_name == ('atmosphere_mass_content_of_cloud_liquid_'
                                  'water')
    assert cube.long_name == 'Liquid Water Path'
    assert cube.units == 'kg m-2'
    assert 'positive' not in cube.attributes

    check_time(cube)
    check_lat_lon(cube)


# Test rsdt and rsut (for positive attribute)


def test_get_rsdt_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'Amon', 'rsdt')
    assert fix == [AllVars(None), GenericFix(None)]


def test_rsdt_fix(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'rsdt')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'rsdt'
    assert cube.standard_name == 'toa_incoming_shortwave_flux'
    assert cube.long_name == 'TOA Incident Shortwave Radiation'
    assert cube.units == 'W m-2'
    assert cube.attributes['positive'] == 'down'

    check_time(cube)
    check_lat_lon(cube)


def test_get_rsut_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'Amon', 'rsut')
    assert fix == [AllVars(None), GenericFix(None)]


def test_rsut_fix(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'rsut')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'rsut'
    assert cube.standard_name == 'toa_outgoing_shortwave_flux'
    assert cube.long_name == 'TOA Outgoing Shortwave Radiation'
    assert cube.units == 'W m-2'
    assert cube.attributes['positive'] == 'up'

    check_time(cube)
    check_lat_lon(cube)


# Test siconc and siconca (for extra_facets, extra fix and typesi coordinate)


def test_get_siconc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'SImon', 'siconc')
    assert fix == [AllVars(None), GenericFix(None)]


def test_siconc_fix(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('SImon', 'siconc')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    cube = check_siconc_metadata(fixed_cubes, 'siconc',
                                 'Sea-Ice Area Percentage (Ocean Grid)')
    check_time(cube)
    check_lat_lon(cube)
    check_typesi(cube)

    np.testing.assert_allclose(
        cube.data,
        [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]],
    )


def test_get_siconca_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'SImon', 'siconca')
    assert fix == [AllVars(None), GenericFix(None)]


def test_siconca_fix(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('SImon', 'siconca')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    cube = check_siconc_metadata(fixed_cubes, 'siconca',
                                 'Sea-Ice Area Percentage (Atmospheric Grid)')
    check_time(cube)
    check_lat_lon(cube)
    check_typesi(cube)

    np.testing.assert_allclose(
        cube.data,
        [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]],
    )


# Test ta (for height and plev coordinate)


def test_get_ta_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'Amon', 'ta')
    assert fix == [AllVars(None), GenericFix(None)]


def test_ta_fix(cubes_3d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'ta')
    fixed_cubes = fix.fix_metadata(cubes_3d)

    cube = check_ta_metadata(fixed_cubes)
    check_time(cube)
    check_height(cube)
    check_lat_lon(cube)


def test_ta_fix_no_plev_bounds(cubes_3d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'ta')
    cubes = CubeList([
        cubes_3d.extract_cube(NameConstraint(var_name='ta')),
        cubes_3d.extract_cube(NameConstraint(var_name='pfull')),
    ])
    fixed_cubes = fix.fix_metadata(cubes)

    cube = check_ta_metadata(fixed_cubes)
    check_time(cube)
    check_height(cube, plev_has_bounds=False)
    check_lat_lon(cube)


# Test tas (for height2m coordinate, no mesh, no shift time)


def test_get_tas_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'Amon', 'tas')
    assert fix == [AllVars(None), GenericFix(None)]


def test_tas_fix(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    cube = check_tas_metadata(fixed_cubes)
    check_time(cube)
    check_lat_lon(cube)
    check_heightxm(cube, 2.0)


def test_tas_spatial_index_coord_already_present(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')

    index_coord = DimCoord(np.arange(8), var_name='ncells')
    cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    cube.add_dim_coord(index_coord, 1)
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    check_lat_lon(cube)


def test_tas_scalar_height2m_already_present(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')

    # Scalar height (with wrong metadata) already present
    height_coord = AuxCoord(2.0, var_name='h', standard_name='height')
    cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    cube.add_aux_coord(height_coord, ())
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.shape == (1, 8)
    check_heightxm(cube, 2.0)


def test_tas_dim_height2m_already_present(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')
    fix.extra_facets['ugrid'] = False
    fixed_cubes = fix.fix_metadata(cubes_2d)

    cube = check_tas_metadata(fixed_cubes)

    assert cube.mesh is None

    assert cube.coords('first spatial index for variables stored on an '
                       'unstructured grid', dim_coords=True)
    i_coord = cube.coord('first spatial index for variables stored on an '
                         'unstructured grid', dim_coords=True)
    assert i_coord.var_name == 'i'
    assert i_coord.standard_name is None
    assert i_coord.long_name == ('first spatial index for variables stored on '
                                 'an unstructured grid')
    assert i_coord.units == '1'
    np.testing.assert_allclose(i_coord.points, [0, 1, 2, 3, 4, 5, 6, 7])
    assert i_coord.bounds is None

    assert cube.coords('latitude', dim_coords=False)
    assert cube.coords('longitude', dim_coords=False)
    lat = cube.coord('latitude', dim_coords=False)
    lon = cube.coord('longitude', dim_coords=False)
    assert len(cube.coord_dims(lat)) == 1
    assert cube.coord_dims(lat) == cube.coord_dims(lon)
    assert cube.coord_dims(lat) == cube.coord_dims(i_coord)


def test_tas_no_mesh(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')

    # Dimensional coordinate height (with wrong metadata) already present
    height_coord = AuxCoord(2.0, var_name='h', standard_name='height')
    cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    cube.add_aux_coord(height_coord, ())
    cube = iris.util.new_axis(cube, scalar_coord='height')
    cube.transpose((1, 0, 2))
    cubes = CubeList([cube])
    fixed_cubes = fix.fix_metadata(cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.shape == (1, 8)
    check_heightxm(cube, 2.0)


def test_tas_no_shift_time(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')
    fix.extra_facets['shift_time'] = False
    fixed_cubes = fix.fix_metadata(cubes_2d)

    cube = check_tas_metadata(fixed_cubes)
    check_lat_lon(cube)
    check_heightxm(cube, 2.0)

    assert cube.coords('time', dim_coords=True)
    time = cube.coord('time', dim_coords=True)
    assert time.var_name == 'time'
    assert time.standard_name == 'time'
    assert time.long_name == 'time'
    assert time.units == Unit('days since 1850-01-01',
                              calendar='proleptic_gregorian')
    np.testing.assert_allclose(time.points, [54786.0])
    assert time.bounds is None
    assert time.attributes == {}


# Test uas (for height10m coordinate)


def test_get_uas_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'Amon', 'uas')
    assert fix == [AllVars(None), GenericFix(None)]


def test_uas_fix(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'uas')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'uas'
    assert cube.standard_name == 'eastward_wind'
    assert cube.long_name == 'Eastward Near-Surface Wind'
    assert cube.units == 'm s-1'
    assert 'positive' not in cube.attributes

    check_time(cube)
    check_lat_lon(cube)
    assert cube.coords('height')
    height = cube.coord('height')
    assert height.var_name == 'height'
    assert height.standard_name == 'height'
    assert height.long_name == 'height'
    assert height.units == 'm'
    assert height.attributes == {'positive': 'up'}
    np.testing.assert_allclose(height.points, [10.0])
    assert height.bounds is None


def test_uas_scalar_height10m_already_present(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'uas')

    # Scalar height (with wrong metadata) already present
    height_coord = AuxCoord(10.0, var_name='h', standard_name='height')
    cube = cubes_2d.extract_cube(NameConstraint(var_name='uas'))
    cube.add_aux_coord(height_coord, ())
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.shape == (1, 8)
    check_heightxm(cube, 10.0)


def test_uas_dim_height10m_already_present(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'uas')

    # Dimensional coordinate height (with wrong metadata) already present
    height_coord = AuxCoord(10.0, var_name='h', standard_name='height')
    cube = cubes_2d.extract_cube(NameConstraint(var_name='uas'))
    cube.add_aux_coord(height_coord, ())
    cube = iris.util.new_axis(cube, scalar_coord='height')
    cube.transpose((1, 0, 2))
    cubes = CubeList([cube])
    fixed_cubes = fix.fix_metadata(cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.shape == (1, 8)
    check_heightxm(cube, 10.0)


# Test fix with regular grid and 2D latitudes and longitude


def test_regular_grid_fix(cubes_regular_grid):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')
    fixed_cubes = fix.fix_metadata(cubes_regular_grid)

    cube = check_tas_metadata(fixed_cubes)
    assert cube.coords('time', dim_coords=True, dimensions=0)
    assert cube.coords('latitude', dim_coords=True, dimensions=1)
    assert cube.coords('longitude', dim_coords=True, dimensions=2)
    assert cube.coords('height', dim_coords=False, dimensions=())


def test_2d_lat_lon_grid_fix(cubes_2d_lat_lon_grid):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')
    fixed_cubes = fix.fix_metadata(cubes_2d_lat_lon_grid)

    cube = check_tas_metadata(fixed_cubes)
    assert cube.coords('time', dim_coords=True, dimensions=0)
    assert cube.coords('latitude', dim_coords=False, dimensions=(1, 2))
    assert cube.coords('longitude', dim_coords=False, dimensions=(1, 2))
    assert cube.coords('height', dim_coords=False, dimensions=())


# Test ch4Clim (for time dimension time2)


def test_get_ch4clim_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'Amon', 'ch4Clim')
    assert fix == [AllVars(None), GenericFix(None)]


def test_ch4clim_fix(cubes_regular_grid):
    """Test fix."""
    cube = cubes_regular_grid[0]
    cube.var_name = 'ch4Clim'
    cube.units = 'mol mol-1'
    cube.coord('time').units = 'no_unit'
    cube.coord('time').attributes['invalid_units'] = 'day as %Y%m%d.%f'
    cube.coord('time').points = [18500201.0]
    cube.coord('time').long_name = 'wrong_time_name'

    fix = get_allvars_fix('Amon', 'ch4Clim')
    fixed_cubes = fix.fix_metadata(cubes_regular_grid)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'ch4Clim'
    assert cube.standard_name == 'mole_fraction_of_methane_in_air'
    assert cube.long_name == 'Mole Fraction of CH4'
    assert cube.units == 'mol mol-1'
    assert 'positive' not in cube.attributes

    time_coord = cube.coord('time')
    assert time_coord.var_name == 'time'
    assert time_coord.standard_name == 'time'
    assert time_coord.long_name == 'time'
    assert time_coord.units == Unit(
        'days since 1850-01-01', calendar='proleptic_gregorian'
    )
    np.testing.assert_allclose(time_coord.points, [15.5])
    np.testing.assert_allclose(time_coord.bounds, [[0.0, 31.0]])


# Test fix with empty standard_name


def test_empty_standard_name_fix(cubes_2d, monkeypatch):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')
    # We know that tas has a standard name, but this being native model output
    # there may be variables with no standard name. The code is designed to
    # handle this gracefully and here we test it with an artificial, but
    # realistic case.
    monkeypatch.setattr(fix.vardef, 'standard_name', '')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'tas'
    assert cube.standard_name is None
    assert cube.long_name == 'Near-Surface Air Temperature'
    assert cube.units == 'K'
    assert 'positive' not in cube.attributes


# Test automatic addition of missing coordinates


def test_add_time(cubes_2d):
    """Test fix."""
    # Remove time from tas cube to test automatic addition
    tas_cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    uas_cube = cubes_2d.extract_cube(NameConstraint(var_name='uas'))
    tas_cube = tas_cube[0]
    tas_cube.remove_coord('time')
    cubes = CubeList([tas_cube, uas_cube])

    fix = get_allvars_fix('Amon', 'tas')
    fixed_cubes = fix.fix_metadata(cubes)

    cube = check_tas_metadata(fixed_cubes)
    assert cube.shape == (1, 8)
    check_time(cube)


def test_add_time_fail():
    """Test fix."""
    fix = get_allvars_fix('Amon', 'ta')
    cube = Cube(1, var_name='ta', units='K')
    cubes = CubeList([
        cube,
        Cube(1, var_name='tas', units='K'),
    ])
    msg = "Cannot add required coordinate 'time' to variable 'ta'"
    with pytest.raises(ValueError, match=msg):
        fix._add_time(cube, cubes)


def test_add_latitude(cubes_2d):
    """Test fix."""
    # Remove latitude from tas cube to test automatic addition
    tas_cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    tas_cube.remove_coord('latitude')
    cubes = CubeList([tas_cube])
    fix = get_allvars_fix('Amon', 'tas')

    assert len(fix._horizontal_grids) == 0
    fixed_cubes = fix.fix_metadata(cubes)

    cube = check_tas_metadata(fixed_cubes)
    assert cube.shape == (1, 8)
    check_lat_lon(cube)
    assert len(fix._horizontal_grids) == 1
    assert TEST_GRID_FILE_NAME in fix._horizontal_grids


def test_add_longitude(cubes_2d):
    """Test fix."""
    # Remove longitude from tas cube to test automatic addition
    tas_cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    tas_cube.remove_coord('longitude')
    cubes = CubeList([tas_cube])
    fix = get_allvars_fix('Amon', 'tas')

    assert len(fix._horizontal_grids) == 0
    fixed_cubes = fix.fix_metadata(cubes)

    cube = check_tas_metadata(fixed_cubes)
    assert cube.shape == (1, 8)
    check_lat_lon(cube)
    assert len(fix._horizontal_grids) == 1
    assert TEST_GRID_FILE_NAME in fix._horizontal_grids


def test_add_latitude_longitude(cubes_2d):
    """Test fix."""
    # Remove latitude and longitude from tas cube to test automatic addition
    tas_cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    tas_cube.remove_coord('latitude')
    tas_cube.remove_coord('longitude')
    cubes = CubeList([tas_cube])
    fix = get_allvars_fix('Amon', 'tas')

    assert len(fix._horizontal_grids) == 0
    fixed_cubes = fix.fix_metadata(cubes)

    cube = check_tas_metadata(fixed_cubes)
    assert cube.shape == (1, 8)
    check_lat_lon(cube)
    assert len(fix._horizontal_grids) == 1
    assert TEST_GRID_FILE_NAME in fix._horizontal_grids


def test_add_latitude_fail(cubes_2d):
    """Test fix."""
    # Remove latitude and grid file attribute from tas cube to test automatic
    # addition
    tas_cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    tas_cube.remove_coord('latitude')
    tas_cube.attributes = {}
    cubes = CubeList([tas_cube])
    fix = get_allvars_fix('Amon', 'tas')

    msg = "Failed to add missing latitude coordinate to cube"
    with pytest.raises(ValueError, match=msg):
        fix.fix_metadata(cubes)


def test_add_longitude_fail(cubes_2d):
    """Test fix."""
    # Remove longitude and grid file attribute from tas cube to test automatic
    # addition
    tas_cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    tas_cube.remove_coord('longitude')
    tas_cube.attributes = {}
    cubes = CubeList([tas_cube])
    fix = get_allvars_fix('Amon', 'tas')

    msg = "Failed to add missing longitude coordinate to cube"
    with pytest.raises(ValueError, match=msg):
        fix.fix_metadata(cubes)


def test_add_coord_from_grid_file_fail_invalid_coord():
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')

    msg = r"coord_name must be one of .* got 'invalid_coord_name'"
    with pytest.raises(ValueError, match=msg):
        fix._add_coord_from_grid_file(mock.sentinel.cube, 'invalid_coord_name')


def test_add_coord_from_grid_file_fail_no_url():
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')

    msg = ("Cube does not contain the attribute 'grid_file_uri' necessary to "
           "download the ICON horizontal grid file")
    with pytest.raises(ValueError, match=msg):
        fix._add_coord_from_grid_file(Cube(0), 'latitude')


def test_add_coord_from_grid_fail_no_unnamed_dim(cubes_2d):
    """Test fix."""
    # Remove latitude from tas cube to test automatic addition
    tas_cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    tas_cube.remove_coord('latitude')
    index_coord = DimCoord(np.arange(8), var_name='ncells')
    tas_cube.add_dim_coord(index_coord, 1)
    fix = get_allvars_fix('Amon', 'tas')

    msg = ("Cannot determine coordinate dimension for coordinate 'latitude', "
           "cube does not contain a single unnamed dimension")
    with pytest.raises(ValueError, match=msg):
        fix._add_coord_from_grid_file(tas_cube, 'latitude')


def test_add_coord_from_grid_fail_two_unnamed_dims(cubes_2d):
    """Test fix."""
    # Remove latitude from tas cube to test automatic addition
    tas_cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    tas_cube.remove_coord('latitude')
    tas_cube = iris.util.new_axis(tas_cube)
    fix = get_allvars_fix('Amon', 'tas')

    msg = ("Cannot determine coordinate dimension for coordinate 'latitude', "
           "cube does not contain a single unnamed dimension")
    with pytest.raises(ValueError, match=msg):
        fix._add_coord_from_grid_file(tas_cube, 'latitude')


# Test get_horizontal_grid


@mock.patch.object(IconFix, '_get_grid_from_facet', autospec=True)
@mock.patch('esmvalcore.cmor._fixes.icon._base_fixes.requests', autospec=True)
def test_get_horizontal_grid_from_attr_cached_in_dict(
    mock_requests,
    mock_get_grid_from_facet,
):
    """Test fix."""
    cube = Cube(0, attributes={'grid_file_uri': 'cached_grid_url.nc'})
    grid_cube = Cube(0)
    fix = get_allvars_fix('Amon', 'tas')
    fix._horizontal_grids['cached_grid_url.nc'] = grid_cube
    fix._horizontal_grids['grid_from_facet.nc'] = mock.sentinel.wrong_grid

    grid = fix.get_horizontal_grid(cube)
    assert len(fix._horizontal_grids) == 2
    assert 'cached_grid_url.nc' in fix._horizontal_grids
    assert 'grid_from_facet.nc' in fix._horizontal_grids  # has not been used
    assert fix._horizontal_grids['cached_grid_url.nc'] == grid
    assert grid == grid_cube
    assert grid is not grid_cube
    assert mock_requests.mock_calls == []
    mock_get_grid_from_facet.assert_not_called()


@mock.patch.object(IconFix, '_get_grid_from_facet', autospec=True)
def test_get_horizontal_grid_from_attr_rootpath(
    mock_get_grid_from_facet, monkeypatch, tmp_path
):
    """Test fix."""
    rootpath = deepcopy(CFG['rootpath'])
    rootpath['ICON'] = str(tmp_path)
    monkeypatch.setitem(CFG, 'rootpath', rootpath)
    cube = Cube(0, attributes={'grid_file_uri': 'grid.nc'})
    grid_cube = Cube(0, var_name='test_grid_cube')
    (tmp_path / 'amip').mkdir(parents=True, exist_ok=True)
    iris.save(grid_cube, tmp_path / 'amip' / 'grid.nc')

    fix = get_allvars_fix('Amon', 'tas')
    fix._horizontal_grids['grid_from_facet.nc'] = mock.sentinel.wrong_grid

    grid = fix.get_horizontal_grid(cube)
    assert len(fix._horizontal_grids) == 2
    assert 'grid.nc' in fix._horizontal_grids
    assert 'grid_from_facet.nc' in fix._horizontal_grids  # has not been used
    assert fix._horizontal_grids['grid.nc'] == grid
    assert len(grid) == 1
    assert grid[0].var_name == 'test_grid_cube'
    assert grid[0].shape == ()
    mock_get_grid_from_facet.assert_not_called()


@mock.patch.object(IconFix, '_get_grid_from_facet', autospec=True)
@mock.patch('esmvalcore.cmor._fixes.icon._base_fixes.requests', autospec=True)
def test_get_horizontal_grid_from_attr_cached_in_file(
    mock_requests,
    mock_get_grid_from_facet,
    tmp_path,
):
    """Test fix."""
    cube = Cube(0, attributes={
        'grid_file_uri': 'https://temporary.url/this/is/the/grid_file.nc'})
    fix = get_allvars_fix('Amon', 'tas')
    assert len(fix._horizontal_grids) == 0

    # Save temporary grid file
    grid_cube = Cube(0, var_name='grid')
    iris.save(grid_cube, str(tmp_path / 'grid_file.nc'))

    grid = fix.get_horizontal_grid(cube)
    assert isinstance(grid, CubeList)
    assert len(grid) == 1
    assert grid[0].var_name == 'grid'
    assert grid[0].shape == ()
    assert len(fix._horizontal_grids) == 1
    assert 'grid_file.nc' in fix._horizontal_grids
    assert fix._horizontal_grids['grid_file.nc'] == grid
    assert mock_requests.mock_calls == []
    mock_get_grid_from_facet.assert_not_called()


@mock.patch.object(IconFix, '_get_grid_from_facet', autospec=True)
def test_get_horizontal_grid_from_attr_cache_file_too_old(
    mock_get_grid_from_facet,
    tmp_path,
    monkeypatch,
):
    """Test fix."""
    cube = Cube(0, attributes={'grid_file_uri': TEST_GRID_FILE_URI})
    fix = get_allvars_fix('Amon', 'tas')
    assert len(fix._horizontal_grids) == 0

    # Save temporary grid file
    grid_cube = Cube(0, var_name='grid')
    iris.save(grid_cube, str(tmp_path / 'icon_grid.nc'))

    # Temporary overwrite default cache location for downloads and cache
    # validity duration
    monkeypatch.setattr(fix, 'CACHE_VALIDITY', -1)

    grid = fix.get_horizontal_grid(cube)
    assert isinstance(grid, CubeList)
    assert len(grid) == 4
    var_names = [cube.var_name for cube in grid]
    assert 'cell_area' in var_names
    assert 'dual_area' in var_names
    assert 'vertex_index' in var_names
    assert 'vertex_of_cell' in var_names
    assert len(fix._horizontal_grids) == 1
    assert TEST_GRID_FILE_NAME in fix._horizontal_grids
    assert fix._horizontal_grids[TEST_GRID_FILE_NAME] == grid
    mock_get_grid_from_facet.assert_not_called()


@mock.patch.object(IconFix, '_get_grid_from_cube_attr', autospec=True)
def test_get_horizontal_grid_from_facet_cached_in_dict(
    mock_get_grid_from_cube_attr,
    tmp_path,
):
    """Test fix."""
    session = CFG.start_session('my session')
    session['auxiliary_data_dir'] = tmp_path

    # Save temporary grid file (this will not be used; however, it is necessary
    # to not raise a FileNotFoundError)
    grid_path = 'grid.nc'
    wrong_grid_cube = Cube(0, var_name='wrong_grid')
    iris.save(wrong_grid_cube, tmp_path / 'grid.nc')

    # Make sure that grid specified by cube attribute is NOT used
    cube = Cube(0, attributes={'grid_file_uri': 'cached_grid_url.nc'})
    grid_cube = Cube(0, var_name='grid')
    fix = get_allvars_fix('Amon', 'tas', session=session)
    fix.extra_facets['horizontal_grid'] = grid_path
    fix._horizontal_grids['cached_grid_url.nc'] = mock.sentinel.wrong_grid
    fix._horizontal_grids['grid.nc'] = grid_cube

    grid = fix.get_horizontal_grid(cube)
    assert len(fix._horizontal_grids) == 2
    assert 'cached_grid_url.nc' in fix._horizontal_grids  # has not been used
    assert 'grid.nc' in fix._horizontal_grids
    assert fix._horizontal_grids['grid.nc'] == grid
    assert grid == grid_cube
    assert grid is not grid_cube
    mock_get_grid_from_cube_attr.assert_not_called()


@pytest.mark.parametrize('grid_path', ['{tmp_path}/grid.nc', 'grid.nc'])
@mock.patch.object(IconFix, '_get_grid_from_cube_attr', autospec=True)
def test_get_horizontal_grid_from_facet(
    mock_get_grid_from_cube_attr,
    grid_path,
    tmp_path,
):
    """Test fix."""
    session = CFG.start_session('my session')
    session['auxiliary_data_dir'] = tmp_path

    # Make sure that grid specified by cube attribute is NOT used
    cube = Cube(0, attributes={'grid_file_uri': 'cached_grid_url.nc'})

    # Save temporary grid file
    grid_path = grid_path.format(tmp_path=tmp_path)
    grid_cube = Cube(0, var_name='grid')
    iris.save(grid_cube, tmp_path / 'grid.nc')

    fix = get_allvars_fix('Amon', 'tas', session=session)
    fix.extra_facets['horizontal_grid'] = grid_path
    fix._horizontal_grids['cached_grid_url.nc'] = mock.sentinel.wrong_grid

    grid = fix.get_horizontal_grid(cube)
    assert isinstance(grid, CubeList)
    assert len(grid) == 1
    assert grid[0].var_name == 'grid'
    assert len(fix._horizontal_grids) == 2
    assert 'cached_grid_url.nc' in fix._horizontal_grids  # has not been used
    assert 'grid.nc' in fix._horizontal_grids
    assert fix._horizontal_grids['grid.nc'] == grid
    mock_get_grid_from_cube_attr.assert_not_called()


def test_get_horizontal_grid_from_facet_fail(tmp_path):
    """Test fix."""
    session = CFG.start_session('my session')
    session['auxiliary_data_dir'] = tmp_path

    cube = Cube(0)
    fix = get_allvars_fix('Amon', 'tas', session=session)
    fix.extra_facets['horizontal_grid'] = '/this/does/not/exist.nc'

    with pytest.raises(FileNotFoundError):
        fix.get_horizontal_grid(cube)


# Test with single-dimension cubes


def test_only_time(monkeypatch):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'ta')
    # We know that ta has dimensions time, plev19, latitude, longitude, but the
    # ICON CMORizer is designed to check for the presence of each dimension
    # individually. To test this, remove all but one dimension of ta to create
    # an artificial, but realistic test case.
    coord_info = CoordinateInfo('time')
    coord_info.standard_name = 'time'
    monkeypatch.setattr(fix.vardef, 'coordinates', {'time': coord_info})

    # Create cube with only a single dimension
    time_coord = DimCoord([0.0, 31.0],
                          var_name='time',
                          standard_name='time',
                          long_name='time',
                          units='days since 1850-01-01')
    cubes = CubeList([
        Cube([1, 1], var_name='ta', units='K',
             dim_coords_and_dims=[(time_coord, 0)]),
    ])
    fixed_cubes = fix.fix_metadata(cubes)

    # Check cube metadata
    cube = check_ta_metadata(fixed_cubes)

    # Check cube data
    assert cube.shape == (2,)
    np.testing.assert_equal(cube.data, [1, 1])

    # Check time metadata
    assert cube.coords('time')
    new_time_coord = cube.coord('time', dim_coords=True)
    assert new_time_coord.var_name == 'time'
    assert new_time_coord.standard_name == 'time'
    assert new_time_coord.long_name == 'time'
    assert new_time_coord.units == 'days since 1850-01-01'

    # Check time data
    np.testing.assert_allclose(new_time_coord.points, [-15.5, 15.5])
    np.testing.assert_allclose(new_time_coord.bounds,
                               [[-31.0, 0.0], [0.0, 31.0]])

    # Check that no mesh has been created
    assert cube.mesh is None


def test_only_height(monkeypatch):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'ta')
    # We know that ta has dimensions time, plev19, latitude, longitude, but the
    # ICON CMORizer is designed to check for the presence of each dimension
    # individually. To test this, remove all but one dimension of ta to create
    # an artificial, but realistic test case.
    coord_info = CoordinateInfo('plev19')
    coord_info.standard_name = 'air_pressure'
    monkeypatch.setattr(fix.vardef, 'coordinates', {'plev19': coord_info})

    # Create cube with only a single dimension
    height_coord = DimCoord([1000.0, 100.0],
                            var_name='height',
                            standard_name='height',
                            units='cm')
    cubes = CubeList([
        Cube([1, 1], var_name='ta', units='K',
             dim_coords_and_dims=[(height_coord, 0)]),
    ])
    fixed_cubes = fix.fix_metadata(cubes)

    # Check cube metadata
    cube = check_ta_metadata(fixed_cubes)

    # Check cube data
    assert cube.shape == (2,)
    np.testing.assert_equal(cube.data, [1, 1])

    # Check height metadata
    assert cube.coords('height', dim_coords=True)
    new_height_coord = cube.coord('height')
    assert new_height_coord.var_name == 'height'
    assert new_height_coord.standard_name == 'height'
    assert new_height_coord.long_name == 'height'
    assert new_height_coord.units == 'm'
    assert new_height_coord.attributes == {'positive': 'up'}

    # Check height data
    np.testing.assert_allclose(new_height_coord.points, [1.0, 10.0])
    assert new_height_coord.bounds is None

    # Check that no air_pressure coordinate has been created
    assert not cube.coords('air_pressure')

    # Check that no mesh has been created
    assert cube.mesh is None


def test_only_latitude(monkeypatch):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'ta')
    # We know that ta has dimensions time, plev19, latitude, longitude, but the
    # ICON CMORizer is designed to check for the presence of each dimension
    # individually. To test this, remove all but one dimension of ta to create
    # an artificial, but realistic test case.
    coord_info = CoordinateInfo('latitude')
    coord_info.standard_name = 'latitude'
    monkeypatch.setattr(fix.vardef, 'coordinates', {'latitude': coord_info})

    # Create cube with only a single dimension
    lat_coord = DimCoord([0.0, 10.0],
                         var_name='lat',
                         standard_name='latitude',
                         units='degrees')
    cubes = CubeList([
        Cube([1, 1], var_name='ta', units='K',
             dim_coords_and_dims=[(lat_coord, 0)]),
    ])
    fixed_cubes = fix.fix_metadata(cubes)

    # Check cube metadata
    cube = check_ta_metadata(fixed_cubes)

    # Check cube data
    assert cube.shape == (2,)
    np.testing.assert_equal(cube.data, [1, 1])

    # Check latitude metadata
    assert cube.coords('latitude', dim_coords=True)
    new_lat_coord = cube.coord('latitude')
    assert new_lat_coord.var_name == 'lat'
    assert new_lat_coord.standard_name == 'latitude'
    assert new_lat_coord.long_name == 'latitude'
    assert new_lat_coord.units == 'degrees_north'

    # Check latitude data
    np.testing.assert_allclose(new_lat_coord.points, [0.0, 10.0])
    assert new_lat_coord.bounds is None

    # Check that no mesh has been created
    assert cube.mesh is None


def test_only_longitude(monkeypatch):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'ta')
    # We know that ta has dimensions time, plev19, latitude, longitude, but the
    # ICON CMORizer is designed to check for the presence of each dimension
    # individually. To test this, remove all but one dimension of ta to create
    # an artificial, but realistic test case.
    coord_info = CoordinateInfo('longitude')
    coord_info.standard_name = 'longitude'
    monkeypatch.setattr(fix.vardef, 'coordinates', {'longitude': coord_info})

    # Create cube with only a single dimension
    lon_coord = DimCoord([0.0, 180.0],
                         var_name='lon',
                         standard_name='longitude',
                         units='degrees')
    cubes = CubeList([
        Cube([1, 1], var_name='ta', units='K',
             dim_coords_and_dims=[(lon_coord, 0)]),
    ])
    fixed_cubes = fix.fix_metadata(cubes)

    # Check cube metadata
    cube = check_ta_metadata(fixed_cubes)

    # Check cube data
    assert cube.shape == (2,)
    np.testing.assert_equal(cube.data, [1, 1])

    # Check longitude metadata
    assert cube.coords('longitude', dim_coords=True)
    new_lon_coord = cube.coord('longitude')
    assert new_lon_coord.var_name == 'lon'
    assert new_lon_coord.standard_name == 'longitude'
    assert new_lon_coord.long_name == 'longitude'
    assert new_lon_coord.units == 'degrees_east'

    # Check longitude data
    np.testing.assert_allclose(new_lon_coord.points, [0.0, 180.0])
    assert new_lon_coord.bounds is None

    # Check that no mesh has been created
    assert cube.mesh is None


# Test variable not available in file


def test_var_not_available_pr(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'pr')
    msg = "Variable 'pr' used to extract 'pr' is not available in input file"
    with pytest.raises(ValueError, match=msg):
        fix.fix_metadata(cubes_2d)


# Test fix with invalid time units


def test_invalid_time_units(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')
    for cube in cubes_2d:
        cube.coord('time').attributes['invalid_units'] = 'month as %Y%m%d.%f'
    msg = "Expected time units"
    with pytest.raises(ValueError, match=msg):
        fix.fix_metadata(cubes_2d)


# Test fix with (sub-)hourly data


def test_hourly_data(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')
    fix.extra_facets['frequency'] = '1hr'
    for cube in cubes_2d:
        cube.coord('time').points = [20041104.5833333]

    fixed_cubes = fix.fix_metadata(cubes_2d)

    cube = check_tas_metadata(fixed_cubes)
    date = cube.coord('time').units.num2date(cube.coord('time').points)
    date_bnds = cube.coord('time').units.num2date(cube.coord('time').bounds)
    np.testing.assert_array_equal(date, [datetime(2004, 11, 4, 13, 30)])
    np.testing.assert_array_equal(
        date_bnds, [[datetime(2004, 11, 4, 13), datetime(2004, 11, 4, 14)]]
    )


@pytest.mark.parametrize(
    'bounds',
    [
        None,
        [
            [20211231.875, 20220101.125],
            [20220101.125, 20220101.375],
        ],
    ],
)
def test_6hourly_data_multiple_points(bounds):
    """Test fix."""
    time_coord = DimCoord(
        [20220101, 20220101.25],
        bounds=bounds,
        standard_name='time',
        attributes={'invalid_units': 'day as %Y%m%d.%f'},
    )
    cube = Cube(
        [1, 2],
        var_name='tas',
        units='K',
        dim_coords_and_dims=[(time_coord, 0)],
    )
    cubes = CubeList([cube])
    fix = get_allvars_fix('Amon', 'tas')
    fix.extra_facets['frequency'] = '6hr'

    fixed_cube = fix._fix_time(cube, cubes)

    points = fixed_cube.coord('time').units.num2date(cube.coord('time').points)
    bounds = fixed_cube.coord('time').units.num2date(cube.coord('time').bounds)
    np.testing.assert_array_equal(
        points,
        [datetime(2021, 12, 31, 21), datetime(2022, 1, 1, 3)],
    )
    np.testing.assert_array_equal(
        bounds,
        [
            [datetime(2021, 12, 31, 18), datetime(2022, 1, 1)],
            [datetime(2022, 1, 1), datetime(2022, 1, 1, 6)],
        ],
    )


def test_subhourly_data_no_shift():
    """Test fix."""
    time_coord = DimCoord(
        [0.5, 1.0],
        standard_name='time',
        units=Unit('hours since 2022-01-01', calendar='proleptic_gregorian'),
    )
    cube = Cube(
        [1, 2],
        var_name='tas',
        units='K',
        dim_coords_and_dims=[(time_coord, 0)],
    )
    cubes = CubeList([cube])
    fix = get_allvars_fix('Amon', 'tas')
    fix.extra_facets['frequency'] = 'subhr'
    fix.extra_facets['shift_time'] = False

    fixed_cube = fix._fix_time(cube, cubes)

    points = fixed_cube.coord('time').units.num2date(cube.coord('time').points)
    bounds = fixed_cube.coord('time').units.num2date(cube.coord('time').bounds)
    np.testing.assert_array_equal(
        points,
        [datetime(2022, 1, 1, 0, 30), datetime(2022, 1, 1, 1)],
    )
    np.testing.assert_array_equal(
        bounds,
        [
            [datetime(2022, 1, 1, 0, 15), datetime(2022, 1, 1, 0, 45)],
            [datetime(2022, 1, 1, 0, 45), datetime(2022, 1, 1, 1, 15)],
        ],
    )


# Test _shift_time_coord


@pytest.mark.parametrize(
    'frequency,dt_in,dt_out,bounds',
    [
        (
            'dec',
            [(2000, 1, 1)],
            [(1995, 1, 1)],
            [[(1990, 1, 1), (2000, 1, 1)]],
        ),
        (
            'yr',
            [(2000, 1, 1), (2001, 1, 1)],
            [(1999, 7, 2, 12), (2000, 7, 2)],
            [[(1999, 1, 1), (2000, 1, 1)], [(2000, 1, 1), (2001, 1, 1)]],
        ),
        (
            'mon',
            [(2000, 1, 1)],
            [(1999, 12, 16, 12)],
            [[(1999, 12, 1), (2000, 1, 1)]],
        ),
        (
            'mon',
            [(2000, 11, 30, 23, 45), (2000, 12, 31, 23)],
            [(2000, 11, 16), (2000, 12, 16, 12)],
            [[(2000, 11, 1), (2000, 12, 1)], [(2000, 12, 1), (2001, 1, 1)]],
        ),
        (
            'day',
            [(2000, 1, 1, 12)],
            [(2000, 1, 1)],
            [[(1999, 12, 31, 12), (2000, 1, 1, 12)]],
        ),
        (
            '6hr',
            [(2000, 1, 5, 14), (2000, 1, 5, 20)],
            [(2000, 1, 5, 11), (2000, 1, 5, 17)],
            [
                [(2000, 1, 5, 8), (2000, 1, 5, 14)],
                [(2000, 1, 5, 14), (2000, 1, 5, 20)],
            ],
        ),
        (
            '3hr',
            [(2000, 1, 1)],
            [(1999, 12, 31, 22, 30)],
            [[(1999, 12, 31, 21), (2000, 1, 1)]],
        ),
        (
            '1hr',
            [(2000, 1, 5, 14), (2000, 1, 5, 15)],
            [(2000, 1, 5, 13, 30), (2000, 1, 5, 14, 30)],
            [
                [(2000, 1, 5, 13), (2000, 1, 5, 14)],
                [(2000, 1, 5, 14), (2000, 1, 5, 15)],
            ],
        ),
    ],
)
def test_shift_time_coord(frequency, dt_in, dt_out, bounds):
    """Test ``_shift_time_coord``."""
    cube = Cube(0, cell_methods=[CellMethod('mean', 'time')])
    datetimes = [datetime(*dt) for dt in dt_in]
    time_units = Unit('days since 1950-01-01', calendar='proleptic_gregorian')
    time_coord = DimCoord(
        time_units.date2num(datetimes),
        standard_name='time',
        var_name='time',
        long_name='time',
        units=time_units,
    )

    fix = get_allvars_fix('Amon', 'tas')
    fix.extra_facets['frequency'] = frequency

    fix._shift_time_coord(cube, time_coord)

    dt_out = [datetime(*dt) for dt in dt_out]
    bounds = [[datetime(*dt1), datetime(*dt2)] for (dt1, dt2) in bounds]
    np.testing.assert_allclose(
        time_coord.points, time_coord.units.date2num(dt_out)
    )
    np.testing.assert_allclose(
        time_coord.bounds, time_coord.units.date2num(bounds)
    )


@pytest.mark.parametrize(
    'frequency,dt_in',
    [
        ('dec', [(2000, 1, 15)]),
        ('yr', [(2000, 1, 1), (2001, 1, 1)]),
        ('mon', [(2000, 6, 15)]),
        ('day', [(2000, 1, 1), (2001, 1, 2)]),
        ('6hr', [(2000, 6, 15, 12)]),
        ('3hr', [(2000, 1, 1, 4), (2000, 1, 1, 7)]),
        ('1hr', [(2000, 1, 1, 4), (2000, 1, 1, 5)]),
    ],
)
def test_shift_time_point_measurement(frequency, dt_in):
    """Test ``_shift_time_coord``."""
    cube = Cube(0, cell_methods=[CellMethod('point', 'time')])
    datetimes = [datetime(*dt) for dt in dt_in]
    time_units = Unit('days since 1950-01-01', calendar='proleptic_gregorian')
    time_coord = DimCoord(
        time_units.date2num(datetimes),
        standard_name='time',
        var_name='time',
        long_name='time',
        units=time_units,
    )

    fix = get_allvars_fix('Amon', 'tas')
    fix.extra_facets['frequency'] = frequency

    fix._shift_time_coord(cube, time_coord)

    np.testing.assert_allclose(
        time_coord.points, time_coord.units.date2num(datetimes)
    )
    assert time_coord.bounds is None


@pytest.mark.parametrize(
    'frequency', ['dec', 'yr', 'yrPt', 'mon', 'monC', 'monPt']
)
def test_shift_time_coord_hourly_data_low_freq_fail(frequency):
    """Test ``_shift_time_coord``."""
    cube = Cube(0, cell_methods=[CellMethod('mean', 'time')])
    time_units = Unit('hours since 1950-01-01', calendar='proleptic_gregorian')
    time_coord = DimCoord(
        [1, 2, 3],
        standard_name='time',
        var_name='time',
        long_name='time',
        units=time_units,
    )

    fix = get_allvars_fix('Amon', 'tas')
    fix.extra_facets['frequency'] = frequency

    msg = (
        "Cannot shift time coordinate: Rounding to closest day failed."
    )
    with pytest.raises(ValueError, match=msg):
        fix._shift_time_coord(cube, time_coord)


@pytest.mark.parametrize(
    'frequency', ['dec', 'yr', 'yrPt', 'mon', 'monC', 'monPt']
)
def test_shift_time_coord_not_first_of_month(frequency):
    """Test ``_get_previous_timestep``."""
    cube = Cube(0, cell_methods=[CellMethod('mean', 'time')])
    time_units = Unit('days since 1950-01-01', calendar='proleptic_gregorian')
    time_coord = DimCoord(
        [1.5],
        standard_name='time',
        var_name='time',
        long_name='time',
        units=time_units,
    )
    fix = get_allvars_fix('Amon', 'tas')
    fix.extra_facets['frequency'] = frequency

    msg = (
        "Cannot shift time coordinate: expected first of the month at 00:00:00"
    )
    with pytest.raises(ValueError, match=msg):
        fix._shift_time_coord(cube, time_coord)


@pytest.mark.parametrize('frequency', ['fx', 'subhrPt', 'invalid_freq'])
def test_shift_time_coord_invalid_freq(frequency):
    """Test ``_get_previous_timestep``."""
    cube = Cube(0, cell_methods=[CellMethod('mean', 'time')])
    time_units = Unit('days since 1950-01-01', calendar='proleptic_gregorian')
    time_coord = DimCoord(
        [1.5, 2.5],
        standard_name='time',
        var_name='time',
        long_name='time',
        units=time_units,
    )
    fix = get_allvars_fix('Amon', 'tas')
    fix.extra_facets['frequency'] = frequency

    msg = (
        "Cannot shift time coordinate: failed to determine previous time step"
    )
    with pytest.raises(ValueError, match=msg):
        fix._shift_time_coord(cube, time_coord)


# Test _get_previous_timestep


@pytest.mark.parametrize(
    'frequency,datetime_in,datetime_out',
    [
        ('dec', (2000, 1, 1), (1990, 1, 1)),
        ('yr', (2000, 1, 1), (1999, 1, 1)),
        ('yrPt', (2001, 6, 1), (2000, 6, 1)),
        ('mon', (2001, 1, 1), (2000, 12, 1)),
        ('mon', (2001, 2, 1), (2001, 1, 1)),
        ('mon', (2001, 3, 1), (2001, 2, 1)),
        ('mon', (2001, 4, 1), (2001, 3, 1)),
        ('monC', (2000, 5, 1), (2000, 4, 1)),
        ('monC', (2000, 6, 1), (2000, 5, 1)),
        ('monC', (2000, 7, 1), (2000, 6, 1)),
        ('monC', (2000, 8, 1), (2000, 7, 1)),
        ('monPt', (2002, 9, 1), (2002, 8, 1)),
        ('monPt', (2002, 10, 1), (2002, 9, 1)),
        ('monPt', (2002, 11, 1), (2002, 10, 1)),
        ('monPt', (2002, 12, 1), (2002, 11, 1)),
        ('day', (2000, 1, 1), (1999, 12, 31)),
        ('day', (2000, 3, 1), (2000, 2, 29)),
        ('day', (2187, 3, 14), (2187, 3, 13)),
        ('6hr', (2000, 3, 14, 15), (2000, 3, 14, 9)),
        ('6hrPt', (2000, 1, 1), (1999, 12, 31, 18)),
        ('6hrCM', (2000, 1, 1, 1), (1999, 12, 31, 19)),
        ('3hr', (2000, 3, 14, 15), (2000, 3, 14, 12)),
        ('3hrPt', (2000, 1, 1), (1999, 12, 31, 21)),
        ('3hrCM', (2000, 1, 1, 1), (1999, 12, 31, 22)),
        ('1hr', (2000, 3, 14, 15), (2000, 3, 14, 14)),
        ('1hrPt', (2000, 1, 1), (1999, 12, 31, 23)),
        ('1hrCM', (2000, 1, 1, 1), (2000, 1, 1)),
        ('hr', (2000, 3, 14), (2000, 3, 13, 23)),
    ],
)
def test_get_previous_timestep(frequency, datetime_in, datetime_out):
    """Test ``_get_previous_timestep``."""
    datetime_in = datetime(*datetime_in)
    datetime_out = datetime(*datetime_out)

    fix = get_allvars_fix('Amon', 'tas')
    fix.extra_facets['frequency'] = frequency

    new_datetime = fix._get_previous_timestep(datetime_in)

    assert new_datetime == datetime_out


# Test mesh creation raises warning because bounds do not match vertices


@mock.patch('esmvalcore.cmor._fixes.icon._base_fixes.logger', autospec=True)
def test_get_mesh_fail_invalid_clat_bounds(mock_logger, cubes_2d):
    """Test fix."""
    # Slightly modify latitude bounds from tas cube to make mesh creation fail
    tas_cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    lat_bnds = tas_cube.coord('latitude').bounds.copy()
    lat_bnds[0, 0] = 40.0
    tas_cube.coord('latitude').bounds = lat_bnds
    cubes = CubeList([tas_cube])
    fix = get_allvars_fix('Amon', 'tas')

    fixed_cubes = fix.fix_metadata(cubes)
    cube = check_tas_metadata(fixed_cubes)

    assert cube.coord('latitude').bounds[0, 0] != 40.0
    mock_logger.warning.assert_called_once_with(
        "Latitude bounds of the face coordinate ('clat_vertices' in "
        "the grid file) differ from the corresponding values "
        "calculated from the connectivity ('vertex_of_cell') and the "
        "node coordinate ('vlat'). Using bounds defined by "
        "connectivity."
    )


@mock.patch('esmvalcore.cmor._fixes.icon._base_fixes.logger', autospec=True)
def test_get_mesh_fail_invalid_clon_bounds(mock_logger, cubes_2d):
    """Test fix."""
    # Slightly modify longitude bounds from tas cube to make mesh creation fail
    tas_cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    lon_bnds = tas_cube.coord('longitude').bounds.copy()
    lon_bnds[0, 1] = 40.0
    tas_cube.coord('longitude').bounds = lon_bnds
    cubes = CubeList([tas_cube])
    fix = get_allvars_fix('Amon', 'tas')

    fixed_cubes = fix.fix_metadata(cubes)
    cube = check_tas_metadata(fixed_cubes)

    assert cube.coord('longitude').bounds[0, 1] != 40.0
    mock_logger.warning.assert_called_once_with(
        "Longitude bounds of the face coordinate ('clon_vertices' in "
        "the grid file) differ from the corresponding values "
        "calculated from the connectivity ('vertex_of_cell') and the "
        "node coordinate ('vlon'). Note that these values are allowed "
        "to differ by 360° or at the poles of the grid. Using bounds "
        "defined by connectivity."
    )


# Test _get_grid_url


def test_get_grid_url():
    """Test fix."""
    cube = Cube(0, attributes={'grid_file_uri': TEST_GRID_FILE_URI})
    fix = get_allvars_fix('Amon', 'tas')
    (grid_url, grid_name) = fix._get_grid_url(cube)
    assert grid_url == TEST_GRID_FILE_URI
    assert grid_name == TEST_GRID_FILE_NAME


def test_get_grid_url_fail():
    """Test fix."""
    cube = Cube(0)
    fix = get_allvars_fix('Amon', 'tas')
    msg = ("Cube does not contain the attribute 'grid_file_uri' necessary to "
           "download the ICON horizontal grid file")
    with pytest.raises(ValueError, match=msg):
        fix._get_grid_url(cube)


# Test get_mesh


def test_get_mesh_cached_from_attr(monkeypatch):
    """Test fix."""
    cube = Cube(0, attributes={'grid_file_uri': TEST_GRID_FILE_URI})
    fix = get_allvars_fix('Amon', 'tas')
    monkeypatch.setattr(fix, '_create_mesh', mock.Mock())
    fix._meshes[TEST_GRID_FILE_NAME] = mock.sentinel.mesh
    mesh = fix.get_mesh(cube)
    assert mesh == mock.sentinel.mesh
    fix._create_mesh.assert_not_called()


def test_get_mesh_not_cached_from_attr(monkeypatch):
    """Test fix."""
    cube = Cube(0, attributes={'grid_file_uri': TEST_GRID_FILE_URI})
    fix = get_allvars_fix('Amon', 'tas')
    monkeypatch.setattr(fix, '_create_mesh', mock.Mock())
    fix.get_mesh(cube)
    fix._create_mesh.assert_called_once_with(cube)


def test_get_mesh_cached_from_facet(monkeypatch, tmp_path):
    """Test fix."""
    session = CFG.start_session('my session')
    session['auxiliary_data_dir'] = tmp_path

    # Save temporary grid file (this will not be used; however, it is necessary
    # to not raise a FileNotFoundError)
    grid_path = 'grid.nc'
    grid_cube = Cube(0, var_name='grid')
    iris.save(grid_cube, tmp_path / 'grid.nc')

    cube = Cube(0, attributes={'grid_file_uri': TEST_GRID_FILE_URI})
    fix = get_allvars_fix('Amon', 'tas', session=session)
    fix.extra_facets['horizontal_grid'] = grid_path
    monkeypatch.setattr(fix, '_create_mesh', mock.Mock())
    fix._meshes[TEST_GRID_FILE_NAME] = mock.sentinel.wrong_mesh
    fix._meshes['grid.nc'] = mock.sentinel.mesh

    mesh = fix.get_mesh(cube)

    assert mesh == mock.sentinel.mesh
    fix._create_mesh.assert_not_called()


def test_get_mesh_not_cached_from_facet(monkeypatch, tmp_path):
    """Test fix."""
    session = CFG.start_session('my session')
    session['auxiliary_data_dir'] = tmp_path

    # Save temporary grid file (this will not be used; however, it is necessary
    # to not raise a FileNotFoundError)
    grid_path = 'grid.nc'
    grid_cube = Cube(0, var_name='grid')
    iris.save(grid_cube, tmp_path / 'grid.nc')

    cube = Cube(0, attributes={'grid_file_uri': TEST_GRID_FILE_URI})
    fix = get_allvars_fix('Amon', 'tas', session=session)
    fix.extra_facets['horizontal_grid'] = grid_path
    monkeypatch.setattr(fix, '_create_mesh', mock.Mock())
    fix._meshes[TEST_GRID_FILE_NAME] = mock.sentinel.wrong_mesh

    fix.get_mesh(cube)

    fix._create_mesh.assert_called_once_with(cube)


# Test _get_path_from_facet


@pytest.mark.parametrize(
    'path,description,output',
    [
        ('{tmp_path}/a.nc', None, '{tmp_path}/a.nc'),
        ('b.nc', 'Grid file', '{tmp_path}/b.nc'),
    ],
)
def test_get_path_from_facet(path, description, output, tmp_path):
    """Test fix."""
    session = CFG.start_session('my session')
    session['auxiliary_data_dir'] = tmp_path
    path = path.format(tmp_path=tmp_path)
    fix = get_allvars_fix('Amon', 'tas', session=session)
    fix.extra_facets['test_path'] = path

    # Create empty dummy file
    output = output.format(tmp_path=tmp_path)
    with open(output, 'w', encoding='utf-8'):
        pass

    out_path = fix._get_path_from_facet('test_path', description=description)

    assert isinstance(out_path, Path)
    assert out_path == Path(output.format(tmp_path=tmp_path))


@pytest.mark.parametrize(
    'path,description',
    [
        ('{tmp_path}/a.nc', None),
        ('b.nc', 'Grid file'),
    ],
)
def test_get_path_from_facet_fail(path, description, tmp_path):
    """Test fix."""
    session = CFG.start_session('my session')
    session['auxiliary_data_dir'] = tmp_path
    path = path.format(tmp_path=tmp_path)
    fix = get_allvars_fix('Amon', 'tas', session=session)
    fix.extra_facets['test_path'] = path

    with pytest.raises(FileNotFoundError, match=description):
        fix._get_path_from_facet('test_path', description=description)


# Test add_additional_cubes


@pytest.mark.parametrize('facet', ['zg_file', 'zghalf_file'])
@pytest.mark.parametrize('path', ['{tmp_path}/a.nc', 'a.nc'])
def test_add_additional_cubes(path, facet, tmp_path):
    """Test fix."""
    session = CFG.start_session('my session')
    session['auxiliary_data_dir'] = tmp_path
    path = path.format(tmp_path=tmp_path)
    fix = get_allvars_fix('Amon', 'tas', session=session)
    fix.extra_facets[facet] = path

    # Save temporary cube
    cube = Cube(0, var_name=facet)
    iris.save(cube, tmp_path / 'a.nc')

    cubes = CubeList([])
    new_cubes = fix.add_additional_cubes(cubes)

    assert new_cubes is cubes
    assert len(cubes) == 1
    assert cubes[0].var_name == facet


@pytest.mark.parametrize('facet', ['zg_file', 'zghalf_file'])
@pytest.mark.parametrize('path', ['{tmp_path}/a.nc', 'a.nc'])
def test_add_additional_cubes_fail(path, facet, tmp_path):
    """Test fix."""
    session = CFG.start_session('my session')
    session['auxiliary_data_dir'] = tmp_path
    path = path.format(tmp_path=tmp_path)
    fix = get_allvars_fix('Amon', 'tas', session=session)
    fix.extra_facets[facet] = path

    cubes = CubeList([])
    with pytest.raises(FileNotFoundError, match='File'):
        fix.add_additional_cubes(cubes)


# Test _fix_height


@pytest.mark.parametrize('bounds', [True, False])
def test_fix_height_plev(bounds, simple_unstructured_cube):
    """Test fix."""
    cube = simple_unstructured_cube[:, 1:, :]
    pfull_cube = simple_unstructured_cube[:, 1:, :]
    pfull_cube.var_name = 'pfull'
    pfull_cube.units = 'Pa'
    cubes = CubeList([cube, pfull_cube])
    if bounds:
        phalf_cube = simple_unstructured_cube.copy()
        phalf_cube.var_name = 'phalf'
        phalf_cube.units = 'Pa'
        cubes.append(phalf_cube)
    fix = get_allvars_fix('Amon', 'ta')

    fixed_cube = fix._fix_height(cube, cubes)

    expected_data = [[[4.0, 5.0], [2.0, 3.0]]]
    np.testing.assert_allclose(fixed_cube.data, expected_data)

    height = check_model_level_metadata(fixed_cube)
    np.testing.assert_array_equal(height.points, [0, 1])
    assert height.bounds is None

    plev = check_air_pressure_metadata(fixed_cube)
    assert fixed_cube.coord_dims('air_pressure') == (0, 1, 2)
    np.testing.assert_allclose(plev.points, expected_data)
    if bounds:
        expected_bnds = [[[[4.0, 2.0], [5.0, 3.0]], [[2.0, 0.0], [3.0, 1.0]]]]
        np.testing.assert_allclose(plev.bounds, expected_bnds)
    else:
        assert plev.bounds is None


@pytest.mark.parametrize('bounds', [True, False])
def test_fix_height_alt16(bounds, simple_unstructured_cube):
    """Test fix."""
    cube = simple_unstructured_cube[:, 1:, :]
    zg_cube = simple_unstructured_cube[0, 1:, :]
    zg_cube.var_name = 'zg'
    zg_cube.units = 'm'
    cubes = CubeList([cube, zg_cube])
    if bounds:
        zghalf_cube = simple_unstructured_cube[0, :, :]
        zghalf_cube.var_name = 'zghalf'
        zghalf_cube.units = 'm'
        cubes.append(zghalf_cube)
    fix = get_allvars_fix('Amon', 'ta')

    fixed_cube = fix._fix_height(cube, cubes)

    expected_data = [[[4.0, 5.0], [2.0, 3.0]]]
    np.testing.assert_allclose(fixed_cube.data, expected_data)

    height = check_model_level_metadata(fixed_cube)
    np.testing.assert_array_equal(height.points, [0, 1])
    assert height.bounds is None

    assert fixed_cube.coords('altitude', dim_coords=False)
    alt16 = fixed_cube.coord('altitude', dim_coords=False)
    assert alt16.var_name == 'alt16'
    assert alt16.standard_name == 'altitude'
    assert alt16.long_name == 'altitude'
    assert alt16.units == 'm'
    assert alt16.attributes == {'positive': 'up'}
    assert fixed_cube.coord_dims('altitude') == (1, 2)
    np.testing.assert_allclose(alt16.points, expected_data[0])
    if bounds:
        expected_bnds = [[[4.0, 2.0], [5.0, 3.0]], [[2.0, 0.0], [3.0, 1.0]]]
        np.testing.assert_allclose(alt16.bounds, expected_bnds)
    else:
        assert alt16.bounds is None


# Test hfls (for extra fix)


def test_get_hfls_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'Amon', 'hfls')
    assert fix == [Hfls(None), AllVars(None), GenericFix(None)]


def test_hfls_fix(cubes_regular_grid):
    """Test fix."""
    cubes = CubeList([cubes_regular_grid[0].copy()])
    cubes[0].var_name = 'hfls'
    cubes[0].units = 'W m-2'

    fixed_cubes = fix_metadata(cubes, 'Amon', 'hfls')

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'hfls'
    assert cube.standard_name == 'surface_upward_latent_heat_flux'
    assert cube.long_name == 'Surface Upward Latent Heat Flux'
    assert cube.units == 'W m-2'
    assert cube.attributes['positive'] == 'up'

    fixed_cube = fix_data(cube, 'Amon', 'hfls')

    np.testing.assert_allclose(fixed_cube.data, [[[0.0, -1.0], [-2.0, -3.0]]])


# Test hfss (for extra fix)


def test_get_hfss_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'Amon', 'hfss')
    assert fix == [Hfss(None), AllVars(None), GenericFix(None)]


def test_hfss_fix(cubes_regular_grid):
    """Test fix."""
    cubes = CubeList([cubes_regular_grid[0].copy()])
    cubes[0].var_name = 'hfss'
    cubes[0].units = 'W m-2'

    fixed_cubes = fix_metadata(cubes, 'Amon', 'hfss')

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'hfss'
    assert cube.standard_name == 'surface_upward_sensible_heat_flux'
    assert cube.long_name == 'Surface Upward Sensible Heat Flux'
    assert cube.units == 'W m-2'
    assert cube.attributes['positive'] == 'up'

    fixed_cube = fix_data(cube, 'Amon', 'hfss')

    np.testing.assert_allclose(fixed_cube.data, [[[0.0, -1.0], [-2.0, -3.0]]])


# Test rtnt (for extra fix)


def test_get_rtnt_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'Amon', 'rtnt')
    assert fix == [Rtnt(None), AllVars(None), GenericFix(None)]


def test_rtnt_fix(cubes_regular_grid):
    """Test fix."""
    cubes = CubeList([
        cubes_regular_grid[0].copy(),
        cubes_regular_grid[0].copy(),
        cubes_regular_grid[0].copy()
    ])
    cubes[0].var_name = 'rsdt'
    cubes[1].var_name = 'rsut'
    cubes[2].var_name = 'rlut'
    cubes[0].units = 'W m-2'
    cubes[1].units = 'W m-2'
    cubes[2].units = 'W m-2'

    fixed_cubes = fix_metadata(cubes, 'Amon', 'rtnt')

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'rtnt'
    assert cube.standard_name is None
    assert cube.long_name == 'TOA Net downward Total Radiation'
    assert cube.units == 'W m-2'
    assert cube.attributes['positive'] == 'down'

    np.testing.assert_allclose(cube.data, [[[0.0, -1.0], [-2.0, -3.0]]])


# Test rtmt (for extra fix)


def test_get_rtmt_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'Amon', 'rtmt')
    assert fix == [Rtmt(None), AllVars(None), GenericFix(None)]


def test_rtmt_fix(cubes_regular_grid):
    """Test fix."""
    cubes = CubeList([
        cubes_regular_grid[0].copy(),
        cubes_regular_grid[0].copy(),
        cubes_regular_grid[0].copy()
    ])
    cubes[0].var_name = 'rsdt'
    cubes[1].var_name = 'rsut'
    cubes[2].var_name = 'rlut'
    cubes[0].units = 'W m-2'
    cubes[1].units = 'W m-2'
    cubes[2].units = 'W m-2'

    fixed_cubes = fix_metadata(cubes, 'Amon', 'rtmt')

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'rtmt'
    assert cube.standard_name == ('net_downward_radiative_flux_at_top_of'
                                  '_atmosphere_model')
    assert cube.long_name == 'Net Downward Radiative Flux at Top of Model'
    assert cube.units == 'W m-2'
    assert cube.attributes['positive'] == 'down'

    np.testing.assert_allclose(cube.data, [[[0.0, -1.0], [-2.0, -3.0]]])
