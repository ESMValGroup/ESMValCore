"""Test the ICON on-the-fly CMORizer."""

from datetime import datetime
from pathlib import Path
from unittest import mock

import iris
import numpy as np
import pytest
from cf_units import Unit
from iris import NameConstraint
from iris.coords import CellMethod, DimCoord
from iris.cube import Cube, CubeList

import esmvalcore.cmor._fixes.native6.oras5
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor._fixes.native6.oras5 import AllVars, Oras5Fix
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info
from esmvalcore.config import CFG
from esmvalcore.dataset import Dataset

TEST_GRID_FILE_URI = (
    "https://github.com/ESMValGroup/ESMValCore/raw/main/tests/integration/"
    "cmor/_fixes/test_data/oras5_grid.nc"
)
TEST_GRID_FILE_NAME = "oras5_grid.nc"


@pytest.fixture(autouse=True)
def tmp_cache_dir(monkeypatch, tmp_path):
    """Use temporary path as cache directory for all tests in this module."""
    monkeypatch.setattr(Oras5Fix, "CACHE_DIR", tmp_path)


# Note that test_data_path is defined in tests/integration/cmor/_fixes/conftest.py


@pytest.fixture
def cubes_2d(test_data_path):
    """2D sample cubes."""
    nc_path = test_data_path / "oras5_2d.nc"
    return iris.load(str(nc_path))


@pytest.fixture
def cubes_3d(test_data_path):
    """3D sample cubes."""
    nc_path = test_data_path / "oras5_3d.nc"
    return iris.load(str(nc_path))


@pytest.fixture
def cubes_grid(test_data_path):
    """Grid description sample cubes."""
    nc_path = test_data_path / "oras5_grid.nc"
    return iris.load(str(nc_path))


def _get_fix(mip, short_name, fix_name, session=None):
    """Load a fix from esmvalcore.cmor._fixes.native6.oras5."""
    dataset = Dataset(
        project="native6",
        dataset="ORAS5",
        mip=mip,
        short_name=short_name,
    )
    extra_facets = dataset._get_extra_facets()
    extra_facets["frequency"] = "mon"
    extra_facets["exp"] = "omip"
    test_data_path = Path(__file__).resolve().parent.parent / "test_data"
    extra_facets["horizontal_grid"] = str(test_data_path / "oras5_grid.nc")
    extra_facets["ugrid"] = True
    vardef = get_var_info(project="native6", mip=mip, short_name=short_name)
    cls = getattr(esmvalcore.cmor._fixes.native6.oras5, fix_name)
    return cls(vardef, extra_facets=extra_facets, session=session)


def get_fix(mip, short_name, session=None):
    """Load a variable fix from esmvalcore.cmor._fixes.native6.oras5."""
    fix_name = short_name[0].upper() + short_name[1:]
    return _get_fix(mip, short_name, fix_name, session=session)


def get_allvars_fix(mip, short_name, session=None):
    """Load the AllVars fix from esmvalcore.cmor._fixes.native6.oras5."""
    return _get_fix(mip, short_name, "AllVars", session=session)


def fix_metadata(cubes, mip, short_name, session=None):
    """Fix metadata of cubes."""
    fix = get_fix(mip, short_name, session=session)
    cubes = fix.fix_metadata(cubes)
    fix = get_allvars_fix(mip, short_name, session=session)
    return fix.fix_metadata(cubes)


def fix_data(cube, mip, short_name, session=None):
    """Fix data of cube."""
    fix = get_fix(mip, short_name, session=session)
    cube = fix.fix_data(cube)
    fix = get_allvars_fix(mip, short_name, session=session)
    return fix.fix_data(cube)


def check_thetao_metadata(cubes):
    """Check thetao metadata."""
    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == "thetao"
    assert cube.standard_name == "sea_water_potential_temperature"
    assert cube.long_name == "Sea Water Potential Temperature"
    assert cube.units == "degC"
    assert "positive" not in cube.attributes
    return cube


def check_tos_metadata(cubes):
    """Check tos metadata."""
    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == "tos"
    assert cube.standard_name == "sea_surface_temperature"
    assert cube.long_name == "Sea Surface Temperature"
    assert cube.units == "degC"
    # assert "positive" not in cube.attributes
    return cube


def check_time(cube):
    """Check time coordinate of cube."""
    assert cube.coords("time", dim_coords=True)
    time = cube.coord("time", dim_coords=True)
    assert time.var_name == "time"
    assert time.standard_name == "time"
    assert time.long_name == "time"
    assert time.attributes == {}


def check_model_level_metadata(cube):
    """Check metadata of model_level coordinate."""
    assert cube.coords("depth", dim_coords=True)
    height = cube.coord("depth", dim_coords=True)
    assert height.var_name == "lev"
    assert height.standard_name is None
    assert height.long_name == "model level number"
    assert height.units == "m"
    assert height.attributes == {"positive": "down"}
    return height


def check_air_pressure_metadata(cube):
    """Check metadata of air_pressure coordinate."""
    assert cube.coords("air_pressure", dim_coords=False)
    plev = cube.coord("air_pressure", dim_coords=False)
    assert plev.var_name == "plev"
    assert plev.standard_name == "air_pressure"
    assert plev.long_name == "pressure"
    assert plev.units == "Pa"
    assert plev.attributes == {"positive": "down"}
    return plev


def check_lat(cube):
    """Check latitude coordinate of cube."""
    assert cube.coords("latitude", dim_coords=False)
    lat = cube.coord("latitude", dim_coords=False)
    assert lat.var_name == "lat"
    assert lat.standard_name == "latitude"
    assert lat.long_name == "latitude"
    assert lat.units == "degrees_north"
    return lat


def check_lon(cube):
    """Check longitude coordinate of cube."""
    assert cube.coords("longitude", dim_coords=False)
    lon = cube.coord("longitude", dim_coords=False)
    assert lon.var_name == "lon"
    assert lon.standard_name == "longitude"
    assert lon.long_name == "longitude"
    assert lon.units == "degrees_east"
    return lon


def check_lat_lon(cube):
    """Check latitude, longitude and mesh of cube."""
    lat = check_lat(cube)
    lon = check_lon(cube)

    # Check that latitude and longitude are mesh coordinates
    assert cube.coords("latitude", mesh_coords=True)
    assert cube.coords("longitude", mesh_coords=True)

    # Check dimensional coordinate describing the mesh
    assert cube.coords(
        "first spatial index for variables stored on an unstructured grid",
        dim_coords=True,
    )
    i_coord = cube.coord(
        "first spatial index for variables stored on an unstructured grid",
        dim_coords=True,
    )
    assert i_coord.var_name == "i"
    assert i_coord.standard_name is None
    assert i_coord.long_name == (
        "first spatial index for variables stored on an unstructured grid"
    )
    assert i_coord.units == "1"
    np.testing.assert_allclose(i_coord.points, list(range(13 * 12)))
    assert i_coord.bounds is None

    assert len(cube.coord_dims(lat)) == 1
    assert cube.coord_dims(lat) == cube.coord_dims(lon)
    assert cube.coord_dims(lat) == cube.coord_dims(i_coord)

    # Check the mesh itself
    assert cube.location == "face"
    mesh = cube.mesh
    check_mesh(mesh)
    return lat, lon


def check_mesh(mesh):
    """Check the mesh."""
    assert mesh is not None
    assert mesh.var_name is None
    assert mesh.standard_name is None
    assert mesh.long_name is None
    assert mesh.units == "unknown"
    assert mesh.attributes == {}
    assert mesh.cf_role == "mesh_topology"
    assert mesh.topology_dimension == 2

    # Check face coordinates
    assert len(mesh.coords(location="face")) == 2

    mesh_face_lat = mesh.coord(location="face", axis="y")
    assert mesh_face_lat.var_name == "lat"
    assert mesh_face_lat.standard_name == "latitude"
    assert mesh_face_lat.long_name == "latitude"
    assert mesh_face_lat.units == "degrees_north"
    assert mesh_face_lat.attributes == {}

    mesh_face_lon = mesh.coord(location="face", axis="x")
    assert mesh_face_lon.var_name == "lon"
    assert mesh_face_lon.standard_name == "longitude"
    assert mesh_face_lon.long_name == "longitude"
    assert mesh_face_lon.units == "degrees_east"
    assert mesh_face_lon.attributes == {}

    # Check node coordinates
    assert len(mesh.coords(location="node")) == 2

    mesh_node_lat = mesh.coord(location="node", axis="y")
    assert mesh_node_lat.var_name == "nlat"
    assert mesh_node_lat.standard_name == "latitude"
    assert mesh_node_lat.long_name == "node latitude"
    assert mesh_node_lat.units == "degrees_north"
    assert mesh_node_lat.attributes == {}
    assert mesh_node_lat.bounds is None

    mesh_node_lon = mesh.coord(location="node", axis="x")
    assert mesh_node_lon.var_name == "nlon"
    assert mesh_node_lon.standard_name == "longitude"
    assert mesh_node_lon.long_name == "node longitude"
    assert mesh_node_lon.units == "degrees_east"
    assert mesh_node_lon.attributes == {}
    assert mesh_node_lon.bounds is None

    # Check connectivity
    assert len(mesh.connectivities()) == 1
    conn = mesh.connectivity()
    assert conn.var_name is None
    assert conn.standard_name is None
    assert conn.long_name is None
    assert conn.units == "unknown"
    assert conn.attributes == {}
    assert conn.cf_role == "face_node_connectivity"
    assert conn.start_index == 0
    assert conn.location_axis == 0
    assert conn.shape == ((13 * 12), 4)


def test_get_thetao_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("native6", "ORAS5", "Omon", "thetao")
    assert fix == [AllVars(None), GenericFix(None)]


def test_thetao_fix(cubes_3d):
    """Test fix."""
    fix = get_allvars_fix("Omon", "thetao")
    fixed_cubes = fix.fix_metadata(cubes_3d)

    cube = check_thetao_metadata(fixed_cubes)
    check_time(cube)
    check_lat_lon(cube)
    assert cube.shape == (1, 75, 13 * 12)


def test_get_tos_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("native6", "ORAS5", "Omon", "tos")
    assert fix == [AllVars(None), GenericFix(None)]


def test_tos_fix(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix("Omon", "tos")
    fixed_cubes = fix.fix_metadata(cubes_2d)

    cube = check_tos_metadata(fixed_cubes)
    check_time(cube)
    lat, lon = check_lat_lon(cube)

    assert cube.coords("latitude", dim_coords=False)
    assert cube.coords("longitude", dim_coords=False)
    assert len(cube.coord_dims(lat)) == 1
    assert len(cube.coord_dims(lon)) == 1
    assert cube.shape == (1, 13 * 12)


def test_tos_no_mesh(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix("Omon", "tos")
    fix.extra_facets["ugrid"] = False
    fixed_cubes = fix.fix_metadata(cubes_2d)

    cube = check_tos_metadata(fixed_cubes)

    assert cube.mesh is None

    lat = check_lat(cube)
    lon = check_lon(cube)

    assert cube.coords("latitude", dim_coords=False)
    assert cube.coords("longitude", dim_coords=False)
    assert len(cube.coord_dims(lat)) == 2
    assert len(cube.coord_dims(lon)) == 2
    assert cube.shape == (1, 13, 12)


def test_tos_no_mesh_unstructured(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix("Omon", "tos")
    fix.extra_facets["ugrid"] = False
    fix.extra_facets["make_unstructured"] = True
    fixed_cubes = fix.fix_metadata(cubes_2d)

    cube = check_tos_metadata(fixed_cubes)

    assert cube.mesh is None

    lat = check_lat(cube)
    lon = check_lon(cube)

    assert cube.coords("latitude", dim_coords=False)
    assert cube.coords("longitude", dim_coords=False)
    assert len(cube.coord_dims(lat)) == 1
    assert len(cube.coord_dims(lon)) == 1
    assert cube.shape == (1, 13 * 12)


def test_empty_standard_name_fix(cubes_2d, monkeypatch):
    """Test fix."""
    fix = get_allvars_fix("Omon", "tos")
    # We know that tas has a standard name, but this being native model output
    # there may be variables with no standard name. The code is designed to
    # handle this gracefully and here we test it with an artificial, but
    # realistic case.
    monkeypatch.setattr(fix.vardef, "standard_name", "")
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == "tos"
    assert cube.standard_name is None
    assert cube.long_name == "Sea Surface Temperature"
    assert cube.units == "degC"
    assert "positive" not in cube.attributes


# Test automatic addition of missing coordinates


def test_add_time(cubes_2d, cubes_3d):
    """Test fix."""
    # Remove time from tas cube to test automatic addition
    tos_cube = cubes_2d.extract_cube(NameConstraint(var_name="sosstsst"))
    thetao_cube = cubes_3d.extract_cube(NameConstraint(var_name="votemper"))
    tos_cube = tos_cube[0]
    tos_cube.remove_coord("time")
    cubes = CubeList([tos_cube, thetao_cube])

    fix = get_allvars_fix("Omon", "tos")
    fixed_cubes = fix.fix_metadata(cubes)
    cube = check_tos_metadata(fixed_cubes)
    assert cube.shape == (1, 13 * 12)
    check_time(cube)


def test_add_time_fail():
    """Test fix."""
    fix = get_allvars_fix("Omon", "tos")
    cube = Cube(1, var_name="sosstsst", units="degC")
    cubes = CubeList(
        [
            cube,
            Cube(1, var_name="sosstsst", units="degC"),
        ],
    )
    msg = "Cannot add required coordinate 'time' to variable 'tos'"
    with pytest.raises(ValueError, match=msg):
        fix._add_time(cube, cubes)


@mock.patch.object(Oras5Fix, "_get_grid_from_cube_attr", autospec=True)
def test_get_horizontal_grid_from_facet_cached_in_dict(
    mock_get_grid_from_cube_attr,
    tmp_path,
):
    """Test fix."""
    session = CFG.start_session("my session")
    session["auxiliary_data_dir"] = tmp_path

    # Save temporary grid file (this will not be used; however, it is necessary
    # to not raise a FileNotFoundError)
    grid_path = "grid.nc"
    wrong_grid_cube = Cube(0, var_name="wrong_grid")
    iris.save(wrong_grid_cube, tmp_path / "grid.nc")

    # Make sure that grid specified by cube attribute is NOT used
    cube = Cube(0, attributes={"grid_file_uri": "cached_grid_url.nc"})
    grid_cube = Cube(0, var_name="grid")
    fix = get_allvars_fix("Omon", "tos", session=session)
    fix.extra_facets["horizontal_grid"] = grid_path
    fix._horizontal_grids["cached_grid_url.nc"] = mock.sentinel.wrong_grid
    fix._horizontal_grids[grid_path] = grid_cube

    grid = fix.get_horizontal_grid(cube)
    assert len(fix._horizontal_grids) == 2
    assert "cached_grid_url.nc" in fix._horizontal_grids  # has not been used
    assert grid_path in fix._horizontal_grids
    assert fix._horizontal_grids[grid_path] == grid
    assert grid is grid_cube
    mock_get_grid_from_cube_attr.assert_not_called()


@pytest.mark.parametrize("grid_path", ["{tmp_path}/grid.nc", "grid.nc"])
@mock.patch.object(Oras5Fix, "_get_grid_from_cube_attr", autospec=True)
def test_get_horizontal_grid_from_facet(
    mock_get_grid_from_cube_attr,
    grid_path,
    tmp_path,
):
    """Test fix."""
    session = CFG.start_session("my session")
    session["auxiliary_data_dir"] = tmp_path

    # Make sure that grid specified by cube attribute is NOT used
    cube = Cube(0, attributes={"grid_file_uri": "cached_grid_url.nc"})

    # Save temporary grid file
    grid_path = grid_path.format(tmp_path=tmp_path)
    grid_cube = Cube(0, var_name="grid")
    iris.save(grid_cube, tmp_path / "grid.nc")

    fix = get_allvars_fix("Omon", "tos", session=session)
    fix.extra_facets["horizontal_grid"] = grid_path
    fix._horizontal_grids["cached_grid_url.nc"] = mock.sentinel.wrong_grid

    grid = fix.get_horizontal_grid(cube)
    assert isinstance(grid, CubeList)
    assert len(grid) == 1
    assert grid[0].var_name == "grid"
    assert len(fix._horizontal_grids) == 2
    assert "cached_grid_url.nc" in fix._horizontal_grids  # has not been used
    assert "grid.nc" in fix._horizontal_grids
    assert fix._horizontal_grids["grid.nc"] == grid
    mock_get_grid_from_cube_attr.assert_not_called()


def test_get_horizontal_grid_from_facet_fail(tmp_path):
    """Test fix."""
    session = CFG.start_session("my session")
    session["auxiliary_data_dir"] = tmp_path

    cube = Cube(0)
    fix = get_allvars_fix("Omon", "tos", session=session)
    fix.extra_facets["horizontal_grid"] = "/this/does/not/exist.nc"

    with pytest.raises(FileNotFoundError):
        fix.get_horizontal_grid(cube)


def test_get_horizontal_grid_none(tmp_path):
    """Test fix."""
    session = CFG.start_session("my session")
    session["auxiliary_data_dir"] = tmp_path

    cube = Cube(0)
    fix = get_allvars_fix("Omon", "tos", session=session)
    del fix.extra_facets["horizontal_grid"]

    msg = "Full path to suitable ORAS5 grid must be specified in facet 'horizontal_grid'"
    with pytest.raises(NotImplementedError, match=msg):
        fix.get_horizontal_grid(cube)


@pytest.mark.parametrize(
    ("frequency", "dt_in", "dt_out", "bounds"),
    [
        (
            "dec",
            [(2000, 1, 1)],
            [(1995, 1, 1)],
            [[(1990, 1, 1), (2000, 1, 1)]],
        ),
        (
            "yr",
            [(2000, 1, 1), (2001, 1, 1)],
            [(1999, 7, 2, 12), (2000, 7, 2)],
            [[(1999, 1, 1), (2000, 1, 1)], [(2000, 1, 1), (2001, 1, 1)]],
        ),
        (
            "mon",
            [(2000, 1, 1)],
            [(1999, 12, 16, 12)],
            [[(1999, 12, 1), (2000, 1, 1)]],
        ),
        (
            "mon",
            [(2000, 11, 30, 23, 45), (2000, 12, 31, 23)],
            [(2000, 11, 16), (2000, 12, 16, 12)],
            [[(2000, 11, 1), (2000, 12, 1)], [(2000, 12, 1), (2001, 1, 1)]],
        ),
        (
            "day",
            [(2000, 1, 1, 12)],
            [(2000, 1, 1)],
            [[(1999, 12, 31, 12), (2000, 1, 1, 12)]],
        ),
        (
            "6hr",
            [(2000, 1, 5, 14), (2000, 1, 5, 20)],
            [(2000, 1, 5, 11), (2000, 1, 5, 17)],
            [
                [(2000, 1, 5, 8), (2000, 1, 5, 14)],
                [(2000, 1, 5, 14), (2000, 1, 5, 20)],
            ],
        ),
        (
            "3hr",
            [(2000, 1, 1)],
            [(1999, 12, 31, 22, 30)],
            [[(1999, 12, 31, 21), (2000, 1, 1)]],
        ),
        (
            "1hr",
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
    cube = Cube(0, cell_methods=[CellMethod("mean", "time")])
    datetimes = [datetime(*dt) for dt in dt_in]
    time_units = Unit("days since 1950-01-01", calendar="proleptic_gregorian")
    time_coord = DimCoord(
        time_units.date2num(datetimes),
        standard_name="time",
        var_name="time",
        long_name="time",
        units=time_units,
    )

    fix = get_allvars_fix("Omon", "tos")
    fix.extra_facets["frequency"] = frequency

    fix._shift_time_coord(cube, time_coord)

    dt_out = [datetime(*dt) for dt in dt_out]
    bounds = [[datetime(*dt1), datetime(*dt2)] for (dt1, dt2) in bounds]
    np.testing.assert_allclose(
        time_coord.points,
        time_coord.units.date2num(dt_out),
    )
    np.testing.assert_allclose(
        time_coord.bounds,
        time_coord.units.date2num(bounds),
    )


@pytest.mark.parametrize(
    ("frequency", "dt_in"),
    [
        ("dec", [(2000, 1, 15)]),
        ("yr", [(2000, 1, 1), (2001, 1, 1)]),
        ("mon", [(2000, 6, 15)]),
        ("day", [(2000, 1, 1), (2001, 1, 2)]),
        ("6hr", [(2000, 6, 15, 12)]),
        ("3hr", [(2000, 1, 1, 4), (2000, 1, 1, 7)]),
        ("1hr", [(2000, 1, 1, 4), (2000, 1, 1, 5)]),
    ],
)
def test_shift_time_point_measurement(frequency, dt_in):
    """Test ``_shift_time_coord``."""
    cube = Cube(0, cell_methods=[CellMethod("point", "time")])
    datetimes = [datetime(*dt) for dt in dt_in]
    time_units = Unit("days since 1950-01-01", calendar="proleptic_gregorian")
    time_coord = DimCoord(
        time_units.date2num(datetimes),
        standard_name="time",
        var_name="time",
        long_name="time",
        units=time_units,
    )

    fix = get_allvars_fix("Omon", "tos")
    fix.extra_facets["frequency"] = frequency

    fix._shift_time_coord(cube, time_coord)

    np.testing.assert_allclose(
        time_coord.points,
        time_coord.units.date2num(datetimes),
    )
    assert time_coord.bounds is None


@pytest.mark.parametrize(
    "frequency",
    ["dec", "yr", "yrPt", "mon", "monC", "monPt"],
)
def test_shift_time_coord_hourly_data_low_freq_fail(frequency):
    """Test ``_shift_time_coord``."""
    cube = Cube(0, cell_methods=[CellMethod("mean", "time")])
    time_units = Unit("hours since 1950-01-01", calendar="proleptic_gregorian")
    time_coord = DimCoord(
        [1, 2, 3],
        standard_name="time",
        var_name="time",
        long_name="time",
        units=time_units,
    )

    fix = get_allvars_fix("Omon", "tos")
    fix.extra_facets["frequency"] = frequency

    msg = "Cannot shift time coordinate: Rounding to closest day failed."
    with pytest.raises(ValueError, match=msg):
        fix._shift_time_coord(cube, time_coord)


@pytest.mark.parametrize(
    "frequency",
    ["dec", "yr", "yrPt", "mon", "monC", "monPt"],
)
def test_shift_time_coord_not_first_of_month(frequency):
    """Test ``_get_previous_timestep``."""
    cube = Cube(0, cell_methods=[CellMethod("mean", "time")])
    time_units = Unit("days since 1950-01-01", calendar="proleptic_gregorian")
    time_coord = DimCoord(
        [1.5],
        standard_name="time",
        var_name="time",
        long_name="time",
        units=time_units,
    )
    fix = get_allvars_fix("Omon", "tos")
    fix.extra_facets["frequency"] = frequency

    msg = (
        "Cannot shift time coordinate: expected first of the month at 00:00:00"
    )
    with pytest.raises(ValueError, match=msg):
        fix._shift_time_coord(cube, time_coord)


@pytest.mark.parametrize("frequency", ["fx", "subhrPt", "invalid_freq"])
def test_shift_time_coord_invalid_freq(frequency):
    """Test ``_get_previous_timestep``."""
    cube = Cube(0, cell_methods=[CellMethod("mean", "time")])
    time_units = Unit("days since 1950-01-01", calendar="proleptic_gregorian")
    time_coord = DimCoord(
        [1.5, 2.5],
        standard_name="time",
        var_name="time",
        long_name="time",
        units=time_units,
    )
    fix = get_allvars_fix("Omon", "tos")
    fix.extra_facets["frequency"] = frequency

    msg = (
        "Cannot shift time coordinate: failed to determine previous time step"
    )
    with pytest.raises(ValueError, match=msg):
        fix._shift_time_coord(cube, time_coord)


# Test _get_previous_timestep


@pytest.mark.parametrize(
    ("frequency", "datetime_in", "datetime_out"),
    [
        ("dec", (2000, 1, 1), (1990, 1, 1)),
        ("yr", (2000, 1, 1), (1999, 1, 1)),
        ("yrPt", (2001, 6, 1), (2000, 6, 1)),
        ("mon", (2001, 1, 1), (2000, 12, 1)),
        ("mon", (2001, 2, 1), (2001, 1, 1)),
        ("mon", (2001, 3, 1), (2001, 2, 1)),
        ("mon", (2001, 4, 1), (2001, 3, 1)),
        ("monC", (2000, 5, 1), (2000, 4, 1)),
        ("monC", (2000, 6, 1), (2000, 5, 1)),
        ("monC", (2000, 7, 1), (2000, 6, 1)),
        ("monC", (2000, 8, 1), (2000, 7, 1)),
        ("monPt", (2002, 9, 1), (2002, 8, 1)),
        ("monPt", (2002, 10, 1), (2002, 9, 1)),
        ("monPt", (2002, 11, 1), (2002, 10, 1)),
        ("monPt", (2002, 12, 1), (2002, 11, 1)),
        ("day", (2000, 1, 1), (1999, 12, 31)),
        ("day", (2000, 3, 1), (2000, 2, 29)),
        ("day", (2187, 3, 14), (2187, 3, 13)),
        ("6hr", (2000, 3, 14, 15), (2000, 3, 14, 9)),
        ("6hrPt", (2000, 1, 1), (1999, 12, 31, 18)),
        ("6hrCM", (2000, 1, 1, 1), (1999, 12, 31, 19)),
        ("3hr", (2000, 3, 14, 15), (2000, 3, 14, 12)),
        ("3hrPt", (2000, 1, 1), (1999, 12, 31, 21)),
        ("3hrCM", (2000, 1, 1, 1), (1999, 12, 31, 22)),
        ("1hr", (2000, 3, 14, 15), (2000, 3, 14, 14)),
        ("1hrPt", (2000, 1, 1), (1999, 12, 31, 23)),
        ("1hrCM", (2000, 1, 1, 1), (2000, 1, 1)),
        ("hr", (2000, 3, 14), (2000, 3, 13, 23)),
    ],
)
def test_get_previous_timestep(frequency, datetime_in, datetime_out):
    """Test ``_get_previous_timestep``."""
    datetime_in = datetime(*datetime_in)
    datetime_out = datetime(*datetime_out)

    fix = get_allvars_fix("Omon", "tos")
    fix.extra_facets["frequency"] = frequency

    new_datetime = fix._get_previous_timestep(datetime_in)

    assert new_datetime == datetime_out


def test_get_grid_url():
    """Test fix."""
    cube = Cube(0, attributes={"grid_file_uri": TEST_GRID_FILE_URI})
    fix = get_allvars_fix("Omon", "tos")
    (grid_url, grid_name) = fix._get_grid_url(cube)
    assert grid_url == TEST_GRID_FILE_URI
    assert grid_name == TEST_GRID_FILE_NAME


def test_get_grid_url_fail():
    """Test fix."""
    cube = Cube(0)
    fix = get_allvars_fix("Omon", "tos")
    msg = (
        "Cube does not contain the attribute 'grid_file_uri' necessary to "
        "download the ICON horizontal grid file"
    )
    with pytest.raises(ValueError, match=msg):
        fix._get_grid_url(cube)


# Test get_mesh


def test_get_mesh_cached_from_attr(monkeypatch):
    """Test fix."""
    cube = Cube(0, attributes={"grid_file_uri": TEST_GRID_FILE_URI})
    fix = get_allvars_fix("Omon", "tos")
    monkeypatch.setattr(fix, "_create_mesh", mock.Mock())
    fix._meshes[TEST_GRID_FILE_NAME] = mock.sentinel.mesh
    mesh = fix.get_mesh(cube)
    assert mesh == mock.sentinel.mesh
    fix._create_mesh.assert_not_called()


def test_get_mesh_not_cached_from_attr(monkeypatch):
    """Test fix."""
    cube = Cube(0, attributes={"grid_file_uri": TEST_GRID_FILE_URI})
    fix = get_allvars_fix("Omon", "tos")
    monkeypatch.setattr(fix, "_create_mesh", mock.Mock())
    fix.get_mesh(cube)
    fix._create_mesh.assert_called_once_with(cube)


def test_get_mesh_cached_from_facet(monkeypatch, tmp_path):
    """Test fix."""
    session = CFG.start_session("my session")
    session["auxiliary_data_dir"] = tmp_path

    # Save temporary grid file (this will not be used; however, it is necessary
    # to not raise a FileNotFoundError)
    grid_path = "grid.nc"
    grid_cube = Cube(0, var_name="grid")
    iris.save(grid_cube, tmp_path / "grid.nc")

    cube = Cube(0, attributes={"grid_file_uri": TEST_GRID_FILE_URI})
    fix = get_allvars_fix("Omon", "tos", session=session)
    fix.extra_facets["horizontal_grid"] = grid_path
    monkeypatch.setattr(fix, "_create_mesh", mock.Mock())
    fix._meshes[TEST_GRID_FILE_NAME] = mock.sentinel.wrong_mesh
    fix._meshes["grid.nc"] = mock.sentinel.mesh

    mesh = fix.get_mesh(cube)

    assert mesh == mock.sentinel.mesh
    fix._create_mesh.assert_not_called()


def test_get_mesh_not_cached_from_facet(monkeypatch, tmp_path):
    """Test fix."""
    session = CFG.start_session("my session")
    session["auxiliary_data_dir"] = tmp_path

    # Save temporary grid file (this will not be used; however, it is necessary
    # to not raise a FileNotFoundError)
    grid_path = "grid.nc"
    grid_cube = Cube(0, var_name="grid")
    iris.save(grid_cube, tmp_path / "grid.nc")

    cube = Cube(0, attributes={"grid_file_uri": TEST_GRID_FILE_URI})
    fix = get_allvars_fix("Omon", "tos", session=session)
    fix.extra_facets["horizontal_grid"] = grid_path
    monkeypatch.setattr(fix, "_create_mesh", mock.Mock())
    fix._meshes[TEST_GRID_FILE_NAME] = mock.sentinel.wrong_mesh

    fix.get_mesh(cube)

    fix._create_mesh.assert_called_once_with(cube)


def test_get_bounds_cached_from_facet(cubes_2d, cubes_3d):
    """Test fix."""
    tos_cube = cubes_2d.extract_cube(NameConstraint(var_name="sosstsst"))
    tos_cube2 = tos_cube.copy()
    cubes = CubeList([tos_cube, tos_cube2])

    fix = get_allvars_fix("Omon", "tos")
    fix.extra_facets["ugrid"] = False
    fixed_cubes = []
    for i in range(len(cubes)):
        fixed_cubes.append(fix.fix_metadata(CubeList([cubes[i]]))[0])
    fixed_cubes = CubeList(fixed_cubes)

    assert fixed_cubes[0].coord("latitude") == fixed_cubes[1].coord("latitude")
    assert fixed_cubes[0].coord("longitude") == fixed_cubes[1].coord(
        "longitude",
    )
    assert (
        fixed_cubes[0].coord("latitude").bounds
        == fixed_cubes[1].coord("latitude").bounds
    ).all()
    assert (
        fixed_cubes[0].coord("latitude").points
        == fixed_cubes[1].coord("latitude").points
    ).all()
    assert (
        fixed_cubes[0].coord("longitude").bounds
        == fixed_cubes[1].coord("longitude").bounds
    ).all()
    assert (
        fixed_cubes[0].coord("longitude").points
        == fixed_cubes[1].coord("longitude").points
    ).all()


def test_get_coord_cached_from_facet(cubes_2d, cubes_3d):
    """Test fix."""
    tos_cube = cubes_2d.extract_cube(NameConstraint(var_name="sosstsst"))
    tos_cube2 = tos_cube.copy()
    cubes = CubeList([tos_cube, tos_cube2])

    fix = get_allvars_fix("Omon", "tos")
    fixed_cubes = []
    for i in range(len(cubes)):
        fixed_cubes.append(fix.fix_metadata(CubeList([cubes[i]]))[0])
    fixed_cubes = CubeList(fixed_cubes)

    assert fixed_cubes[0].coord("latitude") == fixed_cubes[1].coord("latitude")
    assert fixed_cubes[0].coord("longitude") == fixed_cubes[1].coord(
        "longitude",
    )
    assert (
        fixed_cubes[0].coord("latitude").bounds
        == fixed_cubes[1].coord("latitude").bounds
    ).all()
    assert (
        fixed_cubes[0].coord("latitude").points
        == fixed_cubes[1].coord("latitude").points
    ).all()
    assert (
        fixed_cubes[0].coord("longitude").bounds
        == fixed_cubes[1].coord("longitude").bounds
    ).all()
    assert (
        fixed_cubes[0].coord("longitude").points
        == fixed_cubes[1].coord("longitude").points
    ).all()


# Test _get_path_from_facet


@pytest.mark.parametrize(
    ("path", "description", "output"),
    [
        ("{tmp_path}/a.nc", None, "{tmp_path}/a.nc"),
        ("b.nc", "Grid file", "{tmp_path}/b.nc"),
    ],
)
def test_get_path_from_facet(path, description, output, tmp_path):
    """Test fix."""
    session = CFG.start_session("my session")
    session["auxiliary_data_dir"] = tmp_path
    path = path.format(tmp_path=tmp_path)
    fix = get_allvars_fix("Omon", "tos", session=session)
    fix.extra_facets["test_path"] = path

    # Create empty dummy file
    output = output.format(tmp_path=tmp_path)
    with open(output, "w", encoding="utf-8"):
        pass

    out_path = fix._get_path_from_facet("test_path", description=description)

    assert isinstance(out_path, Path)
    assert out_path == Path(output.format(tmp_path=tmp_path))


@pytest.mark.parametrize(
    ("path", "description"),
    [
        ("{tmp_path}/a.nc", None),
        ("b.nc", "Grid file"),
    ],
)
def test_get_path_from_facet_fail(path, description, tmp_path):
    """Test fix."""
    session = CFG.start_session("my session")
    session["auxiliary_data_dir"] = tmp_path
    path = path.format(tmp_path=tmp_path)
    fix = get_allvars_fix("Omon", "tos", session=session)
    fix.extra_facets["test_path"] = path

    with pytest.raises(FileNotFoundError, match=description):
        fix._get_path_from_facet("test_path", description=description)


# Test add_additional_cubes


@pytest.mark.parametrize("facet", ["zg_file", "zghalf_file"])
@pytest.mark.parametrize("path", ["{tmp_path}/a.nc", "a.nc"])
def test_add_additional_cubes(path, facet, tmp_path):
    """Test fix."""
    session = CFG.start_session("my session")
    session["auxiliary_data_dir"] = tmp_path
    path = path.format(tmp_path=tmp_path)
    fix = get_allvars_fix("Omon", "tos", session=session)
    fix.extra_facets[facet] = path

    # Save temporary cube
    cube = Cube(0, var_name=facet)
    iris.save(cube, tmp_path / "a.nc")

    cubes = CubeList([])
    new_cubes = fix.add_additional_cubes(cubes)

    assert new_cubes is cubes
    assert len(cubes) == 1
    assert cubes[0].var_name == facet


@pytest.mark.parametrize("facet", ["zg_file", "zghalf_file"])
@pytest.mark.parametrize("path", ["{tmp_path}/a.nc", "a.nc"])
def test_add_additional_cubes_fail(path, facet, tmp_path):
    """Test fix."""
    session = CFG.start_session("my session")
    session["auxiliary_data_dir"] = tmp_path
    path = path.format(tmp_path=tmp_path)
    fix = get_allvars_fix("Omon", "tos", session=session)
    fix.extra_facets[facet] = path

    cubes = CubeList([])
    with pytest.raises(FileNotFoundError, match="File"):
        fix.add_additional_cubes(cubes)
