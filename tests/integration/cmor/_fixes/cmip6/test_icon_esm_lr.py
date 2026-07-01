"""Tests for the fixes of ICON-ESM-LR."""

import numpy as np
import pytest
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList

from esmvalcore.cmor._fixes.cmip6.icon_esm_lr import AllVars
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor.fix import Fix


@pytest.fixture
def cubes():
    """Cubes to test fix."""
    correct_lat_coord = AuxCoord(
        [0.0],
        var_name="lat",
        standard_name="latitude",
    )
    wrong_lat_coord = AuxCoord(
        [0.0],
        var_name="latitude",
        standard_name="latitude",
    )
    correct_lon_coord = AuxCoord(
        [0.0],
        var_name="lon",
        standard_name="longitude",
    )
    wrong_lon_coord = AuxCoord(
        [0.0],
        var_name="longitude",
        standard_name="longitude",
    )
    correct_cube = Cube(
        [10.0],
        var_name="tas",
        aux_coords_and_dims=[(correct_lat_coord, 0), (correct_lon_coord, 0)],
    )
    wrong_cube = Cube(
        [10.0],
        var_name="pr",
        aux_coords_and_dims=[(wrong_lat_coord, 0), (wrong_lon_coord, 0)],
    )
    return CubeList([correct_cube, wrong_cube])


def test_get_allvars_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "ICON-ESM-LR", "Amon", "tas")
    assert fix == [AllVars(None), GenericFix(None)]


def test_allvars_fix_metadata_lat_lon(cubes):
    """Test ``fix_metadata`` for all variables."""
    fix = AllVars(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        lat_coord = cube.coord("latitude")
        lon_coord = cube.coord("longitude")
        assert lat_coord.var_name == "lat"
        assert lon_coord.var_name == "lon"


def test_allvars_fix_metadata_lat(cubes):
    """Test ``fix_metadata`` for all variables."""
    for cube in cubes:
        cube.remove_coord("longitude")
    fix = AllVars(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        lat_coord = cube.coord("latitude")
        assert lat_coord.var_name == "lat"


def test_allvars_fix_metadata_lon(cubes):
    """Test ``fix_metadata`` for all variables."""
    for cube in cubes:
        cube.remove_coord("latitude")
    fix = AllVars(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        lon_coord = cube.coord("longitude")
        assert lon_coord.var_name == "lon"


def test_allvars_fix_metadata_no_lat_lon(cubes):
    """Test ``fix_metadata`` for all variables."""
    for cube in cubes:
        cube.remove_coord("latitude")
        cube.remove_coord("longitude")
    fix = AllVars(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes


def test_allvars_fix_metadata_adds_mesh():
    lat = AuxCoord(
        [0.5, 0.5],
        standard_name="latitude",
        var_name="latitude",
        units="degrees_north",
        bounds=[[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
    )
    lon = AuxCoord(
        [0.5, 0.5],
        standard_name="longitude",
        var_name="longitude",
        units="degrees_east",
        bounds=[[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    )
    cube = Cube(
        np.array([1.0, 2.0]),
        var_name="tas",
        aux_coords_and_dims=[(lat, 0), (lon, 0)],
    )

    AllVars(None).fix_metadata(CubeList([cube]))

    assert cube.mesh is not None
    assert cube.coords("latitude", mesh_coords=True)
    assert cube.coords("longitude", mesh_coords=True)
    assert cube.mesh.connectivity().shape == (2, 3)


def test_allvars_fix_metadata_merges_pole_vertices():
    """Ensure pole vertices are merged into a single mesh node.

    there must be 4 nodes: North Pole, (0,0), (0,120), (0,240)
    """
    lat = AuxCoord(
        [60.0, 60.0],
        standard_name="latitude",
        var_name="latitude",
        units="degrees_north",
        bounds=[[90.0, 0.0, 0.0], [90.0, 0.0, 0.0]],
    )
    lon = AuxCoord(
        [60.0, 180.0],
        standard_name="longitude",
        var_name="longitude",
        units="degrees_east",
        bounds=[[0.0, 0.0, 120.0], [240.0, 120.0, 240.0]],
    )
    cube = Cube(
        np.array([1.0, 2.0]),
        var_name="tas",
        aux_coords_and_dims=[(lat, 0), (lon, 0)],
    )

    AllVars(None).fix_metadata(CubeList([cube]))

    node_lat = cube.mesh.coord(location="node", axis="y")
    assert len(node_lat.points) == 4
