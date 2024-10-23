"""Tests for the fixes of CESM2."""

import iris
import numpy as np
import pytest
from cf_units import Unit

from esmvalcore.cmor._fixes.cmip6.cesm2 import (
    Cl,
    Cli,
    Clw,
    Fgco2,
    Omon,
    Siconc,
    Tas,
    Tos,
)
from esmvalcore.cmor._fixes.common import SiconcFixScalarCoord
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2", "Amon", "cl")
    assert fix == [Cl(None), GenericFix(None)]


AIR_PRESSURE_POINTS = np.array(
    [
        [
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            [
                [2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0],
            ],
        ]
    ]
)
AIR_PRESSURE_BOUNDS = np.array(
    [
        [
            [
                [[0.0, 1.5], [-1.0, 2.0], [-2.0, 2.5], [-3.0, 3.0]],
                [[-4.0, 3.5], [-5.0, 4.0], [-6.0, 4.5], [-7.0, 5.0]],
                [[-8.0, 5.5], [-9.0, 6.0], [-10.0, 6.5], [-11.0, 7.0]],
            ],
            [
                [[1.5, 3.0], [2.0, 5.0], [2.5, 7.0], [3.0, 9.0]],
                [[3.5, 11.0], [4.0, 13.0], [4.5, 15.0], [5.0, 17.0]],
                [[5.5, 19.0], [6.0, 21.0], [6.5, 23.0], [7.0, 25.0]],
            ],
        ]
    ]
)


def test_cl_fix_file(tmp_path, test_data_path):
    """Test ``fix_file`` for ``cl``."""
    nc_path = test_data_path / "cesm2_cl.nc"
    cubes = iris.load(str(nc_path))

    # Raw cubes
    assert len(cubes) == 5
    var_names = [cube.var_name for cube in cubes]
    assert "cl" in var_names
    assert "a" in var_names
    assert "b" in var_names
    assert "p0" in var_names
    assert "ps" in var_names

    # Raw cl cube
    raw_cube = cubes.extract_cube("cloud_area_fraction_in_atmosphere_layer")
    assert not raw_cube.coords("air_pressure")

    # Apply fix
    vardef = get_var_info("CMIP6", "Amon", "cl")
    fix = Cl(vardef)
    (fixed_cl_cube,) = fix.fix_file(nc_path, tmp_path)
    assert fixed_cl_cube.var_name == "cl"
    fixed_air_pressure_coord = fixed_cl_cube.coord("air_pressure")
    assert fixed_air_pressure_coord.points is not None
    assert fixed_air_pressure_coord.bounds is not None
    np.testing.assert_allclose(
        fixed_air_pressure_coord.points, AIR_PRESSURE_POINTS
    )
    np.testing.assert_allclose(
        fixed_air_pressure_coord.bounds, AIR_PRESSURE_BOUNDS
    )
    lev_coord = fixed_cl_cube.coord(var_name="lev")
    assert lev_coord.units == "1"
    np.testing.assert_allclose(lev_coord.points, [1, 3])
    np.testing.assert_allclose(lev_coord.bounds, [[-1, 2], [2, 5]])


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2", "Amon", "cli")
    assert fix == [Cli(None), GenericFix(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is Cl


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2", "Amon", "clw")
    assert fix == [Clw(None), GenericFix(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is Cl


@pytest.fixture
def tas_cubes():
    """Cubes to test fixes for ``tas``."""
    time_coord = iris.coords.DimCoord(
        [0.0, 1.0],
        var_name="time",
        standard_name="time",
        units="days since 1850-01-01 00:00:00",
    )
    lat_coord = iris.coords.DimCoord(
        [0.0, 1.0], var_name="lat", standard_name="latitude", units="degrees"
    )
    lon_coord = iris.coords.DimCoord(
        [0.0, 1.0], var_name="lon", standard_name="longitude", units="degrees"
    )
    coord_specs = [
        (time_coord, 0),
        (lat_coord, 1),
        (lon_coord, 2),
    ]
    ta_cube = iris.cube.Cube(
        np.ones((2, 2, 2)),
        var_name="ta",
        dim_coords_and_dims=coord_specs,
    )
    tas_cube = iris.cube.Cube(
        np.ones((2, 2, 2)),
        var_name="tas",
        dim_coords_and_dims=coord_specs,
    )

    return iris.cube.CubeList([ta_cube, tas_cube])


@pytest.fixture
def tos_cubes():
    """Cubes to test fixes for ``tos``."""
    time_coord = iris.coords.DimCoord(
        [0.0004, 1.09776],
        var_name="time",
        standard_name="time",
        units="days since 1850-01-01 00:00:00",
    )
    lat_coord = iris.coords.DimCoord(
        [0.0, 1.0], var_name="lat", standard_name="latitude", units="degrees"
    )
    lon_coord = iris.coords.DimCoord(
        [0.0, 1.0], var_name="lon", standard_name="longitude", units="degrees"
    )
    coord_specs = [
        (time_coord, 0),
        (lat_coord, 1),
        (lon_coord, 2),
    ]
    tos_cube = iris.cube.Cube(
        np.ones((2, 2, 2)),
        var_name="tos",
        dim_coords_and_dims=coord_specs,
    )
    tos_cube.attributes = {}
    tos_cube.attributes["mipTable"] = "Omon"

    return iris.cube.CubeList([tos_cube])


@pytest.fixture
def thetao_cubes():
    """Cubes to test fixes for ``thetao``."""
    time_coord = iris.coords.DimCoord(
        [0.0004, 1.09776],
        var_name="time",
        standard_name="time",
        units="days since 1850-01-01 00:00:00",
    )
    lat_coord = iris.coords.DimCoord(
        [0.0, 1.0], var_name="lat", standard_name="latitude", units="degrees"
    )
    lon_coord = iris.coords.DimCoord(
        [0.0, 1.0], var_name="lon", standard_name="longitude", units="degrees"
    )
    lev_coord = iris.coords.DimCoord(
        [500.0, 1000.0],
        bounds=[[2.5, 7.5], [7.5, 12.5]],
        var_name="lev",
        standard_name=None,
        units="cm",
        attributes={"positive": "up"},
    )
    coord_specs = [
        (time_coord, 0),
        (lev_coord, 1),
        (lat_coord, 2),
        (lon_coord, 3),
    ]
    thetao_cube = iris.cube.Cube(
        np.ones((2, 2, 2, 2)),
        var_name="thetao",
        dim_coords_and_dims=coord_specs,
    )
    return iris.cube.CubeList([thetao_cube])


def test_get_tas_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2", "Amon", "tas")
    assert fix == [Tas(None), GenericFix(None)]


def test_get_tos_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2", "Omon", "tos")
    assert fix == [Tos(None), Omon(None), GenericFix(None)]


def test_get_thetao_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2", "Omon", "thetao")
    assert fix == [Omon(None), GenericFix(None)]


def test_get_fgco2_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2", "Omon", "fgco2")
    assert fix == [Fgco2(None), Omon(None), GenericFix(None)]


def test_get_siconc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2", "SImon", "siconc")
    assert fix == [Siconc(None), GenericFix(None)]


def test_tas_fix_metadata(tas_cubes):
    """Test ``fix_metadata`` for ``tas``."""
    for cube in tas_cubes:
        with pytest.raises(iris.exceptions.CoordinateNotFoundError):
            cube.coord("height")
    height_coord = iris.coords.AuxCoord(
        2.0,
        var_name="height",
        standard_name="height",
        long_name="height",
        units=Unit("m"),
        attributes={"positive": "up"},
    )
    vardef = get_var_info("CMIP6", "Amon", "tas")
    fix = Tas(vardef)
    out_cubes = fix.fix_metadata(tas_cubes)
    assert out_cubes is tas_cubes
    for cube in out_cubes:
        assert cube.coord("longitude").has_bounds()
        assert cube.coord("latitude").has_bounds()
        if cube.var_name == "tas":
            coord = cube.coord("height")
            assert coord == height_coord
        else:
            with pytest.raises(iris.exceptions.CoordinateNotFoundError):
                cube.coord("height")


def test_tos_fix_metadata(tos_cubes):
    """Test ``fix_metadata`` for ``tos``."""
    vardef = get_var_info("CMIP6", "Omon", "tos")
    fix = Tos(vardef)
    out_cubes = fix.fix_metadata(tos_cubes)
    assert out_cubes is tos_cubes
    for cube in out_cubes:
        np.testing.assert_equal(cube.coord("time").points, [0.0, 1.1])


def test_thetao_fix_metadata(thetao_cubes):
    """Test ``fix_metadata`` for ``thetao``."""
    vardef = get_var_info("CMIP6", "Omon", "thetao")
    fix = Omon(vardef)
    out_cubes = fix.fix_metadata(thetao_cubes)
    assert out_cubes is thetao_cubes
    assert len(out_cubes) == 1
    out_cube = out_cubes[0]

    # Check metadata of depth coordinate
    depth_coord = out_cube.coord("depth")
    assert depth_coord.standard_name == "depth"
    assert depth_coord.var_name == "lev"
    assert depth_coord.long_name == "ocean depth coordinate"
    assert depth_coord.units == "m"
    assert depth_coord.attributes == {"positive": "down"}

    # Check values of depth coordinate
    np.testing.assert_allclose(depth_coord.points, [5.0, 10.0])
    np.testing.assert_allclose(depth_coord.bounds, [[2.5, 7.5], [7.5, 12.5]])


def test_fgco2_fix_metadata():
    """Test ``fix_metadata`` for ``fgco2``."""
    vardef = get_var_info("CMIP6", "Omon", "fgco2")
    cubes = iris.cube.CubeList(
        [
            iris.cube.Cube(0.0, var_name="fgco2"),
        ]
    )
    fix = Fgco2(vardef)
    out_cubes = fix.fix_metadata(cubes)
    assert out_cubes is cubes
    assert len(out_cubes) == 1
    out_cube = out_cubes[0]

    # Check depth coordinate
    depth_coord = out_cube.coord("depth")
    assert depth_coord.standard_name == "depth"
    assert depth_coord.var_name == "depth"
    assert depth_coord.long_name == "depth"
    assert depth_coord.units == "m"
    assert depth_coord.attributes == {"positive": "down"}

    # Check values of depth coordinate
    np.testing.assert_allclose(depth_coord.points, 0.0)
    assert depth_coord.bounds is None


def test_siconc_fix():
    """Test fix for ``siconc``."""
    assert Siconc is SiconcFixScalarCoord
