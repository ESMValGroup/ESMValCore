"""Tests for the fixes of GFDL-CM4."""

import iris
import numpy as np
import pytest
from cf_units import Unit

from esmvalcore.cmor._fixes.cmip6.gfdl_cm4 import (
    Cl,
    Cli,
    Clw,
    Fgco2,
    Omon,
    Siconc,
    Sos,
    Tas,
    Tos,
    Uas,
)
from esmvalcore.cmor._fixes.common import OceanFixGrid, SiconcFixScalarCoord
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "GFDL-CM4", "Amon", "cl")
    assert fix == [Cl(None), GenericFix(None)]


AIR_PRESSURE_POINTS = np.array(
    [
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[2.0, 3.0], [4.0, 5.0]],
            [[5.0, 8.0], [11.0, 14.0]],
        ],
    ],
)
AIR_PRESSURE_BOUNDS = np.array(
    [
        [
            [[[0.0, 1.5], [-1.0, 2.0]], [[-2.0, 2.5], [-3.0, 3.0]]],
            [[[1.5, 3.0], [2.0, 5.0]], [[2.5, 7.0], [3.0, 9.0]]],
            [[[3.0, 6.0], [5.0, 11.0]], [[7.0, 16.0], [9.0, 21.0]]],
        ],
    ],
)


def test_cl_fix_metadata(test_data_path):
    """Test ``fix_metadata`` for ``cl``."""
    nc_path = test_data_path / "gfdl_cm4_cl.nc"
    cubes = iris.load(str(nc_path))

    # Raw cubes
    assert len(cubes) == 6
    var_names = [cube.var_name for cube in cubes]
    assert "cl" in var_names
    assert "ap" in var_names
    assert "ap_bnds" in var_names
    assert "b" in var_names
    assert "b_bnds" in var_names
    assert "ps" in var_names

    # Raw cl cube
    cl_cube = cubes.extract_cube("cloud_area_fraction_in_atmosphere_layer")
    assert not cl_cube.coords("air_pressure")

    # Apply fix
    vardef = get_var_info("CMIP6", "Amon", "cl")
    fix = Cl(vardef)
    fixed_cubes = fix.fix_metadata(cubes)
    assert len(fixed_cubes) == 1
    fixed_cl_cube = fixed_cubes.extract_cube(
        "cloud_area_fraction_in_atmosphere_layer",
    )
    fixed_air_pressure_coord = fixed_cl_cube.coord("air_pressure")
    assert fixed_air_pressure_coord.points is not None
    assert fixed_air_pressure_coord.bounds is not None
    np.testing.assert_allclose(
        fixed_air_pressure_coord.points,
        AIR_PRESSURE_POINTS,
    )
    np.testing.assert_allclose(
        fixed_air_pressure_coord.bounds,
        AIR_PRESSURE_BOUNDS,
    )


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "GFDL-CM4", "Amon", "cli")
    assert fix == [Cli(None), GenericFix(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is Cl


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "GFDL-CM4", "Amon", "clw")
    assert fix == [Clw(None), GenericFix(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is Cl


def test_get_fgco2_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "GFDL-CM4", "Omon", "fgco2")
    assert fix == [Fgco2(None), Omon(None), GenericFix(None)]


def test_get_siconc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "GFDL-CM4", "SImon", "siconc")
    assert fix == [Siconc(None), GenericFix(None)]


def test_siconc_fix():
    """Test fix for ``siconc``."""
    assert Siconc is SiconcFixScalarCoord


def test_get_sos_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "GFDL-CM4", "Omon", "sos")
    assert fix == [Sos(None), Omon(None), GenericFix(None)]


def test_sos_fix():
    """Test fix for ``sos``."""
    assert Sos is OceanFixGrid


def test_get_tos_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "GFDL-CM4", "Omon", "tos")
    assert fix == [Tos(None), Omon(None), GenericFix(None)]


def test_tos_fix():
    """Test fix for ``tos``."""
    assert Tos is OceanFixGrid


@pytest.fixture
def tas_cubes():
    correct_lat_coord = iris.coords.DimCoord(
        [0.0],
        var_name="lat",
        standard_name="latitude",
    )
    wrong_lat_coord = iris.coords.DimCoord(
        [0.0],
        var_name="latitudeCoord",
        standard_name="latitude",
    )
    correct_lon_coord = iris.coords.DimCoord(
        [0.0],
        var_name="lon",
        standard_name="longitude",
    )
    wrong_lon_coord = iris.coords.DimCoord(
        [0.0],
        var_name="longitudeCoord",
        standard_name="longitude",
    )
    correct_cube = iris.cube.Cube(
        [[2.0]],
        var_name="tas",
        dim_coords_and_dims=[(correct_lat_coord, 0), (correct_lon_coord, 1)],
    )
    wrong_cube = iris.cube.Cube(
        [[2.0]],
        var_name="ta",
        dim_coords_and_dims=[(wrong_lat_coord, 0), (wrong_lon_coord, 1)],
    )
    scalar_cube = iris.cube.Cube(0.0, var_name="ps")
    return iris.cube.CubeList([correct_cube, wrong_cube, scalar_cube])


def test_get_tas_fix():
    fixes = Fix.get_fixes("CMIP6", "GFDL-CM4", "day", "tas")
    assert Tas(None) in fixes


def test_tas_fix_metadata(tas_cubes):
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
    vardef = get_var_info("CMIP6", "day", "tas")
    fix = Tas(vardef)

    out_cubes = fix.fix_metadata(tas_cubes)
    assert out_cubes[0].var_name == "tas"
    coord = out_cubes[0].coord("height")
    assert coord == height_coord


@pytest.fixture
def uas_cubes():
    correct_lat_coord = iris.coords.DimCoord(
        [0.0],
        var_name="lat",
        standard_name="latitude",
    )
    wrong_lat_coord = iris.coords.DimCoord(
        [0.0],
        var_name="latitudeCoord",
        standard_name="latitude",
    )
    correct_lon_coord = iris.coords.DimCoord(
        [0.0],
        var_name="lon",
        standard_name="longitude",
    )
    wrong_lon_coord = iris.coords.DimCoord(
        [0.0],
        var_name="longitudeCoord",
        standard_name="longitude",
    )
    correct_cube = iris.cube.Cube(
        [[10.0]],
        var_name="uas",
        dim_coords_and_dims=[(correct_lat_coord, 0), (correct_lon_coord, 1)],
    )
    wrong_cube = iris.cube.Cube(
        [[10.0]],
        var_name="ua",
        dim_coords_and_dims=[(wrong_lat_coord, 0), (wrong_lon_coord, 1)],
    )
    scalar_cube = iris.cube.Cube(0.0, var_name="ps")
    return iris.cube.CubeList([correct_cube, wrong_cube, scalar_cube])


def test_get_uas_fix():
    fixes = Fix.get_fixes("CMIP6", "GFDL-CM4", "day", "uas")
    assert Uas(None) in fixes


def test_uas_fix_metadata(uas_cubes):
    for cube in uas_cubes:
        with pytest.raises(iris.exceptions.CoordinateNotFoundError):
            cube.coord("height")
    height_coord = iris.coords.AuxCoord(
        10.0,
        var_name="height",
        standard_name="height",
        long_name="height",
        units=Unit("m"),
        attributes={"positive": "up"},
    )
    vardef = get_var_info("CMIP6", "day", "uas")
    fix = Uas(vardef)

    out_cubes = fix.fix_metadata(uas_cubes)
    assert out_cubes[0].var_name == "uas"
    coord = out_cubes[0].coord("height")
    assert coord == height_coord
