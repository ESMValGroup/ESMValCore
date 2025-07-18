"""Test fixes for KIOST-ESM."""

import iris
import numpy as np
import pytest
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from esmvalcore.cmor._fixes.cmip6.kiost_esm import SfcWind, Siconc, Tas
from esmvalcore.cmor._fixes.common import SiconcFixScalarCoord
from esmvalcore.cmor._fixes.fix import Fix, GenericFix
from esmvalcore.cmor.table import get_var_info


@pytest.fixture
def sfcwind_cubes():
    correct_lat_coord = DimCoord(
        [0.0],
        var_name="lat",
        standard_name="latitude",
    )
    wrong_lat_coord = DimCoord(
        [0.0],
        var_name="latitudeCoord",
        standard_name="latitude",
    )
    correct_lon_coord = DimCoord(
        [0.0],
        var_name="lon",
        standard_name="longitude",
    )
    wrong_lon_coord = DimCoord(
        [0.0],
        var_name="longitudeCoord",
        standard_name="longitude",
    )
    correct_cube = Cube(
        [[10.0]],
        var_name="sfcWind",
        dim_coords_and_dims=[(correct_lat_coord, 0), (correct_lon_coord, 1)],
    )
    wrong_cube = Cube(
        [[10.0]],
        var_name="ta",
        dim_coords_and_dims=[(wrong_lat_coord, 0), (wrong_lon_coord, 1)],
        attributes={"parent_time_units": "days since 0000-00-00 (noleap)"},
    )
    scalar_cube = Cube(0.0, var_name="ps")
    return CubeList([correct_cube, wrong_cube, scalar_cube])


@pytest.fixture
def tas_cubes():
    correct_lat_coord = DimCoord(
        [0.0],
        var_name="lat",
        standard_name="latitude",
    )
    wrong_lat_coord = DimCoord(
        [0.0],
        var_name="latitudeCoord",
        standard_name="latitude",
    )
    correct_lon_coord = DimCoord(
        [0.0],
        var_name="lon",
        standard_name="longitude",
    )
    wrong_lon_coord = DimCoord(
        [0.0],
        var_name="longitudeCoord",
        standard_name="longitude",
    )
    correct_cube = Cube(
        [[10.0]],
        var_name="tas",
        dim_coords_and_dims=[(correct_lat_coord, 0), (correct_lon_coord, 1)],
    )
    wrong_cube = Cube(
        [[10.0]],
        var_name="ta",
        dim_coords_and_dims=[(wrong_lat_coord, 0), (wrong_lon_coord, 1)],
        attributes={"parent_time_units": "days since 0000-00-00 (noleap)"},
    )
    scalar_cube = Cube(0.0, var_name="ps")
    return CubeList([correct_cube, wrong_cube, scalar_cube])


def test_get_sfcwind_fix():
    fix = Fix.get_fixes("CMIP6", "KIOST-ESM", "Amon", "sfcWind")
    assert fix == [SfcWind(None), GenericFix(None)]


def test_sfcwind_fix_metadata(sfcwind_cubes):
    for cube in sfcwind_cubes:
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
    vardef = get_var_info("CMIP6", "Amon", "sfcWind")
    fix = SfcWind(vardef)

    # Check fix
    out_cubes = fix.fix_metadata(sfcwind_cubes)
    assert out_cubes[0].var_name == "sfcWind"
    coord = out_cubes[0].coord("height")
    assert coord == height_coord

    # Check that height coordinate is not added twice
    out_cubes_2 = fix.fix_metadata(out_cubes)
    assert out_cubes_2[0].var_name == "sfcWind"
    coord = out_cubes_2[0].coord("height")
    assert coord == height_coord


def test_get_siconc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "KIOST-ESM", "SImon", "siconc")
    assert fix == [Siconc(None), GenericFix(None)]


def test_siconc_fix():
    """Test fix for ``siconc``."""
    assert issubclass(Siconc, SiconcFixScalarCoord)


def test_siconc_fix_data():
    """Test fix for ``siconc``."""
    vardef = get_var_info("CMIP6", "SImon", "siconc")
    fix = Siconc(vardef)

    cube = Cube([0.0, np.nan, 1.0], var_name="siconc")
    assert not np.ma.is_masked(cube.data)

    out_cube = fix.fix_data(cube)
    np.testing.assert_array_almost_equal(out_cube.data, [0.0, np.nan, 1.0])
    np.testing.assert_array_equal(out_cube.data.mask, [False, True, False])


def test_get_tas_fix():
    fix = Fix.get_fixes("CMIP6", "KIOST-ESM", "Amon", "tas")
    assert fix == [Tas(None), GenericFix(None)]


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
    vardef = get_var_info("CMIP6", "Amon", "tas")
    fix = Tas(vardef)

    # Check fix
    out_cubes = fix.fix_metadata(tas_cubes)
    assert out_cubes[0].var_name == "tas"
    coord = out_cubes[0].coord("height")
    assert coord == height_coord

    # Check that height coordinate is not added twice
    out_cubes_2 = fix.fix_metadata(out_cubes)
    assert out_cubes_2[0].var_name == "tas"
    coord = out_cubes_2[0].coord("height")
    assert coord == height_coord
