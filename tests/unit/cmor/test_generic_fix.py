"""Unit tests for generic fixes."""

from unittest.mock import sentinel

import numpy as np
import pytest
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList

from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor.table import get_var_info


@pytest.fixture
def generic_fix():
    """Create a GenericFix object."""
    vardef = get_var_info("CMIP6", "CFmon", "ta")
    extra_facets = {"short_name": "ta", "project": "CMIP6", "dataset": "MODEL"}
    return GenericFix(vardef, extra_facets=extra_facets)


def test_generic_fix_empty_long_name(generic_fix, monkeypatch):
    """Test ``GenericFix``."""
    # Artificially set long_name to empty string for test
    monkeypatch.setattr(generic_fix.vardef, "long_name", "")

    cube = generic_fix._fix_long_name(sentinel.cube)

    assert cube == sentinel.cube


def test_generic_fix_empty_units(generic_fix, monkeypatch):
    """Test ``GenericFix``."""
    # Artificially set latitude units to empty string for test
    coord_info = generic_fix.vardef.coordinates["latitude"]
    monkeypatch.setattr(coord_info, "units", "")

    ret = generic_fix._fix_coord_units(
        sentinel.cube,
        coord_info,
        sentinel.cube_coord,
    )

    assert ret is None


def test_generic_fix_no_generic_lev_coords(generic_fix, monkeypatch):
    """Test ``GenericFix``."""
    # Artificially remove generic_lev_coords
    monkeypatch.setattr(
        generic_fix.vardef.coordinates["alevel"],
        "generic_lev_coords",
        {},
    )

    cube = generic_fix._fix_alternative_generic_level_coords(sentinel.cube)

    assert cube == sentinel.cube


def test_requested_levels_2d_coord(generic_fix, mocker):
    """Test ``GenericFix``."""
    cube_coord = AuxCoord([[0]], standard_name="latitude", units="rad")
    cmor_coord = mocker.Mock(requested=True)

    ret = generic_fix._fix_requested_coord_values(
        sentinel.cube,
        cmor_coord,
        cube_coord,
    )

    assert ret is None


def test_requested_levels_invalid_arr(generic_fix, mocker):
    """Test ``GenericFix``."""
    cube_coord = AuxCoord([0], standard_name="latitude", units="rad")
    cmor_coord = mocker.Mock(requested=["a", "b"])

    ret = generic_fix._fix_requested_coord_values(
        sentinel.cube,
        cmor_coord,
        cube_coord,
    )

    assert ret is None


def test_lon_no_fix_needed(generic_fix):
    """Test ``GenericFix``."""
    cube_coord = AuxCoord(
        [0.0, 180.0, 360.0],
        standard_name="longitude",
        units="rad",
    )

    ret = generic_fix._fix_longitude_0_360(
        sentinel.cube,
        sentinel.cmor_coord,
        cube_coord,
    )

    assert ret == (sentinel.cube, cube_coord)


def test_lon_too_low_to_fix(generic_fix):
    """Test ``GenericFix``."""
    cube_coord = AuxCoord(
        [-370.0, 0.0],
        standard_name="longitude",
        units="rad",
    )

    ret = generic_fix._fix_longitude_0_360(
        sentinel.cube,
        sentinel.cmor_coord,
        cube_coord,
    )

    assert ret == (sentinel.cube, cube_coord)


def test_lon_too_high_to_fix(generic_fix):
    """Test ``GenericFix``."""
    cube_coord = AuxCoord([750.0, 0.0], standard_name="longitude", units="rad")

    ret = generic_fix._fix_longitude_0_360(
        sentinel.cube,
        sentinel.cmor_coord,
        cube_coord,
    )

    assert ret == (sentinel.cube, cube_coord)


def test_fix_direction_2d_coord(generic_fix):
    """Test ``GenericFix``."""
    cube_coord = AuxCoord([[0]], standard_name="latitude", units="rad")

    ret = generic_fix._fix_coord_direction(
        sentinel.cube,
        sentinel.cmor_coord,
        cube_coord,
    )

    assert ret == (sentinel.cube, cube_coord)


def test_fix_direction_string_coord(generic_fix):
    """Test ``GenericFix``."""
    cube_coord = AuxCoord(["a"], standard_name="latitude", units="rad")

    ret = generic_fix._fix_coord_direction(
        sentinel.cube,
        sentinel.cmor_coord,
        cube_coord,
    )

    assert ret == (sentinel.cube, cube_coord)


def test_fix_direction_no_stored_direction(generic_fix, mocker):
    """Test ``GenericFix``."""
    cube = Cube(0)
    cube_coord = AuxCoord([0, 1], standard_name="latitude", units="rad")
    cmor_coord = mocker.Mock(stored_direction="")

    ret = generic_fix._fix_coord_direction(cube, cmor_coord, cube_coord)

    assert ret == (cube, cube_coord)


def test_fix_metadata_not_fail_with_empty_cube(generic_fix):
    """Generic fixes should not fail with empty cubes."""
    fixed_cubes = generic_fix.fix_metadata([Cube(0)])

    assert isinstance(fixed_cubes, CubeList)
    assert len(fixed_cubes) == 1
    assert fixed_cubes[0] == Cube(
        0,
        standard_name="air_temperature",
        long_name="Air Temperature",
    )


@pytest.mark.parametrize(
    "extra_facets",
    [{}, {"project": "P", "dataset": "D"}],
)
def test_fix_metadata_multiple_cubes_fail(extra_facets):
    """Generic fixes should fail when multiple invalid cubes are given."""
    vardef = get_var_info("CMIP6", "Amon", "ta")
    fix = GenericFix(vardef, extra_facets=extra_facets)
    with pytest.raises(ValueError):
        fix.fix_metadata([Cube(0), Cube(0)])


def test_fix_metadata_no_extra_facets():
    """Generic fixes should not fail when no extra facets are given."""
    vardef = get_var_info("CMIP6", "Amon", "ta")
    fix = GenericFix(vardef)
    fixed_cubes = fix.fix_metadata([Cube(0)])

    assert isinstance(fixed_cubes, CubeList)
    assert len(fixed_cubes) == 1
    assert fixed_cubes[0] == Cube(
        0,
        standard_name="air_temperature",
        long_name="Air Temperature",
    )


def test_fix_data_not_fail_with_empty_cube(generic_fix):
    """Generic fixes should not fail with empty cubes."""
    fixed_cube = generic_fix.fix_data(Cube(0))

    assert isinstance(fixed_cube, Cube)
    assert fixed_cube == Cube(0)


def test_fix_data_no_extra_facets():
    """Generic fixes should not fail when no extra facets are given."""
    vardef = get_var_info("CMIP6", "Amon", "ta")
    fix = GenericFix(vardef)
    fixed_cube = fix.fix_data(Cube(0))

    assert isinstance(fixed_cube, Cube)
    assert fixed_cube == Cube(0)


def _get_multidim_lat_lon_cube(lat_var_name, lon_var_name):
    """Create a cube with two-dimensional latitude and longitude."""
    lat = AuxCoord(
        [[0.0, 0.0], [1.0, 1.0]],
        var_name=lat_var_name,
        standard_name="latitude",
        units="degrees_north",
    )
    lon = AuxCoord(
        [[0.0, 1.0], [0.0, 1.0]],
        var_name=lon_var_name,
        standard_name="longitude",
        units="degrees_east",
    )
    return Cube(
        np.zeros((2, 2)),
        aux_coords_and_dims=[(lat, (0, 1)), (lon, (0, 1))],
    )


@pytest.mark.parametrize(
    ("project", "mip"),
    [("CMIP5", "Amon"), ("CMIP6", "Amon"), ("CORDEX", "mon")],
)
def test_fix_multidim_lat_lon_coord(project, mip):
    """Test that multidimensional lat/lon coords are renamed."""
    vardef = get_var_info(project, mip, "tas")
    fix = GenericFix(vardef, extra_facets={"project": project})
    cube = _get_multidim_lat_lon_cube("latitude", "longitude")

    fix._fix_regular_coord_names(cube)

    assert cube.coord("latitude").var_name == "lat"
    assert cube.coord("longitude").var_name == "lon"


def test_fix_multidim_lat_lon_coord_no_fix_needed():
    """Test that correctly named multidimensional lat/lon are kept."""
    vardef = get_var_info("CMIP5", "Amon", "tas")
    fix = GenericFix(vardef, extra_facets={"project": "CMIP5"})
    cube = _get_multidim_lat_lon_cube("lat", "lon")

    fix._fix_regular_coord_names(cube)

    assert cube.coord("latitude").var_name == "lat"
    assert cube.coord("longitude").var_name == "lon"


def test_fix_onedim_lat_lon_coord_not_renamed():
    """Test that one-dimensional lat/lon are not renamed by this fix."""
    vardef = get_var_info("CMIP5", "Amon", "tas")
    fix = GenericFix(vardef, extra_facets={"project": "CMIP5"})
    lat = AuxCoord([0.0, 1.0], var_name="latitude", standard_name="latitude")
    cube = Cube(np.zeros(2), aux_coords_and_dims=[(lat, 0)])

    fix._fix_regular_coord_names(cube)

    assert cube.coord("latitude").var_name == "latitude"
