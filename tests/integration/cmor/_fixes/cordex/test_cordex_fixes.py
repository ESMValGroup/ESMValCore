"""Tests for general CORDEX fixes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cordex as cx
import iris
import iris.cube
import numpy as np
import pytest
from cf_units import Unit

from esmvalcore.cmor._fixes.cordex.cordex_fixes import (
    AllVars,
    CLMcomCCLM4817,
    MOHCHadREM3GA705,
    TimeLongName,
)
from esmvalcore.cmor.fix import Fix

if TYPE_CHECKING:
    import pytest_mock

    from esmvalcore.config import Session


@pytest.fixture
def cubes():
    correct_time_coord = iris.coords.DimCoord(
        [0.0],
        var_name="time",
        standard_name="time",
        long_name="time",
    )
    wrong_time_coord = iris.coords.DimCoord(
        [0.0],
        var_name="time",
        standard_name="time",
        long_name="wrong",
    )
    correct_lat_coord = iris.coords.DimCoord(
        [0.0, 1.0],
        var_name="lat",
        standard_name="latitude",
        long_name="latitude",
    )
    wrong_lat_coord = iris.coords.DimCoord(
        [0.0, 1.0],
        var_name="latitudeCoord",
        standard_name="latitude",
        long_name="latitude",
    )
    correct_lon_coord = iris.coords.DimCoord(
        [0.0],
        var_name="lon",
        standard_name="longitude",
        long_name="longitude",
    )
    wrong_lon_coord = iris.coords.DimCoord(
        [0.0],
        var_name="longitudeCoord",
        standard_name="longitude",
        long_name="longitude",
    )
    correct_cube = iris.cube.Cube(
        [[[10.0], [10.0]]],
        var_name="tas",
        dim_coords_and_dims=[
            (correct_time_coord, 0),
            (correct_lat_coord, 1),
            (correct_lon_coord, 2),
        ],
    )
    wrong_cube = iris.cube.Cube(
        [[[10.0], [10.0]]],
        var_name="tas",
        dim_coords_and_dims=[
            (wrong_time_coord, 0),
            (wrong_lat_coord, 1),
            (wrong_lon_coord, 2),
        ],
    )
    return iris.cube.CubeList([correct_cube, wrong_cube])


@pytest.fixture
def cordex_cubes():
    coord_system = iris.coord_systems.RotatedGeogCS(
        grid_north_pole_latitude=39.25,
        grid_north_pole_longitude=-162,
    )
    time = iris.coords.DimCoord(
        np.arange(0, 3),
        var_name="time",
        standard_name="time",
    )

    rlat = iris.coords.DimCoord(
        np.arange(0, 412),
        var_name="rlat",
        standard_name="grid_latitude",
        coord_system=coord_system,
    )
    rlon = iris.coords.DimCoord(
        np.arange(0, 424),
        var_name="rlon",
        standard_name="grid_longitude",
        coord_system=coord_system,
    )
    lat = iris.coords.AuxCoord(
        np.ones((412, 424)),
        var_name="lat",
        standard_name="latitude",
    )
    lon = iris.coords.AuxCoord(
        np.ones((412, 424)),
        var_name="lon",
        standard_name="longitude",
    )

    cube = iris.cube.Cube(
        np.ones((3, 412, 424)),
        var_name="tas",
        dim_coords_and_dims=[(time, 0), (rlat, 1), (rlon, 2)],
        aux_coords_and_dims=[(lat, (1, 2)), (lon, (1, 2))],
    )
    return iris.cube.CubeList([cube])


@pytest.mark.parametrize(
    ("coord", "var_name", "long_name"),
    [
        ("time", "time", "time"),
        ("latitude", "lat", "latitude"),
        ("longitude", "lon", "longitude"),
    ],
)
def test_mohchadrem3ga705_fix_metadata(cubes, coord, var_name, long_name):
    fix = MOHCHadREM3GA705(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        assert cube.coord(standard_name=coord).var_name == var_name
        assert cube.coord(standard_name=coord).long_name == long_name


def test_mohchadrem3ga705_fix_metadata_no_time_coord(
    cubes: iris.cube.CubeList,
    session: Session,
) -> None:
    for cube in cubes:
        cube.remove_coord("time")
    fix = MOHCHadREM3GA705(None, session=session)  # type: ignore[arg-type]
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        assert cube.coord(standard_name="latitude").var_name == "lat"
        assert cube.coord(standard_name="longitude").var_name == "lon"


def test_timelongname_fix_metadata(cubes):
    fix = TimeLongName(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        assert cube.coord("time").long_name == "time"


@pytest.mark.parametrize("has_time_coord", [True, False])
def test_clmcomcclm4817_fix_metadata_time(
    cubes: iris.cube.CubeList,
    session: Session,
    has_time_coord: bool,
) -> None:
    if has_time_coord:
        cubes[0].coord("time").units = Unit(
            "days since 1850-1-1 00:00:00",
            calendar="proleptic_gregorian",
        )
        cubes[1].coord("time").units = Unit(
            "days since 1850-1-1 00:00:00",
            calendar="standard",
        )
    else:
        for cube in cubes:
            cube.remove_coord("time")
    for cube in cubes:
        cube.data = cube.lazy_data().astype(">f4", casting="same_kind")
    for coord in cubes[1].coords():
        coord.points = coord.core_points().astype(">f8", casting="same_kind")
    lat = cubes[1].coord("latitude")
    lat.guess_bounds()
    lat.bounds = lat.core_bounds().astype(">f4", casting="same_kind")

    fix = CLMcomCCLM4817(None, session=session)  # type: ignore[arg-type]
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        if has_time_coord:
            assert cube.coord("time").units == Unit(
                "days since 1850-1-1 00:00:00",
                calendar="proleptic_gregorian",
            )
        assert cube.has_lazy_data()
        assert cube.lazy_data().dtype == np.float32
        for coord in cube.coords():
            assert coord.points.dtype == np.float64


def test_rotated_grid_fix(cordex_cubes):
    fix = AllVars(
        vardef=None,
        extra_facets={
            "domain": "EUR-11",
            "dataset": "DATASET",
            "driver": "DRIVER",
            "use_standard_grid": True,
        },
    )
    domain = cx.cordex_domain("EUR-11", bounds=True)
    for cube in cordex_cubes:
        for coord in ["rlat", "rlon", "lat", "lon"]:
            cube_coord = cube.coord(var_name=coord)
            cube_coord.points = domain[coord].data + 1e-6
    out_cubes = fix.fix_metadata(cordex_cubes)
    assert len(out_cubes) == len(cordex_cubes)
    for out_cube in out_cubes:
        for coord in ["rlat", "rlon", "lat", "lon"]:
            cube_coord = out_cube.coord(var_name=coord)
            domain_coord = domain[coord].data
            np.testing.assert_array_equal(cube_coord.points, domain_coord)


def test_rotated_grid_fix_warning(cordex_cubes, caplog):
    fix = AllVars(
        vardef=None,
        extra_facets={
            "domain": "EUR-11",
            "dataset": "DATASET",
            "driver": "DRIVER",
            "use_standard_grid": True,
        },
    )
    domain = cx.cordex_domain("EUR-11", bounds=True)
    for cube in cordex_cubes:
        for coord in ["rlat", "rlon", "lat", "lon"]:
            cube_coord = cube.coord(var_name=coord)
            cube_coord.points = domain[coord].data + 1.0
    fix.fix_metadata(cordex_cubes)
    msg = (
        "Maximum difference between original grid_latitude points and standard "
        "EUR-11 domain points for variable tas from dataset DATASET is 1.0"
    )
    assert msg in caplog.text


@pytest.mark.parametrize("use_standard_grid", [True, False])
def test_lambert_conformal_grid_fix(use_standard_grid: bool) -> None:
    fixes = Fix.get_fixes(
        project="CORDEX",
        dataset="DATASET",
        mip="mon",
        short_name="tas",
        extra_facets={
            "domain": "EUR-11",
            "dataset": "DATASET",
            "driver": "DRIVER",
            "use_standard_grid": use_standard_grid,
        },
    )
    fix = fixes[0]
    assert isinstance(fix, AllVars)

    # Prepare some test data on a wrong Lambert Conformal grid.
    lambert_crs = iris.coord_systems.LambertConformal(
        central_lat=49.5,
        central_lon=10.5,
        secant_latitudes=(49.5,),
    )
    cube = iris.cube.Cube(
        np.ones((3, 453, 453)),
        var_name="tas",
        units="K",
        dim_coords_and_dims=[
            (
                iris.coords.DimCoord(
                    np.arange(0, 3),
                    var_name="time",
                    standard_name="time",
                    units="days since 1850-1-1 00:00:00",
                ),
                0,
            ),
            (
                iris.coords.DimCoord(
                    np.arange(0, 453),
                    var_name="projection_y_coordinate",
                    standard_name="projection_y_coordinate",
                    coord_system=lambert_crs,
                    units="km",
                ),
                1,
            ),
            (
                iris.coords.DimCoord(
                    np.arange(0, 453),
                    var_name="projection_x_coordinate",
                    standard_name="projection_x_coordinate",
                    coord_system=lambert_crs,
                    units="km",
                ),
                2,
            ),
        ],
        aux_coords_and_dims=[
            (
                iris.coords.AuxCoord(
                    np.ones((453, 453)),
                    var_name="latitude",
                    standard_name="latitude",
                    units="degrees_north",
                ),
                (1, 2),
            ),
            (
                iris.coords.AuxCoord(
                    np.ones((453, 453)),
                    var_name="longitude",
                    standard_name="longitude",
                    units="degrees_east",
                ),
                (1, 2),
            ),
        ],
    )

    (result,) = fix.fix_metadata([cube])
    if not use_standard_grid:
        assert result == cube
        return

    for coord_name in [
        "projection_x_coordinate",
        "projection_y_coordinate",
        "latitude",
        "longitude",
    ]:
        assert len(result.coords(coord_name)) == 1

    x_coord = result.coord("projection_x_coordinate")
    y_coord = result.coord("projection_y_coordinate")
    assert x_coord.units == "m"
    assert y_coord.units == "m"
    assert x_coord.points.dtype == np.float64
    assert y_coord.points.dtype == np.float64
    assert x_coord.bounds is not None
    assert y_coord.bounds is not None
    assert x_coord.bounds.dtype == np.float64
    assert y_coord.bounds.dtype == np.float64
    assert x_coord.coord_system == lambert_crs
    assert y_coord.coord_system == lambert_crs
    np.testing.assert_array_almost_equal(
        x_coord.points[[0, 1, 226, -1]],
        [-2825000.0, -2812500.0, 0.0, 2825000.0],
    )
    np.testing.assert_array_almost_equal(
        y_coord.points[[0, 1, 226, -1]],
        [-2825000.0, -2812500.0, 0.0, 2825000.0],
    )

    lon_coord = result.coord("longitude")
    lat_coord = result.coord("latitude")
    assert lon_coord.units == "degrees_east"
    assert lat_coord.units == "degrees_north"
    assert lon_coord.points.dtype == np.float64
    assert lat_coord.points.dtype == np.float64
    assert lon_coord.bounds is not None
    assert lat_coord.bounds is not None
    assert lon_coord.bounds.dtype == np.float64
    assert lat_coord.bounds.dtype == np.float64
    assert lon_coord.coord_system is None
    assert lat_coord.coord_system is None
    np.testing.assert_array_almost_equal(lon_coord.points[0, 0], -14.26627)
    np.testing.assert_array_almost_equal(
        lon_coord.bounds[0, 0],
        [
            -14.29979978,
            -14.19799928,
            -14.23267916,
            -14.33460134,
        ],
    )
    np.testing.assert_array_almost_equal(lat_coord.points[0, 0], 20.922545)
    np.testing.assert_array_almost_equal(
        lat_coord.bounds[0, 0],
        [
            20.858380098,
            20.890987413,
            20.986725409,
            20.954050931,
        ],
    )


def test_lambert_conformal_grid_fix_domain_with_unknown_spacing(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test that the fix does not fail if the domain spacing is unknown."""
    fixes = Fix.get_fixes(
        project="CORDEX",
        dataset="DATASET",
        mip="mon",
        short_name="tas",
        extra_facets={
            "domain": "EUR-11i",
            "dataset": "DATASET",
            "driver": "DRIVER",
            "use_standard_grid": True,
        },
    )
    fix = fixes[0]
    assert isinstance(fix, AllVars)
    mocker.patch.object(AllVars, "_check_grid_differences")
    cube = iris.cube.Cube(
        [0],
        dim_coords_and_dims=[
            (
                iris.coords.DimCoord(
                    [0],
                    standard_name="projection_x_coordinate",
                    coord_system=iris.coord_systems.LambertConformal(
                        central_lat=49.5,
                        central_lon=10.5,
                        secant_latitudes=(49.5,),
                    ),
                    units="m",
                ),
                0,
            ),
        ],
    )
    (result,) = fix.fix_metadata([cube])
    assert result == cube


def test_lambert_grid_warning(cubes, caplog):
    fix = AllVars(
        vardef=None,
        extra_facets={
            "domain": "EUR-11",
            "dataset": "DATASET",
            "driver": "DRIVER",
        },
    )
    for cube in cubes:
        cube.coord_system = iris.coord_systems.LambertConformal
    fix.fix_metadata(cubes)
    msg = (
        "Support for CORDEX datasets in a Lambert Conformal "
        "coordinate system is ongoing. Certain preprocessor "
        "functions may fail."
    )
    assert msg in caplog.text
