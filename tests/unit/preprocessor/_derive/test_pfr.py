"""Test derivation of ``pfr``."""

import copy

import cf_units
import iris
import numpy as np
import pytest
from iris import NameConstraint

from esmvalcore.preprocessor._derive import pfr


@pytest.fixture
def cubes():
    time_coord = iris.coords.DimCoord(
        [
            0.0,
            31.0,
            59.0,
            90.0,
            120.0,
            151.0,
            181.0,
            212.0,
            243.0,
            273.0,
            304.0,
            334.0,
            365.0,
            396.0,
            424.0,
            455.0,
            485.0,
            516.0,
            546.0,
            577.0,
            608.0,
            638.0,
            669.0,
            699.0,
        ],
        standard_name="time",
        var_name="time",
        units="days since 1950-01-01 00:00:00",
    )
    dpth_coord = iris.coords.DimCoord(
        [1.0, 2.0, 5.0],
        standard_name="depth",
        var_name="depth",
        units="m",
        attributes={"positive": "down"},
    )
    lat_coord = iris.coords.DimCoord(
        [45.0, 60.0],
        standard_name="latitude",
        var_name="lat",
        units="degrees",
    )
    lon_coord = iris.coords.DimCoord(
        [10.0, 20.0],
        standard_name="longitude",
        var_name="lon",
        units="degrees",
    )
    coord_specs = [
        (time_coord, 0),
        (dpth_coord, 1),
        (lat_coord, 2),
        (lon_coord, 3),
    ]
    tsl_data = np.zeros(shape=(24, 3, 2, 2))
    tsl_data[:, 0, :, :] = 280.0
    tsl_data[:, 1, :, :] = 270.0
    tsl_data[:, 2, :, :] = 260.0
    tsl_cube = iris.cube.Cube(
        tsl_data,
        dim_coords_and_dims=coord_specs,
        var_name="tsl",
        units="K",
        standard_name="soil_temperature",
    )
    coord_specs = [
        (time_coord, 0),
        (copy.deepcopy(lat_coord), 1),
        (copy.deepcopy(lon_coord), 2),
    ]
    mrsos_data = np.zeros(shape=(24, 2, 2))
    mrsos_data[:, :, :] = 10.0
    mrsos_cube = iris.cube.Cube(
        mrsos_data,
        dim_coords_and_dims=coord_specs,
        var_name="mrsos",
        units="kg m-2",
        standard_name="mass_content_of_water_in_soil_layer",
    )
    coord_specs = [
        (copy.deepcopy(lat_coord), 0),
        (copy.deepcopy(lon_coord), 1),
    ]
    sftlf_data = np.zeros(shape=(2, 2))
    sftlf_data[:, :] = 100.0
    sftlf_cube = iris.cube.Cube(
        sftlf_data,
        dim_coords_and_dims=coord_specs,
        var_name="sftlf",
        units="%",
        standard_name="land_area_fraction",
    )

    return iris.cube.CubeList([tsl_cube, sftlf_cube, mrsos_cube])


def test_pfr_calculation(cubes):
    """Test function ``calculate``."""
    derived_var = pfr.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    assert out_cube.units == cf_units.Unit("%")
    out_data = out_cube.data
    expected = 100.0 * np.ones_like(out_cube.data)
    np.testing.assert_array_equal(out_data, expected)


def test_pfr_calculation_minor_latlon_differences(cubes):
    """Test function ``calculate``."""
    # small differences (i.e. < 1.0e-4) in lat/lon coordinates
    # in sftlf and mrsos
    derived_var = pfr.DerivedVariable()
    for cube in cubes:
        if cube.coords("year"):
            cube.remove_coord("year")
    sftlf_cube = cubes.extract_cube(NameConstraint(var_name="sftlf"))
    x_coord = sftlf_cube.coord(axis="X")
    y_coord = sftlf_cube.coord(axis="Y")
    x_coord.points = x_coord.core_points() + 1.0e-5
    y_coord.points = y_coord.core_points() + 1.0e-5
    out_cube = derived_var.calculate(cubes)
    # small differences are corrected automatically --> expect same results
    out_data = out_cube.data
    expected = 100.0 * np.ones_like(out_cube.data)
    np.testing.assert_array_equal(out_data, expected)


def test_pfr_calculation_major_lat_differences(cubes):
    """Test function ``calculate``."""
    # larger deviations in latitudes should trigger an error
    derived_var = pfr.DerivedVariable()
    for cube in cubes:
        if cube.coords("year"):
            cube.remove_coord("year")
    sftlf_cube = cubes.extract_cube(NameConstraint(var_name="sftlf"))
    y_coord = sftlf_cube.coord(axis="Y")
    org_y_pts = copy.deepcopy(y_coord.core_points())
    y_coord.points = y_coord.core_points() + 1.0e-2
    with pytest.raises(ValueError):
        derived_var.calculate(cubes)
    y_coord.points = org_y_pts


def test_pfr_calculation_major_lon_differences(cubes):
    """Test function ``calculate``."""
    # larger deviations in longitudes should trigger an error
    derived_var = pfr.DerivedVariable()
    for cube in cubes:
        if cube.coords("year"):
            cube.remove_coord("year")
    sftlf_cube = cubes.extract_cube(NameConstraint(var_name="sftlf"))
    x_coord = sftlf_cube.coord(axis="X")
    org_x_pts = copy.deepcopy(x_coord.core_points())
    x_coord.points = x_coord.core_points() + 1.0e-2
    with pytest.raises(ValueError):
        derived_var.calculate(cubes)
    x_coord.points = org_x_pts


def test_pfr_required():
    """Test function ``required``."""
    derived_var = pfr.DerivedVariable()
    output = derived_var.required(None)
    assert output == [
        {"short_name": "tsl", "mip": "Lmon"},
        {"short_name": "sftlf", "mip": "fx"},
        {"short_name": "mrsos", "mip": "Lmon"},
    ]
