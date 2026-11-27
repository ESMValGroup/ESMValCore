"""Unit test for :func:`esmvalcore.preprocessor._regrid`."""

from decimal import Decimal

import iris.coords
import iris.cube
import numpy as np
import pytest

from esmvalcore.preprocessor._regrid import (
    _global_stock_cube,
    _regional_stock_cube,
    _spec_to_latlonvals,
    regrid,
)


@pytest.fixture(autouse=True)
def clear_lru_cache():
    """Make sure to clear LRU cache for ``_global_stock_cube``.

    See https://github.com/ESMValGroup/ESMValCore/issues/2320.

    """
    yield  # execute tests
    _global_stock_cube.cache_clear()


SPEC_KEYS = (
    "start_longitude",
    "end_longitude",
    "step_longitude",
    "start_latitude",
    "end_latitude",
    "step_latitude",
)
PASSING_SPECS = tuple(
    dict(zip(SPEC_KEYS, spec, strict=True))
    for spec in (
        (0, 360, 5, -90, 90, 5),
        (0, 360, 20, -90, 90, 20),
        (0, 21, 5, -90, 90, 1),
        (0, 360, 5, -90, -70, 5),
        (0, 360, 5, -90, -69, 5),
        (350, 370, 1, -90, 90, 1),
        (-20, -5, 1, -90, 90, 1),
        (100, 50, -5, -90, 90, 5),
        (0, 360, 5, 40, 20, -5),
        (0, 359, 10, -90, 89, 10),
        (0, 0, 5, -90, 90, 5),
        (0, 360, 5, 0, 0, 5),
        (0, 9, 0.1, 45, 54, 0.1),
        (3.75, 11.75, 0.5, 46.25, 52.25, 0.5),
    )
)

FAILING_SPECS = tuple(
    dict(zip(SPEC_KEYS, spec, strict=True))
    for spec in (
        (0, 360, 5, -90, 180, 5),
        (0, 360, 5, -180, 90, 5),
        (0, 360, 5, -90, 90, -5),
        (0, 360, -5, -90, 90, 5),
        (0, -360, 5, -90, 90, 5),
        (0, 360, 0, -90, 90, 5),
        (0, 360, 5, -90, 90, 0),
    )
)


@pytest.mark.parametrize("spec", PASSING_SPECS)
def test_extract_regional_grid_passing(spec):
    """Test regridding with regional target spec."""
    global_cube = _global_stock_cube("10x10")
    scheme = "linear"

    result_cube = regrid(global_cube, target_grid=spec, scheme=scheme)

    expected_latvals, expected_lonvals = _spec_to_latlonvals(**spec)

    lat_coord = result_cube.coord("latitude")
    lon_coord = result_cube.coord("longitude")

    np.testing.assert_array_equal(lat_coord.points, expected_latvals)
    np.testing.assert_array_equal(lon_coord.points, expected_lonvals)

    assert lat_coord.has_bounds()
    assert lon_coord.has_bounds()


@pytest.mark.parametrize("spec", FAILING_SPECS)
def test_extract_regional_grid_failing(spec):
    """Test failing input for spec."""
    global_cube = _global_stock_cube("10x10")
    scheme = "linear"

    with pytest.raises(ValueError):
        _ = regrid(global_cube, target_grid=spec, scheme=scheme)


@pytest.mark.parametrize("spec", PASSING_SPECS)
def test_spec_to_latlonvals(spec):
    """Test lat/lon val specification."""
    latvals, lonvals = _spec_to_latlonvals(**spec)

    lat_step = spec["step_latitude"]
    assert latvals[0] == spec["start_latitude"]
    lat_diff = latvals[-1] - latvals[0]
    assert Decimal(lat_diff) % Decimal(str(lat_step)) == 0
    np.testing.assert_allclose(np.diff(latvals), lat_step)
    assert spec["end_latitude"] >= latvals[-1]

    lon_step = spec["step_longitude"]
    assert lonvals[0] == spec["start_longitude"]
    lon_diff = lonvals[-1] - lonvals[0]
    assert Decimal(lon_diff) % Decimal(str(lon_step)) == 0
    np.testing.assert_allclose(np.diff(lonvals), lon_step)
    assert spec["end_longitude"] >= lonvals[-1]


@pytest.mark.parametrize(
    ("longitudes", "circular"),
    [
        ((-52, 73, 0.25), False),
        ((350, 370, 1), False),
        ((0, 359, 1), True),
        ((0, 360, 5), True),
    ],
)
def test_regional_stock_cube_circular(longitudes, circular):
    """Test that only a grid covering the full great circle is circular."""
    start_longitude, end_longitude, step_longitude = longitudes
    spec = {
        "start_longitude": start_longitude,
        "end_longitude": end_longitude,
        "step_longitude": step_longitude,
        "start_latitude": -10,
        "end_latitude": 10,
        "step_latitude": 1,
    }

    cube = _regional_stock_cube(spec)

    assert cube.coord("longitude").circular is circular
    assert cube.coord("latitude").circular is False


def test_regrid_curvilinear_to_regional_grid():
    """Test that a regional target grid does not wrap around.

    Target grid cells outside the source grid should be masked.
    """
    # A curvilinear source grid with 1 degree cells covering 0-10E, 0-10N
    lon_bounds = np.arange(0, 11)
    lat_bounds = np.arange(0, 11)
    lon_points, lat_points = np.meshgrid(
        lon_bounds[:-1] + 0.5,
        lat_bounds[:-1] + 0.5,
    )
    corners_lon, corners_lat = np.meshgrid(lon_bounds, lat_bounds)

    def vertices(corners):
        return np.stack(
            [
                corners[:-1, :-1],
                corners[:-1, 1:],
                corners[1:, 1:],
                corners[1:, :-1],
            ],
            axis=-1,
        )

    source = iris.cube.Cube(
        np.ones((10, 10), dtype=np.float32),
        var_name="tas",
        units="K",
        dim_coords_and_dims=[
            (iris.coords.DimCoord(np.arange(10), var_name="y"), 0),
            (iris.coords.DimCoord(np.arange(10), var_name="x"), 1),
        ],
        aux_coords_and_dims=[
            (
                iris.coords.AuxCoord(
                    lat_points,
                    bounds=vertices(corners_lat),
                    standard_name="latitude",
                    units="degrees_north",
                ),
                (0, 1),
            ),
            (
                iris.coords.AuxCoord(
                    lon_points,
                    bounds=vertices(corners_lon),
                    standard_name="longitude",
                    units="degrees_east",
                ),
                (0, 1),
            ),
        ],
    )
    # A regional target grid extending 10 degrees east of the source grid
    spec = {
        "start_longitude": 0.5,
        "end_longitude": 19.5,
        "step_longitude": 1,
        "start_latitude": 0.5,
        "end_latitude": 9.5,
        "step_latitude": 1,
    }

    result = regrid(source, target_grid=spec, scheme="area_weighted")

    mask = np.ma.getmaskarray(result.data)
    lons = result.coord("longitude").points
    assert not mask[:, lons < 10].any()
    assert mask[:, lons > 10].all()
