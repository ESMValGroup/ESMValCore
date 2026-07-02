"""Unit tests for CORDEX domain target grids in regridding."""

import cordex as cx
import iris
import iris.coords
import numpy as np
import pytest

from esmvalcore.preprocessor._regrid import (
    _cordex_stock_cube,
    _get_target_grid_cube,
    _global_stock_cube,
    is_cordex_domain,
    parse_cell_spec,
)


@pytest.fixture(autouse=True)
def clear_lru_cache():
    """Clear LRU caches for stock cube helpers."""
    yield
    _global_stock_cube.cache_clear()
    _cordex_stock_cube.cache_clear()


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("EUR-11", True),
        ("EUR-44", True),
        ("not-a-grid", False),
        ("1x1", False),
        ("RCA4", False),
    ],
)
def test_is_cordex_domain(spec, expected):
    """Test CORDEX domain name detection."""
    assert is_cordex_domain(spec) is expected


def test_is_cordex_domain_rejects_non_string():
    """Test CORDEX domain detection rejects non-string input."""
    assert is_cordex_domain(11) is False
    assert is_cordex_domain({"domain": "EUR-11"}) is False


def test_parse_cell_spec_rejects_cordex_domain():
    """CORDEX domains must not be parsed as MxN cell specifications."""
    with pytest.raises(ValueError, match="Invalid MxN cell specification"):
        parse_cell_spec("EUR-11")


def test_cordex_stock_cube_eur11():
    """Test stock cube for EUR-11 matches the official domain grid."""
    domain = cx.cordex_domain("EUR-11", bounds=True)
    cube = _cordex_stock_cube("EUR-11")

    np.testing.assert_array_equal(
        cube.coord(var_name="rlat").points,
        domain["rlat"].data,
    )
    np.testing.assert_array_equal(
        cube.coord(var_name="rlon").points,
        domain["rlon"].data,
    )
    np.testing.assert_array_equal(
        cube.coord(var_name="lat").points,
        domain["lat"].data,
    )
    np.testing.assert_array_equal(
        cube.coord(var_name="lon").points,
        domain["lon"].data,
    )
    assert cube.coord(var_name="rlat").has_bounds()
    assert cube.coord(var_name="rlon").has_bounds()
    assert cube.coord(var_name="lat").has_bounds()
    assert cube.coord(var_name="lon").has_bounds()


@pytest.fixture
def global_cube():
    """Return a simple regular global cube for target-grid construction tests."""
    lat_coord = iris.coords.DimCoord(
        np.linspace(-85, 85, 18),
        standard_name="latitude",
        units="degrees",
    )
    lon_coord = iris.coords.DimCoord(
        np.linspace(5, 355, 36),
        standard_name="longitude",
        units="degrees",
    )
    lat_coord.guess_bounds()
    lon_coord.guess_bounds()
    return iris.cube.Cube(
        np.zeros((18, 36), dtype=np.float32),
        dim_coords_and_dims=[(lat_coord, 0), (lon_coord, 1)],
    )


def test_get_target_grid_cube_cordex_domain(global_cube):
    """Test target grid cube construction for a CORDEX domain."""
    target = _get_target_grid_cube(global_cube, "EUR-11")
    assert target.coords(var_name="rlat")
    assert target.coords(var_name="rlon")
