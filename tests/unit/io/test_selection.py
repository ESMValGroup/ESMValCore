"""Tests for selecting preferred samples across catalog data sources."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import pytest
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube

from esmvalcore.io._selection import select_cubes, version_key
from esmvalcore.io.local import LocalFile

if TYPE_CHECKING:
    from esmvalcore.typing import FacetValue


def _element(name: str, **facets: FacetValue) -> LocalFile:
    file = LocalFile(name)
    file.facets.update(facets)
    return file


def _cube(points: list[int], value: float = 1.0) -> Cube:
    return Cube(
        da.full((len(points), 2), value, chunks=(1, 2)),
        var_name="tas",
        standard_name="air_temperature",
        units="K",
        dim_coords_and_dims=[
            (
                DimCoord(
                    points,
                    standard_name="time",
                    units=Unit("days since 2000-01-01", calendar="360_day"),
                ),
                0,
            ),
            (
                DimCoord(
                    [0.0, 10.0],
                    standard_name="latitude",
                    units="degrees_north",
                ),
                1,
            ),
        ],
    )


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("v20200101", (1, 20200101, "")),
        (20200101, (1, 20200101, "")),
        ("v10", (1, 10, "")),
        ("release-a", (0, 0, "release-a")),
        ("", (0, 0, "")),
    ],
)
def test_version_key(
    version: FacetValue,
    expected: tuple[int, int, str],
) -> None:
    assert version_key(_element("data", version=version)) == expected


@pytest.mark.parametrize("preferred_times", [[1, 2], [0, 2]])
@pytest.mark.parametrize("time_axis", [0, 1])
def test_partial_overlap_and_internal_gaps(
    preferred_times: list[int],
    time_axis: int,
) -> None:
    """Retain lower-priority samples wherever the preferred source has gaps."""
    preferred = _cube(preferred_times, 10.0)
    fallback = _cube([0, 1, 2, 3], 20.0)
    if time_axis == 1:
        preferred.transpose()
        fallback.transpose()
    result = select_cubes(
        [_element("catalog", ensemble="r1"), _element("local", ensemble="r1")],
        [[preferred], [fallback]],
        {},
    )
    assert all(cube.has_lazy_data() for cube in result)
    np.testing.assert_array_equal(
        np.concatenate([cube.coord("time").points for cube in result]),
        [0, 1, 2, 3],
    )
    values = np.concatenate([cube.data for cube in result], axis=time_axis)
    np.testing.assert_array_equal(
        np.moveaxis(values, time_axis, 0)[:, 0],
        [10.0 if time in preferred_times else 20.0 for time in range(4)],
    )


def test_same_times_with_different_origins() -> None:
    preferred = _cube([0, 1], 10.0)
    fallback = _cube([0, 1], 20.0)
    fallback.coord("time").convert_units(
        Unit("hours since 1999-01-01", calendar="360_day"),
    )
    result = select_cubes(
        [_element("a"), _element("b")],
        [[preferred], [fallback]],
        {},
    )
    assert len(result) == 1
    np.testing.assert_array_equal(result[0].data, 10.0)


@pytest.mark.parametrize("facet", ["ensemble", "grid", "exp", "short_name"])
def test_distinct_identities_are_preserved(facet: str) -> None:
    first, second = _cube([0]), _cube([0], 2.0)
    result = select_cubes(
        [_element("a", **{facet: "one"}), _element("b", **{facet: "two"})],
        [[first], [second]],
        {},
    )
    assert result == [first, second]


def test_recipe_supplies_missing_identity_facet() -> None:
    result = select_cubes(
        [_element("a", ensemble="r1"), _element("b")],
        [[_cube([0])], [_cube([0], 2.0)]],
        {"ensemble": "r1"},
    )
    assert len(result) == 1
    np.testing.assert_array_equal(result[0].data, 1.0)


def test_static_field_uses_preferred_source() -> None:
    first, second = _cube([0])[0], _cube([0], 2.0)[0]
    first.remove_coord("time")
    second.remove_coord("time")
    result = select_cubes(
        [_element("a"), _element("b")],
        [[first], [second]],
        {},
    )
    assert result == [first]
    assert result[0].has_lazy_data()


@pytest.mark.parametrize("histories", [(None, "new"), ("old", "new")])
def test_coordinate_history_preserves_samples_and_input_metadata(
    histories: tuple[str | None, str],
) -> None:
    """Ignore coordinate history without modifying the original cubes."""
    first, second = _cube([0, 1], 10.0), _cube([1, 2], 20.0)
    for cube, history in zip((first, second), histories, strict=True):
        if history is not None:
            cube.coord("latitude").attributes["history"] = history
    result = select_cubes(
        [_element("a"), _element("b")],
        [[first], [second]],
        {},
    )
    assert all(cube.has_lazy_data() for cube in [first, second, *result])
    assert (
        tuple(
            cube.coord("latitude").attributes.get("history")
            for cube in (first, second)
        )
        == histories
    )
    np.testing.assert_array_equal(
        np.concatenate([cube.coord("time").points for cube in result]),
        [0, 1, 2],
    )
    np.testing.assert_array_equal(
        np.concatenate([cube.data[:, 0] for cube in result]),
        [10.0, 10.0, 20.0],
    )


def test_coordinate_history_does_not_hide_other_metadata_differences() -> None:
    first, second = _cube([0, 1]), _cube([0, 1], 2.0)
    first.coord("latitude").attributes["history"] = "old"
    second.coord("latitude").attributes.update(
        {"history": "new", "uncertainty": 0.1},
    )
    with pytest.raises(
        ValueError,
        match="different coordinates, units or calendars",
    ):
        select_cubes([_element("a"), _element("b")], [[first], [second]], {})


@pytest.mark.parametrize(
    "difference",
    ["units", "grid", "calendar", "time", "scalar"],
)
def test_incompatible_coordinates_fail(difference: str) -> None:
    first, second = _cube([0, 1]), _cube([0, 1], 2.0)
    if difference == "units":
        second.units = "degC"
    elif difference == "grid":
        second.coord("latitude").points = [20.0, 30.0]
    elif difference == "calendar":
        second.coord("time").units = Unit(
            "days since 2000-01-01",
            calendar="standard",
        )
    elif difference == "time":
        second.remove_coord("time")
    else:
        second = second[0]
    with pytest.raises(
        ValueError,
        match="different coordinates, units or calendars",
    ):
        select_cubes([_element("a"), _element("b")], [[first], [second]], {})
