"""Select overlapping data using source preference and actual coordinates."""

from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from iris.cube import Cube

    from esmvalcore.io.protocol import DataElement
    from esmvalcore.typing import Facets


def version_key(element: DataElement) -> tuple[int, int, str]:
    """Compare numeric releases with or without the conventional v prefix."""
    version = str(element.facets.get("version", ""))
    if version.startswith("v") and version[1:].isdigit():
        version = version[1:]
    if version.isdigit():
        return (1, int(version), "")
    return (0, 0, version)


def select_cubes(
    elements: Sequence[DataElement],
    batches: Sequence[Sequence[Cube]],
    facets: Facets,
) -> list[Cube]:
    """Keep preferred samples, retaining complementary periods from others.

    Elements are ordered by preference. Selection happens after metadata
    fixes, without realizing the data arrays or assuming continuous coverage.
    Facets distinguish datasets even when their filenames or catalog keys
    differ. Recipe facets supply identity values absent from a source.
    """
    keys = sorted(
        {key for element in elements for key in element.facets}
        - {"version", "timerange"},
    )
    groups: dict[tuple, list[Cube]] = {}
    for element, cubes in zip(elements, batches, strict=True):
        identity = {**facets, **element.facets}
        key = tuple((name, str(identity.get(name))) for name in keys)
        for cube in cubes:
            groups.setdefault((key, cube.var_name, cube.name()), []).append(
                cube,
            )

    return [
        cube for group in groups.values() for cube in _select_samples(group)
    ]


def _select_samples(cubes: Sequence[Cube]) -> list[Cube]:
    """Select each timestamp once, then slice into chronological runs."""
    if len(cubes) == 1:
        return list(cubes)

    reference = cubes[0]
    has_time = bool(reference.coords("time"))
    locations: dict[tuple[int, ...], tuple[int, int]] = {}
    for cube_index, cube in enumerate(cubes):
        _check_coordinates(reference, cube)
        if not has_time:
            continue
        time = cube.coord("time")
        for index, date in enumerate(time.units.num2date(time.points)):
            key = (
                date.year,
                date.month,
                date.day,
                date.hour,
                date.minute,
                date.second,
                date.microsecond,
            )
            locations.setdefault(key, (cube_index, index))

    if not has_time:
        return [reference]

    # Split at changes of owner or gaps in an input cube's indices. Keeping
    # separate runs prevents concatenate from discarding interleaved periods.
    ordered = [locations[date] for date in sorted(locations)]
    result = []
    for _, run in groupby(
        enumerate(ordered),
        key=lambda item: (item[1][0], item[1][1] - item[0]),
    ):
        indices = [item[1] for item in run]
        cube = cubes[indices[0][0]]
        dimension = cube.coord_dims("time")[0]
        selection = [slice(None)] * cube.ndim
        selection[dimension] = slice(indices[0][1], indices[-1][1] + 1)
        result.append(cube[tuple(selection)])
    return result


def _check_coordinates(reference: Cube, cube: Cube) -> None:
    """Do not silently discard incompatible grids, calendars or variables."""

    def coordinates(item: Cube) -> list:
        time_dims = item.coord_dims("time") if item.coords("time") else ()
        result = []
        for coord in item.coords():
            dims = item.coord_dims(coord)
            if coord.name() == "time" or set(dims).intersection(time_dims):
                continue
            comparison_coord = coord
            if "history" in coord.attributes:
                # Concatenate also ignores CMOR coordinate history. Compare
                # copies so selection does not change the input metadata.
                comparison_coord = coord.copy()
                del comparison_coord.attributes["history"]
            result.append((comparison_coord, dims))
        return result

    reference_time = reference.coords("time")
    cube_time = cube.coords("time")
    if (
        cube.units != reference.units
        or coordinates(cube) != coordinates(reference)
        or bool(cube_time) != bool(reference_time)
        or (
            cube_time
            and (
                len(cube.coord_dims("time")) != 1
                or cube_time[0].units.calendar
                != reference_time[0].units.calendar
            )
        )
    ):
        msg = (
            "Cannot select overlapping data with different coordinates, "
            "units or calendars. Check the source facet mappings and data."
        )
        raise ValueError(msg)
