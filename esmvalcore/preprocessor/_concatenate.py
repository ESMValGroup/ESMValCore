"""Module containing :func:`esmvalcore.preprocessor.concatenate`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NamedTuple, Self

import cftime
import iris.exceptions
import numpy as np
from iris.cube import CubeList

from esmvalcore.cmor.check import CheckLevels
from esmvalcore.esgf.facets import FACETS
from esmvalcore.iris_helpers import merge_cube_attributes
from esmvalcore.preprocessor._shared import _rechunk_aux_factory_dependencies

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from iris.coords import Coord, DimCoord
    from iris.cube import Cube

logger = logging.getLogger(__name__)


def _delete_attributes(iris_object: Cube | Coord, atts: Iterable[str]) -> None:
    """Delete attributes from Iris cube or coordinate."""
    for att in atts:
        if att in iris_object.attributes:
            del iris_object.attributes[att]


def _concatenate_cubes(
    cubes: Iterable[Cube],
    check_level: CheckLevels,
) -> CubeList:
    """Concatenate cubes according to the check_level."""
    kwargs = {
        "check_aux_coords": True,
        "check_cell_measures": True,
        "check_ancils": True,
        "check_derived_coords": True,
    }

    if check_level > CheckLevels.DEFAULT:
        kwargs = dict.fromkeys(kwargs, False)
        logger.debug(
            "Concatenation will be performed without checking "
            "auxiliary coordinates, cell measures, ancillaries "
            "and derived coordinates present in the cubes.",
        )

    return CubeList(cubes).concatenate(**kwargs)


class _TimesHelper:
    def __init__(self, time: DimCoord) -> None:
        self.times = time.core_points()
        self.units = str(time.units)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.times, name)

    def __len__(self) -> int:
        return len(self.times)

    def __getitem__(self, key: Any) -> Any:
        return self.times[key]


def _remove_time_overlaps(cubes: CubeList) -> CubeList:
    """Handle time overlaps.

    Parameters
    ----------
    cubes : iris.cube.CubeList
        A list of cubes belonging to a single timeseries,
        ordered by starting point with possible overlaps.

    Returns
    -------
    iris.cube.CubeList
        A list of cubes belonging to a single timeseries,
        ordered by starting point with no overlaps.
    """
    if len(cubes) < 2:
        return cubes

    class _TrackedCube(NamedTuple):
        cube: Cube
        times: iris.coords.DimCoord
        start: float
        end: float

        @classmethod
        def from_cube(cls, cube: Cube) -> Self:
            """Construct tracked cube."""
            times = cube.coord("time")
            start, end = times.core_points()[[0, -1]]
            return cls(cube, times, start, end)

    new_cubes = CubeList()
    current_cube = _TrackedCube.from_cube(cubes[0])
    for new_cube in map(_TrackedCube.from_cube, cubes[1:]):
        if new_cube.start > current_cube.end:
            # no overlap, use current cube and start again from new cube
            logger.debug("Using %s", current_cube.cube)
            new_cubes.append(current_cube.cube)
            current_cube = new_cube
            continue
        # overlap
        if current_cube.end > new_cube.end:
            # current cube ends after new one, just forget new cube
            logger.debug(
                "Discarding %s because the time range "
                "is already covered by %s",
                new_cube.cube,
                current_cube.cube,
            )
            continue
        if new_cube.start == current_cube.start:
            # new cube completely covers current one
            # forget current cube
            current_cube = new_cube
            logger.debug(
                "Discarding %s because the time range is covered by %s",
                current_cube.cube,
                new_cube.cube,
            )
            continue
        # new cube ends after current one,
        # use all of new cube, and shorten current cube to
        # eliminate overlap with new cube
        cut_index = (
            cftime.time2index(
                new_cube.start,
                _TimesHelper(current_cube.times),
                current_cube.times.units.calendar,
                select="before",
            )
            + 1
        )
        logger.debug(
            "Using %s shortened to %s due to overlap",
            current_cube.cube,
            current_cube.times.cell(cut_index).point,
        )
        new_cubes.append(current_cube.cube[:cut_index])
        current_cube = new_cube

    logger.debug("Using %s", current_cube.cube)
    new_cubes.append(current_cube.cube)

    return new_cubes


def _fix_calendars(cubes: Sequence[Cube]) -> None:
    """Check and homogenise calendars, if possible."""
    calendars = [cube.coord("time").units.calendar for cube in cubes]
    unique_calendars = np.unique(calendars)

    calendar_ocurrences = np.array(
        [calendars.count(calendar) for calendar in unique_calendars],
    )
    calendar_index = int(
        np.argwhere(calendar_ocurrences == calendar_ocurrences.max()),
    )

    for cube in cubes:
        time_coord = cube.coord("time")
        old_calendar = time_coord.units.calendar
        if old_calendar != unique_calendars[calendar_index]:
            new_unit = time_coord.units.change_calendar(
                unique_calendars[calendar_index],
            )
            time_coord.units = new_unit


def _raise_concatenation_exception(cubes: Sequence[Cube]) -> None:
    """Raise an error for concatenation."""
    # Concatenation not successful -> retrieve exact error message
    try:
        CubeList(cubes).concatenate_cube()
    except iris.exceptions.ConcatenateError as exc:
        msg = str(exc)
        logger.error("Can not concatenate cubes into a single one: %s", msg)
        logger.error("Resulting cubes:")
        for cube in cubes:
            logger.error(cube)
            time = cube.coord("time")
            logger.error("From %s to %s", time.cell(0), time.cell(-1))

        msg = f"Can not concatenate cubes: {msg}"
        raise ValueError(msg) from exc


def _sort_cubes_by_time(cubes: Iterable[Cube]) -> list[Cube]:
    """Sort CubeList by time coordinate."""
    try:
        cubes = sorted(cubes, key=lambda c: c.coord("time").cell(0).point)
    except iris.exceptions.CoordinateNotFoundError as exc:
        msg = f"One or more cubes {cubes} are missing time coordinate: {exc!s}"
        raise ValueError(msg) from exc
    except TypeError as error:
        msg = f"Cubes cannot be sorted due to differing time units: {error!s}"
        raise TypeError(msg) from error
    return cubes


def _concatenate_cubes_by_experiment(cubes: Sequence[Cube]) -> Sequence[Cube]:
    """Concatenate cubes by experiment.

    This ensures overlapping (branching) experiments are handled correctly.
    """
    # get the possible facet names in CMIP3, 5, 6 for exp
    # currently these are 'experiment', 'experiment_id'
    exp_facet_names = {
        project["exp"] for project in FACETS.values() if "exp" in project
    }

    def get_exp(cube: Cube) -> Any:
        for key in exp_facet_names:
            if key in cube.attributes:
                return cube.attributes[key]
        return ""

    experiments = {get_exp(cube) for cube in cubes}
    if len(experiments) > 1:
        # first do experiment-wise concatenation, then time-based
        cubes = [
            concatenate([cube for cube in cubes if get_exp(cube) == exp])
            for exp in experiments
        ]

    return cubes


def concatenate(
    cubes: Sequence[Cube],
    check_level: CheckLevels = CheckLevels.DEFAULT,
) -> Cube:
    """Concatenate all cubes after fixing metadata.

    Parameters
    ----------
    cubes: iterable of iris.cube.Cube
        Data cubes to be concatenated
    check_level: CheckLevels
        Level of strictness of the checks in the concatenation.

    Returns
    -------
    cube: iris.cube.Cube
        Resulting concatenated cube.

    Raises
    ------
    ValueError
        Concatenation was not possible.
    """
    if not cubes:
        return cubes
    if len(cubes) == 1:
        return cubes[0]

    for cube in cubes:
        # Remove attributes that cause issues with merging and concatenation
        _delete_attributes(
            cube,
            ("creation_date", "tracking_id", "history", "comment"),
        )
        for coord in cube.coords():
            # CMOR sometimes adds a history to the coordinates.
            _delete_attributes(coord, ("history",))

    cubes = _concatenate_cubes_by_experiment(cubes)

    merge_cube_attributes(cubes)
    cubes = _sort_cubes_by_time(cubes)
    _fix_calendars(cubes)
    cubes = _remove_time_overlaps(cubes)
    cubes = [_rechunk_aux_factory_dependencies(cube) for cube in cubes]
    result = _concatenate_cubes(cubes, check_level=check_level)

    if len(result) == 1:
        result = result[0]
    else:
        _raise_concatenation_exception(result)

    return result
