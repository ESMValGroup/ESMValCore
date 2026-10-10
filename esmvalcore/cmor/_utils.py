"""Utilities for CMOR module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

import esmvalcore.cmor.table

if TYPE_CHECKING:
    from collections.abc import Sequence

    from iris.coords import Coord
    from iris.cube import Cube

    from esmvalcore.cmor.table import (
        CoordinateInfo,
        InfoBase,
        VariableInfo,
    )
    from esmvalcore.config import Config, Session

logger = logging.getLogger(__name__)

# Standard names of the coordinates that can be computed from parametric
# vertical coordinates, see Appendix D of the CF conventions
# https://cfconventions.org/cf-conventions/cf-conventions.html#parametric-v-coord
# and `cf.constants.formula_terms_computed_standard_names` in cf-python.
# For some parametric coordinates, the computed standard name depends on the
# standard name of one of the formula terms. The CMOR tables use the terms
# relative to the geoid (orog: surface_altitude, deptho:
# sea_floor_depth_below_geoid), so the computed standard name is altitude.
_COMPUTED_STANDARD_NAMES: dict[str, str] = {
    "atmosphere_ln_pressure_coordinate": "air_pressure",
    "atmosphere_sigma_coordinate": "air_pressure",
    "atmosphere_hybrid_sigma_pressure_coordinate": "air_pressure",
    # Used in the CMIP7 CMOR tables (formula p = p0*lev*(ps/p0)**b). This is a
    # CF standard name, but CF-1.12 Appendix D does not define its formula.
    "atmosphere_hybrid_sigma_ln_pressure_coordinate": "air_pressure",
    "atmosphere_hybrid_height_coordinate": "altitude",
    "atmosphere_sleve_coordinate": "altitude",
    "ocean_sigma_coordinate": "altitude",
    "ocean_s_coordinate": "altitude",
    "ocean_s_coordinate_g1": "altitude",
    "ocean_s_coordinate_g2": "altitude",
    "ocean_sigma_z_coordinate": "altitude",
    "ocean_double_sigma_coordinate": "altitude",
}


def _get_computed_standard_names(
    tables: InfoBase,
    coord_name: str,
) -> set[str]:
    """Get the standard names that can be computed for a generic level.

    Parameters
    ----------
    tables:
        The CMOR tables.
    coord_name:
        Name of the generic level coordinate, e.g. ``alevel``.

    Returns
    -------
    set[str]
        The standard names of the coordinates that can be computed from
        the parametric vertical coordinates that are allowed for the
        generic level coordinate.
    """
    parametric_coords = [
        coord
        for coord in tables.coords.values()
        if coord.generic_lev_name == coord_name
    ]
    if not parametric_coords:
        # CMOR tables in the CMIP5 format do not link the parametric
        # coordinates to the generic level coordinates, so use all
        # atmosphere or ocean parametric coordinates instead.
        prefix = {"a": "atmosphere_", "o": "ocean_"}.get(coord_name[:1], "")
        parametric_coords = [
            coord
            for coord in tables.coords.values()
            if coord.standard_name.startswith(prefix)
        ]
    return {
        _COMPUTED_STANDARD_NAMES[coord.standard_name]
        for coord in parametric_coords
        if coord.standard_name in _COMPUTED_STANDARD_NAMES
    }


def _count_requested_values(
    cmor_coord: CoordinateInfo,
    cube_coord: Coord,
) -> int:
    """Count how many of the requested values are present in the cube."""
    if not cmor_coord.requested or cube_coord.ndim != 1:
        return 0
    try:
        requested = np.array(cmor_coord.requested, dtype=float)
        points = cube_coord.units.convert(
            np.asarray(cube_coord.points, dtype=float),
            cmor_coord.units,
        )
    except (TypeError, ValueError):
        # Non-numeric values or units that cannot be converted.
        return 0
    return int(np.isclose(requested[:, np.newaxis], points).any(axis=1).sum())


def _rank_alternative(
    cmor_coord: CoordinateInfo,
    cube_coord: Coord,
) -> tuple[int, int, int]:
    """Rank how well a cube coordinate matches a CMOR coordinate.

    Lower values indicate a better match.
    """
    n_requested = len(cmor_coord.requested)
    n_points = len(cube_coord.points) if cube_coord.ndim == 1 else None
    if n_requested == 0:
        # No requested values, so any number of levels is fine.
        rank, difference = 1, 0
    elif n_requested == n_points:
        rank, difference = 0, 0
    else:
        rank = 2
        difference = 0 if n_points is None else abs(n_requested - n_points)
    return (
        rank,
        difference,
        -_count_requested_values(cmor_coord, cube_coord),
    )


def _get_alternative_generic_lev_coord(
    cube: Cube,
    coord_name: str,
    project: str,
    session: Session | Config,
) -> tuple[CoordinateInfo, Coord]:
    """Find alternative generic level coordinate in cube.

    Generic level coordinates (e.g. ``alevel``) stand for parametric vertical
    coordinates (e.g. ``standard_hybrid_sigma``). As an alternative, a
    regular vertical coordinate with the standard name of the coordinate that
    can be computed from the parametric coordinate (e.g. ``air_pressure``)
    can be used. The alternative is selected from the coordinates in the
    CMOR tables of the project. If several coordinates in the CMOR tables
    match, the one where the number of requested values matches the number of
    points in the cube best is selected. Remaining ties are resolved by
    selecting the coordinate with the largest number of requested values
    present in the cube and finally by the order in the CMOR tables.

    Parameters
    ----------
    cube:
        Cube to be checked.
    coord_name:
        Name of the generic level coordinate.
    project:
        The project that the data belongs to.
    session:
        The configuration.

    Returns
    -------
    tuple[CoordinateInfo, Coord]
        Coordinate information from the CMOR tables and the corresponding
        coordinate in the cube.

    Raises
    ------
    ValueError
        No valid alternative generic level coordinate present in cube.

    """
    tables = esmvalcore.cmor.table.get_tables(session, project)
    standard_names = _get_computed_standard_names(tables, coord_name)
    matches = [
        (cmor_coord, cube.coord(var_name=cmor_coord.out_name))
        for cmor_coord in tables.coords.values()
        if cmor_coord.standard_name in standard_names
        and cmor_coord.axis == "Z"
        and not cmor_coord.value  # Exclude single levels, e.g. p850
        and not cmor_coord.generic_level
        and cube.coords(var_name=cmor_coord.out_name)
    ]
    if not matches:
        msg = (
            f"Found no valid alternative coordinate for generic level "
            f"coordinate '{coord_name}'"
        )
        raise ValueError(msg)
    # `min` returns the first best match, so ties are resolved by the order in
    # the CMOR tables.
    return min(matches, key=lambda match: _rank_alternative(*match))


def _get_generic_lev_coord_names(
    cube: Cube,
    cmor_coord: CoordinateInfo,
) -> tuple[str | None, str | None, str | None]:
    """Try to get names of a generic level coordinate.

    Parameters
    ----------
    cube:
        Cube to be checked.
    cmor_coord:
        Coordinate information from the CMOR table with a non-emmpty
        `generic_lev_coords` :obj:`dict`.

    Returns
    -------
    tuple[str | None, str | None, str | None]
        Tuple of `standard_name`, `out_name`, and `name` of the generic level
        coordinate present in the cube. Values are ``None`` if generic level
        coordinate has not been found in cube.

    """
    standard_name = None
    out_name = None
    name = None

    # Iterate over all possible generic level coordinates
    for coord in cmor_coord.generic_lev_coords.values():
        # First, try to use var_name to find coordinate
        if cube.coords(var_name=coord.out_name):
            cube_coord = cube.coord(var_name=coord.out_name)
            out_name = coord.out_name
            if cube_coord.standard_name == coord.standard_name:
                standard_name = coord.standard_name
                name = coord.name

        # Second, try to use standard_name to find coordinate
        elif cube.coords(coord.standard_name):
            standard_name = coord.standard_name
            name = coord.name

    return (standard_name, out_name, name)


def _get_new_generic_level_coord(
    var_info: VariableInfo,
    generic_level_coord: CoordinateInfo,
    generic_level_coord_name: str,
    new_coord_name: str,
) -> CoordinateInfo:
    """Get new generic level coordinate.

    There are a variety of possible options for each generic level coordinate
    (e.g., `alevel`) which is actually present in a cube, for example,
    `hybrid_height` or `standard_hybrid_sigma`. This function returns the new
    coordinate (e.g., `new_coord_name=hybrid_height`) with the relevant
    metadata.

    Note
    ----
    This alters the corresponding entry of the original generic level
    coordinate's `generic_level_coords` attribute (i.e.,
    ``generic_level_coord.generic_level_coords[new_coord_name]`) in-place!

    Parameters
    ----------
    var_info:
        CMOR variable information.
    generic_level_coord:
        Original generic level coordinate.
    generic_level_coord_name:
        Original name of the generic level coordinate (e.g., `alevel`).
    new_coord_name:
        Name of the new generic level coordinate (e.g., `hybrid_height`).

    Returns
    -------
    CoordinateInfo
        New generic level coordinate.

    """
    new_coord = generic_level_coord.generic_lev_coords[new_coord_name]
    new_coord.generic_level = True
    new_coord.generic_lev_coords = var_info.coordinates[
        generic_level_coord_name
    ].generic_lev_coords
    return new_coord


def _get_simplified_calendar(calendar: str) -> str:
    """Simplify calendar."""
    calendar_aliases = {
        "all_leap": "366_day",
        "noleap": "365_day",
        "gregorian": "standard",
    }
    return calendar_aliases.get(calendar, calendar)


def _get_single_cube(
    cube_list: Sequence[Cube],
    short_name: str,
    dataset_str: str | None = None,
) -> Cube:
    if len(cube_list) == 1:
        return cube_list[0]
    cube = None
    for raw_cube in cube_list:
        if raw_cube.var_name == short_name:
            cube = raw_cube
            break

    dataset_str = "" if dataset_str is None else f" in {dataset_str}"

    if not cube:
        msg = (
            f"More than one cube found for variable {short_name}{dataset_str} "
            f"but none of their var_names match the expected.\nFull list of "
            f"cubes encountered: {cube_list}"
        )
        raise ValueError(msg)
    logger.warning(
        "Found variable %s%s, but there were other present in the file. Those "
        "extra variables are usually metadata (cell area, latitude "
        "descriptions) that was not saved according to CF-conventions. It is "
        "possible that errors appear further on because of this.\nFull list "
        "of cubes encountered: %s",
        short_name,
        dataset_str,
        cube_list,
    )
    return cube
