"""CMOR module."""
from __future__ import annotations

from iris.coords import Coord
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError

from .table import CMOR_TABLES, CoordinateInfo, VariableInfo

_ALTERNATIVE_GENERIC_LEV_COORDS = {
    'alevel': {
        'CMIP5': ['alt40', 'plevs'],
        'CMIP6': ['alt16', 'plev3'],
        'obs4MIPs': ['alt16', 'plev3'],
    },
    'zlevel': {
        'CMIP3': ['pressure'],
    },
}


def _get_alternative_generic_lev_coord(
    cube: Cube,
    coord_name: str,
    cmor_table_type: str,
) -> tuple[CoordinateInfo, Coord]:
    """Find alternative generic level coordinate in cube.

    Parameters
    ----------
    cube:
        Cube to be checked.
    coord_name:
        Name of the generic level coordinate.
    cmor_table_type:
        CMOR table type, e.g., CMIP3, CMIP5, CMIP6. Note: This is NOT the
        project of the dataset, but rather the entry `cmor_type` in
        `config-developer.yml`.

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
    alternatives_for_coord = _ALTERNATIVE_GENERIC_LEV_COORDS.get(
        coord_name, {}
    )
    allowed_alternatives = alternatives_for_coord.get(cmor_table_type, [])

    # Check if any of the allowed alternative coordinates is present in the
    # cube
    for allowed_alternative in allowed_alternatives:
        cmor_coord = CMOR_TABLES[cmor_table_type].coords[allowed_alternative]
        if cube.coords(var_name=cmor_coord.out_name):
            cube_coord = cube.coord(var_name=cmor_coord.out_name)
            return (cmor_coord, cube_coord)

    raise ValueError(
        f"Found no valid alternative coordinate for generic level coordinate "
        f"'{coord_name}'"
    )


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
    new_coord_name: str | None,
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
    new_coord.generic_lev_coords = (
        var_info.coordinates[generic_level_coord_name].generic_lev_coords
    )
    return new_coord


def _get_simplified_calendar(calendar: str) -> str:
    """Simplify calendar."""
    calendar_aliases = {
        'all_leap': '366_day',
        'noleap': '365_day',
        'gregorian': 'standard',
    }
    return calendar_aliases.get(calendar, calendar)


def _is_unstructured_grid(cube: Cube) -> bool:
    """Check if cube uses unstructured grid."""
    try:
        lat = cube.coord('latitude')
        lon = cube.coord('longitude')
    except CoordinateNotFoundError:
        pass
    else:
        if lat.ndim == 1 and (cube.coord_dims(lat) == cube.coord_dims(lon)):
            return True
    return False
