"""Contains class for automatic dataset fixes."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
from cf_units import Unit
from iris.coords import Coord, CoordExtent
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError
from iris.util import reverse

from esmvalcore.iris_helpers import date2num

from ..table import CMOR_TABLES, CoordinateInfo, VariableInfo, get_var_info

logger = logging.getLogger(__name__)


ALTERNATIVE_GENERIC_LEV_COORDS = {
    'alevel': {
        'CMIP5': ['alt40', 'plevs'],
        'CMIP6': ['alt16', 'plev3'],
        'obs4MIPs': ['alt16', 'plev3'],
    },
    'zlevel': {
        'CMIP3': ['pressure'],
    },
}


def get_alternative_generic_lev_coord(
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
    alternatives_for_coord = ALTERNATIVE_GENERIC_LEV_COORDS.get(coord_name, {})
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


def get_generic_lev_coord_names(
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


def get_next_month(month: int, year: int) -> tuple[int, int]:
    """Get next month and year.

    Parameters
    ----------
    month:
        Current month.
    year:
        Current year.

    Returns
    -------
    tuple[int, int]
        Next month and next year.

    """
    if month != 12:
        return month + 1, year
    return 1, year + 1


def get_time_bounds(time: Coord, freq: str):
    """Get bounds for time coordinate.

    Parameters
    ----------
    time:
        Time coordinate.
    freq:
        Frequency.

    Returns
    -------
    np.ndarray
        Time bounds

    Raises
    ------
    NotImplementedError
        Non-supported frequency is given.

    """
    bounds = []
    dates = time.units.num2date(time.points)
    for step, date in enumerate(dates):
        month = date.month
        year = date.year
        if freq in ['mon', 'mo']:
            next_month, next_year = get_next_month(month, year)
            min_bound = date2num(datetime(year, month, 1, 0, 0),
                                 time.units, time.dtype)
            max_bound = date2num(datetime(next_year, next_month, 1, 0, 0),
                                 time.units, time.dtype)
        elif freq == 'yr':
            min_bound = date2num(datetime(year, 1, 1, 0, 0),
                                 time.units, time.dtype)
            max_bound = date2num(datetime(year + 1, 1, 1, 0, 0),
                                 time.units, time.dtype)
        elif freq == 'dec':
            min_bound = date2num(datetime(year, 1, 1, 0, 0),
                                 time.units, time.dtype)
            max_bound = date2num(datetime(year + 10, 1, 1, 0, 0),
                                 time.units, time.dtype)
        else:
            delta = {
                'day': 12 / 24,
                '6hr': 3 / 24,
                '3hr': 1.5 / 24,
                '1hr': 0.5 / 24,
            }
            if freq not in delta:
                raise NotImplementedError(
                    f"Cannot guess time bounds for frequency '{freq}'"
                )
            point = time.points[step]
            min_bound = point - delta[freq]
            max_bound = point + delta[freq]
        bounds.append([min_bound, max_bound])

    return np.array(bounds)


def is_unstructured_grid(cube: Cube) -> bool:
    """Check if cube uses unstructured grid.

    Parameters
    ----------
    cube:
        Cube to check.

    Returns
    -------
    bool
        ``True`` if cube uses unstructured grid, ```False`` if not.

    """
    try:
        lat = cube.coord('latitude')
        lon = cube.coord('longitude')
    except CoordinateNotFoundError:
        pass
    else:
        if lat.ndim == 1 and (cube.coord_dims(lat) == cube.coord_dims(lon)):
            return True
    return False


def simplify_calendar(calendar: str) -> str:
    """Simplify calendar.

    Parameters
    ----------
    calendar:
        Input calendar.

    Returns
    -------
    str
        Simplified calendar.

    """
    calendar_aliases = {
        'all_leap': '366_day',
        'noleap': '365_day',
        'gregorian': 'standard',
    }
    return calendar_aliases.get(calendar, calendar)


class AutomaticFix:
    """Class providing automatic fixes for all datasets."""

    def __init__(
        self,
        var_info: VariableInfo | None,
        frequency: Optional[str] = None,
    ) -> None:
        """Initialize class member.

        Parameters
        ----------
        var_info:
            Variable information from the CMOR table.
        frequency:
            Expected frequency of the variable. If not given, use the one from
            the variable information.

        Raises
        ------
        ValueError
            Variable information is not available (i.e., `var_info=None`):

        """
        if var_info is None:
            raise ValueError(
                "Cannot setup automatic fix if no variable information is "
                "given (i.e., var_info=None)"
            )
        if frequency is None:
            frequency = var_info.frequency

        self.var_info = var_info
        self.frequency = frequency

    @classmethod
    def from_facets(
        cls,
        project: str,
        mip: str,
        short_name: str,
        frequency: Optional[str] = None,
    ) -> AutomaticFix:
        """Get dataset's facets (i.e., `project`, `mip`, and `short_name`).

        Parameters
        ----------
        project:
            The dataset's project.
        mip:
            The variable's MIP.
        short_name:
            The variable's short name.
        frequency:
            Expected frequency of the variable. If not given, use the one from
            the CMOR table.

        Returns
        -------
        AutomaticFix
            AutomaticFix object.

        """
        var_info = get_var_info(project, mip, short_name)
        return cls(var_info, frequency=frequency)

    @staticmethod
    def _msg_prefix(cube: Cube) -> str:
        """Get prefix for log messages."""
        if 'source_file' in cube.attributes:
            return f"For file {cube.attributes['source_file']}: "
        return f"For variable {cube.var_name}: "

    def _debug_msg(self, cube: Cube, msg: str, *args) -> None:
        """Print debug message."""
        msg = self._msg_prefix(cube) + msg
        logger.debug(msg, *args)

    def _warning_msg(self, cube: Cube, msg: str, *args) -> None:
        """Print debug message."""
        msg = self._msg_prefix(cube) + msg
        logger.warning(msg, *args)

    @staticmethod
    def _set_range_in_0_360(array: np.ndarray) -> np.ndarray:
        """Convert longitude coordinate to [0, 360]."""
        return (array + 360.0) % 360.0

    def _reverse_coord(self, cube: Cube, coord: Coord) -> tuple[Cube, Coord]:
        """Reverse cube along a given coordinate."""
        if coord.ndim == 1:
            cube = reverse(cube, cube.coord_dims(coord))
            reversed_coord = cube.coord(var_name=coord.var_name)
            if reversed_coord.has_bounds():
                bounds = reversed_coord.bounds
                right_bounds = bounds[:-2, 1]
                left_bounds = bounds[1:-1, 0]
                if np.all(right_bounds != left_bounds):
                    reversed_coord.bounds = np.fliplr(bounds)
                    coord = reversed_coord
            self._debug_msg(
                cube,
                "Coordinate %s values have been reversed",
                coord.var_name,
            )
        return (cube, coord)

    def _get_effective_units(self) -> str:
        """Get effective units."""
        if self.var_info.units.lower() == 'psu':
            return '1'
        return self.var_info.units

    def fix_metadata(self, cube: Cube) -> Cube:
        """Fix cube metadata.

        Parameters
        ----------
        cube:
            Cube to be fixed.

        Returns
        -------
        Cube
            Fixed cube.

        """
        cube = self.fix_standard_name(cube)
        cube = self.fix_long_name(cube)
        cube = self.fix_psu_units(cube)

        cube = self.fix_regular_coord_names(cube)
        cube = self.fix_alternative_generic_level_coords(cube)
        cube = self.fix_coords(cube)
        cube = self.fix_time_coord(cube)

        return cube

    def fix_data(self, cube: Cube) -> Cube:
        """Fix cube data.

        Parameters
        ----------
        cube:
            Cube to be fixed.

        Returns
        -------
        Cube
            Fixed cube.

        """
        cube = self.fix_units(cube)

        return cube

    def fix_units(self, cube: Cube) -> Cube:
        """Fix cube units.

        Parameters
        ----------
        cube:
            Cube to be fixed.

        Returns
        -------
        Cube
            Fixed cube.

        """
        if self.var_info.units:
            units = self._get_effective_units()

            # We use str(cube.units) in the following to catch `degrees` !=
            # `degrees_north`
            if str(cube.units) != units:
                old_units = cube.units
                cube.convert_units(units)
                self._warning_msg(
                    cube,
                    "Converted cube units from '%s' to '%s'",
                    old_units,
                    cube.units,
                )
        return cube

    def fix_standard_name(self, cube: Cube) -> Cube:
        """Fix standard_name.

        Parameters
        ----------
        cube:
            Cube to be fixed.

        Returns
        -------
        Cube
            Fixed cube.

        """
        # Do not change empty standard names
        if not self.var_info.standard_name:
            return cube

        if cube.standard_name != self.var_info.standard_name:
            self._warning_msg(
                cube,
                "Standard name changed from '%s' to '%s'",
                cube.standard_name,
                self.var_info.standard_name,
            )
            cube.standard_name = self.var_info.standard_name

        return cube

    def fix_long_name(self, cube: Cube) -> Cube:
        """Fix long_name.

        Parameters
        ----------
        cube:
            Cube to be fixed.

        Returns
        -------
        Cube
            Fixed cube.

        """
        # Do not change empty long names
        if not self.var_info.long_name:
            return cube

        if cube.long_name != self.var_info.long_name:
            self._warning_msg(
                cube,
                "Long name changed from '%s' to '%s'",
                cube.long_name,
                self.var_info.long_name,
            )
            cube.long_name = self.var_info.long_name

        return cube

    def fix_psu_units(self, cube: Cube) -> Cube:
        """Fix psu units.

        Parameters
        ----------
        cube:
            Cube to be fixed.

        Returns
        -------
        Cube
            Fixed cube.

        """
        if cube.attributes.get('invalid_units', '').lower() == 'psu':
            cube.units = '1'
            cube.attributes.pop('invalid_units')
            self._debug_msg(cube, "Units converted from 'psu' to '1'")
        return cube

    def fix_regular_coord_names(self, cube: Cube) -> Cube:
        """Fix regular (non-generic-level) coordinate names.

        Parameters
        ----------
        cube:
            Cube to be fixed.

        Returns
        -------
        Cube
            Fixed cube.

        """
        for cmor_coord in self.var_info.coordinates.values():
            if cmor_coord.generic_level:
                continue  # Ignore generic level coordinate in this function
            if cube.coords(var_name=cmor_coord.out_name):
                continue  # Coordinate found -> fine
            if cube.coords(cmor_coord.standard_name):
                cube_coord = cube.coord(cmor_coord.standard_name)
                self.fix_cmip6_multidim_lat_lon_coord(
                    cube, cmor_coord, cube_coord
                )
        return cube

    def fix_alternative_generic_level_coords(self, cube: Cube) -> Cube:
        """Fix alternative generic level coordinates.

        Parameters
        ----------
        cube:
            Cube to be fixed.

        Returns
        -------
        Cube
            Fixed cube.

        """
        for (coord_name, cmor_coord) in self.var_info.coordinates.items():
            if not cmor_coord.generic_level:
                continue  # Ignore non-generic-level coordinates
            if not cmor_coord.generic_lev_coords:
                continue  # Cannot fix anything without coordinate info

            # Extract names of the generic level coordinates present in the
            # cube; if coordinates have been found and we don't need
            # alternatives (= names are not ``None``), do nothing
            (standard_name, out_name, _) = get_generic_lev_coord_names(
                cube, cmor_coord
            )
            if standard_name or out_name:
                continue

            # Search for alternative coordinates (i.e., regular level
            # coordinates); if none found, do nothing
            try:
                (alternative_coord,
                 cube_coord) = get_alternative_generic_lev_coord(
                    cube, coord_name, self.var_info.table_type
                )
            except ValueError:  # no alternatives found
                continue

            # Fix alternative coord
            (cube, cube_coord) = self.fix_coord(
                cube, alternative_coord, cube_coord
            )

        return cube

    def fix_cmip6_multidim_lat_lon_coord(
        self,
        cube: Cube,
        cmor_coord: CoordinateInfo,
        cube_coord: Coord,
    ) -> None:
        """Fix CMIP6 multidimensional latitude and longitude coordinates.

        Parameters
        ----------
        cube:
            Cube to be fixed.
        cmor_coord:
            Coordinate information from the CMOR table.
        cube_coord:
            Corresponding coordinate of the cube.

        """
        is_cmip6_multidim_lat_lon = all([
            'CMIP6' in self.var_info.table_type,
            cube_coord.ndim > 1,
            cube_coord.standard_name in ('latitude', 'longitude'),
        ])
        if is_cmip6_multidim_lat_lon:
            self._debug_msg(
                cube,
                "Multidimensional %s coordinate is not set in CMOR standard, "
                "ESMValTool will change the original value of '%s' to '%s' to "
                "match the one-dimensional case",
                cube_coord.standard_name,
                cube_coord.var_name,
                cmor_coord.out_name,
            )
            cube_coord.var_name = cmor_coord.out_name

    def fix_coord_units(
        self,
        cube: Cube,
        cmor_coord: CoordinateInfo,
        cube_coord: Coord,
    ) -> None:
        """Fix coordinate units.

        Parameters
        ----------
        cube:
            Cube to be fixed.
        cmor_coord:
            Coordinate information from the CMOR table.
        cube_coord:
            Corresponding coordinate of the cube.

        """
        if not cmor_coord.units:
            return

        # We use str(cube_coord.units) in the following to catch `degrees` !=
        # `degrees_north`
        if str(cube_coord.units) != cmor_coord.units:
            old_units = cube_coord.units
            try:
                cube_coord.convert_units(cmor_coord.units)
            except ValueError:
                self._warning_msg(
                    cube,
                    "Failed to convert units of coordinate %s from '%s' to "
                    "'%s'",
                    cmor_coord.out_name,
                    old_units,
                    cmor_coord.units,
                )
            else:
                self._warning_msg(
                    cube,
                    "Coordinate %s units '%s' converted to '%s'",
                    cmor_coord.out_name,
                    old_units,
                    cmor_coord.units,
                )

    def fix_requested_coord_values(
        self,
        cube: Cube,
        cmor_coord: CoordinateInfo,
        cube_coord: Coord,
    ) -> None:
        """Fix requested coordinate values.

        Parameters
        ----------
        cube:
            Cube to be fixed.
        cmor_coord:
            Coordinate information from the CMOR table.
        cube_coord:
            Corresponding coordinate of the cube.

        """
        if not cmor_coord.requested:
            return

        # Cannot fix non-1D points
        if cube_coord.core_points().ndim != 1:
            return

        # Get requested CMOR values
        try:
            cmor_points = np.array(cmor_coord.requested, dtype=float)
        except ValueError:
            return

        # Align coordinate points with CMOR values if possible
        if cube_coord.core_points().shape != cmor_points.shape:
            return
        atol = 1e-7 * np.mean(cmor_points)
        align_coords = np.allclose(
            cube_coord.core_points(),
            cmor_points,
            rtol=1e-7,
            atol=atol,
        )
        if align_coords:
            cube_coord.points = cmor_points
            self._debug_msg(
                cube,
                "Aligned %s points with CMOR points",
                cmor_coord.out_name,
            )

    def fix_longitude_0_360(
        self,
        cube: Cube,
        cmor_coord: CoordinateInfo,
        cube_coord: Coord,
    ) -> tuple[Cube, Coord]:
        """Fix longitude coordinate to be in [0, 360].

        Parameters
        ----------
        cube:
            Cube to be fixed.
        cmor_coord:
            Coordinate information from the CMOR table.
        cube_coord:
            Corresponding coordinate of the cube.

        Returns
        -------
        tuple[Cube, Coord]
            Fixed cube and coordinate.

        """
        if not cube_coord.standard_name == 'longitude':
            return (cube, cube_coord)

        # Cannot fix longitudes outside [-360, 720]
        if np.any(cube_coord.core_points() < -360.0):
            return (cube, cube_coord)
        if np.any(cube_coord.core_points() > 720.0):
            return (cube, cube_coord)

        # cube.intersection only works for cells with 0 or 2 bounds
        # Note: nbounds==0 means there are no bounds given, nbounds==2
        # implies a regular grid with bounds in the grid direction,
        # nbounds>2 implies an irregular grid with bounds given as vertices
        # of the cell polygon.
        if cube_coord.ndim == 1 and cube_coord.nbounds in (0, 2):
            lon_extent = CoordExtent(cube_coord, 0.0, 360., True, False)
            cube = cube.intersection(lon_extent)
        else:
            new_lons = cube_coord.core_points().copy()
            new_lons = self._set_range_in_0_360(new_lons)
            if cube_coord.bounds is not None:
                new_bounds = cube_coord.bounds.copy()
                new_bounds = self._set_range_in_0_360(new_bounds)
            else:
                new_bounds = None
            new_coord = cube_coord.copy(new_lons, new_bounds)
            dims = cube.coord_dims(cube_coord)
            cube.remove_coord(cube_coord)
            cube.add_aux_coord(new_coord, dims)
        new_coord = cube.coord(var_name=cmor_coord.out_name)
        self._debug_msg(cube, "Shifted longitude to [0, 360]")

        return (cube, new_coord)

    def fix_coord_bounds(
        self,
        cube: Cube,
        cmor_coord: CoordinateInfo,
        cube_coord: Coord,
    ) -> None:
        """Fix coordinate bounds.

        Parameters
        ----------
        cube:
            Cube to be fixed.
        cmor_coord:
            Coordinate information from the CMOR table.
        cube_coord:
            Corresponding coordinate of the cube.

        """
        if cmor_coord.must_have_bounds != 'yes' or cube_coord.has_bounds():
            return
        try:
            cube_coord.guess_bounds()
        except ValueError as exc:
            self._warning_msg(
                cube,
                "Cannot guess bounds for coordinate %s: %s",
                cube_coord.var_name,
                cube.var_name,
                str(exc),
            )

    def fix_coord_direction(
        self,
        cube: Cube,
        cmor_coord: CoordinateInfo,
        cube_coord: Coord,
    ) -> tuple[Cube, Coord]:
        """Fix coordinate direction (increasing vs. decreasing).

        Parameters
        ----------
        cube:
            Cube to be fixed.
        cmor_coord:
            Coordinate information from the CMOR table.
        cube_coord:
            Corresponding coordinate of the cube.

        Returns
        -------
        tuple[Cube, Coord]
            Fixed cube and coordinate.

        """
        # Skip fix for a variety of reasons
        if cube_coord.ndim > 1:
            return (cube, cube_coord)
        if cube_coord.dtype.kind == 'U':
            return (cube, cube_coord)
        if is_unstructured_grid(cube) and cube_coord.standard_name in (
                'latitude', 'longitude'
        ):
            return (cube, cube_coord)
        if len(cube_coord.core_points()) == 1:
            return (cube, cube_coord)
        if not cmor_coord.stored_direction:
            return (cube, cube_coord)

        # Fix coordinates with wrong direction
        if cmor_coord.stored_direction == 'increasing':
            if cube_coord.core_points()[0] > cube_coord.core_points()[1]:
                (cube, cube_coord) = self._reverse_coord(cube, cube_coord)
        elif cmor_coord.stored_direction == 'decreasing':
            if cube_coord.core_points()[0] < cube_coord.core_points()[1]:
                (cube, cube_coord) = self._reverse_coord(cube, cube_coord)

        return (cube, cube_coord)

    def fix_time_units(self, cube: Cube, cube_coord: Coord) -> None:
        """Fix time units and attributes.

        Parameters
        ----------
        cube:
            Cube to be fixed.
        cube_coord:
            Time coordinate of the cube.

        """
        # Fix cube units
        old_units = cube_coord.units
        cube_coord.convert_units(
            Unit(
                'days since 1850-1-1 00:00:00',
                calendar=cube_coord.units.calendar,
            )
        )
        simplified_cal = simplify_calendar(cube_coord.units.calendar)
        cube_coord.units = Unit(
            cube_coord.units.origin, calendar=simplified_cal
        )

        # Fix units of time-related cube attributes
        attrs = cube.attributes
        parent_time = 'parent_time_units'
        if parent_time in attrs:
            if attrs[parent_time] in 'no parent':
                pass
            else:
                try:
                    parent_units = Unit(attrs[parent_time], simplified_cal)
                except ValueError:
                    pass
                else:
                    attrs[parent_time] = 'days since 1850-1-1 00:00:00'

                    branch_parent = 'branch_time_in_parent'
                    if branch_parent in attrs:
                        attrs[branch_parent] = parent_units.convert(
                            attrs[branch_parent], cube_coord.units)

                    branch_child = 'branch_time_in_child'
                    if branch_child in attrs:
                        attrs[branch_child] = old_units.convert(
                            attrs[branch_child], cube_coord.units)

    def fix_time_bounds(self, cube: Cube, cube_coord: Coord) -> None:
        """Fix time bounds.

        Parameters
        ----------
        cube:
            Cube to be fixed.
        cube_coord:
            Time coordinate of the cube.

        """
        times = {'time', 'time1', 'time2', 'time3'}
        key = times.intersection(self.var_info.coordinates)
        cmor = self.var_info.coordinates[' '.join(key)]
        if cmor.must_have_bounds == 'yes' and not cube_coord.has_bounds():
            cube_coord.bounds = get_time_bounds(cube_coord, self.frequency)
            self._warning_msg(
                cube,
                "Added guessed bounds to coordinate %s from var %s",
                cube_coord.var_name,
                self.var_info.short_name,
            )

    def fix_time_coord(self, cube: Cube) -> Cube:
        """Fix time coordinate.

        Parameters
        ----------
        cube:
            Cube to be fixed.

        Returns
        -------
        Cube
            Fixed cube.

        """
        # Make sure to get dimensional time coordinate if possible
        if cube.coords('time', dim_coords=True):
            cube_coord = cube.coord('time', dim_coords=True)
        elif cube.coords('time'):
            cube_coord = cube.coord('time')
        else:
            return cube

        # Cannot fix wrong time that are not references
        if not cube_coord.units.is_time_reference():
            return cube

        # Fix time units
        self.fix_time_units(cube, cube_coord)

        # Remove time_origin from coordinate attributes
        cube_coord.attributes.pop('time_origin', None)

        # Fix time bounds
        self.fix_time_bounds(cube, cube_coord)

        return cube

    def fix_coord(
        self,
        cube: Cube,
        cmor_coord: CoordinateInfo,
        cube_coord: Coord,
    ) -> tuple[Cube, Coord]:
        """Fix non-time coordinate.

        Parameters
        ----------
        cube:
            Cube to be fixed.
        cmor_coord:
            Coordinate information from the CMOR table.
        cube_coord:
            Corresponding coordinate of the cube.

        Returns
        -------
        tuple[Cube, Coord]
            Fixed cube and coordinate.

        """
        self.fix_coord_units(cube, cmor_coord, cube_coord)
        self.fix_requested_coord_values(cube, cmor_coord, cube_coord)
        (cube, cube_coord) = self.fix_longitude_0_360(
            cube, cmor_coord, cube_coord
        )
        self.fix_coord_bounds(cube, cmor_coord, cube_coord)
        (cube, cube_coord) = self.fix_coord_direction(
            cube, cmor_coord, cube_coord
        )
        return (cube, cube_coord)

    def fix_coords(self, cube: Cube) -> Cube:
        """Fix all coordinates.

        Parameters
        ----------
        cube:
            Cube to be fixed.

        Returns
        -------
        Cube
            Fixed cube.

        """
        for cmor_coord in self.var_info.coordinates.values():

            # Cannot fix generic level coords with no unique CMOR information
            if cmor_coord.generic_level and not cmor_coord.out_name:
                continue

            # Try to get coordinate from cube; if it does not exists, skip
            if not cube.coords(var_name=cmor_coord.out_name):
                continue
            cube_coord = cube.coord(var_name=cmor_coord.out_name)

            # Fixes for time coord are done separately
            if cube_coord.var_name == 'time':
                continue

            # Fixes
            (cube, cube_coord) = self.fix_coord(cube, cmor_coord, cube_coord)

        return cube
