"""Module for checking iris cubes against their CMOR definitions."""
from __future__ import annotations

import logging
import warnings
from collections import namedtuple
from collections.abc import Callable
from enum import IntEnum
from functools import cached_property
from typing import Optional

import cf_units
import dask
import iris.coord_categorisation
import iris.coords
import iris.exceptions
import iris.util
import numpy as np
from iris.coords import Coord
from iris.cube import Cube

from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor._utils import (
    _get_alternative_generic_lev_coord,
    _get_generic_lev_coord_names,
    _get_new_generic_level_coord,
    _get_simplified_calendar,
)
from esmvalcore.cmor.table import CoordinateInfo, get_var_info
from esmvalcore.exceptions import ESMValCoreDeprecationWarning
from esmvalcore.iris_helpers import has_unstructured_grid


class CheckLevels(IntEnum):
    """Level of strictness of the checks."""

    DEBUG = 1
    """Report any debug message that the checker wants to communicate."""

    STRICT = 2
    """Fail if there are warnings regarding compliance of CMOR standards."""

    DEFAULT = 3
    """Fail if cubes present any discrepancy with CMOR standards."""

    RELAXED = 4
    """Fail if cubes present severe discrepancies with CMOR standards."""

    IGNORE = 5
    """Do not fail for any discrepancy with CMOR standards."""


class CMORCheckError(Exception):
    """Exception raised when a cube does not pass the CMORCheck."""


class CMORCheck():
    """Class used to check the CMOR-compliance of the data.

    Parameters
    ----------
    cube: iris.cube.Cube:
        Iris cube to check.
    var_info: variables_info.VariableInfo
        Variable info to check.
    frequency: str
        Expected frequency for the data. If not given, use the one from the
        variable information.
    fail_on_error: bool
        If true, CMORCheck stops on the first error. If false, it collects
        all possible errors before stopping.
    automatic_fixes: bool
        If True, CMORCheck will try to apply automatic fixes for any
        detected error, if possible.

        .. deprecated:: 2.10.0
            This option has been deprecated in ESMValCore version 2.10.0 and is
            scheduled for removal in version 2.12.0. Please use the functions
            :func:`~esmvalcore.preprocessor.fix_metadata`,
            :func:`~esmvalcore.preprocessor.fix_data`, or
            :meth:`esmvalcore.dataset.Dataset.load` (which automatically
            includes the first two functions) instead. Fixes and CMOR checks
            have been clearly separated in ESMValCore version 2.10.0.
    check_level: CheckLevels
        Level of strictness of the checks.

    Attributes
    ----------
    frequency: str
        Expected frequency for the data.
    """

    _attr_msg = '{}: {} should be {}, not {}'
    _does_msg = '{}: does not {}'
    _is_msg = '{}: is not {}'
    _vals_msg = '{}: has values {} {}'
    _contain_msg = '{}: does not contain {} {}'

    def __init__(self,
                 cube,
                 var_info,
                 frequency=None,
                 fail_on_error=False,
                 check_level=CheckLevels.DEFAULT,
                 automatic_fixes=False):

        self._cube = cube
        self._failerr = fail_on_error
        self._check_level = check_level
        self._logger = logging.getLogger(__name__)
        self._errors = list()
        self._warnings = list()
        self._debug_messages = list()

        self._cmor_var = var_info
        if not frequency:
            frequency = self._cmor_var.frequency
        self.frequency = frequency
        self.automatic_fixes = automatic_fixes

        # Deprecate automatic_fixes (remove in v2.12)
        if automatic_fixes:
            msg = (
                "The option `automatic_fixes` has been deprecated in "
                "ESMValCore version 2.10.0 and is scheduled for removal in "
                "version 2.12.0. Please use the functions "
                "esmvalcore.preprocessor.fix_metadata(), "
                "esmvalcore.preprocessor.fix_data(), or "
                "esmvalcore.dataset.Dataset.load() (which automatically "
                "includes the first two functions) instead. Fixes and CMOR "
                "checks have been clearly separated in ESMValCore version "
                "2.10.0."
            )
            warnings.warn(msg, ESMValCoreDeprecationWarning)

        # TODO: remove in v2.12

        self._generic_fix = GenericFix(var_info, frequency=frequency)

    @cached_property
    def _unstructured_grid(self) -> bool:
        """Cube uses unstructured grid."""
        return has_unstructured_grid(self._cube)

    def check_metadata(self, logger: Optional[logging.Logger] = None) -> Cube:
        """Check the cube metadata.

        It will also report some warnings in case of minor errors.


        Parameters
        ----------
        logger:
            Given logger.

        Returns
        -------
        iris.cube.Cube
            Checked cube.

        Raises
        ------
        CMORCheckError
            If errors are found. If fail_on_error attribute is set to True,
            raises as soon as an error is detected. If set to False, it perform
            all checks and then raises.

        """
        if logger is not None:
            self._logger = logger

        # TODO: remove in v2.12
        if self.automatic_fixes:
            [self._cube] = self._generic_fix.fix_metadata([self._cube])

        self._check_var_metadata()
        self._check_fill_value()
        self._check_multiple_coords_same_stdname()
        self._check_dim_names()
        self._check_coords()
        if self.frequency != 'fx':
            self._check_time_coord()

        self._check_rank()

        self.report_debug_messages()
        self.report_warnings()
        self.report_errors()

        return self._cube

    def check_data(self, logger: Optional[logging.Logger] = None) -> Cube:
        """Check the cube data.

        Assumes that metadata is correct, so you must call check_metadata prior
        to this.

        It will also report some warnings in case of minor errors.

        Parameters
        ----------
        logger:
            Given logger.

        Returns
        -------
        iris.cube.Cube
            Checked cube.

        Raises
        ------
        CMORCheckError
            If errors are found. If fail_on_error attribute is set to True,
            raises as soon as an error is detected. If set to False, it perform
            all checks and then raises.

        """
        if logger is not None:
            self._logger = logger

        # TODO: remove in v2.12
        if self.automatic_fixes:
            self._cube = self._generic_fix.fix_data(self._cube)

        self._check_coords_data()

        self.report_debug_messages()
        self.report_warnings()
        self.report_errors()

        return self._cube

    def report_errors(self):
        """Report detected errors.

        Raises
        ------
        CMORCheckError
            If any errors were reported before calling this method.
        """
        if self.has_errors():
            msg = '\n'.join([
                f'There were errors in variable {self._cube.var_name}:',
                ' ' + '\n '.join(self._errors),
                'in cube:',
                f'{self._cube}',
                'loaded from file ' +
                self._cube.attributes.get('source_file', ''),
            ])
            raise CMORCheckError(msg)

    def report_warnings(self):
        """Report detected warnings to the given logger."""
        if self.has_warnings():
            msg = '\n'.join([
                f'There were warnings in variable {self._cube.var_name}:',
                ' ' + '\n '.join(self._warnings),
                'loaded from file ' +
                self._cube.attributes.get('source_file', ''),
            ])
            self._logger.warning(msg)

    def report_debug_messages(self):
        """Report detected debug messages to the given logger."""
        if self.has_debug_messages():
            msg = '\n'.join([
                f'There were metadata changes in variable '
                f'{self._cube.var_name}:',
                ' ' + '\n '.join(self._debug_messages),
                'loaded from file ' +
                self._cube.attributes.get('source_file', ''),
            ])
            self._logger.debug(msg)

    def _check_fill_value(self):
        """Check fill value."""
        # Iris removes _FillValue/missing_value information if data has none
        #  of these values. If there are values == _FillValue then it will
        #  be encoded in the numpy.ma object created.
        #
        #  => Very difficult to check!

    def _check_var_metadata(self):
        """Check metadata of variable."""
        # Check standard_name
        if self._cmor_var.standard_name:
            if self._cube.standard_name != self._cmor_var.standard_name:
                self.report_error(self._attr_msg, self._cube.var_name,
                                  'standard_name',
                                  self._cmor_var.standard_name,
                                  self._cube.standard_name)
        # Check long_name
        if self._cmor_var.long_name:
            if self._cube.long_name != self._cmor_var.long_name:
                self.report_error(self._attr_msg, self._cube.var_name,
                                  'long_name', self._cmor_var.long_name,
                                  self._cube.long_name)

        # Check units
        if self._cmor_var.units:
            units = self._get_effective_units()
            if self._cube.units != units:
                self.report_error(self._attr_msg, self._cube.var_name,
                                  'units', self._cmor_var.units,
                                  self._cube.units)

        # Check other variable attributes that match entries in cube.attributes
        attrs = ('positive', )
        for attr in attrs:
            attr_value = getattr(self._cmor_var, attr)
            if attr_value:
                if attr not in self._cube.attributes:
                    self.report_warning('{}: attribute {} not present',
                                        self._cube.var_name, attr)
                elif self._cube.attributes[attr] != attr_value:
                    self.report_error(self._attr_msg, self._cube.var_name,
                                      attr, attr_value,
                                      self._cube.attributes[attr])

    def _get_effective_units(self):
        """Get effective units."""
        # TODO: remove entire function in v2.12
        if self._cmor_var.units.lower() == 'psu':
            units = '1.0'
        else:
            units = self._cmor_var.units
        return units

    def _check_rank(self):
        """Check rank, excluding scalar dimensions."""
        rank = 0
        dimensions = []
        for coordinate in self._cmor_var.coordinates.values():
            if coordinate.generic_level:
                rank += 1
            elif not coordinate.value:
                try:
                    for dim in self._cube.coord_dims(coordinate.standard_name):
                        dimensions.append(dim)
                except iris.exceptions.CoordinateNotFoundError:
                    # Error reported at other stages
                    pass
        rank += len(set(dimensions))

        # Check number of dimension coords matches rank
        if self._cube.ndim != rank:
            self.report_error(self._does_msg, self._cube.var_name,
                              'match coordinate rank')

    def _check_multiple_coords_same_stdname(self):
        standard_names = set()
        for coord in self._cube.coords():
            if coord.standard_name:
                if coord.standard_name in standard_names:
                    coords = [
                        c.var_name for c in self._cube.coords(
                            standard_name=coord.standard_name)
                    ]
                    self.report_error(
                        'There are multiple coordinates with '
                        f'standard_name "{coord.standard_name}": {coords}')
                else:
                    standard_names.add(coord.standard_name)

    def _check_dim_names(self):
        """Check dimension names."""
        cmor_var_coordinates = self._cmor_var.coordinates.copy()
        link = 'https://github.com/ESMValGroup/ESMValCore/discussions/1587'
        for (key, coordinate) in cmor_var_coordinates.items():
            if coordinate.generic_level:
                self._check_generic_level_dim_names(key, coordinate)
            else:
                try:
                    cube_coord = self._cube.coord(var_name=coordinate.out_name)
                    if (cube_coord.standard_name is None
                            and coordinate.standard_name == ''):
                        pass
                    elif cube_coord.standard_name != coordinate.standard_name:
                        self.report_critical(
                            self._attr_msg,
                            coordinate.out_name,
                            'standard_name',
                            coordinate.standard_name,
                            cube_coord.standard_name,
                        )
                except iris.exceptions.CoordinateNotFoundError:
                    try:
                        coord = self._cube.coord(coordinate.standard_name)
                        if coord.standard_name in ['region', 'area_type']:
                            self.report_debug_message(
                                'Coordinate {0} has var name {1} '
                                'instead of {2}. '
                                "But that's considered OK and ignored. "
                                'See also {3}',
                                coordinate.name,
                                coord.var_name,
                                coordinate.out_name,
                                link
                            )
                        else:
                            self.report_error(
                                'Coordinate {0} has var name {1} '
                                'instead of {2}',
                                coordinate.name,
                                coord.var_name,
                                coordinate.out_name,
                            )
                    except iris.exceptions.CoordinateNotFoundError:
                        if coordinate.standard_name in ['time', 'latitude',
                                                        'longitude'] or \
                           coordinate.requested:
                            self.report_critical(self._does_msg,
                                                 coordinate.name, 'exist')
                        else:
                            self.report_error(self._does_msg, coordinate.name,
                                              'exist')

    def _check_generic_level_dim_names(self, key, coordinate):
        """Check name of generic level coordinate."""
        if coordinate.generic_lev_coords:
            (standard_name, out_name, name) = _get_generic_lev_coord_names(
                self._cube, coordinate
            )
            if standard_name:
                if not out_name:
                    self.report_error(
                        f'Generic level coordinate {key} has wrong var_name.')
                level = _get_new_generic_level_coord(
                    self._cmor_var, coordinate, key, name
                )
                self._cmor_var.coordinates[key] = level
                self.report_debug_message(f'Generic level coordinate {key} '
                                          'will be checked against '
                                          f'{name} coordinate information')
            else:
                if out_name:
                    self.report_critical(
                        f'Generic level coordinate {key} with out_name '
                        f'{out_name} has wrong standard_name or is not set.')
                else:
                    self._check_alternative_dim_names(key)

    def _check_alternative_dim_names(self, key):
        """Check for viable alternatives to generic level coordinates.

        Generic level coordinates are used to calculate high-dimensional (e.g.,
        3D or 4D) regular level coordinates (like pressure or altitude) from
        lower-dimensional (e.g., 2D or 1D) arrays in order to save disk space.
        In order to also support regular level coordinates, search for allowed
        alternatives here.  A detailed explanation of this can be found here:
            https://github.com/ESMValGroup/ESMValCore/issues/1029

        Only the projects CMIP3, CMIP5, CMIP6 and obs4MIPs support generic
        level coordinates. Right now, only alternative level coordinates for
        the atmosphere ('alevel' or 'zlevel') are supported.

        Note that only the "simplest" CMOR table entry per coordinate is
        specified (e.g., only 'plev3' for the pressure level coordinate and
        'alt16' for the altitude coordinate). These different versions (e.g.,
        'plev3', 'plev19', 'plev39', etc.) only differ in the requested values.
        We are mainly interested in the metadata of the coordinates (names,
        units), which is equal for all coordinate versions. In the DEFAULT
        strictness or lower, differing requested values only produce a warning.
        A stricter setting (such as STRICT) does not allow this feature (i.e.,
        the use of alternative level coordinates) in the first place, so we do
        not need to worry about differing requested values for the levels in
        this case.

        In the future, this might be extended: For ``cmor_strict=True``
        projects (like CMIP) the level coordinate's ``len`` might be used to
        search for the correct coordinate version and then check against this.
        For ``cmor_strict=False`` project (like OBS) the check for requested
        values might be disabled.
        """
        try:
            (alternative_coord,
             cube_coord) = _get_alternative_generic_lev_coord(
                self._cube, key, self._cmor_var.table_type
            )

        # No valid alternative coordinate found -> critical error
        except ValueError:
            self.report_critical(self._does_msg, key, 'exist')
            return

        # Wrong standard_name -> error
        if cube_coord.standard_name != alternative_coord.standard_name:
            self.report_error(
                f"Found alternative coordinate '{alternative_coord.out_name}' "
                f"for generic level coordinate '{key}' with wrong "
                f"standard_name {cube_coord.standard_name}' (expected "
                f"'{alternative_coord.standard_name}')"
            )
            return

        # Valid alternative coordinate found -> perform checks on it
        self.report_warning(
            f"Found alternative coordinate '{alternative_coord.out_name}' "
            f"for generic level coordinate '{key}'. Subsequent warnings about "
            f"levels that are not contained in '{alternative_coord.out_name}' "
            f"can be safely ignored.")
        self._check_coord(alternative_coord, cube_coord, cube_coord.var_name)

    def _check_coords(self):
        """Check coordinates."""
        coords = []
        for coordinate in self._cmor_var.coordinates.values():
            # Cannot check generic_level coords with no CMOR information
            if coordinate.generic_level and not coordinate.out_name:
                continue
            var_name = coordinate.out_name

            # Get coordinate var_name as it exists!
            try:
                coord = self._cube.coord(var_name=var_name)
            except iris.exceptions.CoordinateNotFoundError:
                continue

            self._check_coord(coordinate, coord, var_name)
            coords.append((coordinate, coord))

        self._check_coord_ranges(coords)

    def _check_coord_ranges(self, coords: list[tuple[CoordinateInfo, Coord]]):
        """Check coordinate value are inside valid ranges."""
        Limit = namedtuple('Limit', ['name', 'type', 'limit', 'value'])

        limits = []
        for coord_info, coord in coords:
            points = coord.core_points()
            for limit_type in 'min', 'max':
                valid = getattr(coord_info, f'valid_{limit_type}')
                if valid != "":
                    limit = Limit(
                        name=coord_info.out_name,
                        type=limit_type,
                        limit=float(valid),
                        value=getattr(points, limit_type)(),
                    )
                    limits.append(limit)

        limits = dask.compute(*limits)
        for limit in limits:
            if limit.type == 'min' and limit.value < limit.limit:
                self.report_critical(self._vals_msg, limit.name,
                                     '< valid_min =', limit.limit)
            if limit.type == 'max' and limit.value > limit.limit:
                self.report_critical(self._vals_msg, limit.name,
                                     '> valid_max =', limit.limit)

    def _check_coords_data(self):
        """Check coordinate data."""
        for coordinate in self._cmor_var.coordinates.values():
            # Cannot check generic_level coords as no CMOR information
            if coordinate.generic_level:
                continue
            var_name = coordinate.out_name

            # Get coordinate var_name as it exists!
            try:
                coord = self._cube.coord(var_name=var_name, dim_coords=True)
            except iris.exceptions.CoordinateNotFoundError:
                continue

            # TODO: remove in v2.12
            if self.automatic_fixes:
                (self._cube, coord) = self._generic_fix._fix_coord_direction(
                    self._cube, coordinate, coord
                )

            self._check_coord_monotonicity_and_direction(
                coordinate, coord, var_name)

    def _check_coord(self, cmor, coord, var_name):
        """Check single coordinate."""
        if coord.var_name == 'time':
            return
        if cmor.units:
            if str(coord.units) != cmor.units:
                self.report_critical(self._attr_msg, var_name, 'units',
                                     cmor.units, coord.units)
        self._check_coord_points(cmor, coord, var_name)

    def _check_coord_bounds(self, cmor, coord, var_name):
        if cmor.must_have_bounds == 'yes' and not coord.has_bounds():
            self.report_warning(
                'Coordinate {0} from var {1} does not have bounds',
                coord.var_name, var_name)

    def _check_time_bounds(self, time):
        times = {'time', 'time1', 'time2', 'time3'}
        key = times.intersection(self._cmor_var.coordinates)
        cmor = self._cmor_var.coordinates[" ".join(key)]
        if cmor.must_have_bounds == 'yes' and not time.has_bounds():
            self.report_warning(
                'Coordinate {0} from var {1} does not have bounds',
                time.var_name, self._cmor_var.short_name)

    def _check_coord_monotonicity_and_direction(self, cmor, coord, var_name):
        """Check monotonicity and direction of coordinate."""
        if coord.ndim > 1:
            return
        if coord.dtype.kind == 'U':
            return

        if (self._unstructured_grid and
                coord.standard_name in ['latitude', 'longitude']):
            self.report_debug_message(
                f'Coordinate {coord.standard_name} appears to belong to '
                'an unstructured grid. Skipping monotonicity and '
                'direction tests.')
            return

        if not coord.is_monotonic():
            self.report_critical(self._is_msg, var_name, 'monotonic')

        if len(coord.core_points()) == 1:
            return

        if cmor.stored_direction:
            if cmor.stored_direction == 'increasing':
                if coord.core_points()[0] > coord.core_points()[1]:
                    self.report_critical(self._is_msg, var_name, 'increasing')
            elif cmor.stored_direction == 'decreasing':
                if coord.core_points()[0] < coord.core_points()[1]:
                    self.report_critical(self._is_msg, var_name, 'decreasing')

    def _check_coord_points(self, coord_info, coord, var_name):
        """Check coordinate points: values, bounds and monotonicity."""
        self._check_requested_values(coord, coord_info, var_name)
        self._check_coord_bounds(coord_info, coord, var_name)
        self._check_coord_monotonicity_and_direction(coord_info, coord,
                                                     var_name)

    def _check_requested_values(self, coord, coord_info, var_name):
        """Check requested values."""
        if coord_info.requested:
            if coord.core_points().ndim != 1:
                self.report_warning(
                    "Cannot check requested values of {}D coordinate {} since "
                    "it is not 1D", coord.core_points().ndim, var_name)
                return
            try:
                cmor_points = np.array(coord_info.requested, dtype=float)
            except ValueError:
                cmor_points = coord_info.requested
            for point in cmor_points:
                if point not in coord.core_points():
                    self.report_warning(self._contain_msg, var_name,
                                        str(point), str(coord.units))

    def _check_time_coord(self):
        """Check time coordinate."""
        try:
            coord = self._cube.coord('time', dim_coords=True)
        except iris.exceptions.CoordinateNotFoundError:
            try:
                coord = self._cube.coord('time')
            except iris.exceptions.CoordinateNotFoundError:
                return

        var_name = coord.var_name
        if not coord.is_monotonic():
            self.report_error('Time coordinate for var {} is not monotonic',
                              var_name)

        if not coord.units.is_time_reference():
            self.report_critical(self._does_msg, var_name,
                                 'have time reference units')
        else:
            simplified_cal = _get_simplified_calendar(coord.units.calendar)
            attrs = self._cube.attributes
            parent_time = 'parent_time_units'
            if parent_time in attrs:
                if attrs[parent_time] in 'no parent':
                    pass
                else:
                    try:
                        cf_units.Unit(attrs[parent_time], simplified_cal)
                    except ValueError:
                        self.report_warning('Attribute parent_time_units has '
                                            'a wrong format and cannot be '
                                            'read by cf_units. A fix needs to '
                                            'be added to convert properly '
                                            'attributes branch_time_in_parent '
                                            'and branch_time_in_child.')

        # Check frequency
        tol = 0.001
        intervals = {'dec': (3600, 3660), 'day': (1, 1)}
        freq = self.frequency
        if freq.lower().endswith('pt'):
            freq = freq[:-2]
        if freq in ['mon', 'mo']:
            dates = coord.units.num2date(coord.points)
            for i in range(len(coord.points) - 1):
                first = dates[i]
                second = dates[i + 1]
                second_month = first.month + 1
                second_year = first.year
                if second_month == 13:
                    second_month = 1
                    second_year += 1
                if second_month != second.month or \
                   second_year != second.year:
                    msg = '{}: Frequency {} does not match input data'
                    self.report_error(msg, var_name, freq)
                    break
        elif freq == 'yr':
            dates = coord.units.num2date(coord.points)
            for i in range(len(coord.points) - 1):
                first = dates[i]
                second = dates[i + 1]
                second_month = first.month + 1
                if first.year + 1 != second.year:
                    msg = '{}: Frequency {} does not match input data'
                    self.report_error(msg, var_name, freq)
                    break
        else:
            if freq in intervals:
                interval = intervals[freq]
                target_interval = (interval[0] - tol, interval[1] + tol)
            elif freq.endswith('hr'):
                if freq == 'hr':
                    freq = '1hr'
                frequency = freq[:-2]
                if frequency == 'sub':
                    frequency = 1.0 / 24
                    target_interval = (-tol, frequency + tol)
                else:
                    frequency = float(frequency) / 24
                    target_interval = (frequency - tol, frequency + tol)
            else:
                msg = '{}: Frequency {} not supported by checker'
                self.report_error(msg, var_name, freq)
                return
            for i in range(len(coord.points) - 1):
                interval = coord.points[i + 1] - coord.points[i]
                if (interval < target_interval[0]
                        or interval > target_interval[1]):
                    msg = '{}: Frequency {} does not match input data'
                    self.report_error(msg, var_name, freq)
                    break

        self._check_time_bounds(coord)

    def has_errors(self):
        """Check if there are reported errors.

        Returns
        -------
        bool:
            True if there are pending errors, False otherwise.
        """
        return len(self._errors) > 0

    def has_warnings(self):
        """Check if there are reported warnings.

        Returns
        -------
        bool:
            True if there are pending warnings, False otherwise.
        """
        return len(self._warnings) > 0

    def has_debug_messages(self):
        """Check if there are reported debug messages.

        Returns
        -------
        bool:
            True if there are pending debug messages, False otherwise.
        """
        return len(self._debug_messages) > 0

    def report(self, level, message, *args):
        """Report a message from the checker.

        Parameters
        ----------
        level : CheckLevels
            Message level
        message : str
            Message to report
        args :
            String format args for the message

        Raises
        ------
        CMORCheckError
            If fail on error is set, it is thrown when registering an error
            message
        """
        msg = message.format(*args)
        if level == CheckLevels.DEBUG:
            if self._failerr:
                self._logger.debug(msg)
            else:
                self._debug_messages.append(msg)
        elif level < self._check_level:
            if self._failerr:
                self._logger.warning(msg)
            else:
                self._warnings.append(msg)
        else:
            if self._failerr:
                raise CMORCheckError(msg +
                                     '\n in cube:\n{}'.format(self._cube))
            self._errors.append(msg)

    def report_critical(self, message, *args):
        """Report an error.

        If fail_on_error is set to True, raises automatically.
        If fail_on_error is set to False, stores it for later reports.

        Parameters
        ----------
        message: str: unicode
            Message for the error.
        *args:
            arguments to format the message string.
        """
        self.report(CheckLevels.RELAXED, message, *args)

    def report_error(self, message, *args):
        """Report a normal error.

        Parameters
        ----------
        message: str: unicode
            Message for the error.
        *args:
            arguments to format the message string.
        """
        self.report(CheckLevels.DEFAULT, message, *args)

    def report_warning(self, message, *args):
        """Report a warning level error.

        Parameters
        ----------
        message: str: unicode
            Message for the warning.
        *args:
            arguments to format the message string.
        """
        self.report(CheckLevels.STRICT, message, *args)

    def report_debug_message(self, message, *args):
        """Report a debug message.

        Parameters
        ----------
        message: str: unicode
            Message for the debug logger.
        *args:
            arguments to format the message string
        """
        self.report(CheckLevels.DEBUG, message, *args)


def _get_cmor_checker(
    project: str,
    mip: str,
    short_name: str,
    frequency: None | str = None,
    fail_on_error: bool = False,
    check_level: CheckLevels = CheckLevels.DEFAULT,
    automatic_fixes: bool = False,  # TODO: remove in v2.12
) -> Callable[[Cube], CMORCheck]:
    """Get a CMOR checker."""
    var_info = get_var_info(project, mip, short_name)

    def _checker(cube: Cube) -> CMORCheck:
        return CMORCheck(cube,
                         var_info,
                         frequency=frequency,
                         fail_on_error=fail_on_error,
                         check_level=check_level,
                         automatic_fixes=automatic_fixes)

    return _checker


def cmor_check_metadata(
    cube: Cube,
    cmor_table: str,
    mip: str,
    short_name: str,
    frequency: Optional[str] = None,
    check_level: CheckLevels = CheckLevels.DEFAULT,
) -> Cube:
    """Check if metadata conforms to variable's CMOR definition.

    None of the checks at this step will force the cube to load the data.

    Parameters
    ----------
    cube:
        Data cube to check.
    cmor_table:
        CMOR definitions to use (i.e., the variable's project).
    mip:
        Variable's MIP.
    short_name:
        Variable's short name.
    frequency:
        Data frequency. If not given, use the one from the CMOR table of the
        variable.
    check_level:
        Level of strictness of the checks.

    Returns
    -------
    iris.cube.Cube
        Checked cube.

    """
    checker = _get_cmor_checker(
        cmor_table,
        mip,
        short_name,
        frequency=frequency,
        check_level=check_level,
    )
    cube = checker(cube).check_metadata()
    return cube


def cmor_check_data(
    cube: Cube,
    cmor_table: str,
    mip: str,
    short_name: str,
    frequency: Optional[str] = None,
    check_level: CheckLevels = CheckLevels.DEFAULT,
) -> Cube:
    """Check if data conforms to variable's CMOR definition.

    Parameters
    ----------
    cube:
        Data cube to check.
    cmor_table:
        CMOR definitions to use (i.e., the variable's project).
    mip:
        Variable's MIP.
    short_name:
        Variable's short name
    frequency:
        Data frequency. If not given, use the one from the CMOR table of the
        variable.
    check_level:
        Level of strictness of the checks.

    Returns
    -------
    iris.cube.Cube
        Checked cube.

    """
    checker = _get_cmor_checker(
        cmor_table,
        mip,
        short_name,
        frequency=frequency,
        check_level=check_level,
    )
    cube = checker(cube).check_data()
    return cube


def cmor_check(
    cube: Cube,
    cmor_table: str,
    mip: str,
    short_name: str,
    frequency: Optional[str] = None,
    check_level: CheckLevels = CheckLevels.DEFAULT,
) -> Cube:
    """Check if cube conforms to variable's CMOR definition.

    Equivalent to calling :func:`cmor_check_metadata` and
    :func:`cmor_check_data` consecutively.

    Parameters
    ----------
    cube:
        Data cube to check.
    cmor_table:
        CMOR definitions to use (i.e., the variable's project).
    mip:
        Variable's MIP.
    short_name:
        Variable's short name.
    frequency:
        Data frequency. If not given, use the one from the CMOR table of the
        variable.
    check_level:
        Level of strictness of the checks.

    Returns
    -------
    iris.cube.Cube
        Checked cube.

    """
    cube = cmor_check_metadata(
        cube,
        cmor_table,
        mip,
        short_name,
        frequency=frequency,
        check_level=check_level,
    )
    cube = cmor_check_data(
        cube,
        cmor_table,
        mip,
        short_name,
        frequency=frequency,
        check_level=check_level,
    )
    return cube
