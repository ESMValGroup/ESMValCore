"""Fixes for IFW."""

import datetime
import logging

import numpy as np
from iris.cube import CubeList
from iris.util import reverse

from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.cmor._fixes.shared import (
    add_scalar_height_coord,
    add_scalar_lambda550nm_coord,
)
from esmvalcore.cmor.table import CMOR_TABLES
from esmvalcore.iris_helpers import (
    date2num,
    has_unstructured_grid,
    safe_convert_units,
)

logger = logging.getLogger(__name__)


def fix_hourly_time_coordinate(cube, frequency):
    """Shift aggregated variables 30 minutes back in time."""
    # While the frequency for aggregated variables is "1hr", the most common frequency
    # in the CMIP6 E1hr table is "1hrPt" and in the E1hrClimMon table is "1hrCM".
    # We could set the frequency to "1hr" using the extra_facets_native6.yml configuration
    # file, but this would be backward incompatible for users who have already
    # stored the data under a directory with the name 1hrPt. Therefore, apply
    # this fix to any frequency starting with "1hr".
    #
    # Note that comparing instantaneous variables from CMIP6 to averaged
    # variables from ERA5 may lead to some differences.
    if frequency.startswith("1hr"):
        time = cube.coord(axis="T")
        if str(time.units).startswith("hours since"):
            shift = 0.5
        elif str(time.units).startswith("days since"):
            shift = 1.0 / 48.0
        else:
            msg = f"Unexpected time units {time.units} encountered for ERA5 data."
            raise ValueError(msg)
        time.points = time.points - shift
    return cube


def fix_accumulated_units(cube, frequency):
    """Convert accumulations to fluxes."""
    # While the frequency for aggregated variables is "1hr", the most common frequency
    # in the CMIP6 E1hr table is "1hrPt" and in the E1hrClimMon table is "1hrCM".
    # We could set the frequency to "1hr" using the extra_facets_native6.yml configuration
    # file, but this would be backward incompatible for users who have already
    # stored the data under a directory with the name 1hrPt. Therefore, apply
    # this fix to any frequency starting with "1hr".
    #
    # Note that comparing instantaneous variables from CMIP6 to averaged
    # variables from ERA5 may lead to some differences.
    if frequency == "mon":
        cube.units = cube.units * "d-1"
    elif frequency.startswith("1hr"):
        cube.units = cube.units * "h-1"
    elif frequency == "day":
        msg = (
            f"Fixing of accumulated units of cube "
            f"{cube.summary(shorten=True)} is not implemented for daily data"
        )
        raise NotImplementedError(msg)
    return cube


def multiply_with_density(cube, density=1000):
    """Convert precipitatin from m to kg/m2."""
    cube.data = cube.core_data() * density
    cube.units *= "kg m**-3"
    return cube


def remove_time_coordinate(cube):
    """Remove time coordinate for invariant parameters."""
    cube = cube[0]
    cube.remove_coord("time")
    return cube


def divide_by_gravity(cube):
    """Convert geopotential to height."""
    cube.units = cube.units / "m s-2"
    cube.data = cube.core_data() / 9.80665
    return cube


class Clt(Fix):
    """Fixes for clt."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            # Invalid input cube units (ignored on load) were '0-1'
            cube.units = "%"
            cube.data = cube.core_data() * 100.0

        return cubes

class Clivi(Fix):
    """Fixes for prw."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            cube.units = "kg m-2"
        return cubes


class Lwp(Fix):
    """Fixes for prw."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            cube.units = "kg m-2"
        return cubes


class Od550aer(Fix):
    """Fixes for od550aer."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            cube.units = "1"
        return cubes


class Prw(Fix):
    """Fixes for prw."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            cube.units = "kg m-2"
        return cubes


class AllVars(Fix):
    """Fixes for all variables."""

    def _fix_coordinates(  # noqa: C901
        self,
        cube,
    ):
        """Fix coordinates."""
        # Add scalar height coordinates
        if "height2m" in self.vardef.dimensions:
            add_scalar_height_coord(cube, 2.0)
        if "height10m" in self.vardef.dimensions:
            add_scalar_height_coord(cube, 10.0)
        if "lambda550nm" in self.vardef.dimensions:
            add_scalar_lambda550nm_coord(cube)

        # Fix coord metadata
        for coord_def in self.vardef.coordinates.values():
            axis = coord_def.axis
            # ERA5 uses regular pressure level coordinate. In case the cmor
            # variable requires a hybrid level coordinate, we replace this with
            # a regular pressure level coordinate.
            # (https://github.com/ESMValGroup/ESMValCore/issues/1029)
            if axis == "" and coord_def.name == "alevel":
                axis = "Z"
                coord_def = CMOR_TABLES["CMIP6"].coords["plev19"]  # noqa: PLW2901
            if axis == "" and coord_def.name == "lambda550nm":
                continue
            coord = cube.coord(axis=axis)
            if axis == "T":
                coord.convert_units("days since 1850-1-1 00:00:00.0")
            if axis in ("X", "Y", "Z"):
                coord.convert_units(coord_def.units)
            coord.standard_name = coord_def.standard_name
            coord.var_name = coord_def.out_name
            coord.long_name = coord_def.long_name
            coord.points = coord.core_points().astype("float64")
            if (
                not coord.has_bounds()
                and len(coord.core_points()) > 1
                and coord_def.must_have_bounds == "yes"
            ):
                # Do not guess bounds for lat and lon on unstructured grids
                if not (
                    coord.name() in ("latitude", "longitude")
                    and has_unstructured_grid(cube)
                ):
                    coord.guess_bounds()

        self._fix_monthly_time_coord(cube, self.frequency)

        # Fix coordinate increasing direction
        if cube.coords("latitude") and not has_unstructured_grid(cube):
            lat = cube.coord("latitude")
            if lat.points[0] > lat.points[-1]:
                cube = reverse(cube, "latitude")
        if cube.coords("air_pressure"):
            plev = cube.coord("air_pressure")
            if plev.points[0] < plev.points[-1]:
                cube = reverse(cube, "air_pressure")

        return cube

    @staticmethod
    def _fix_monthly_time_coord(cube, frequency):
        """Set the monthly time coordinates to the middle of the month."""
        if frequency in ("monthly", "mon"):
            coord = cube.coord(axis="T")
            end = []
            for cell in coord.cells():
                month = cell.point.month + 1
                year = cell.point.year
                if month == 13:
                    month = 1
                    year = year + 1
                end.append(cell.point.replace(month=month, year=year))
            end = date2num(end, coord.units)
            start = coord.points
            coord.points = 0.5 * (start + end)
            coord.bounds = np.column_stack([start, end])

    def fix_metadata(self, cubes):
        """Fix metadata."""
        fixed_cubes = CubeList()
        for orig_cube in cubes:
            cube = orig_cube.copy()
            cube.var_name = self.vardef.short_name
            if self.vardef.standard_name:
                cube.standard_name = self.vardef.standard_name
            cube.long_name = self.vardef.long_name
            cube = self._fix_coordinates(cube)
            cube = safe_convert_units(cube, self.vardef.units)
            cube.data = cube.core_data().astype("float32")
            year = datetime.datetime.now().year
            cube.attributes["comment"] = (
                "Contains modified Copernicus Climate Change "
                f"Service Information {year}"
            )
            if "GRIB_PARAM" in cube.attributes:
                cube.attributes["GRIB_PARAM"] = str(
                    cube.attributes["GRIB_PARAM"],
                )

            fixed_cubes.append(cube)

        return fixed_cubes
