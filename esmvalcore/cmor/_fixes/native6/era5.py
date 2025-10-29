"""Fixes for ERA5."""

import datetime
import logging

import iris
import numpy as np
from iris.cube import CubeList
from iris.util import reverse

from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.cmor._fixes.shared import add_scalar_height_coord
from esmvalcore.cmor.table import CMOR_TABLES
from esmvalcore.iris_helpers import (
    date2num,
    has_unstructured_grid,
    safe_convert_units,
)

logger = logging.getLogger(__name__)


def get_frequency(cube):
    """Determine time frequency of input cube."""
    try:
        time = cube.coord(axis="T")
    except iris.exceptions.CoordinateNotFoundError:
        return "fx"

    time.convert_units("days since 1850-1-1 00:00:00.0")
    if len(time.points) == 1:
        acceptable_long_names = (
            "Geopotential",
            "Percentage of the Grid Cell Occupied by Land (Including Lakes)",
        )
        if cube.long_name not in acceptable_long_names:
            msg = (
                "Unable to infer frequency of cube "
                f"with length 1 time dimension: {cube}"
            )
            raise ValueError(
                msg,
            )
        return "fx"

    interval = time.points[1] - time.points[0]

    if interval - 1 / 24 < 1e-4:
        return "hourly"
    if interval - 1.0 < 1e-4:
        return "daily"
    return "monthly"


def fix_hourly_time_coordinate(cube):
    """Shift aggregated variables 30 minutes back in time."""
    if get_frequency(cube) == "hourly":
        time = cube.coord(axis="T")
        time.points = time.points - 1 / 48
    return cube


def fix_accumulated_units(cube):
    """Convert accumulations to fluxes."""
    if get_frequency(cube) == "monthly":
        cube.units = cube.units * "d-1"
    elif get_frequency(cube) == "hourly":
        cube.units = cube.units * "h-1"
    elif get_frequency(cube) == "daily":
        msg = (
            f"Fixing of accumulated units of cube "
            f"{cube.summary(shorten=True)} is not implemented for daily data"
        )
        raise NotImplementedError(
            msg,
        )
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


class Albsn(Fix):
    """Fixes for albsn."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            # Invalid input cube units (ignored on load) were '0-1'
            cube.units = "1"
        return cubes


class Cli(Fix):
    """Fixes for cli."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            cube.units = "kg kg-1"
        return cubes


class Clt(Fix):
    """Fixes for clt."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            # Invalid input cube units (ignored on load) were '0-1'
            cube.units = "%"
            cube.data = cube.core_data() * 100.0

        return cubes


class Clw(Fix):
    """Fixes for clw."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            cube.units = "kg kg-1"
        return cubes


class Cl(Fix):
    """Fixes for cl."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            # Invalid input cube units (ignored on load) were '0-1'
            cube.units = "%"
            cube.data = cube.core_data() * 100.0

        return cubes


class Evspsbl(Fix):
    """Fixes for evspsbl."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            # Set input cube units for invalid units were ignored on load
            cube.units = "m"
            fix_hourly_time_coordinate(cube)
            fix_accumulated_units(cube)
            multiply_with_density(cube)
            # Correct sign to align with CMOR standards
            cube.data = cube.core_data() * -1.0

        return cubes


class Evspsblpot(Fix):
    """Fixes for evspsblpot."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            # Set input cube units for invalid units were ignored on load
            cube.units = "m"
            fix_hourly_time_coordinate(cube)
            fix_accumulated_units(cube)
            multiply_with_density(cube)
            # Correct sign to align with CMOR standards
            cube.data = cube.core_data() * -1.0

        return cubes


class Hus(Fix):
    """Fixes for hus."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            cube.units = "kg kg-1"
        return cubes


class Mrro(Fix):
    """Fixes for mrro."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            fix_hourly_time_coordinate(cube)
            fix_accumulated_units(cube)
            multiply_with_density(cube)

        return cubes


class O3(Fix):
    """Fixes for o3."""

    def fix_metadata(self, cubes):
        """Convert mass mixing ratios to mole fractions."""
        for cube in cubes:
            # Original units are kg kg-1. Convert these to molar mixing ratios,
            # which is almost identical to mole fraction for small amounts of
            # substances (which we have here)
            cube.data = cube.core_data() * 28.9644 / 47.9982
            cube.units = "mol mol-1"
        return cubes


class Orog(Fix):
    """Fixes for orography."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        fixed_cubes = []
        for orig_cube in cubes:
            cube = orig_cube.copy()
            cube = remove_time_coordinate(cube)
            divide_by_gravity(cube)
            fixed_cubes.append(cube)
        return CubeList(fixed_cubes)


class Pr(Fix):
    """Fixes for pr."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            fix_hourly_time_coordinate(cube)
            fix_accumulated_units(cube)
            multiply_with_density(cube)

        return cubes


class Prc(Pr):
    """Fix for Prc."""


class Prsn(Fix):
    """Fixes for prsn."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            # Set input cube units for invalid units were ignored on load
            cube.units = "m"
            fix_hourly_time_coordinate(cube)
            fix_accumulated_units(cube)
            multiply_with_density(cube)

        return cubes


class Prw(Fix):
    """Fixes for prw."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            cube.units = "kg m-2"
        return cubes


class Ps(Fix):
    """Fixes for ps."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            cube.units = "Pa"
        return cubes


class Ptype(Fix):
    """Fixes for ptype."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            cube.units = 1

        return cubes


class Rainmxrat27(Fix):
    """Fixes for rainmxrat27."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            cube.units = "kg kg-1"
        return cubes


class Rlds(Fix):
    """Fixes for Rlds."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            fix_hourly_time_coordinate(cube)
            fix_accumulated_units(cube)
            cube.attributes["positive"] = "down"

        return cubes


class Rlns(Fix):
    """Fixes for Rlns."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            fix_hourly_time_coordinate(cube)
            fix_accumulated_units(cube)
            cube.attributes["positive"] = "down"

        return cubes


class Rls(Fix):
    """Fixes for Rls."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            fix_hourly_time_coordinate(cube)
            cube.attributes["positive"] = "down"

        return cubes


class Rlus(Fix):
    """Fixes for Rlus."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            fix_hourly_time_coordinate(cube)
            fix_accumulated_units(cube)
            cube.attributes["positive"] = "up"

        return cubes


class Rlut(Fix):
    """Fixes for Rlut."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        fixed_cubes = CubeList()
        for cube in cubes:
            cube.attributes["positive"] = "up"
            fixed_cubes.append(-cube)

        return fixed_cubes


class Rlutcs(Fix):
    """Fixes for Rlutcs."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        fixed_cubes = CubeList()
        for cube in cubes:
            cube.attributes["positive"] = "up"
            fixed_cubes.append(-cube)

        return fixed_cubes


class Rsds(Fix):
    """Fixes for Rsds."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            fix_hourly_time_coordinate(cube)
            fix_accumulated_units(cube)
            cube.attributes["positive"] = "down"

        return cubes


class Rsns(Fix):
    """Fixes for Rsns."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            fix_hourly_time_coordinate(cube)
            fix_accumulated_units(cube)
            cube.attributes["positive"] = "down"

        return cubes


class Rsus(Fix):
    """Fixes for Rsus."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            fix_hourly_time_coordinate(cube)
            fix_accumulated_units(cube)
            cube.attributes["positive"] = "up"

        return cubes


class Rsdt(Fix):
    """Fixes for Rsdt."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            fix_hourly_time_coordinate(cube)
            fix_accumulated_units(cube)
            cube.attributes["positive"] = "down"

        return cubes


class Rss(Fix):
    """Fixes for Rss."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            fix_hourly_time_coordinate(cube)
            fix_accumulated_units(cube)
            cube.attributes["positive"] = "down"

        return cubes


class Sftlf(Fix):
    """Fixes for sftlf."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            # Invalid input cube units (ignored on load) were '0-1'
            cube.units = "1"
        return cubes


class Snowmxrat27(Fix):
    """Fixes for snowmxrat27."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            cube.units = "kg kg-1"
        return cubes


class Tasmax(Fix):
    """Fixes for tasmax."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            fix_hourly_time_coordinate(cube)
        return cubes


class Tasmin(Fix):
    """Fixes for tasmin."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            fix_hourly_time_coordinate(cube)
        return cubes


class Toz(Fix):
    """Fixes for toz."""

    def fix_metadata(self, cubes):
        """Convert 'kg m-2' to 'm'."""
        for cube in cubes:
            # Original units are kg m-2. Convert these to m here.
            # 1 DU = 0.4462 mmol m-2 = 21.415 mg m-2 = 2.1415e-5 kg m-2
            # (assuming O3 molar mass of 48 g mol-1)
            # Since 1 mm of pure O3 layer is defined as 100 DU
            # --> 1m ~ 2.1415 kg m-2
            cube.data = cube.core_data() / 2.1415
            cube.units = "m"
        return cubes


class Zg(Fix):
    """Fixes for Geopotential."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            divide_by_gravity(cube)
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

        self._fix_monthly_time_coord(cube)

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
    def _fix_monthly_time_coord(cube):
        """Set the monthly time coordinates to the middle of the month."""
        if get_frequency(cube) == "monthly":
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
