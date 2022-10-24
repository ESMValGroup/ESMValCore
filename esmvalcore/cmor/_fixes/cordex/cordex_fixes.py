"""Fixes that are shared between datasets and drivers."""
from cf_units import Unit
import numpy as np

from esmvalcore.cmor.fix import Fix


class MOHCHadREM3GA705(Fix):
    """General fix for MOHC-HadREM3-GA7-05."""

    def fix_metadata(self, cubes):
        """Fix time long_name, and latitude and longitude var_name.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            cube.coord('latitude').var_name = 'lat'
            cube.coord('longitude').var_name = 'lon'
            cube.coord('time').long_name = 'time'

        return cubes


class TimeLongName(Fix):
    """Fixes for time coordinate."""

    def fix_metadata(self, cubes):
        """
        Fix time long_name.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            cube.coord('time').long_name = 'time'

        return cubes


class CLMcomCCLM4817(Fix):
    """Fixes for CLMcom-CCLM4-8-17."""

    def fix_metadata(self, cubes):
        """ Fix calendars.

        Set calendar to 'proleptic_gregorian' to avoid
        concatenation issues between historical and
        scenario runs.

        Fix dtype value of coordinates and coordinate bounds.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            if cube.coord('time').units.calendar == 'gregorian':
                cube.coord('time').units = Unit(
                    'days since 1850-1-1 00:00:00',
                    calendar='proleptic_gregorian'
                )
            for coord in cube.coords():
                if coord.dtype in ['>f8', '>f4']:
                    coord.points = coord.core_points().astype(
                        np.float64, casting='same_kind')
                    if coord.bounds is not None:
                        coord.bounds = coord.core_bounds().astype(
                            np.float64, casting='same_kind')
        # Further issues appear, maybe because historical data lat/lon
        # does not have bounds whereas scenario data has them.
        return cubes
