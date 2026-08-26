"""Derivation of variable `pfr`."""

import logging

import cftime
import dask.array as da
import iris
import numpy as np
from iris import NameConstraint
from iris.time import PartialDateTime

from esmvalcore.preprocessor._rolling_window import rolling_window_statistics

from ._baseclass import DerivedVariableBase

logger = logging.getLogger(__name__)

# Constants
THRESH_TEMPERATURE = 273.15
FROZEN_YEARS = 2


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `pfr` (permafrost extent)."""

    @staticmethod
    def required(project):  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [
            {"short_name": "tsl", "mip": "Lmon"},
            {"short_name": "sftlf", "mip": "fx"},
            {"short_name": "mrsos", "mip": "Lmon"},
        ]

    @staticmethod
    def calculate(cubes):
        """Compute permafrost extent.

        Permafrost is assumed if
          - soil temperature at depth=2m is < 0°C
            Note: In Burke et al., soil temperature < 0°C in the deepest
                  level is used. This has been modified to depth=2m to be
                  consistent with ESACCI-PERMAFROST data.
          - for at least 24 consecutive months
          - ice covered part of grid cell is excluded
        Reference: Burke, E. J., Y. Zhang, and G. Krinner:
        Evaluating permafrost physics in the Coupled Model
        Intercomparison Project 6 (CMIP6) models and their
        sensitivity to climate change, The Cryosphere, 14,
        3155-3174, doi: 10.5194/tc-14-3155-2020, 2020.
        """
        # create a mask of land fraction (%) over ice-free grid cells
        # use soil moisture as proxy for ice / ice-free grid cells
        #   1) annual mean of fraction of grid cell covered with ice (%)
        #      assumption: top soil moisture = 0 --> ice covered
        mrsos = cubes.extract_cube(NameConstraint(var_name="mrsos"))
        iris.coord_categorisation.add_year(mrsos, "time")
        mrsos_yr = mrsos.aggregated_by(["year"], iris.analysis.MEAN)
        mrsos_yr.data = da.where(mrsos_yr.core_data() < 0.001, 0.0, 1.0)
        #   2) fraction of land cover of grid cell (%) (constant)
        landfrac = cubes.extract_cube(NameConstraint(var_name="sftlf"))
        #   3) create mask with fraction of ice-free land (%)

        # latitude/longitude coordinates of mrsos and sftlf sometimes
        # differ by a very small amount for some models (probably because
        # of rounding errors) preventing iris to do the math
        # --> overwrite latitudes/longitudes in sftlf

        # fix longitudes if maximum differences are smaller than 1.0e-4
        x_coord1 = mrsos.coord(axis="X")
        x_coord2 = landfrac.coord(axis="X")
        if np.allclose(
            x_coord1.core_points(),
            x_coord2.core_points(),
            atol=1.0e-4,
        ):
            x_coord2.points = x_coord1.points
            x_coord2.bounds = x_coord1.bounds
        else:
            logger.error(
                "Longitudes of mrsos and stflf fields differ more than 1e-4 degrees.",
            )

        # fix latitudes if maximum differences are smaller than 1.0e-4
        y_coord1 = mrsos.coord(axis="Y")
        y_coord2 = landfrac.coord(axis="Y")
        if np.allclose(
            y_coord1.core_points(),
            y_coord2.core_points(),
            atol=1.0e-4,
        ):
            y_coord2.points = y_coord1.points
            y_coord2.bounds = y_coord1.bounds
        else:
            logger.error(
                "Latitudes of mrsos and stflf fields differ more than 1e-4 degrees.",
            )

        mask = iris.analysis.maths.multiply(mrsos_yr, landfrac)

        # extract soil temperature at depth=2m
        soiltemp = cubes.extract_cube(NameConstraint(var_name="tsl"))
        z_coord = soiltemp.coord(axis="Z")
        soiltemp = soiltemp.interpolate(
            [(z_coord.standard_name, 2.0)],
            scheme=iris.analysis.Linear(),
        )
        # create mask (0 = soil temperature >= 0°C, 1 = soil temperature < 0°C)
        soiltemp.data = da.where(
            soiltemp.core_data() < THRESH_TEMPERATURE,
            1,
            0,
        )

        iris.coord_categorisation.add_year(soiltemp, "time")
        # prepare cube for permafrost extent with yearly time steps
        pfr_yr = soiltemp.aggregated_by(["year"], iris.analysis.MEAN)

        # create cube containing a copy of the first year
        # but with a modified time coordinate saying "year-1"

        # get first year
        year = pfr_yr.coord("year").points[0]
        # create Iris constraint to select first year from time series
        pdt1 = PartialDateTime(year=year, month=1, day=1)
        pdt2 = PartialDateTime(year=year + 1, month=1, day=1)
        yr_range = iris.Constraint(
            time=lambda cell, pdt1=pdt1, pdt2=pdt2: pdt1 <= cell.point < pdt2,
        )
        first_yr = pfr_yr.extract(yr_range)
        aux_coord = first_yr.coord("time")
        # promote time coordinate to axis for concatenation
        first_yr = iris.util.new_axis(first_yr, aux_coord)
        time_coord = first_yr.coord("time")
        # shift time coordinate by one -1 year
        dtime = time_coord.units.num2date(time_coord.points)[0]
        shifted_datetime = cftime.datetime(
            dtime.year - 1,
            dtime.month,
            dtime.day,
            dtime.hour,
            dtime.minute,
            dtime.second,
            calendar=time_coord.units.calendar,
        )
        time_coord.points = np.asarray(
            time_coord.units.date2num(shifted_datetime),
            dtype=np.float64,
        )
        # update time bounds accordingly
        bnds = time_coord.units.num2date(time_coord.bounds)
        shifted_bnds = [
            (
                dt[0].replace(year=dt[0].year - 1),
                dt[1].replace(year=dt[1].year - 1),
            )
            for dt in bnds
        ]
        time_coord.bounds = time_coord.units.date2num(shifted_bnds).astype(
            np.float64,
        )
        # also update aux_coordinate "year"
        first_yr.remove_coord("year")
        iris.coord_categorisation.add_year(first_yr, "time")

        # now concatenate cube with time shifted by -1 years
        # and the original yearly time series
        new_cube = iris.cube.CubeList([first_yr, pfr_yr]).concatenate_cube()

        # calculate rolling window statistics on cube with yearly time steps
        # window length = 2 --> 24 months
        pfr_rws = rolling_window_statistics(
            new_cube,
            coordinate="time",
            operator="mean",
            window_length=FROZEN_YEARS,
        )

        # The window (length = 2) of the "rollowing window statistics"
        # for time step t uses the period [t, t+1]. As we inserted a
        # copy of the first year as new first time step in "new_cube",
        # we get:
        # pfr_rws(0) = mean(year_1, year_1),
        # pfr_rws(1) = mean(year_1, year_2),
        # pfr_rws(2) = mean(year_2, year_3)
        # pfr_rws(n-1) = mean(year_n-2, year_n-1)
        # Out of n time steps, "rolling window statistics" with a window
        # length of 2 returns n-1 new time steps. This is the same number
        # of time steps contained in the original cube "pfr_yr".

        # pfr_yr(t) = 1 --> T_soil <= 0°C during all of year(t-1) and year(t)
        #           = 0 --> T_soil was > 0°C for at least one month in the
        #                   two year time period
        pfr_yr.data = da.where(pfr_rws.core_data() > 0.99, 1, 0)

        # mask out glaciated grid cells
        pfr_yr = pfr_yr * mask
        # update metadata
        pfr_yr.units = "%"
        pfr_yr.rename("Permafrost extent")
        pfr_yr.var_name = "pfr"

        return pfr_yr
