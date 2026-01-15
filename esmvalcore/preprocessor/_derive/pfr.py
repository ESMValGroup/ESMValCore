"""Derivation of variable `pfr`."""

import logging

import dask.array as da
import iris
import numpy as np
from iris import NameConstraint
from iris.time import PartialDateTime

from ._baseclass import DerivedVariableBase

logger = logging.getLogger(__name__)

# Constants
THRESH_TEMPERATURE = 273.15
FROZEN_MONTHS = 24  # valid range: 12-36


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
          - soil temperature in the deepest level is < 0°C
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
        delta_x_max = np.amax(x_coord1.core_points() - x_coord2.core_points())
        if delta_x_max != 0.0:
            if abs(delta_x_max) < 1.0e-4:
                x_coord2.points = x_coord1.points
                x_coord2.bounds = x_coord1.bounds
            else:
                logger.error(
                    "Longitudes of mrsos and stflf fields differ (max = %f).",
                    delta_x_max,
                )

        # fix latitudes if maximum differences are smaller than 1.0e-4
        y_coord1 = mrsos.coord(axis="Y")
        y_coord2 = landfrac.coord(axis="Y")
        delta_y_max = np.amax(y_coord1.core_points() - y_coord2.core_points())
        if delta_y_max != 0.0:
            if abs(delta_y_max) < 1.0e-4:
                y_coord2.points = y_coord1.points
                y_coord2.bounds = y_coord1.bounds
            else:
                logger.error(
                    "Latitudes of mrsos and stflf fields differ (max = %f).",
                    delta_y_max,
                )

        mask = mrsos_yr * landfrac

        # extract deepest soil level
        soiltemp = cubes.extract_cube(NameConstraint(var_name="tsl"))
        z_coord = soiltemp.coord(axis="Z")
        zmax = np.max(z_coord.core_points())
        soiltemp = soiltemp.extract(iris.Constraint(depth=zmax))
        soiltemp.data = da.where(
            soiltemp.core_data() < THRESH_TEMPERATURE,
            1,
            0,
        )
        iris.coord_categorisation.add_year(soiltemp, "time")

        # prepare cube for permafrost extent with yearly time steps
        pfr_yr = soiltemp.aggregated_by(["year"], iris.analysis.MEAN)
        # get years to process
        year_coord = pfr_yr.coord("year")
        # calculate time period before and after current year to include
        # in test for permafrost
        test_period = (FROZEN_MONTHS - 12) / 2
        # loop over all years and test if frost is present throughout
        # the whole test period, i.e. [year-test_period, year+test_period]

        for tidx, year in enumerate(year_coord.points):
            # extract test period
            pdt1 = PartialDateTime(
                year=year - 1,
                month=13 - test_period,
                day=1,
            )
            pdt2 = PartialDateTime(year=year + 1, month=test_period + 1, day=1)
            daterange = iris.Constraint(
                time=lambda cell, pdt1=pdt1, pdt2=pdt2: pdt1
                <= cell.point
                < pdt2,
            )
            soiltemp_window = soiltemp.extract(daterange)
            # remove auxiliary coordinate 'year' to avoid lots of warnings
            # from iris
            soiltemp_window.remove_coord("year")
            # calculate mean over test period
            test_cube = soiltemp_window.collapsed("time", iris.analysis.MEAN)
            # if all months in test period show soil tempeatures below zero
            # then mark grid cell with "1" as permafrost and "0" otherwise
            pfr_yr.data[tidx, :, :] = da.where(test_cube.data > 0.99, 1, 0)

        pfr_yr = pfr_yr * mask
        pfr_yr.units = "%"
        pfr_yr.rename("Permafrost extent")
        pfr_yr.var_name = "pfr"

        return pfr_yr
