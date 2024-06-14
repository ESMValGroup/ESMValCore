"""Derivation of variable `pfr`."""

import iris
import numpy as np
from iris import NameConstraint
from iris.time import PartialDateTime
import dask.array as da

from ._baseclass import DerivedVariableBase

# Constants
THRESH_TEMPERATURE = 273.15
FROZEN_MONTHS = 24  # valid range: 12-36


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `pfr` (permafrost extent)."""
    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [{'short_name': 'tsl', 'mip': 'Lmon'},
                    {'short_name': 'sftlf', 'mip': 'fx'},
                    {'short_name': 'sftgif', 'mip': 'LImon'}]
        return required

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
        #   1) annual mean of fraction of grid cell covered with ice (%)
        icefrac = cubes.extract_cube(NameConstraint(var_name='sftgif'))
        iris.coord_categorisation.add_year(icefrac, 'time')
        icefrac_yr = icefrac.aggregated_by(['year'], iris.analysis.MEAN)
        #   2) fraction of land cover of grid cell (%) (constant)
        landfrac = cubes.extract_cube(NameConstraint(var_name='sftlf'))
        #   3) create mask with fraction of ice-free land (%)
        mask = iris.analysis.maths.subtract(landfrac, icefrac_yr)
        # remove slightly negative values that might occur because of
        # rounding errors between ice and land fractions
        mask.data = da.where(mask.data < 0.0, 0.0, mask.data)

        # extract deepest soil level
        soiltemp = cubes.extract_cube(NameConstraint(var_name='tsl'))
        z_coord = soiltemp.coord(axis='Z')
        zmax = np.amax(z_coord.core_points())
        soiltemp = soiltemp.extract(iris.Constraint(depth=zmax))
        soiltemp.data = da.where(soiltemp.data < THRESH_TEMPERATURE, 1, 0)
        iris.coord_categorisation.add_year(soiltemp, 'time')

        # prepare cube for permafrost extent with yearly time steps
        pfr_yr = soiltemp.aggregated_by(['year'], iris.analysis.MEAN)
        # get years to process
        year_coord = pfr_yr.coord('year')
        # calculate time period before and after current year to include
        # in test for permafrost
        test_period = (FROZEN_MONTHS - 12) / 2
        # loop over all years and test if frost is present throughout
        # the whole test period, i.e. [year-test_period, year+test_period]
        tidx = 0

        for year in year_coord.points:
            # extract test period
            pdt1 = PartialDateTime(year=year-1, month=13-test_period, day=1)
            pdt2 = PartialDateTime(year=year+1, month=test_period+1, day=1)
            daterange = iris.Constraint(
                time=lambda cell: pdt1 <= cell.point < pdt2)
            soiltemp_window = soiltemp.extract(daterange)
            # remove auxiliary coordinate 'year' to avoid lots of warnings
            # from iris
            soiltemp_window.remove_coord('year')
            # calculate mean over test period
            test_cube = soiltemp_window.collapsed('time', iris.analysis.MEAN)
            # if all months in test period show soil tempeatures below zero
            # then mark grid cell with "1" as permafrost and "0" otherwise
            pfr_yr.data[tidx, :, :] = da.where(test_cube.data > 0.99, 1, 0)
            tidx += 1

        pfr_yr = pfr_yr * mask
        pfr_yr.units = "%"
        pfr_yr.rename('Permafrost extent')
        pfr_yr.var_name = "pfr"

        return pfr_yr
