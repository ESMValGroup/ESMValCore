"""Derivation of variable `cdd22`."""

import logging

import dask.array as da
import iris
import numpy as np
from iris import NameConstraint

from ._baseclass import DerivedVariableBase

logger = logging.getLogger(__name__)

# Constants
BASE_TEMPERATURE = 22.0  # degC


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `cdd22` (cooling degree days)."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        if project == "CMIP7":
            return [
                {
                    "short_name": "tas",
                    "branding_suffix": "tavg-h2m-hxy-u",
                    "mip": "atmos",
                },
            ]
        return [
            {"short_name": "tas", "mip": "day"},
        ]

    @staticmethod
    def calculate(cubes):
        """Compute cooling degree days."""

        tas_cube = cubes.extract_cube(NameConstraint(var_name="tas"))
        tas_cube.convert_units("degC")
        
        # Calculate daily CDD: max(0, tas - 22.0)
        tas_cube.data = da.where(tas_cube.core_data() > BASE_TEMPERATURE, tas_cube.core_data() - BASE_TEMPERATURE, 0.0)
#        daily_data = np.maximum(0.0, tas_cube.core_data() - BASE_TEMPERATURE)
#        daily_cdd = tas_cube.copy(data=daily_data)
#
        # Aggregate daily values into annual sums
#        add_year(daily_cdd, "time", name="year")
#        annual_cdd = daily_cdd.aggregated_by("year", iris.analysis.SUM)
        iris.coord_categorisation.add_year(tas_cube, "time", name="year")
        annual_cdd = tas_cube.aggregated_by("year", iris.analysis.SUM)

        # Clean up
        annual_cdd.remove_coord("year")
        annual_cdd.units = "degC day"
        annual_cdd.standard_name = None
        annual_cdd.long_name = f"Annual Cooling Degree Days (base {BASE_TEMPERATURE} degC)"

        return annual_cdd
