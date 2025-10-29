"""Derivation of variable `hurs`."""

import cf_units
import dask.array as da
import iris
from iris import NameConstraint

from ._baseclass import DerivedVariableBase

# Constants
GAS_CONSTANT_WV = 461.5  # JK-1kg-1
ENTALPY_OF_VAPORIZATION = 2.501e6  # Jkg-1


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `hurs`."""

    @staticmethod
    def required(project):  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [
            {"short_name": "tdps"},
            {"short_name": "tas"},
        ]

    @staticmethod
    def calculate(cubes):
        """Compute relative humidity.

        Relative humidity computed from dewpoint temperature and
        surface air temperature following Bohren and Albrecht 1998.
        """
        tdps_cube = cubes.extract_cube(NameConstraint(var_name="tdps"))
        tas_cube = cubes.extract_cube(NameConstraint(var_name="tas"))

        cubes_difference = tas_cube - tdps_cube
        cubes_product = tas_cube * tdps_cube

        log_humidity_cube = (
            -ENTALPY_OF_VAPORIZATION
            * cubes_difference
            / (GAS_CONSTANT_WV * cubes_product)
        )

        hurs_cube = 100 * iris.analysis.maths.exp(log_humidity_cube)

        hurs_cube.units = cf_units.Unit("%")

        hurs_cube.data = da.ma.where(
            hurs_cube.core_data() > 100.0,
            100.0,
            hurs_cube.core_data(),
        )

        return hurs_cube
