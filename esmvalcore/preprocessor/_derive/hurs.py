"""Derivation of variable `hurs`."""

from iris import NameConstraint
import iris
import numpy as np
import cf_units

from ._baseclass import DerivedVariableBase

# Constants
GAS_CONSTANT_WV = 461.5 # JK-1kg-1
ENTALPY_OF_VAPORIZATION = 2.501e6 # Jkg-1

class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `hurs`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {'short_name': 'tdps'},
            {'short_name': 'tas'},]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute relative humidity.

        Relative humidity computed from dewpoint temperature and 
        surface air temperature following Bohren and Albrecht 1998.
        """
        tdps_cube = cubes.extract_cube(NameConstraint(var_name='tdps'))
        tas_cube = cubes.extract_cube(NameConstraint(var_name='tas'))

        cubes_difference = iris.analysis.maths.subtract(tas_cube, tdps_cube) 
        cubes_product = iris.analysis.maths.multiply(tas_cube, tdps_cube)

        log_humidity_cube = iris.analysis.maths.divide(-ENTALPY_OF_VAPORIZATION * cubes_difference,
                                                        GAS_CONSTANT_WV * cubes_product) 

        hurs_cube = 100 * iris.analysis.maths.exp(log_humidity_cube)

        hurs_cube.units = cf_units.Unit('%')

        return hurs_cube