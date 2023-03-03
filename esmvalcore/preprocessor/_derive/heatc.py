"""Derivation of variable `sithick`."""
import logging

import iris
from cf_units import Unit
from iris import Constraint

from ._baseclass import DerivedVariableBase

logger = logging.getLogger(__name__)

DENSITY = iris.coords.AuxCoord(1025, units=Unit('kg m-3'))
HEAT_CAPACITY = iris.coords.AuxCoord(3850, units=Unit('J kg K-1'))

class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `siextent`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                'short_name': 'thetao',
            },]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute sea ice extent.

        Returns the sea water potential temperature multiplied by the
        sea water specific heat.

        Use in combination with the preprocessor `extract_volume` and
        `axis_statistics(axis='Z', operator='sum')` to obtain the global values
        over a given watercolumn. Additionally, use in combination with the
        preprocessor `area_statistics(operator='mean') in order to obtain
        spatially averaged values.

        Arguments
        ---------
            cubes: cubelist containing sea water potential temperature.

        Returns
        -------
            Cube containing heat content.
        """
        
        thetao = cubes.extract_cube(Constraint(standard_name='sea_water_potential_temperature'))
        thetao.convert_units('K')
        heatc = thetao * DENSITY * HEAT_CAPACITY

        return heatc
