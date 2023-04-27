"""Derivation of variable `heatc`."""
import logging

import iris
from cf_units import Unit
from iris import Constraint

from ._baseclass import DerivedVariableBase
from .._supplementary_vars import add_supplementary_variables

logger = logging.getLogger(__name__)

DENSITY = iris.coords.AuxCoord(1025, units=Unit('kg m-3'))
HEAT_CAPACITY = iris.coords.AuxCoord(3985, units=Unit('J kg-1 K-1'))


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `heatc`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                'short_name': 'thetao',
            }, ]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute ocean heat content.

        Returns the sea water potential temperature multiplied by the
        sea water specific heat.

        Use in combination with the preprocessor `extract_volume` and
        `axis_statistics(axis='Z', operator='sum')` to obtain the global values
        over a given water column. Additionally, use in combination with the
        preprocessor `area_statistics(operator='mean') in order to obtain
        spatially averaged values.

        Arguments
        ---------
            cubes: cubelist containing sea water potential temperature.

        Returns
        -------
            Cube containing heat content.
        """

        thetao = cubes.extract_cube(
            Constraint(name='sea_water_potential_temperature')
        )
        thetao.convert_units('K')
        heatc = thetao * DENSITY * HEAT_CAPACITY
        heatc.convert_units('J m-3')
        if thetao.cell_measures():
            add_supplementary_variables(heatc, thetao.cell_measures())
        if thetao.ancillary_variables():
            add_supplementary_variables(heatc, thetao.ancillary_variables())

        return heatc
