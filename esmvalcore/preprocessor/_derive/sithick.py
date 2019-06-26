"""Derivation of variable `sithick`."""

from iris import Constraint
from iris.coords import DimCoord

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `sithick`."""

    # Required variables
    required = [
        {'short_name': 'sit', },
        {'short_name': 'sic', }
    ]

    @staticmethod
    def calculate(cubes):
        """
        Compute sea ice thickness from volume and concentration

        Arguments
        ----
            cubes: cubelist containing volume and concentration components.

        Returns
        -------
            Cube containing sea ice speed.

        """
        siconc = cubes.extract_strict(Constraint(name='sea_ice_thickness'))
        siconc.remove_coord('year')
        siconc.remove_coord('day_of_year')

        sivol = cubes.extract_strict(Constraint(name='sea_ice_y_velocity'))
        sivol.remove_coord('year')
        sivol.remove_coord('day_of_year')

        sithick = siconc * sivol
        return sithick
