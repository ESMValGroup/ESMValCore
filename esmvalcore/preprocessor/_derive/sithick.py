"""Derivation of variable `sithick`."""

from iris import Constraint

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
        sivol = cubes.extract_strict(Constraint(name='sea_ice_thickness'))
        siconc = cubes.extract_strict(Constraint(name='sea_ice_area_fraction'))
        siconc.convert_units(1.0)

        sithick = siconc * sivol
        return sithick
