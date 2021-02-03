"""Derivation of variable `sithick`."""

from iris import Constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `sithick`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [{
            'short_name': 'sit',
        }, {
            'short_name': 'sic',
        }]
        return required

    @staticmethod
    def calculate(cubes):
        """
        Compute sea ice thickness from volume and concentration.

        In CMIP5, `sit` is called `sea_ice_thickness` but it is not real
        thickness. It is ice volume per area unit. In CMIP6, it is called
        `sivol` and the real thickness is called `sithick`

        Arguments
        ---------
            cubes: cubelist containing volume and concentration components.

        Returns
        -------
            Cube containing sea ice speed.

        """
        sivol = cubes.extract_cube(Constraint(name='sea_ice_thickness'))
        siconc = cubes.extract_cube(Constraint(name='sea_ice_area_fraction'))
        siconc.convert_units('1.0')

        sithick = sivol / siconc
        return sithick
