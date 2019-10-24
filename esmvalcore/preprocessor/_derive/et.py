"""Derivation of variable `et`."""

import cf_units
from iris import Constraint

from ._baseclass import DerivedVariableBase

# Constants
LATENT_HEAT_VAPORIZATION = 2.465E6


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `et`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [{'short_name': 'hfls', 'mip': 'Amon'}]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute evapotranspiration."""
        hfls_cube = cubes.extract_strict(
            Constraint(name='surface_upward_latent_heat_flux'))

        et_cube = hfls_cube * 24.0 * 3600.0 / LATENT_HEAT_VAPORIZATION
        et_cube.units = cf_units.Unit('mm day-1')

        return et_cube
