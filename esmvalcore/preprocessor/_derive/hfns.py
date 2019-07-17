"""Derivation of variable `hfns`."""

from iris import Constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `hfns`."""

    # Required variables
    required = [
        {
            'short_name': 'hfls'
        },
        {
            'short_name': 'hfss'
        },
    ]

    @staticmethod
    def calculate(cubes):
        """Compute surface net heat flux."""
        hfls_cube = cubes.extract_strict(
            Constraint(name='surface_upward_latent_heat_flux'))
        hfss_cube = cubes.extract_strict(
            Constraint(name='surface_upward_sensible_heat_flux'))

        hfns_cube = hfls_cube + hfss_cube

        return hfns_cube
