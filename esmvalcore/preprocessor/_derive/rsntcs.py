"""Derivation of variable `rsntcs`."""
from iris import Constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `rsnt`."""

    # Required variables
    required = [
        {
            'short_name': 'rsdt'
        },
        {
            'short_name': 'rsutcs'
        },
    ]

    @staticmethod
    def calculate(cubes):
        """Compute toa net downward shortwave radiation assuming clear sky."""
        rsdt_cube = cubes.extract_strict(
            Constraint(name='toa_incoming_shortwave_flux'))
        rsutcs_cube = cubes.extract_strict(
            Constraint(name='toa_outgoing_shortwave_flux_assuming_clear_sky'))
        rsntcs_cube = rsdt_cube - rsutcs_cube
        rsntcs_cube.attributes['positive'] = 'down'
        return rsntcs_cube
