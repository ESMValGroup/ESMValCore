"""Derivation of variable `rsutcs`.

Calculate TOA outgoing clear-sky shortwave flux (rsutcs) from
- TOA incoming shortwave flux (rsdt) and
- TOA net upward solar clear-sky radiation (ERA5: rsnutcs) as
rsut = rsdt - rsnutcs

authors:
    - axel_lauer

"""
from iris import NameConstraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `rsutcs` from ERA5 variables."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                'short_name': 'rsdt'
            },
            {
                'short_name': 'rsnutcs'
            },
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute outgoing shortwave flux from incoming and net."""
        rsdt_cube = cubes.extract_cube(NameConstraint(var_name='rsdt'))
        rsnutcs_cube = cubes.extract_cube(NameConstraint(var_name='rsnutcs'))

        rsutcs_cube = rsdt_cube.copy()
        rsutcs_cube.data = rsdt_cube.data - rsnutcs_cube.data
        rsutcs_cube.attributes['positive'] = 'up'

        return rsutcs_cube
