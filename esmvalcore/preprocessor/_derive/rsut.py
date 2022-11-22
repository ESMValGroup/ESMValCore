"""Derivation of variable `rsut`.

Calculate TOA outgoing shortwave flux (rsut) from
- TOA incoming shortwave flux (rsdt) and
- TOA net upward solar radiation (ERA5: rsnut) as
rsut = rsdt - rsnut

authors:
    - axel_lauer

"""
from iris import NameConstraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `rsut` from ERA5 variables."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                'short_name': 'rsdt'
            },
            {
                'short_name': 'rsnut'
            },
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute outgoing shortwave flux from incoming and net."""
        rsdt_cube = cubes.extract_cube(NameConstraint(var_name='rsdt'))
        rsnut_cube = cubes.extract_cube(NameConstraint(var_name='rsnut'))

        rsut_cube = rsdt_cube.copy()
        rsut_cube.data = rsdt_cube.data - rsnut_cube.data
        rsut_cube.attributes['positive'] = 'up'

        return rsut_cube
