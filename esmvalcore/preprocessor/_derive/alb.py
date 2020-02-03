"""Derivation of variable `alb`.

authors:
    - crez_ba

"""

from ._baseclass import DerivedVariableBase
from ._shared import _var_name_constraint


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `alb`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                'short_name': 'rsdscs'
            },
            {
                'short_name': 'rsuscs'
            },
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute surface albedo."""
        rsdscs_cube = cubes.extract_strict(_var_name_constraint('rsdscs'))
        rsuscs_cube = cubes.extract_strict(_var_name_constraint('rsuscs'))

        rsnscs_cube = rsuscs_cube / rsdscs_cube

        return rsnscs_cube
