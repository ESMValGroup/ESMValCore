"""Derivation of variable `vegFrac`."""

import iris
from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `vegFrac`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [{
            'short_name': 'baresoilFrac',
        }]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute vegetation fraction from bare soil fraction."""
        baresoilfrac_cube = cubes.extract_strict(
            iris.Constraint(name='area_fraction'))

        baresoilfrac_cube.data = 1. - baresoilfrac_cube.core_data()
        return baresoilfrac_cube
