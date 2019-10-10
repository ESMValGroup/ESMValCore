"""Derivation of variable `vegFrac`."""

import iris
from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `vegFrac`."""

    # Required variables
    required = [{'short_name': 'baresoilFrac', }]

    @staticmethod
    def calculate(cubes):
        """Compute vegetation fraction from bare soil fraction.

        """
        baresoilFrac_cube = cubes.extract_strict(
            iris.Constraint(name='area_fraction'))

        baresoilFrac_cube.data = 1. - baresoilFrac_cube.data
        return baresoilFrac_cube
