"""Derivation of variable `vegFrac`."""

import iris
from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `vegFrac`."""

    required = [{
        'short_name': 'baresoilFrac',
    }]

    @staticmethod
    def calculate(cubes):
        """Compute vegetation fraction from bare soil fraction."""
        baresoilfrac_cube = cubes.extract_strict(
            iris.Constraint(name='area_fraction'))

        baresoilfrac_cube.data = 1. - baresoilfrac_cube.core_data()
        return baresoilfrac_cube
